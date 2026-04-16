"""Stage 1: Detection — call model servers, filter, dedup, route."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from ..checkpoint import CheckpointManager
from ..config import AutoAnnotationV3Config, ClassConfig, DetectorConfig
from ..contracts import (
    BoundingBox,
    Candidate,
    CandidateStatus,
    DetectorName,
    DetectorRequest,
    DetectorResponse,
    DetectResult,
    DetectRouting,
    FinalAction,
    FinalAnnotation,
    ProposalResult,
    StageMessage,
)
from ..output import OutputWriter
from ..utils import (
    apply_cross_class_rules,
    build_class_alias_map,
    cluster_and_collapse,
    filter_by_model_score,
    geometric_filter,
    get_image_size,
    limit_per_class,
    route_candidates,
)
from ..workers.base import StageWorker

logger = logging.getLogger("data_miner.auto_annotation_v3.detect")


class DetectWorker(StageWorker):
    """Stage 1: call detection model servers, merge, filter, dedup, and route."""

    stage = "detect"

    def __init__(
        self,
        config: AutoAnnotationV3Config,
        broker: Any,
        checkpoint_mgr: CheckpointManager,
        output_writer: OutputWriter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, broker, checkpoint_mgr, **kwargs)
        self.output_writer = output_writer
        # Build alias map once — maps lowercased aliases → canonical class name.
        self._alias_map: dict[str, str] = build_class_alias_map(config.classes)

    # ------------------------------------------------------------------
    # Core process method (called by StageWorker.run loop)
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> StageMessage | None:
        image_path = msg.image_path
        image_w, image_h = get_image_size(image_path)

        # 1. Call all detection servers in parallel, respecting per-model cache.
        model_results: dict[str, list[Candidate]] = await self._run_all_detectors(
            msg.image_id, image_path
        )

        if not any(model_results.values()):
            self.logger.info(
                "No detections from any model for %s — forwarding to done",
                msg.image_id,
            )
            detect_result = self._build_detect_result(
                msg.image_id,
                image_path,
                image_w,
                image_h,
                model_results,
                candidates=[],
                routing_result={
                    "auto_accepted": [],
                    "needs_evaluation": [],
                    "confusion_flags": [],
                },
            )
            self.save_checkpoint(msg.image_id, "detect", detect_result)
            if self.output_writer:
                self._write_auto_accepted_output(msg.image_id, detect_result)
            return msg.forward("finalize")

        # 2. Merge all per-model candidates.
        all_candidates: list[Candidate] = []
        for candidates in model_results.values():
            all_candidates.extend(candidates)

        total_proposed = len(all_candidates)

        # 3. Geometric filtering (area, aspect ratio, edge distance).
        filtered = geometric_filter(all_candidates, self.config)
        after_geometric = len(filtered)

        # 3.5. Per-model score floor: drop candidates below their model's
        #      calibrated threshold before they can influence dedup or
        #      cross-class suppression.
        filtered = filter_by_model_score(
            filtered, self.config.filtering.per_model_score
        )
        after_model_score = len(filtered)

        # 4. Per-class cluster-and-collapse: groups detections of the same
        #    class across all models by IoU>=threshold, attaches agreement
        #    metadata, and picks one representative per cluster via the
        #    tiebreak cascade (replaces the old dedup_by_iou + compute_agreement
        #    pair that ran agreement after witnesses had been suppressed).
        deduped = cluster_and_collapse(filtered, self.config.filtering.iou_dedup)
        after_dedup = len(deduped)

        # 5. Per-class cap (keeps top-N by score).
        capped = limit_per_class(deduped, self.config.filtering.max_per_class)
        after_cap = len(capped)

        # 6. Cross-class suppression (suppress lower-score box when non-confused
        #    classes heavily overlap, unless a class is globally exempt).
        cross_filtered = apply_cross_class_rules(capped, self.config)

        # 7. Route: auto_accept (tier-1, high agreement) vs needs_evaluation.
        routing_result = route_candidates(cross_filtered, self.config)

        # 9. Build and checkpoint DetectResult.
        detect_result = self._build_detect_result(
            msg.image_id,
            image_path,
            image_w,
            image_h,
            model_results,
            candidates=cross_filtered,
            routing_result=routing_result,
            filter_stats={
                "total_proposed": total_proposed,
                "after_geometric_filter": after_geometric,
                "after_model_score_filter": after_model_score,
                "after_iou_dedup": after_dedup,
                "after_per_class_cap": after_cap,
                "after_cross_class_rules": len(cross_filtered),
                "auto_accepted": len(routing_result["auto_accepted"]),
                "sent_to_vlm": len(routing_result["needs_evaluation"]),
            },
        )
        self.save_checkpoint(msg.image_id, "detect", detect_result)

        # 10. Forward based on routing.
        if routing_result["needs_evaluation"]:
            self.logger.info(
                "%s: %d auto-accepted, %d sent to evaluate",
                msg.image_id,
                len(routing_result["auto_accepted"]),
                len(routing_result["needs_evaluation"]),
            )
            return msg.forward("evaluate")

        # All surviving candidates were auto-accepted by detect — skip VLM.
        # If any of them belongs to a class with a refine rule, forward to
        # refine (which will adjudicate without an evaluate checkpoint).
        # Otherwise write final output and finish.
        refine_classes = set(self.config.refine_rules.classes.keys())
        auto_accept_ids = set(routing_result["auto_accepted"])
        needs_refine = any(
            c.candidate_id in auto_accept_ids and c.class_name in refine_classes
            for c in cross_filtered
        )
        if needs_refine:
            self.logger.info(
                "%s: %d auto-accepted, forwarding to refine for class-rule extension",
                msg.image_id,
                len(auto_accept_ids),
            )
            return msg.forward("refine")

        self.logger.info(
            "%s: all %d candidates auto-accepted — forwarding to done",
            msg.image_id,
            len(routing_result["auto_accepted"]),
        )
        return msg.forward("finalize")

    # ------------------------------------------------------------------
    # Parallel model calls
    # ------------------------------------------------------------------

    async def _run_all_detectors(
        self,
        image_id: str,
        image_path: str,
    ) -> dict[str, list[Candidate]]:
        """Call every enabled detector uniformly. One HTTP request per
        (image, model); per-prompt strategy lives inside each server."""
        enabled = self.config.servers.enabled_detectors()
        model_results: dict[str, list[Candidate]] = {}
        active_classes = list(self.config.classes)

        # Partition: cached vs pending.
        pending: dict[DetectorName, Any] = {}
        for name, cfg in enabled.items():
            key = name.value
            if self.checkpoint.proposal_exists(image_id, key):
                self.logger.info("Reusing cached proposal %s/%s", image_id, key)
            else:
                pending[name] = cfg

        if pending and active_classes:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = {
                    name: asyncio.create_task(
                        _call_detector(session, name, cfg, image_path, active_classes)
                    )
                    for name, cfg in pending.items()
                }
                for name, task in tasks.items():
                    key = name.value
                    t0 = time.monotonic()
                    try:
                        candidates = await task
                        latency_ms = (time.monotonic() - t0) * 1000
                        self.checkpoint.save_proposal(
                            image_id,
                            key,
                            ProposalResult(
                                model=key,
                                image_id=image_id,
                                image_size=list(get_image_size(image_path)),
                                latency_ms=latency_ms,
                                candidates=candidates,
                            ),
                        )
                        model_results[key] = candidates
                        self.logger.info(
                            "Model %s returned %d candidates in %.0f ms",
                            key,
                            len(candidates),
                            latency_ms,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Model %s failed for %s: %s", key, image_id, exc
                        )
                        model_results[key] = []

        # Load anything cached (skipped in the pending loop above).
        for name in enabled:
            key = name.value
            if key not in model_results:
                cached = self.checkpoint.load_proposal(image_id, key, ProposalResult)
                model_results[key] = cached.candidates if cached is not None else []

        return model_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_detect_result(
        image_id: str,
        image_path: str,
        image_w: int,
        image_h: int,
        model_results: dict[str, list[Candidate]],
        candidates: list[Candidate],
        routing_result: dict,
        filter_stats: dict | None = None,
    ) -> DetectResult:
        return DetectResult(
            image_id=image_id,
            image_path=str(image_path),
            image_size=[image_w, image_h],
            models_used=list(model_results.keys()),
            candidates=candidates,
            routing=DetectRouting(
                auto_accepted=routing_result.get("auto_accepted", []),
                needs_evaluation=routing_result.get("needs_evaluation", []),
                confusion_flags=routing_result.get("confusion_flags", []),
            ),
            filter_stats=filter_stats or {},
            stage_timing_ms=0.0,  # actual timing tracked by StageWorker._handle_message
        )

    # ------------------------------------------------------------------
    # Auto-accept output path (skips VLM entirely)
    # ------------------------------------------------------------------

    def _write_auto_accepted_output(
        self, image_id: str, detect_result: DetectResult
    ) -> None:
        """Write YOLO labels and trace when all candidates are auto-accepted."""
        if self.output_writer is None:
            return

        auto_accepted_ids: set[str] = set(detect_result.routing.auto_accepted)
        class_map: dict[str, int] = {c.name: c.id for c in self.config.classes}

        annotations: list[FinalAnnotation] = []
        for cand in detect_result.candidates:
            if cand.candidate_id not in auto_accepted_ids:
                continue
            annotations.append(
                FinalAnnotation(
                    candidate_id=cand.candidate_id,
                    class_name=cand.class_name,
                    class_id=class_map.get(cand.class_name, -1),
                    bbox=cand.bbox,
                    confidence=cand.score,
                    action=FinalAction.ACCEPT,
                    source_model=cand.source_model,
                    was_refined=False,
                    trace=[
                        f"auto_accepted: agreement={cand.agreement}, "
                        f"score={cand.score:.4f}, "
                        f"agreeing_models={cand.agreeing_models}"
                    ],
                )
            )

        self.output_writer.write_yolo_labels(image_id, annotations, class_map)
        self.output_writer.write_trace(
            image_id,
            {
                "image_id": image_id,
                "stages": ["detect"],
                "detect": detect_result.model_dump(mode="json"),
                "annotations": [a.model_dump(mode="json") for a in annotations],
            },
        )
        self.logger.info(
            "Wrote %d auto-accepted annotations for %s",
            len(annotations),
            image_id,
        )


# ---------------------------------------------------------------------------
# Uniform detector call (one HTTP request per (image, model)).
# ---------------------------------------------------------------------------


async def _call_detector(
    session: aiohttp.ClientSession,
    name: DetectorName,
    cfg: DetectorConfig,
    image_path: str,
    classes: list[ClassConfig],
) -> list[Candidate]:
    """POST a DetectorRequest; parse DetectorResponse → Candidates.

    Server is responsible for per-prompt iteration (if needed) and for
    echoing back the original prompt string in each detection's label.
    """
    req = DetectorRequest(
        image_path=image_path,
        prompts=[c.prompt for c in classes],
    )
    url = f"http://localhost:{cfg.port}/predict"
    try:
        async with session.post(url, json=req.model_dump()) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logger.warning("%s POST failed: %s", name.value, exc)
        return []
    return _to_candidates(DetectorResponse.model_validate(data), name, classes)


def _to_candidates(
    resp: DetectorResponse,
    model: DetectorName,
    classes: list[ClassConfig],
) -> list[Candidate]:
    """Map server-echoed labels back to canonical class names by prompt match."""
    prompt_to_class = {c.prompt.lower().strip(): c.name for c in classes}

    out: list[Candidate] = []
    for i, (box, score, label) in enumerate(zip(resp.boxes, resp.scores, resp.labels)):
        if len(box) != 4:
            continue
        cls_name = prompt_to_class.get(str(label).lower().strip())
        if cls_name is None:
            logger.debug(
                "%s: unrecognised label %r; dropped (known: %s)",
                model.value,
                label,
                sorted(prompt_to_class),
            )
            continue
        out.append(
            Candidate(
                candidate_id=f"{model.value}:{cls_name}:{i}",
                class_name=cls_name,
                label=str(label),
                source_model=model.value,
                expression=str(label),
                bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                score=float(score),
                status=CandidateStatus.PROPOSED,
            )
        )
    return out
