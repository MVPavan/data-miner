"""Stage 1: Detection — call model servers, filter, dedup, route."""

from __future__ import annotations

import asyncio
import time
import logging
from pathlib import Path
from typing import Any

import aiohttp

from ..workers.base import StageWorker
from ..contracts import (
    StageMessage,
    Candidate,
    ProposalResult,
    DetectResult,
    DetectRouting,
    BoundingBox,
    CandidateStatus,
    FinalAnnotation,
    FinalAction,
)
from ..config import AutoAnnotationV3Config
from ..checkpoint import CheckpointManager
from ..output import OutputWriter
from ..utils import (
    geometric_filter,
    dedup_by_iou,
    limit_per_class,
    compute_agreement,
    route_candidates,
    apply_cross_class_rules,
    get_image_size,
    build_class_alias_map,
    resolve_canonical_class,
)

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
                msg.image_id, image_path, image_w, image_h,
                model_results, candidates=[], routing_result={
                    "auto_accepted": [],
                    "needs_evaluation": [],
                    "confusion_flags": [],
                }
            )
            self.save_checkpoint(msg.image_id, "detect", detect_result)
            if self.output_writer:
                self._write_auto_accepted_output(msg.image_id, detect_result)
            return msg.forward("done")

        # 2. Merge all per-model candidates.
        all_candidates: list[Candidate] = []
        for candidates in model_results.values():
            all_candidates.extend(candidates)

        total_proposed = len(all_candidates)

        # 3. Geometric filtering (area, aspect ratio, edge distance).
        filtered = geometric_filter(all_candidates, self.config)
        after_geometric = len(filtered)

        # 4. Per-class IoU dedup (keeps highest-score box when IoU > threshold).
        deduped = dedup_by_iou(filtered, self.config.filtering.iou_dedup_threshold)
        after_dedup = len(deduped)

        # 5. Per-class cap (keeps top-N by score).
        capped = limit_per_class(deduped, self.config.filtering.max_per_class)
        after_cap = len(capped)

        # 6. Cross-class suppression (suppress lower-score box when non-confused
        #    classes heavily overlap, unless a class is globally exempt).
        cross_filtered = apply_cross_class_rules(capped, self.config)

        # 7. Compute model agreement for each surviving candidate.
        compute_agreement(cross_filtered)

        # 8. Route: auto_accept (tier-1, high agreement) vs needs_evaluation.
        routing_result = route_candidates(cross_filtered, self.config)

        # 9. Build and checkpoint DetectResult.
        detect_result = self._build_detect_result(
            msg.image_id, image_path, image_w, image_h,
            model_results, candidates=cross_filtered,
            routing_result=routing_result,
            filter_stats={
                "total_proposed": total_proposed,
                "after_geometric_filter": after_geometric,
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
        else:
            # All surviving candidates are auto-accepted — skip VLM entirely.
            self.logger.info(
                "%s: all %d candidates auto-accepted — forwarding to done",
                msg.image_id,
                len(routing_result["auto_accepted"]),
            )
            if self.output_writer:
                self._write_auto_accepted_output(msg.image_id, detect_result)
            return msg.forward("done")

    # ------------------------------------------------------------------
    # Parallel model calls
    # ------------------------------------------------------------------

    async def _run_all_detectors(
        self,
        image_id: str,
        image_path: str,
    ) -> dict[str, list[Candidate]]:
        """Call every configured detection server; respect per-model proposal cache."""
        server_map = self._get_detection_servers()
        model_results: dict[str, list[Candidate]] = {}

        # Identify which models still need calling (not yet cached).
        pending: dict[str, Any] = {}
        for model_name, server_cfg in server_map.items():
            if self.checkpoint.proposal_exists(image_id, model_name):
                self.logger.info(
                    "Reusing cached proposal %s/%s", image_id, model_name
                )
                # Will be loaded below.
            else:
                pending[model_name] = server_cfg

        # Fire all pending calls concurrently.
        if pending:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = {
                    model_name: asyncio.create_task(
                        self._call_detector(session, model_name, cfg, image_path)
                    )
                    for model_name, cfg in pending.items()
                }
                # Gather results, absorbing individual model failures.
                for model_name, task in tasks.items():
                    t0 = time.monotonic()
                    try:
                        candidates = await task
                        latency_ms = (time.monotonic() - t0) * 1000
                        proposal = ProposalResult(
                            model=model_name,
                            image_id=image_id,
                            image_size=list(get_image_size(image_path)),
                            latency_ms=latency_ms,
                            candidates=candidates,
                        )
                        self.checkpoint.save_proposal(image_id, model_name, proposal)
                        model_results[model_name] = candidates
                        self.logger.info(
                            "Model %s returned %d candidates in %.0f ms",
                            model_name, len(candidates), latency_ms,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Model %s failed for %s: %s",
                            model_name, image_id, exc,
                        )
                        model_results[model_name] = []

        # Load all cached proposals (skipped models + any that already existed).
        for model_name in server_map:
            if model_name not in model_results:
                cached = self.checkpoint.load_proposal(
                    image_id, model_name, ProposalResult
                )
                if cached is not None:
                    model_results[model_name] = cached.candidates
                    self.logger.debug(
                        "Loaded %d cached candidates for %s/%s",
                        len(cached.candidates), image_id, model_name,
                    )
                else:
                    self.logger.warning(
                        "No proposal found for %s/%s after cache check",
                        image_id, model_name,
                    )
                    model_results[model_name] = []

        return model_results

    # ------------------------------------------------------------------
    # Per-model HTTP call
    # ------------------------------------------------------------------

    async def _call_detector(
        self,
        session: aiohttp.ClientSession,
        model_name: str,
        server_cfg: Any,
        image_path: str,
    ) -> list[Candidate]:
        """POST to one model server and parse the response into Candidates."""
        url = f"http://localhost:{server_cfg.port}/predict"
        payload = self._build_payload(model_name, image_path)

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data: dict = await resp.json()

        return self._parse_response(data, model_name)

    def _build_payload(self, model_name: str, image_path: str) -> dict:
        """Construct the JSON payload for each model's expected API."""
        if model_name in ("grounding_dino", "falcon"):
            text_prompt = " . ".join(c.prompt for c in self.config.classes)
            return {"image_path": image_path, "text_prompt": text_prompt}

        if model_name == "sam3":
            text_prompt = " . ".join(c.prompt for c in self.config.classes)
            return {
                "image_path": image_path,
                "text_prompt": text_prompt,
                "mode": "proposal",
            }

        if model_name == "owlvit2":
            queries = [f"a photo of a {c.prompt}" for c in self.config.classes]
            return {"image_path": image_path, "text_queries": queries}

        raise ValueError(f"Unknown model name: {model_name!r}")

    def _parse_response(self, data: dict, model_name: str) -> list[Candidate]:
        """Convert raw server response dict into a list of Candidate objects."""
        boxes: list = data.get("boxes", [])
        scores: list = data.get("scores", [])
        labels: list = data.get("labels", [])

        if not boxes:
            return []

        candidates: list[Candidate] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Resolve label string to canonical class name via alias map.
            class_name = self._resolve_label(label)
            if class_name is None:
                self.logger.debug(
                    "Model %s: unrecognised label %r at index %d — skipped",
                    model_name, label, i,
                )
                continue

            # Validate box has 4 elements.
            if len(box) != 4:
                self.logger.debug(
                    "Model %s: malformed box at index %d: %r — skipped",
                    model_name, i, box,
                )
                continue

            label_str = label if isinstance(label, str) else class_name
            candidate_id = f"{model_name}:{class_name}:{i}"

            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    class_name=class_name,
                    label=label_str,
                    source_model=model_name,
                    expression=label_str,
                    bbox=BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    score=float(score),
                    status=CandidateStatus.PROPOSED,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_detection_servers(self) -> dict[str, Any]:
        """Return ServerConfig objects for the four detection models (not VLM)."""
        servers: dict[str, Any] = {}
        server_map = self.config.servers
        for name in ("grounding_dino", "falcon", "sam3", "owlvit2"):
            cfg = getattr(server_map, name, None)
            if cfg is not None:
                servers[name] = cfg
        return servers

    def _resolve_label(self, label: Any) -> str | None:
        """Map a raw model label to a canonical class name, or None if unknown."""
        if not isinstance(label, str):
            return None
        return resolve_canonical_class(label, self._alias_map)

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
        class_map: dict[str, int] = {
            c.name: c.id for c in self.config.classes
        }

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
            len(annotations), image_id,
        )
