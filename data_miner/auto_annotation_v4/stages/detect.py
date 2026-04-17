"""Stage 1b: Detection merge — load proposals, filter, dedup, route.

Phase 2 splits the monolithic detect stage: per-model DetectModelWorker
instances (in detect_model.py) call individual model servers and save
proposals. This DetectMergeWorker loads all proposals, runs the full
filtering pipeline, and routes to the next stage.

Replaces the Phase 1 DetectWorker.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from ..configs import (
    AutoAnnotationV4Config,
    Candidate,
    ClassConfig,
    DetectResult,
    DetectRouting,
    FinalAction,
    FinalAnnotation,
    ProposalResult,
    Stage,
    StageMessage,
)
from ..output import OutputWriter
from ..utils import (
    apply_cross_class_rules,
    cluster_and_collapse,
    filter_by_model_score,
    geometric_filter,
    get_image_size,
    limit_per_class,
    normalize_class_alias,
    route_candidates,
)
from ..workers.base import StageWorker

logger = logging.getLogger("data_miner.auto_annotation_v4.detect")


def _build_v4_alias_map(classes: dict[str, ClassConfig]) -> dict[str, str]:
    """Build alias map from v4 dict-keyed class registry.

    Maps lowercased/normalised synonyms -> canonical class name.
    """
    alias_map: dict[str, str] = {}
    for name, cfg in classes.items():
        alias_map[normalize_class_alias(name)] = name
        for syn in cfg.synonyms:
            alias_map[normalize_class_alias(syn)] = name
    return alias_map


def _run_filtering_pipeline(
    candidates: list[Candidate],
    config: AutoAnnotationV4Config,
) -> tuple[list[Candidate], dict, dict]:
    """Run the full filtering chain. Pure CPU — safe for run_in_executor.

    Returns (cross_filtered, routing_result, filter_stats).
    """
    filtered = geometric_filter(candidates, config)
    after_geometric = len(filtered)

    filtered = filter_by_model_score(filtered, config.filtering.per_model_score)
    after_model_score = len(filtered)

    deduped = cluster_and_collapse(filtered, config.filtering.iou_dedup)
    after_dedup = len(deduped)

    capped = limit_per_class(deduped, config.filtering.max_per_class)
    after_cap = len(capped)

    cross_filtered = apply_cross_class_rules(capped, config)

    routing_result = route_candidates(cross_filtered, config)

    filter_stats = {
        "total_proposed": len(candidates),
        "after_geometric_filter": after_geometric,
        "after_model_score_filter": after_model_score,
        "after_iou_dedup": after_dedup,
        "after_per_class_cap": after_cap,
        "after_cross_class_rules": len(cross_filtered),
        "auto_accepted": len(routing_result["auto_accepted"]),
        "sent_to_vlm": len(routing_result["needs_evaluation"]),
    }

    return cross_filtered, routing_result, filter_stats


class DetectMergeWorker(StageWorker):
    """Stage 1b: load per-model proposals, merge, filter, dedup, route.

    Claims work from ``"detect:merge"`` queue. Loads proposals saved by
    DetectModelWorker instances, runs the full filtering pipeline, and
    saves a DetectResult checkpoint under the canonical ``Stage.DETECT``.

    This worker does ZERO HTTP calls — all model server communication
    happens in DetectModelWorker.
    """

    stage = "detect:merge"
    needs_session = False

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        db: Any,
        *,
        output_writer: OutputWriter | None = None,
        worker_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(config, db, output_writer=output_writer, worker_id=worker_id, job_id=job_id)
        # Build alias map once — maps lowercased aliases -> canonical class name.
        self._alias_map: dict[str, str] = _build_v4_alias_map(config.classes)

    def _checkpoint_stage(self) -> Stage:
        """Checkpoint stored as 'detect' (not 'detect:merge')."""
        return Stage.DETECT

    # ------------------------------------------------------------------
    # Core process method (called by StageWorker.run loop)
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> BaseModel:
        image_path = msg.image_path
        image_w, image_h = get_image_size(image_path)

        # 1. Load all per-model proposals from DB.
        proposals: dict[str, ProposalResult] = await self.db.load_all_proposals(
            msg.image_id, ProposalResult
        )

        # Build model_results: model_name -> list[Candidate]
        model_results: dict[str, list[Candidate]] = {}
        for model_name, proposal in proposals.items():
            model_results[model_name] = proposal.candidates

        if not any(model_results.values()):
            self.logger.info(
                "No detections from any model for %s — forwarding empty",
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
            if self.output_writer:
                self._write_auto_accepted_output(msg.image_id, detect_result)
            return detect_result

        # 2. Merge all per-model candidates.
        all_candidates: list[Candidate] = []
        for candidates in model_results.values():
            all_candidates.extend(candidates)
        total_proposed = len(all_candidates)

        # 3-7. Filtering pipeline (geometric -> score floor -> dedup -> cap -> cross-class -> route).
        if total_proposed > 100:
            # Offload CPU-bound filtering to thread pool to avoid blocking
            # the event loop when processing large detection sets.
            loop = asyncio.get_running_loop()
            cross_filtered, routing_result, filter_stats = await loop.run_in_executor(
                None,
                _run_filtering_pipeline,
                all_candidates,
                self.config,
            )
        else:
            cross_filtered, routing_result, filter_stats = _run_filtering_pipeline(
                all_candidates, self.config,
            )

        # 8. Build DetectResult.
        detect_result = self._build_detect_result(
            msg.image_id,
            image_path,
            image_w,
            image_h,
            model_results,
            candidates=cross_filtered,
            routing_result=routing_result,
            filter_stats=filter_stats,
        )

        # Write auto-accepted output if applicable.
        if self.output_writer and not routing_result["needs_evaluation"]:
            self._write_auto_accepted_output(msg.image_id, detect_result)

        # Log routing decisions (base class handles save_and_forward).
        if routing_result["needs_evaluation"]:
            self.logger.info(
                "%s: %d auto-accepted, %d sent to evaluate",
                msg.image_id,
                len(routing_result["auto_accepted"]),
                len(routing_result["needs_evaluation"]),
            )
        else:
            self.logger.info(
                "%s: all %d candidates auto-accepted",
                msg.image_id,
                len(routing_result["auto_accepted"]),
            )

        return detect_result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        detect_result: DetectResult = result  # type: ignore[assignment]

        # If candidates need VLM evaluation -> evaluate
        if detect_result.routing.needs_evaluation:
            return Stage.EVALUATE

        # All surviving candidates were auto-accepted by detect — skip VLM.
        # If any of them belongs to a class with a refine rule, forward to
        # refine (which will adjudicate without an evaluate checkpoint).
        refine_classes = set(self.config.refine_rules.classes.keys())
        auto_accept_ids = set(detect_result.routing.auto_accepted)
        needs_refine = any(
            c.candidate_id in auto_accept_ids and c.class_name in refine_classes
            for c in detect_result.candidates
        )
        if needs_refine:
            return Stage.REFINE

        return Stage.FINALIZE

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
            stage_timing_ms=0.0,
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
        # v4: config.classes is dict[str, ClassConfig]
        class_map: dict[str, int] = {
            name: cfg.id for name, cfg in self.config.classes.items()
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
            len(annotations),
            image_id,
        )
