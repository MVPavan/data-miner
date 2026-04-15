"""Stage 4: Finalize — post-refine canonical list + dedup/geometry recheck.

Consolidates output writing (YOLO labels, traces, review queue) into a single
stage that re-runs filtering invariants on the canonical post-refine candidate
list. Catches:
  - relabel collisions (forklift→palletjack now overlapping a real palletjack)
  - refine-induced overlaps (an extension created a new overlap with a
    separately-accepted candidate)
  - refined bbox now violating geometric filters (max_area, aspect, edge)
  - per-class overflow after relabels

Pipeline runs detect → evaluate → refine → finalize → done.
Detect / evaluate / refine no longer write final outputs themselves — they
emit checkpoints only. Finalize is the single sink.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any

from ..workers.base import StageWorker
from ..contracts import (
    BoundingBox,
    Candidate,
    DetectResult,
    EvaluateResult,
    FinalAction,
    FinalAnnotation,
    FinalizeDrop,
    FinalizeResult,
    RefineResult,
    RefinementResult,
    StageMessage,
)
from ..config import AutoAnnotationV3Config
from ..checkpoint import CheckpointManager
from ..output import OutputWriter
from ..utils import (
    apply_cross_class_rules,
    cluster_and_collapse,
    geometric_filter,
    limit_per_class,
)

logger = logging.getLogger("data_miner.auto_annotation_v3.finalize")


class FinalizeWorker(StageWorker):
    """Stage 4: build canonical annotation list, re-check invariants, write outputs."""

    stage = "finalize"

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

    async def process(self, msg: StageMessage) -> StageMessage | None:
        t0 = time.monotonic()

        detect: DetectResult | None = self.load_checkpoint(
            msg.image_id, "detect", DetectResult
        )
        if detect is None:
            raise RuntimeError(f"No detect checkpoint for {msg.image_id}")

        evaluate: EvaluateResult | None = self.load_checkpoint(
            msg.image_id, "evaluate", EvaluateResult
        )
        refine: RefineResult | None = self.load_checkpoint(
            msg.image_id, "refine", RefineResult
        )

        # 1. Build canonical candidate list (relabel + refined bbox applied,
        #    rejected dropped, was_refined flag attached).
        canonical, refine_review = self._build_canonical(detect, evaluate, refine)
        before_geometric = len(canonical)

        # 2. Re-run invariants in the same order as detect's filtering chain.
        after_geom = geometric_filter(canonical, self.config)
        geom_dropped = _diff_drop(canonical, after_geom, "geometric_filter")

        after_dedup = cluster_and_collapse(
            after_geom, self.config.filtering.iou_dedup
        )
        dedup_dropped = _diff_drop(after_geom, after_dedup, "dedup")

        after_cap = limit_per_class(after_dedup, self.config.filtering.max_per_class)
        cap_dropped = _diff_drop(after_dedup, after_cap, "per_class_cap")

        after_cross = apply_cross_class_rules(after_cap, self.config)
        cross_dropped = _diff_drop(after_cap, after_cross, "cross_class")

        all_drops = geom_dropped + dedup_dropped + cap_dropped + cross_dropped

        # 3. Build FinalAnnotation list + review items.
        class_map: dict[str, int] = {c.name: c.id for c in self.config.classes}
        annotations: list[FinalAnnotation] = []
        for c in after_cross:
            annotations.append(FinalAnnotation(
                candidate_id=c.candidate_id,
                class_name=c.class_name,
                class_id=class_map.get(c.class_name, -1),
                bbox=c.bbox,
                confidence=c.score,
                action=FinalAction.ACCEPT,
                source_model=c.source_model,
                was_refined=bool(c.metadata.get("was_refined", False)),
                trace=list(c.notes),
            ))

        review_items: list[dict] = []
        # Upstream review (evaluate.review for non-refined classes).
        review_items.extend(self._evaluate_review_items(detect, evaluate, refine))
        # Refine adjudication review verdicts.
        review_items.extend(refine_review)

        elapsed_ms = (time.monotonic() - t0) * 1000
        result = FinalizeResult(
            image_id=msg.image_id,
            final_annotations=annotations,
            review_items=review_items,
            dropped=all_drops,
            filter_stats={
                "before_geometric": before_geometric,
                "after_geometric": len(after_geom),
                "after_dedup": len(after_dedup),
                "after_per_class_cap": len(after_cap),
                "after_cross_class": len(after_cross),
                "review_items": len(review_items),
            },
            stage_timing_ms=elapsed_ms,
        )
        self.save_checkpoint(msg.image_id, "finalize", result)

        # 4. Write outputs.
        if self.output_writer is not None:
            self.output_writer.write_yolo_labels(
                msg.image_id, annotations, class_map
            )
            self.output_writer.write_trace(msg.image_id, {
                "image_id": msg.image_id,
                "stages": [
                    "detect", "evaluate" if evaluate else None,
                    "refine" if refine else None, "finalize",
                ],
                "detect": detect.model_dump(mode="json"),
                "evaluate": evaluate.model_dump(mode="json") if evaluate else None,
                "refine": refine.model_dump(mode="json") if refine else None,
                "finalize": result.model_dump(mode="json"),
                "annotations": [a.model_dump(mode="json") for a in annotations],
            })
            if review_items:
                self.output_writer.write_review(msg.image_id, review_items)

        self.logger.info(
            "%s: finalize wrote %d annotations, %d review items, %d drops, %.0f ms",
            msg.image_id, len(annotations), len(review_items),
            len(all_drops), elapsed_ms,
        )
        return msg.forward("done")

    # ------------------------------------------------------------------
    # Canonical list builder
    # ------------------------------------------------------------------

    def _build_canonical(
        self,
        detect: DetectResult,
        evaluate: EvaluateResult | None,
        refine: RefineResult | None,
    ) -> tuple[list[Candidate], list[dict]]:
        """Produce the canonical post-refine candidate list, applying:
          - evaluate.relabels (class swaps)
          - refine results: refined_bbox if final_verdict==accept and source==refined,
                            else original_bbox if final_verdict==accept,
                            else dropped (review items captured separately).
          - evaluate.rejected → dropped
          - detect auto-accepted not in evaluate or refine → kept as-is
        Adds metadata={"was_refined": bool} on the survivor.

        Returns (canonical_list, refine_review_items).
        """
        accepted_set: set[str] = set()
        review_set: set[str] = set()
        rejected_set: set[str] = set()
        relabels: dict[str, str] = {}

        if evaluate is not None:
            accepted_set = set(evaluate.accepted)
            review_set = set(evaluate.review)
            rejected_set = set(evaluate.rejected)
            relabels = dict(evaluate.relabels)

        # Detect-stage auto-accepts that bypassed evaluate are also "accepted".
        accepted_set |= set(detect.routing.auto_accepted)

        refine_by_id: dict[str, RefinementResult] = {}
        if refine is not None:
            refine_by_id = {r.candidate_id: r for r in refine.results}

        canonical: list[Candidate] = []
        refine_review: list[dict] = []

        for cand in detect.candidates:
            if cand.candidate_id in rejected_set:
                continue
            if (
                cand.candidate_id not in accepted_set
                and cand.candidate_id not in review_set
            ):
                continue

            cls_post = relabels.get(cand.candidate_id, cand.class_name)
            ref = refine_by_id.get(cand.candidate_id)

            if ref is not None:
                if ref.final_verdict == "reject":
                    continue
                if ref.final_verdict == "review":
                    bbox_for_review = (
                        ref.refined_bbox
                        if ref.final_bbox_source == "refined" and ref.refined_bbox
                        else ref.original_bbox
                    )
                    refine_review.append({
                        "candidate_id": cand.candidate_id,
                        "class_name": cls_post,
                        "bbox": bbox_for_review.model_dump(),
                        "reason": "refine_review",
                        "adjudicate_verdict": ref.adjudicate_verdict,
                        "iou_with_original": ref.iou_with_original,
                    })
                    # Don't include in canonical labels — review-only.
                    continue
                # final_verdict == "accept"
                final_bbox = (
                    ref.refined_bbox
                    if ref.final_bbox_source == "refined" and ref.refined_bbox
                    else ref.original_bbox
                )
                was_refined = ref.final_bbox_source == "refined"
            else:
                # Not refined.
                if cand.candidate_id in review_set:
                    # Evaluate-only review (class not in refine_rules) — handled
                    # separately by _evaluate_review_items, skip here.
                    continue
                final_bbox = cand.bbox
                was_refined = False

            survivor = deepcopy(cand)
            survivor.class_name = cls_post
            survivor.bbox = final_bbox
            survivor.metadata = dict(survivor.metadata)
            survivor.metadata["was_refined"] = was_refined
            survivor.notes = list(survivor.notes)
            if was_refined:
                survivor.notes.append("refined")
            if cls_post != cand.class_name:
                survivor.notes.append(f"relabel:{cand.class_name}->{cls_post}")
            canonical.append(survivor)

        return canonical, refine_review

    @staticmethod
    def _evaluate_review_items(
        detect: DetectResult,
        evaluate: EvaluateResult | None,
        refine: RefineResult | None,
    ) -> list[dict]:
        if evaluate is None or not evaluate.review:
            return []
        refine_ids: set[str] = (
            {r.candidate_id for r in refine.results} if refine else set()
        )
        review = []
        for cand in detect.candidates:
            if cand.candidate_id not in evaluate.review:
                continue
            if cand.candidate_id in refine_ids:
                continue  # handled by refine_review
            cls_post = evaluate.relabels.get(cand.candidate_id, cand.class_name)
            review.append({
                "candidate_id": cand.candidate_id,
                "class_name": cls_post,
                "bbox": cand.bbox.model_dump(),
                "reason": "evaluate_review",
            })
        return review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff_drop(
    before: list[Candidate],
    after: list[Candidate],
    reason: str,
) -> list[FinalizeDrop]:
    """Items in `before` but not in `after` (by candidate_id) become FinalizeDrop entries."""
    after_ids = {c.candidate_id for c in after}
    return [
        FinalizeDrop(
            candidate_id=c.candidate_id,
            class_name=c.class_name,
            reason=reason,
            bbox=c.bbox,
        )
        for c in before
        if c.candidate_id not in after_ids
    ]
