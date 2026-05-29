"""Stage 4: Finalize — post-refine canonical list + dedup/geometry recheck.

Consolidates output writing (YOLO labels, traces, review queue) into a single
stage that re-runs filtering invariants on the canonical post-refine candidate
list. Catches:
  - relabel collisions (forklift->palletjack now overlapping a real palletjack)
  - refine-induced overlaps (an extension created a new overlap with a
    separately-accepted candidate)
  - refined bbox now violating geometric filters (max_area, aspect, edge)
  - per-class overflow after relabels

Pipeline runs detect -> evaluate -> refine -> finalize -> done.
Detect / evaluate / refine no longer write final outputs themselves — they
emit checkpoints only. Finalize is the single sink.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from ..workers.base import StageWorker
from ..configs import (
    AutoAnnotationV4Config,
    BboxSource,
    Candidate,
    EvaluateResult,
    FilterContext,
    FilterResult,
    FinalAction,
    FinalAnnotation,
    FinalizeDrop,
    FinalizeResult,
    RefineResult,
    RefinementResult,
    Stage,
    StageMessage,
    Verdict,
)
from ..filters import FilterPipeline
from ..output import OutputWriter

logger = logging.getLogger("data_miner.auto_annotation_v4.finalize")


class FinalizeWorker(StageWorker):
    """Stage 4: build canonical annotation list, re-check invariants, write outputs."""

    stage = Stage.FINALIZE
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
        # Cache a single FilterPipeline per worker — reused for every
        # PRE_FINALIZE pass on the canonical post-refine candidate list.
        self._filter_pipeline = FilterPipeline(self.config)

    async def process(self, msg: StageMessage) -> BaseModel:
        t0 = time.monotonic()

        filtered: FilterResult | None = await self.load_checkpoint(
            msg.image_id, Stage.FILTER, FilterResult
        )
        if filtered is None:
            raise RuntimeError(f"No filter checkpoint for {msg.image_id}")

        evaluate: EvaluateResult | None = await self.load_checkpoint(
            msg.image_id, Stage.EVALUATE, EvaluateResult
        )
        refine: RefineResult | None = await self.load_checkpoint(
            msg.image_id, Stage.REFINE, RefineResult
        )

        # 1. Build canonical candidate list (relabel + refined bbox applied,
        #    rejected dropped, was_refined flag attached).
        canonical, refine_review = self._build_canonical(filtered, evaluate, refine)
        before_geometric = len(canonical)

        # 2. Re-run filter invariants via FilterPipeline (PRE_FINALIZE subset:
        #    geometric -> score_floor -> dedup -> per_class_cap -> cross_class).
        kept, filter_drops = self._filter_pipeline.run(
            canonical, FilterContext.PRE_FINALIZE,
        )
        # Convert FilterDrops -> FinalizeDrops so existing FinalizeResult shape
        # is preserved. A canonical-id lookup preserves class_name + bbox for
        # the drop record.
        canonical_by_id: dict[str, Candidate] = {c.candidate_id: c for c in canonical}
        all_drops: list[FinalizeDrop] = []
        for d in filter_drops:
            src = canonical_by_id.get(d.candidate_id)
            all_drops.append(FinalizeDrop(
                candidate_id=d.candidate_id,
                class_name=src.class_name if src else "",
                reason=d.reason,
                bbox=src.bbox if src else None,
            ))

        # 3. Build FinalAnnotation list + review items.
        # v4: config.classes is dict[str, ClassConfig]
        class_map: dict[str, int] = {
            name: cfg.id for name, cfg in self.config.classes.items()
        }
        annotations: list[FinalAnnotation] = []
        for c in kept:
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
        review_items.extend(self._evaluate_review_items(filtered, evaluate, refine))
        # Refine adjudication review verdicts.
        review_items.extend(refine_review)

        elapsed_ms = (time.monotonic() - t0) * 1000
        result = FinalizeResult(
            image_id=msg.image_id,
            final_annotations=annotations,
            review_items=review_items,
            dropped=all_drops,
            filter_stats={
                "before_filter": before_geometric,
                "after_filter": len(kept),
                "dropped": len(all_drops),
                "review_items": len(review_items),
            },
            stage_timing_ms=elapsed_ms,
        )
        await self.save_checkpoint(msg.image_id, Stage.FINALIZE, result)

        # 4. Write outputs.
        if self.output_writer is not None:
            self.output_writer.write_yolo_labels(
                msg.image_id, annotations, class_map
            )
            self.output_writer.write_trace(msg.image_id, {
                "image_id": msg.image_id,
                "stages": [
                    "filter", "evaluate" if evaluate else None,
                    "refine" if refine else None, "finalize",
                ],
                "filter": filtered.model_dump(mode="json"),
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
        return result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        return Stage.DONE

    # ------------------------------------------------------------------
    # Canonical list builder
    # ------------------------------------------------------------------

    def _build_canonical(
        self,
        filtered: FilterResult,
        evaluate: EvaluateResult | None,
        refine: RefineResult | None,
    ) -> tuple[list[Candidate], list[dict]]:
        """Produce the canonical post-refine candidate list, applying:
          - evaluate.relabels (class swaps)
          - refine results: refined_bbox if final_verdict==accept and source==refined,
                            else original_bbox if final_verdict==accept,
                            else dropped (review items captured separately).
          - evaluate.rejected -> dropped
          - filter auto-accepted not in evaluate or refine -> kept as-is
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

        # Filter-stage auto-accepts that bypassed evaluate are also "accepted".
        accepted_set |= set(filtered.routing.auto_accepted)

        refine_by_id: dict[str, RefinementResult] = {}
        if refine is not None:
            refine_by_id = {r.candidate_id: r for r in refine.results}

        canonical: list[Candidate] = []
        refine_review: list[dict] = []

        for cand in filtered.candidates:
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
                if ref.final_verdict == Verdict.REJECT:
                    continue
                if ref.final_verdict == Verdict.REVIEW:
                    bbox_for_review = (
                        ref.refined_bbox
                        if ref.final_bbox_source == BboxSource.REFINED and ref.refined_bbox
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
                    if ref.final_bbox_source == BboxSource.REFINED and ref.refined_bbox
                    else ref.original_bbox
                )
                was_refined = ref.final_bbox_source == BboxSource.REFINED
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
        filtered: FilterResult,
        evaluate: EvaluateResult | None,
        refine: RefineResult | None,
    ) -> list[dict]:
        if evaluate is None or not evaluate.review:
            return []
        refine_ids: set[str] = (
            {r.candidate_id for r in refine.results} if refine else set()
        )
        review = []
        for cand in filtered.candidates:
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


