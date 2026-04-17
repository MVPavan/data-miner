"""Stage 1b (split from detect): Filter — run CPU filter chain and route.

Phase 2a splits the monolithic detect stage into a raw-merge DetectMergeWorker
(produces unfiltered candidates) and this FilterWorker which runs the
:class:`FilterPipeline` in :data:`FilterContext.POST_DETECT` context, applies
routing (auto-accept vs VLM evaluation), writes auto-accepted YOLO labels if
the image can short-circuit the VLM, and forwards the image to the next
pipeline stage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel

from ..configs import (
    AutoAnnotationV4Config,
    DetectResult,
    DetectRouting,
    FilterContext,
    FilterResult,
    FinalAction,
    FinalAnnotation,
    Stage,
    StageMessage,
)
from ..filters import FilterPipeline
from ..output import OutputWriter
from ..utils import route_candidates
from ..workers.base import StageWorker

logger = logging.getLogger("data_miner.auto_annotation_v4.filter")


class FilterWorker(StageWorker):
    """Stage ``filter`` worker: CPU-only filter chain + routing.

    Loads the raw merged :class:`DetectResult` produced upstream, runs the
    ``POST_DETECT`` :class:`FilterPipeline`, calls :func:`route_candidates`
    to split survivors into auto-accept vs needs-evaluation buckets, and
    writes YOLO labels for the fast-path (auto-accept only) case. Performs
    zero HTTP calls.
    """

    stage = Stage.FILTER
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
        super().__init__(
            config,
            db,
            output_writer=output_writer,
            worker_id=worker_id,
            job_id=job_id,
        )
        # Single pipeline instance reused for every image — FilterPipeline
        # is stateless apart from the (immutable) config reference.
        self._filter_pipeline = FilterPipeline(config)

    # ------------------------------------------------------------------
    # Core process method (called by StageWorker.run loop)
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> BaseModel:
        """Filter raw detect candidates, route them, and optionally fast-path.

        Loads the detect checkpoint, runs the POST_DETECT filter pipeline,
        applies routing, writes auto-accepted YOLO labels when no candidate
        needs VLM evaluation, and returns a :class:`FilterResult` for the
        base class to persist and forward.
        """
        t0 = time.perf_counter()

        detect_result: DetectResult | None = await self.db.load_stage(
            msg.image_id, Stage.DETECT, DetectResult
        )
        if detect_result is None:
            raise RuntimeError(
                f"No detect checkpoint for {msg.image_id} — upstream did not run."
            )

        raw_candidates = detect_result.candidates
        total_proposed = len(raw_candidates)

        # Offload CPU-bound filtering to a thread when the candidate set is
        # large enough that a single-threaded run could starve other tasks.
        if total_proposed > 100:
            loop = asyncio.get_running_loop()
            kept, drops = await loop.run_in_executor(
                None,
                self._filter_pipeline.run,
                raw_candidates,
                FilterContext.POST_DETECT,
            )
        else:
            kept, drops = self._filter_pipeline.run(
                raw_candidates, FilterContext.POST_DETECT
            )

        routing_dict = route_candidates(kept, self.config)
        routing = DetectRouting(
            auto_accepted=routing_dict.get("auto_accepted", []),
            needs_evaluation=routing_dict.get("needs_evaluation", []),
            confusion_flags=routing_dict.get("confusion_flags", []),
        )

        filter_stats: dict[str, Any] = {
            "total_proposed": total_proposed,
            "after_filters": len(kept),
            "dropped": len(drops),
            "auto_accepted": len(routing.auto_accepted),
            "sent_to_vlm": len(routing.needs_evaluation),
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        filter_result = FilterResult(
            image_id=msg.image_id,
            candidates=kept,
            drops=drops,
            routing=routing,
            filter_stats=filter_stats,
            config_hash=self._config_hash,
            created_at=time.time(),
            stage_timing_ms=elapsed_ms,
        )

        # Fast path: all survivors were auto-accepted → write YOLO labels now.
        if (
            self.output_writer is not None
            and routing.auto_accepted
            and not routing.needs_evaluation
        ):
            self._write_auto_accepted_output(
                msg.image_id, detect_result, filter_result
            )

        if routing.needs_evaluation:
            self.logger.info(
                "%s: %d auto-accepted, %d sent to evaluate (%d dropped)",
                msg.image_id,
                len(routing.auto_accepted),
                len(routing.needs_evaluation),
                len(drops),
            )
        else:
            self.logger.info(
                "%s: all %d candidates auto-accepted (%d dropped)",
                msg.image_id,
                len(routing.auto_accepted),
                len(drops),
            )

        return filter_result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        """Pick the next stage given a :class:`FilterResult`.

        - Any candidate needing VLM evaluation → :attr:`Stage.EVALUATE`.
        - Else, if any auto-accepted candidate's class has a refine rule →
          :attr:`Stage.REFINE` (skipping VLM classification).
        - Else → :attr:`Stage.FINALIZE`.
        """
        filter_result: FilterResult = result  # type: ignore[assignment]

        if filter_result.routing.needs_evaluation:
            return Stage.EVALUATE

        refine_classes = set(self.config.refine_rules.classes.keys())
        auto_accept_ids = set(filter_result.routing.auto_accepted)
        needs_refine = any(
            c.candidate_id in auto_accept_ids and c.class_name in refine_classes
            for c in filter_result.candidates
        )
        if needs_refine:
            return Stage.REFINE

        return Stage.FINALIZE

    # ------------------------------------------------------------------
    # Auto-accept fast path (skips VLM entirely)
    # ------------------------------------------------------------------

    def _write_auto_accepted_output(
        self,
        image_id: str,
        detect_result: DetectResult,
        filter_result: FilterResult,
    ) -> None:
        """Write YOLO labels + trace for the all-auto-accepted short-circuit."""
        if self.output_writer is None:
            return

        auto_accepted_ids: set[str] = set(filter_result.routing.auto_accepted)
        class_map: dict[str, int] = {
            name: cfg.id for name, cfg in self.config.classes.items()
        }

        annotations: list[FinalAnnotation] = []
        for cand in filter_result.candidates:
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
                "stages": ["detect", "filter"],
                "detect": detect_result.model_dump(mode="json"),
                "filter": filter_result.model_dump(mode="json"),
                "annotations": [a.model_dump(mode="json") for a in annotations],
            },
        )
        self.logger.info(
            "Wrote %d auto-accepted annotations for %s",
            len(annotations),
            image_id,
        )
