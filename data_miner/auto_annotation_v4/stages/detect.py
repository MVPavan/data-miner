"""Stage 1b: Detection merge — load per-model proposals and combine them.

Phase 2a split the former monolithic detect stage: filtering and routing now
live in :class:`~.filter.FilterWorker`. This worker only loads per-model
proposals saved by :class:`~.detect_model.DetectModelWorker` instances,
merges them into a single raw candidate list, and saves a
:class:`DetectResult` checkpoint with empty ``routing`` / ``filter_stats``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from ..configs import (
    AutoAnnotationV4Config,
    Candidate,
    ClassConfig,
    DetectResult,
    DetectRouting,
    ProposalResult,
    Stage,
    StageMessage,
)
from ..output import OutputWriter
from ..utils import (
    filter_by_source_model,
    get_image_size,
    normalize_class_alias,
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


class DetectMergeWorker(StageWorker):
    """Stage ``detect:merge`` worker: load per-model proposals and merge.

    Claims work from the ``"detect:merge"`` queue, loads proposals saved by
    :class:`~.detect_model.DetectModelWorker` instances, merges them into a
    single raw candidate list, and saves a :class:`DetectResult` checkpoint
    under the canonical :attr:`Stage.DETECT` key. Filtering and routing run
    downstream in :class:`~.filter.FilterWorker`.

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
        """Merge per-model proposals into a single raw DetectResult.

        No filtering or routing runs here — the saved checkpoint carries
        every raw candidate from every model. :class:`FilterWorker` handles
        the full filter + route chain downstream.
        """
        image_path = msg.image_path
        image_w, image_h = get_image_size(image_path)

        proposals: dict[str, ProposalResult] = await self.db.load_all_proposals(
            msg.image_id, ProposalResult
        )

        model_results: dict[str, list[Candidate]] = {
            model_name: proposal.candidates
            for model_name, proposal in proposals.items()
        }

        all_candidates: list[Candidate] = []
        for candidates in model_results.values():
            all_candidates.extend(candidates)

        # Primary application of the source_model allowlist. Downstream stages
        # inherit clean input; the filter stage also re-applies it defensively
        # for re-runs after the allowlist flips.
        allowed = list(
            getattr(self.config.filtering, "allowed_source_models", []) or []
        )
        if allowed:
            before = len(all_candidates)
            all_candidates = filter_by_source_model(all_candidates, allowed)
            if before != len(all_candidates):
                self.logger.info(
                    "%s: source_model allowlist %s dropped %d/%d candidates",
                    msg.image_id, allowed, before - len(all_candidates), before,
                )

        if not all_candidates:
            self.logger.info(
                "No detections from any model for %s — forwarding empty DetectResult",
                msg.image_id,
            )
        else:
            self.logger.info(
                "%s: merged %d raw candidates from %d model(s)",
                msg.image_id,
                len(all_candidates),
                len(model_results),
            )

        return self._build_detect_result(
            msg.image_id,
            image_path,
            image_w,
            image_h,
            model_results,
            candidates=all_candidates,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        """Always forward to :attr:`Stage.FILTER`; routing happens there."""
        return Stage.FILTER

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
    ) -> DetectResult:
        return DetectResult(
            image_id=image_id,
            image_path=str(image_path),
            image_size=[image_w, image_h],
            models_used=list(model_results.keys()),
            candidates=candidates,
            routing=DetectRouting(),
            filter_stats={},
            stage_timing_ms=0.0,
        )
