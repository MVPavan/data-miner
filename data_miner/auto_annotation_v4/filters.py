"""Context-aware filter pipeline for auto_annotation_v4.

Wraps the existing filter functions in :mod:`utils` behind a single
``FilterPipeline`` class that selects the appropriate subset of filters based
on where in the pipeline it is being invoked (``FilterContext``).

This module intentionally holds no filter logic of its own — it only orders
and tags drops produced by util functions. If a util returns only the kept
list, drops are recovered by set-difference on ``candidate_id``.

Filter subsets per context:
    POST_DETECT:  geometric -> score_floor -> dedup -> per_class_cap -> cross_class
    POST_REVIEW:  cross_class -> per_class_cap   (relabels may break these)
    POST_REFINE:  geometric -> dedup             (bbox changes may break these)
    PRE_FINALIZE: geometric -> score_floor -> dedup -> per_class_cap -> cross_class
"""

from __future__ import annotations

from typing import Any

from .configs.contracts import Candidate, FilterDrop
from .configs.enums import DropReason, FilterContext
from .utils import (
    apply_cross_class_rules,
    cluster_and_collapse,
    filter_by_model_score,
    filter_by_source_model,
    geometric_filter,
    get_logger,
    limit_per_class,
)

__all__ = ["FilterPipeline"]


# Ordered filter plan per context. Each step is a short string key resolved
# inside FilterPipeline.run. ``source_model`` runs first in every plan so a
# post-detect allowlist change only requires a filter-stage re-run.
_PLAN: dict[FilterContext, tuple[str, ...]] = {
    FilterContext.POST_DETECT: (
        "source_model",
        "geometric",
        "score_floor",
        "dedup",
        "per_class_cap",
        "cross_class",
    ),
    FilterContext.POST_REVIEW: (
        "source_model",
        "cross_class",
        "per_class_cap",
    ),
    FilterContext.POST_REFINE: (
        "source_model",
        "geometric",
        "dedup",
    ),
    FilterContext.PRE_FINALIZE: (
        "source_model",
        "geometric",
        "score_floor",
        "dedup",
        "per_class_cap",
        "cross_class",
    ),
}


# Map filter step -> drop reason recorded on candidates removed by that step.
_REASON: dict[str, DropReason] = {
    "source_model": DropReason.SOURCE_MODEL,
    "geometric": DropReason.GEOMETRIC_FILTER,
    "score_floor": DropReason.SCORE_FLOOR,
    "dedup": DropReason.DEDUP,
    "per_class_cap": DropReason.PER_CLASS_CAP,
    "cross_class": DropReason.CROSS_CLASS,
}


def _diff_drops(
    before: list[Candidate],
    after: list[Candidate],
    reason: DropReason,
    context: FilterContext,
) -> list[FilterDrop]:
    """Return FilterDrop entries for candidates in ``before`` but not ``after``."""
    kept_ids = {c.candidate_id for c in after}
    return [
        FilterDrop(candidate_id=c.candidate_id, reason=reason, context=context)
        for c in before
        if c.candidate_id not in kept_ids
    ]


class FilterPipeline:
    """Run the CPU filter chain appropriate for a given FilterContext.

    The pipeline does not own filter logic — it orders and tags drops
    produced by the utility functions in :mod:`utils`. A single ``config``
    object is held so that the wrapped util functions (which read
    ``config.filtering``, ``config.co_existence``, ``config.classes``,
    ``config.class_registry``) continue to work unchanged.

    For Phase 2: a Stage.FILTER worker instantiates this once and calls
    ``run(candidates, FilterContext.POST_DETECT)``. Evaluate/refine/finalize
    instantiate their own and call ``run(..., POST_REVIEW | POST_REFINE |
    PRE_FINALIZE)`` respectively.
    """

    def __init__(self, config: Any) -> None:
        """Hold the full config — util functions already know how to access
        ``config.filtering``, ``config.co_existence``, etc.

        ``config`` is typically an :class:`AutoAnnotationV4Config` but any
        duck-typed object exposing ``.filtering`` (with ``.per_model_score``,
        ``.iou_dedup``, ``.max_per_class``, plus geometric thresholds),
        ``.co_existence``, ``.classes``, and ``.class_registry`` works.
        """
        self.config = config
        self.logger = get_logger("filters.FilterPipeline")

    # ------------------------------------------------------------------
    # Individual filter steps — thin wrappers that return (kept, drops).
    # ------------------------------------------------------------------

    def _step_source_model(
        self, cands: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        allowed = getattr(self.config.filtering, "allowed_source_models", []) or []
        kept = filter_by_source_model(cands, allowed)
        return kept, _diff_drops(cands, kept, DropReason.SOURCE_MODEL, context)

    def _step_geometric(
        self, cands: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        kept = geometric_filter(cands, self.config)
        return kept, _diff_drops(cands, kept, DropReason.GEOMETRIC_FILTER, context)

    def _step_score_floor(
        self, cands: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        per_model = getattr(self.config.filtering, "per_model_score", {}) or {}
        kept = filter_by_model_score(cands, per_model)
        return kept, _diff_drops(cands, kept, DropReason.SCORE_FLOOR, context)

    def _step_dedup(
        self, cands: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        kept = cluster_and_collapse(cands, self.config.filtering.iou_dedup)
        return kept, _diff_drops(cands, kept, DropReason.DEDUP, context)

    def _step_per_class_cap(
        self, cands: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        kept = limit_per_class(cands, self.config.filtering.max_per_class)
        return kept, _diff_drops(cands, kept, DropReason.PER_CLASS_CAP, context)

    def _step_cross_class(
        self, cands: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        kept = apply_cross_class_rules(cands, self.config)
        return kept, _diff_drops(cands, kept, DropReason.CROSS_CLASS, context)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self, candidates: list[Candidate], context: FilterContext
    ) -> tuple[list[Candidate], list[FilterDrop]]:
        """Run the filter subset appropriate for ``context``.

        Returns ``(kept_candidates, drops)``. Drops preserve the order in
        which filters removed them and carry both ``reason`` (which filter)
        and ``context`` (which pipeline point).
        """
        plan = _PLAN.get(context)
        if plan is None:
            raise ValueError(f"Unknown FilterContext: {context!r}")

        step_fns = {
            "source_model": self._step_source_model,
            "geometric": self._step_geometric,
            "score_floor": self._step_score_floor,
            "dedup": self._step_dedup,
            "per_class_cap": self._step_per_class_cap,
            "cross_class": self._step_cross_class,
        }

        kept: list[Candidate] = list(candidates)
        all_drops: list[FilterDrop] = []
        for step in plan:
            fn = step_fns[step]
            before = len(kept)
            kept, drops = fn(kept, context)
            all_drops.extend(drops)
            self.logger.debug(
                "context=%s step=%s %d -> %d (%d dropped)",
                context.value,
                step,
                before,
                len(kept),
                len(drops),
            )

        self.logger.info(
            "FilterPipeline[%s]: %d -> %d candidates (%d drops total)",
            context.value,
            len(candidates),
            len(kept),
            len(all_drops),
        )
        return kept, all_drops
