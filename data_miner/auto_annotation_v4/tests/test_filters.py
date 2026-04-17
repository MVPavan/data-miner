"""Tier 1 unit tests for FilterPipeline — no GPU, pure Python, synthetic data."""

from __future__ import annotations

import pytest

from data_miner.auto_annotation_v4.configs.contracts import BoundingBox, Candidate
from data_miner.auto_annotation_v4.configs.enums import (
    CandidateStatus,
    DropReason,
    FilterContext,
)
from data_miner.auto_annotation_v4.configs.settings import (
    AutoAnnotationV4Config,
    ClassConfig,
    CoExistenceConfig,
    FilterConfig,
    IouDedupConfig,
)
from data_miner.auto_annotation_v4.filters import FilterPipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_config() -> AutoAnnotationV4Config:
    """Synthetic config with two confusable classes and one exempt class.

    - class_small_cap: capped to 1 per image so a duplicate triggers PER_CLASS_CAP.
    - class_A / class_B: distinct classes not sharing confusion tags; used to
      trigger CROSS_CLASS suppression when they overlap > 0.5 IoU.
    - class_geom_fail: bbox below min_area → GEOMETRIC_FILTER drop.
    - class_score_fail: score below per-model floor → SCORE_FLOOR drop.
    - class_dup: two overlapping candidates (same class) → DEDUP collapse.
    """
    iou = IouDedupConfig(
        threshold=0.5,
        tiebreak_by=["agreement", "model_priority", "score"],
        model_priority=["gdino", "sam3"],
    )
    filtering = FilterConfig(
        min_area=0.001,
        max_area=0.95,
        min_aspect_ratio=0.1,
        max_aspect_ratio=10.0,
        min_edge_distance=0.0,
        per_model_score={"gdino": 0.2, "sam3": 0.0},
        iou_dedup=iou,
        max_per_class=1,
    )
    class_registry = {
        "class_small_cap": ClassConfig(id=1, tier=1, prompts=["small_cap"], tags=[]),
        "class_A": ClassConfig(id=2, tier=1, prompts=["A"], tags=[]),
        "class_B": ClassConfig(id=3, tier=1, prompts=["B"], tags=[]),
        "class_geom_fail": ClassConfig(id=4, tier=1, prompts=["geom"], tags=[]),
        "class_score_fail": ClassConfig(id=5, tier=1, prompts=["sc"], tags=[]),
        "class_dup": ClassConfig(id=6, tier=1, prompts=["dup"], tags=[]),
    }
    co_existence = CoExistenceConfig(
        overlap_exempt_tags=[],
        confusion_tags=[],
        extra_confusion_pairs=[],
    )
    return AutoAnnotationV4Config(
        class_registry=class_registry,
        co_existence=co_existence,
        filtering=filtering,
    )


def _bbox(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _build_candidates() -> list[Candidate]:
    """Candidates engineered to exercise each of the 5 filter reasons."""
    return [
        # 1) GEOMETRIC_FILTER — area far below min_area=0.001.
        Candidate(
            candidate_id="geom1",
            class_name="class_geom_fail",
            label="geom_fail",
            source_model="gdino",
            expression="tiny",
            bbox=_bbox(0.10, 0.10, 0.101, 0.101),  # area ~1e-6
            score=0.9,
        ),
        # 2) SCORE_FLOOR — gdino floor 0.2, score 0.1.
        Candidate(
            candidate_id="scor1",
            class_name="class_score_fail",
            label="low_score",
            source_model="gdino",
            expression="low",
            bbox=_bbox(0.20, 0.20, 0.30, 0.30),
            score=0.1,
        ),
        # 3) DEDUP — two heavily overlapping same-class candidates (IoU > 0.5).
        Candidate(
            candidate_id="dup_A",
            class_name="class_dup",
            label="dup",
            source_model="gdino",
            expression="dup",
            bbox=_bbox(0.40, 0.40, 0.60, 0.60),
            score=0.8,
        ),
        Candidate(
            candidate_id="dup_B",
            class_name="class_dup",
            label="dup",
            source_model="sam3",
            expression="dup",
            bbox=_bbox(0.41, 0.41, 0.60, 0.60),
            score=0.75,
        ),
        # 4) PER_CLASS_CAP — max_per_class=1 with two class_small_cap cands.
        Candidate(
            candidate_id="cap1",
            class_name="class_small_cap",
            label="cap1",
            source_model="gdino",
            expression="cap",
            bbox=_bbox(0.70, 0.05, 0.80, 0.15),
            score=0.9,
        ),
        Candidate(
            candidate_id="cap2",
            class_name="class_small_cap",
            label="cap2",
            source_model="gdino",
            expression="cap",
            bbox=_bbox(0.05, 0.70, 0.15, 0.80),
            score=0.85,
        ),
        # 5) CROSS_CLASS — overlapping different classes, no confusion tags,
        #    IoU > 0.5 → loser is suppressed. Keep scores distinguishable and
        #    source model unique enough that the tiebreak picks one loser.
        Candidate(
            candidate_id="cross_A",
            class_name="class_A",
            label="A",
            source_model="gdino",
            expression="A",
            bbox=_bbox(0.50, 0.05, 0.70, 0.25),
            score=0.9,
        ),
        Candidate(
            candidate_id="cross_B",
            class_name="class_B",
            label="B",
            source_model="sam3",
            expression="B",
            bbox=_bbox(0.51, 0.06, 0.70, 0.25),
            score=0.8,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_detect_runs_all_5():
    config = _build_config()
    pipeline = FilterPipeline(config)
    candidates = _build_candidates()

    kept, drops = pipeline.run(candidates, context=FilterContext.POST_DETECT)

    reasons = {d.reason for d in drops}
    expected = {
        DropReason.GEOMETRIC_FILTER,
        DropReason.SCORE_FLOOR,
        DropReason.DEDUP,
        DropReason.PER_CLASS_CAP,
        DropReason.CROSS_CLASS,
    }
    assert expected.issubset(reasons), (
        f"POST_DETECT missing reasons: expected {expected}, got {reasons}. "
        f"Drops: {[(d.candidate_id, d.reason) for d in drops]}"
    )
    assert len(kept) < len(candidates)


def test_post_review_runs_subset():
    config = _build_config()
    pipeline = FilterPipeline(config)
    candidates = _build_candidates()

    _, drops = pipeline.run(candidates, context=FilterContext.POST_REVIEW)

    reasons = {d.reason for d in drops}
    assert DropReason.CROSS_CLASS in reasons
    assert DropReason.PER_CLASS_CAP in reasons
    # Geometric/score/dedup are intentionally skipped in POST_REVIEW.
    for forbidden in (
        DropReason.GEOMETRIC_FILTER,
        DropReason.SCORE_FLOOR,
        DropReason.DEDUP,
    ):
        assert forbidden not in reasons, (
            f"POST_REVIEW produced forbidden drop reason: {forbidden}"
        )


def test_drops_are_context_tagged():
    config = _build_config()
    pipeline = FilterPipeline(config)
    candidates = _build_candidates()

    for ctx in (FilterContext.POST_DETECT, FilterContext.POST_REVIEW):
        _, drops = pipeline.run(candidates, context=ctx)
        assert drops, f"Expected drops for context {ctx}"
        for d in drops:
            assert d.context == ctx, (
                f"Drop {d.candidate_id} has context={d.context}, expected {ctx}"
            )
            assert isinstance(d.reason, DropReason), (
                f"Drop {d.candidate_id} reason is {type(d.reason)}, "
                f"not DropReason enum member"
            )
