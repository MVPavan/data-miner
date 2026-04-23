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
    AutoAcceptConfig,
    AutoAnnotationV4Config,
    ClassConfig,
    CoExistenceConfig,
    FilterConfig,
    IouDedupConfig,
)
from data_miner.auto_annotation_v4.filters import FilterPipeline
from data_miner.auto_annotation_v4.utils import route_candidates


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
        model_priority=["grounding_dino", "sam3"],
    )
    filtering = FilterConfig(
        min_area=0.001,
        max_area=0.95,
        min_aspect_ratio=0.1,
        max_aspect_ratio=10.0,
        min_edge_distance=0.0,
        per_model_score={"grounding_dino": 0.2, "sam3": 0.0},
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
            source_model="grounding_dino",
            expression="tiny",
            bbox=_bbox(0.10, 0.10, 0.101, 0.101),  # area ~1e-6
            score=0.9,
        ),
        # 2) SCORE_FLOOR — gdino floor 0.2, score 0.1.
        Candidate(
            candidate_id="scor1",
            class_name="class_score_fail",
            label="low_score",
            source_model="grounding_dino",
            expression="low",
            bbox=_bbox(0.20, 0.20, 0.30, 0.30),
            score=0.1,
        ),
        # 3) DEDUP — two heavily overlapping same-class candidates (IoU > 0.5).
        Candidate(
            candidate_id="dup_A",
            class_name="class_dup",
            label="dup",
            source_model="grounding_dino",
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
            source_model="grounding_dino",
            expression="cap",
            bbox=_bbox(0.70, 0.05, 0.80, 0.15),
            score=0.9,
        ),
        Candidate(
            candidate_id="cap2",
            class_name="class_small_cap",
            label="cap2",
            source_model="grounding_dino",
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
            source_model="grounding_dino",
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


# ---------------------------------------------------------------------------
# route_candidates — auto-accept high-confidence shortcut
# ---------------------------------------------------------------------------


def _build_config_with_hi_conf(hi_conf: dict[str, float]) -> AutoAnnotationV4Config:
    """Minimal config for route_candidates tests.

    Tier-1 eligible ``person``, per-model floor 0.5 for sam3_dart, and the
    high-confidence shortcut under test. ``min_model_agreement=2`` so the
    agreement path never fires for single-detector cases.
    """
    return AutoAnnotationV4Config(
        class_registry={
            "person": ClassConfig(id=0, tier=1, prompts=["person"], tags=[]),
        },
        co_existence=CoExistenceConfig(),
        filtering=FilterConfig(
            per_model_score={"sam3_dart": 0.5, "grounding_dino": 0.35},
            iou_dedup=IouDedupConfig(
                threshold=0.5,
                tiebreak_by=["agreement", "model_priority", "score"],
                model_priority=["sam3_dart", "grounding_dino"],
            ),
        ),
        auto_accept=AutoAcceptConfig(
            min_model_agreement=2,
            tiers=[1],
            high_confidence_scores=hi_conf,
        ),
    )


def _mk(cid: str, model: str, score: float, agreement: int = 1) -> Candidate:
    return Candidate(
        candidate_id=cid, class_name="person", label="person",
        source_model=model, expression="person",
        bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5),
        score=score, agreement=agreement, agreeing_models=[model],
    )


def test_auto_accept_hi_conf_single_model():
    """sam3_dart alone (agreement=1) auto-accepts when score >= 0.85."""
    cfg = _build_config_with_hi_conf({"sam3_dart": 0.85})
    out = route_candidates(
        [
            _mk("hi",  "sam3_dart", 0.90, agreement=1),  # passes shortcut
            _mk("eq",  "sam3_dart", 0.85, agreement=1),  # boundary passes
            _mk("low", "sam3_dart", 0.84, agreement=1),  # just under -> VLM
        ],
        cfg,
    )
    assert out["auto_accepted"] == ["hi", "eq"]
    assert out["needs_evaluation"] == ["low"]


def test_auto_accept_hi_conf_respects_per_model_floor():
    """Per-model floor is still enforced even if hi_conf_score passes."""
    cfg = _build_config_with_hi_conf({"sam3_dart": 0.40})  # pathological
    out = route_candidates(
        [_mk("under_floor", "sam3_dart", 0.45, agreement=1)],  # hi_conf ok, floor 0.5 fails
        cfg,
    )
    assert out["auto_accepted"] == []
    assert out["needs_evaluation"] == ["under_floor"]


def test_auto_accept_hi_conf_other_models_unaffected():
    """Models without a hi_conf entry fall back to the agreement path."""
    cfg = _build_config_with_hi_conf({"sam3_dart": 0.85})
    out = route_candidates(
        [
            _mk("gdino_hi",  "grounding_dino", 0.99, agreement=1),  # no hi_conf -> VLM
            _mk("gdino_agr", "grounding_dino", 0.99, agreement=2),  # agreement path -> auto
        ],
        cfg,
    )
    assert out["auto_accepted"] == ["gdino_agr"]
    assert out["needs_evaluation"] == ["gdino_hi"]


def test_auto_accept_hi_conf_empty_is_legacy():
    """Empty high_confidence_scores preserves the pre-change behavior."""
    cfg = _build_config_with_hi_conf({})  # no shortcut
    out = route_candidates(
        [_mk("strong", "sam3_dart", 0.99, agreement=1)],
        cfg,
    )
    assert out["auto_accepted"] == []
    assert out["needs_evaluation"] == ["strong"]


# ---------------------------------------------------------------------------
# allowed_source_models — pipeline-wide source_model suppression
# ---------------------------------------------------------------------------


def _build_config_with_allowlist(allowed: list[str]) -> AutoAnnotationV4Config:
    return AutoAnnotationV4Config(
        class_registry={
            "person": ClassConfig(id=0, tier=1, prompts=["person"], tags=[]),
        },
        co_existence=CoExistenceConfig(),
        filtering=FilterConfig(
            per_model_score={"sam3_dart": 0.0, "grounding_dino": 0.0},
            iou_dedup=IouDedupConfig(
                threshold=0.9,
                tiebreak_by=["agreement", "model_priority", "score"],
                model_priority=["sam3_dart", "grounding_dino"],
            ),
            max_per_class=100,
            allowed_source_models=allowed,
        ),
    )


def _mk_cand(cid: str, model: str, x_offset: float = 0.0) -> Candidate:
    """Non-overlapping bbox per ``x_offset`` so dedup doesn't cluster these."""
    return Candidate(
        candidate_id=cid, class_name="person", label="person",
        source_model=model, expression="person",
        bbox=BoundingBox(
            x1=0.05 + x_offset, y1=0.05,
            x2=0.15 + x_offset, y2=0.15,
        ),
        score=0.8,
    )


def test_allowed_source_models_empty_is_passthrough():
    """Empty allowlist = no-op (preserves existing config behavior)."""
    cfg = _build_config_with_allowlist([])
    pipeline = FilterPipeline(cfg)
    cands = [_mk_cand("gd", "grounding_dino", 0.0), _mk_cand("s3", "sam3_dart", 0.5)]

    kept, drops = pipeline.run(cands, FilterContext.POST_DETECT)

    source_drops = [d for d in drops if d.reason == DropReason.SOURCE_MODEL]
    assert source_drops == []
    assert {c.candidate_id for c in kept} == {"gd", "s3"}


def test_allowed_source_models_suppresses_disallowed():
    """Non-empty allowlist drops anything outside it with SOURCE_MODEL reason."""
    cfg = _build_config_with_allowlist(["sam3_dart"])
    pipeline = FilterPipeline(cfg)
    cands = [
        _mk_cand("gd1", "grounding_dino", 0.0),
        _mk_cand("gd2", "grounding_dino", 0.3),
        _mk_cand("s3", "sam3_dart", 0.6),
    ]

    kept, drops = pipeline.run(cands, FilterContext.POST_DETECT)

    source_drops = [d for d in drops if d.reason == DropReason.SOURCE_MODEL]
    assert {d.candidate_id for d in source_drops} == {"gd1", "gd2"}
    assert all(d.context == FilterContext.POST_DETECT for d in source_drops)
    assert {c.candidate_id for c in kept} == {"s3"}
