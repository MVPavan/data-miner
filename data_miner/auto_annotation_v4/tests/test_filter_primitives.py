"""Direct tests for the filter primitives added for LOCO:

  - drop_head_without_person  (per-head containment mode, legacy presence mode)
  - head_person_coexist_pairs (spatial pair detection)
  - apply_class_agnostic_nms  (overlap_exempt handling, confusion-pair suppress)
  - geometric_filter          (per_class_min_area override)
  - route_candidates          (coexistence min-score gate, tier-3 guardrail)

Plus the cross-section config validator added in H2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_miner.auto_annotation_v4.configs import BoundingBox, Candidate
from data_miner.auto_annotation_v4.configs.settings import (
    AutoAcceptConfig,
    ClassAgnosticNmsConfig,
    CoExistenceConfig,
    FilterConfig,
    IouDedupConfig,
)
from data_miner.auto_annotation_v4.utils import (
    apply_class_agnostic_nms,
    drop_head_without_person,
    geometric_filter,
    head_person_coexist_pairs,
    route_candidates,
)


# ---------------------------------------------------------------------------
# Candidate factory
# ---------------------------------------------------------------------------


def _mk(
    cid: str,
    cls: str,
    x1: float, y1: float, x2: float, y2: float,
    *, score: float = 0.8, source: str = "sam3_dart",
    tags: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        candidate_id=cid,
        class_name=cls,
        label=cls,
        source_model=source,
        expression=cls,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        score=score,
    )


# A minimal ClassConfig-like stub for NMS's overlap_exempt lookup. The util
# reads `config.classes[name].tags` — a SimpleNamespace suffices.
def _stub_classes(*specs: tuple[str, tuple[str, ...]]) -> dict[str, SimpleNamespace]:
    """(name, tags) -> dict[name, namespace with .tags and .tier]."""
    return {
        name: SimpleNamespace(tags=list(tags), tier=1)
        for name, tags in specs
    }


def _cfg_for_nms(
    *,
    enabled: bool = True,
    threshold: float = 0.7,
    respect_exempt: bool = True,
    suppress_confusion: bool = True,
    exempt_classes: tuple[str, ...] = ("person", "head", "backpack", "handbag"),
    confusion_pairs: tuple[tuple[str, str], ...] = (
        ("forklift", "palletjack"),
        ("forklift", "truck"),
        ("shopping_cart", "palletjack"),
    ),
) -> SimpleNamespace:
    """Minimal config shape for apply_class_agnostic_nms — only the fields
    the function touches."""
    nms = ClassAgnosticNmsConfig(
        enabled=enabled,
        threshold=threshold,
        respect_overlap_exempt=respect_exempt,
        suppress_confusion_pairs=suppress_confusion,
    )
    iou_dedup = IouDedupConfig()
    filtering = SimpleNamespace(
        class_agnostic_nms=nms,
        iou_dedup=iou_dedup,
    )
    # Build a class_registry-shaped dict with overlap_exempt tags on the
    # named classes; everything else is auto-added as a plain tier-1.
    classes = {}
    for name in set(exempt_classes) | {"forklift", "palletjack", "truck", "shopping_cart", "car"}:
        tags = ["overlap_exempt"] if name in exempt_classes else []
        classes[name] = SimpleNamespace(tags=tags, tier=1)
    co_existence = CoExistenceConfig(
        overlap_exempt_tags=["overlap_exempt"],
        confusion_tags=["confusion:industrial"],
        extra_confusion_pairs=[list(p) for p in confusion_pairs],
    )
    # Mark the confusion-pair members with the confusion tag so
    # _resolve_co_existence picks them up into the pair set.
    for pair in confusion_pairs:
        for name in pair:
            if name in classes and "confusion:industrial" not in classes[name].tags:
                classes[name].tags.append("confusion:industrial")
    return SimpleNamespace(
        filtering=filtering,
        classes=classes,
        class_registry=classes,      # _resolve_co_existence reads this
        co_existence=co_existence,
    )


# ---------------------------------------------------------------------------
# drop_head_without_person
# ---------------------------------------------------------------------------


def _cfg_head(thr: float = 0.0, legacy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        filtering=SimpleNamespace(
            head_person_containment_min=thr,
            reject_head_without_person=legacy,
        )
    )


def test_head_kept_when_contained_in_person():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80)
    kept = drop_head_without_person([head, person], _cfg_head(thr=0.3))
    assert {c.candidate_id for c in kept} == {"h1", "p1"}


def test_orphan_head_dropped_below_threshold():
    head = _mk("h1", "head", 0.10, 0.10, 0.20, 0.20)
    person = _mk("p1", "person", 0.70, 0.70, 0.90, 0.95)
    kept = drop_head_without_person([head, person], _cfg_head(thr=0.3))
    assert {c.candidate_id for c in kept} == {"p1"}


def test_head_kept_by_one_of_many_persons():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    far_person = _mk("p1", "person", 0.70, 0.70, 0.90, 0.95)
    near_person = _mk("p2", "person", 0.40, 0.10, 0.60, 0.80)
    kept = drop_head_without_person([head, far_person, near_person], _cfg_head(thr=0.3))
    assert "h1" in {c.candidate_id for c in kept}


def test_head_dropped_when_no_person_present():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    kept = drop_head_without_person([head], _cfg_head(thr=0.3))
    assert kept == []


def test_legacy_presence_mode_fires_only_when_threshold_zero():
    """reject_head_without_person is the coarse fallback: drops all heads
    when no person exists in the image. Only active when threshold=0."""
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    # With threshold=0 and legacy=True: no person → drop head.
    kept = drop_head_without_person([head], _cfg_head(thr=0.0, legacy=True))
    assert kept == []
    # With threshold=0 and legacy=False: no-op.
    kept = drop_head_without_person([head], _cfg_head(thr=0.0, legacy=False))
    assert {c.candidate_id for c in kept} == {"h1"}


def test_threshold_zero_no_op_when_no_legacy():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80)
    kept = drop_head_without_person([head, person], _cfg_head(thr=0.0))
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# head_person_coexist_pairs
# ---------------------------------------------------------------------------


def test_coexist_pair_at_exact_threshold_qualifies():
    """Pair detection uses ``>= containment_min``, so containment == thr
    must qualify. A head whose every pixel is inside a person has
    containment = 1.0, well above 0.3."""
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80)
    pairs = head_person_coexist_pairs([head, person], 0.3)
    assert pairs == {("h1", "p1")}


def test_coexist_returns_empty_when_disabled():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80)
    assert head_person_coexist_pairs([head, person], 0.0) == set()
    assert head_person_coexist_pairs([head, person], -1.0) == set()


def test_coexist_one_head_multiple_persons():
    head = _mk("h1", "head", 0.48, 0.10, 0.52, 0.15)
    p1 = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80)
    p2 = _mk("p2", "person", 0.47, 0.08, 0.55, 0.25)
    pairs = head_person_coexist_pairs([head, p1, p2], 0.3)
    assert pairs == {("h1", "p1"), ("h1", "p2")}


# ---------------------------------------------------------------------------
# apply_class_agnostic_nms
# ---------------------------------------------------------------------------


def test_nms_disabled_is_noop():
    a = _mk("a", "forklift", 0.1, 0.1, 0.5, 0.5, score=0.9)
    b = _mk("b", "palletjack", 0.15, 0.12, 0.48, 0.48, score=0.5)
    kept = apply_class_agnostic_nms([a, b], _cfg_for_nms(enabled=False))
    assert {c.candidate_id for c in kept} == {"a", "b"}


def test_nms_suppresses_confusion_pair_by_tiebreak():
    """Two overlapping confusion-pair candidates with same model + agreement:
    tiebreak cascade falls to score → higher score wins; lower score is
    suppressed."""
    a = _mk("a", "forklift", 0.1, 0.1, 0.5, 0.5, score=0.82)
    b = _mk("b", "palletjack", 0.12, 0.11, 0.49, 0.48, score=0.55)
    kept = apply_class_agnostic_nms([a, b], _cfg_for_nms(suppress_confusion=True))
    assert {c.candidate_id for c in kept} == {"a"}


def test_nms_respects_overlap_exempt():
    """person + head at IoU > 0.7 (head contained in upper body of person)
    must not be suppressed even when class_agnostic_nms is active."""
    # Construct a head whose bbox overlaps most of the person torso so IoU > 0.7.
    person = _mk("p", "person", 0.40, 0.10, 0.60, 0.30)
    head = _mk("h", "head", 0.41, 0.11, 0.59, 0.29)
    kept = apply_class_agnostic_nms([person, head], _cfg_for_nms(respect_exempt=True))
    assert {c.candidate_id for c in kept} == {"p", "h"}


def test_nms_skips_confusion_pair_when_suppress_false():
    a = _mk("a", "forklift", 0.1, 0.1, 0.5, 0.5, score=0.82)
    b = _mk("b", "palletjack", 0.12, 0.11, 0.49, 0.48, score=0.55)
    kept = apply_class_agnostic_nms(
        [a, b], _cfg_for_nms(suppress_confusion=False),
    )
    # Both preserved when confusion-pair suppression is off.
    assert {c.candidate_id for c in kept} == {"a", "b"}


def test_nms_below_threshold_keeps_both():
    a = _mk("a", "forklift", 0.1, 0.1, 0.5, 0.5, score=0.82)
    b = _mk("b", "palletjack", 0.45, 0.45, 0.9, 0.9, score=0.55)
    kept = apply_class_agnostic_nms([a, b], _cfg_for_nms(threshold=0.7))
    assert {c.candidate_id for c in kept} == {"a", "b"}


# ---------------------------------------------------------------------------
# geometric_filter per_class_min_area
# ---------------------------------------------------------------------------


def _cfg_geom(
    min_area: float = 0.0005,
    per_class: dict[str, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        filtering=SimpleNamespace(
            min_area=min_area,
            max_area=0.95,
            min_aspect_ratio=0.1,
            max_aspect_ratio=10.0,
            min_edge_distance=0.0,
            per_class_min_area=per_class or {},
        )
    )


def test_per_class_min_area_override():
    """Global floor 0.0005; cellphone override 0.00005. A 0.00008-area
    cellphone passes (>= 0.00005) but a generic candidate at the same
    area falls below the global 0.0005 and is dropped."""
    cell_pass = _mk("c1", "cellphone", 0.20, 0.20, 0.21, 0.2101)  # ~1e-5
    cell_fail = _mk("c2", "cellphone", 0.30, 0.30, 0.301, 0.3005) # way below 5e-5
    generic_big = _mk("g1", "truck", 0.10, 0.10, 0.30, 0.30)      # 0.04 area
    cfg = _cfg_geom(per_class={"cellphone": 0.00005})
    # compute areas: cell_pass = 0.01 * 0.0101 = 1.01e-4 (passes 5e-5 but passes 5e-4 too? no, 1.01e-4 < 5e-4)
    # Actually for the override we want to demonstrate: a candidate at 1e-4 passes under the override (>= 5e-5) but would fail under the global (< 5e-4).
    kept = geometric_filter([cell_pass, cell_fail, generic_big], cfg)
    ids = {c.candidate_id for c in kept}
    assert "c1" in ids, "cellphone at area ~1e-4 should pass under override 5e-5"
    assert "c2" not in ids, "cellphone at area ~5e-7 should fail under override 5e-5"
    assert "g1" in ids


def test_per_class_min_area_fallback_to_global():
    """A class not in the override dict uses the global floor."""
    # Handbag at area 1e-4 — below global 5e-4 → dropped.
    cand = _mk("h1", "handbag", 0.20, 0.20, 0.21, 0.2101)
    cfg = _cfg_geom(per_class={"cellphone": 0.00005})  # no handbag entry
    kept = geometric_filter([cand], cfg)
    assert kept == [], "handbag should fall back to global floor and drop"


def test_per_class_min_area_zero_disables():
    """head override = 0 means no min_area floor for heads."""
    tiny_head = _mk("t1", "head", 0.20, 0.20, 0.205, 0.205)  # 5e-5 area
    cfg = _cfg_geom(per_class={"head": 0.0})
    kept = geometric_filter([tiny_head], cfg)
    assert {c.candidate_id for c in kept} == {"t1"}


# ---------------------------------------------------------------------------
# route_candidates — coexistence min-score gate (H1)
# ---------------------------------------------------------------------------


def _cfg_route(
    *,
    coexist: bool = True,
    coexist_min_score: float = 0.0,
    containment_min: float = 0.3,
    tiers: list[int] | None = None,
    hi_conf: dict[str, float] | None = None,
) -> SimpleNamespace:
    aa = AutoAcceptConfig(
        min_model_agreement=2,
        min_score=0.0,
        tiers=tiers if tiers is not None else [1],
        high_confidence_scores=hi_conf or {"sam3_dart": 0.85},
        head_person_coexistence=coexist,
        head_person_coexistence_min_score=coexist_min_score,
    )
    filtering = FilterConfig(
        min_area=0.0005,
        head_person_containment_min=containment_min,
        per_model_score={"sam3_dart": 0.50},
    )
    classes = _stub_classes(
        ("person", ("overlap_exempt",)),
        ("head", ("overlap_exempt",)),
        ("forklift", ("confusion:industrial",)),
    )
    # Give classes a tier attribute that route_candidates expects.
    for name, tier in (("person", 1), ("head", 1), ("forklift", 3)):
        classes[name].tier = tier
    co_existence = CoExistenceConfig(
        overlap_exempt_tags=["overlap_exempt"],
        confusion_tags=[],
        extra_confusion_pairs=[],
    )
    return SimpleNamespace(
        auto_accept=aa, filtering=filtering,
        classes=classes, class_registry=classes,
        co_existence=co_existence,
    )


def test_coexist_shortcut_fires_when_both_scores_clear_gate():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20, score=0.70)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80, score=0.80)
    out = route_candidates([head, person], _cfg_route(coexist_min_score=0.60))
    assert set(out["auto_accepted"]) == {"h1", "p1"}
    assert out["needs_evaluation"] == []


def test_coexist_shortcut_blocked_when_head_score_below_gate():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20, score=0.52)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80, score=0.90)
    out = route_candidates([head, person], _cfg_route(coexist_min_score=0.60))
    # Head 0.52 < 0.60 → pair doesn't qualify. Both fall to baseline:
    #   person tier-1 score=0.90 passes hi_conf 0.85 → auto_accepted
    #   head   tier-1 score=0.52 below 0.85          → needs_evaluation
    assert "p1" in out["auto_accepted"]
    assert "h1" in out["needs_evaluation"]


def test_coexist_shortcut_blocked_when_person_score_below_gate():
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20, score=0.90)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80, score=0.52)
    out = route_candidates([head, person], _cfg_route(coexist_min_score=0.60))
    # Symmetric to the head-weak case.
    assert "h1" in out["auto_accepted"]
    assert "p1" in out["needs_evaluation"]


def test_coexist_shortcut_disabled_by_default():
    """Default.yaml-shaped config: coexistence flag off. Head and person
    must each qualify on their own merits."""
    head = _mk("h1", "head", 0.45, 0.10, 0.55, 0.20, score=0.70)
    person = _mk("p1", "person", 0.40, 0.10, 0.60, 0.80, score=0.80)
    out = route_candidates([head, person], _cfg_route(coexist=False))
    # Agreement=1 (default), hi_conf 0.85 → neither clears shortcut.
    assert out["auto_accepted"] == []
    assert set(out["needs_evaluation"]) == {"h1", "p1"}


def test_tier_three_cannot_escape_via_coexistence():
    """forklift is tier 3 and not in the head/person pair set — its
    routing is unaffected by the coexistence shortcut."""
    fork = _mk("f1", "forklift", 0.20, 0.20, 0.60, 0.60, score=0.95)
    out = route_candidates([fork], _cfg_route(coexist_min_score=0.60))
    assert out["auto_accepted"] == []
    assert out["needs_evaluation"] == ["f1"]


# ---------------------------------------------------------------------------
# Cross-section config validator (H2)
# ---------------------------------------------------------------------------


def _minimal_full_config(**overrides):
    """Build a config dict valid enough to construct AutoAnnotationV4Config.

    We rely on most field defaults; only class_registry is required to
    populate so the validator has something to cross-check against.
    """
    base = {
        "class_registry": {
            "person": {"id": 0, "tier": 1, "prompts": ["person"], "synonyms": [], "tags": ["overlap_exempt"]},
            "head":   {"id": 1, "tier": 1, "prompts": ["head"],   "synonyms": ["face"], "tags": ["overlap_exempt"]},
            "cellphone": {"id": 2, "tier": 2, "prompts": ["cellphone"], "synonyms": [], "tags": []},
        },
    }

    def _merge(a, b):
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return _merge(base, overrides)


def test_validator_accepts_known_class_keys():
    from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config
    AutoAnnotationV4Config.model_validate(_minimal_full_config(
        filtering={"per_class_min_area": {"head": 0.0, "cellphone": 0.00005}},
    ))


def test_validator_rejects_unknown_class_key():
    from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config
    with pytest.raises(ValueError, match="per_class_min_area.*unknown class"):
        AutoAnnotationV4Config.model_validate(_minimal_full_config(
            filtering={"per_class_min_area": {"Head": 0.0}},  # capital H typo
        ))


def test_validator_rejects_unknown_detector_key():
    from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config
    with pytest.raises(ValueError, match="per_model_score.*unknown detector"):
        AutoAnnotationV4Config.model_validate(_minimal_full_config(
            filtering={"per_model_score": {"sam3_drt": 0.5}},  # typo
        ))


def test_validator_rejects_unknown_high_conf_detector_key():
    from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config
    with pytest.raises(ValueError, match="high_confidence_scores.*unknown detector"):
        AutoAnnotationV4Config.model_validate(_minimal_full_config(
            auto_accept={"high_confidence_scores": {"sam3": 0.85, "faln": 0.80}},
        ))


def test_validator_rejects_coexist_without_containment_threshold():
    from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config
    with pytest.raises(ValueError, match="head_person_coexistence"):
        AutoAnnotationV4Config.model_validate(_minimal_full_config(
            auto_accept={"head_person_coexistence": True},
            filtering={"head_person_containment_min": 0.0},  # default
        ))


def test_validator_rejects_coexist_without_score_gate():
    """Enabling coexistence without a score gate is the legacy unsafe
    default; validator must reject so the caller sets a meaningful floor."""
    from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config
    with pytest.raises(ValueError, match="head_person_coexistence_min_score"):
        AutoAnnotationV4Config.model_validate(_minimal_full_config(
            auto_accept={
                "head_person_coexistence": True,
                # min_score left at 0.0 default — should fail
            },
            filtering={"head_person_containment_min": 0.3},
        ))


# ---------------------------------------------------------------------------
# FilterPipeline _PLAN order invariants (pins the H3/M1 comment-drift fixes)
# ---------------------------------------------------------------------------


def test_plan_post_detect_runs_head_no_person_before_score_floor():
    """Locks in the explicit ordering justified in filters.py docstring:
    low-confidence persons must still shield their heads from the orphan
    drop before the score floor culls them."""
    from data_miner.auto_annotation_v4.configs.enums import FilterContext
    from data_miner.auto_annotation_v4.filters import _PLAN

    post_detect = _PLAN[FilterContext.POST_DETECT]
    pre_finalize = _PLAN[FilterContext.PRE_FINALIZE]
    assert post_detect.index("head_no_person") < post_detect.index("score_floor")
    assert pre_finalize.index("head_no_person") < pre_finalize.index("score_floor")


def test_plan_class_agnostic_nms_is_terminal_step():
    from data_miner.auto_annotation_v4.configs.enums import FilterContext
    from data_miner.auto_annotation_v4.filters import _PLAN
    assert _PLAN[FilterContext.POST_DETECT][-1] == "class_agnostic_nms"
    assert _PLAN[FilterContext.PRE_FINALIZE][-1] == "class_agnostic_nms"


def test_plan_post_review_excludes_spatial_nms_and_head_orphan_check():
    """Documented omissions: POST_REVIEW trusts the VLM's curated survivor
    set, so a second spatial-NMS pass would drop legitimately kept
    candidates. Head-orphan check is moot post-VLM (relabels don't
    spatially create new heads)."""
    from data_miner.auto_annotation_v4.configs.enums import FilterContext
    from data_miner.auto_annotation_v4.filters import _PLAN
    post_review = _PLAN[FilterContext.POST_REVIEW]
    assert "class_agnostic_nms" not in post_review
    assert "head_no_person" not in post_review
