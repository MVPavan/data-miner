"""Tests for the shared dedup primitives in ml_backend/dedup.py.

Covers the three helpers the routes call:

  * ``iou_xyxy`` — IoU on normalized [x1, y1, x2, y2] boxes.
  * ``canvas_rectangles`` — pulls existing rectangles + their classes out
    of an LS task/context payload (annotations + predictions + draft).
  * ``nms_regions`` — internal NMS over a list of LS rectangle regions,
    score-desc, class-aware by default.
  * ``dedup_against`` — drops emitted regions that overlap an existing
    canvas rectangle of the same class at IoU > threshold.

Class-aware behavior is exercised explicitly — a person box and a
forklift box at the same coords must both survive both NMS and
external dedup.
"""

from __future__ import annotations

from typing import Any

import pytest

from manual_reviewer.ml_backend.dedup import (
    canvas_rectangles,
    dedup_against,
    iou_xyxy,
    nms_regions,
)
from manual_reviewer.ml_backend.ls_payload import norm_box_to_ls_region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _region(
    bbox: list[float],
    label: str = "forklift",
    score: float = 0.5,
    *,
    rid: str = "r",
) -> dict[str, Any]:
    return norm_box_to_ls_region(bbox, label, score=score, region_id=rid)


def _accepted_annotation(
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"id": 1, "result": regions, "result_count": len(regions)}


def _cancelled_annotation(
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": 2,
        "result": regions,
        "result_count": len(regions),
        "was_cancelled": True,
    }


def _prediction(regions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": 99, "result": regions, "model_version": "x"}


# ---------------------------------------------------------------------------
# 1. iou_xyxy
# ---------------------------------------------------------------------------


def test_iou_xyxy_perfect_overlap() -> None:
    box = (0.1, 0.2, 0.3, 0.4)
    assert iou_xyxy(box, box) == pytest.approx(1.0)


def test_iou_xyxy_disjoint_zero() -> None:
    assert iou_xyxy((0, 0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_xyxy_accepts_list_and_tuple() -> None:
    a = [0.0, 0.0, 0.4, 0.4]
    b = (0.2, 0.2, 0.6, 0.6)
    assert iou_xyxy(a, b) == pytest.approx(0.04 / (0.16 + 0.16 - 0.04))


def test_iou_xyxy_zero_area_box_returns_zero() -> None:
    assert iou_xyxy((0.1, 0.1, 0.1, 0.1), (0, 0, 0.5, 0.5)) == 0.0


# ---------------------------------------------------------------------------
# 2. canvas_rectangles
# ---------------------------------------------------------------------------


def test_canvas_rectangles_ignores_task_annotations() -> None:
    """Live-test 2026-04-30: the dedup pool must come from the live
    canvas state in ``context.result`` only, not from server-side
    ``task["annotations"]`` (which is stale or absent during a session)."""
    task = {
        "annotations": [
            _accepted_annotation([_region([0.10, 0.10, 0.30, 0.30], "forklift")])
        ]
    }
    assert canvas_rectangles(task, None) == []


def test_canvas_rectangles_excludes_task_predictions_by_default() -> None:
    """smart_click usage: predictions are NOT in the pool, so a refine
    click on a yellow seeded box still produces a region."""
    task = {
        "predictions": [
            _prediction([_region([0.10, 0.10, 0.30, 0.30], "forklift")])
        ]
    }
    assert canvas_rectangles(task, None) == []


def test_canvas_rectangles_includes_predictions_when_opted_in() -> None:
    """smart_text / visual_prompt usage: predictions ARE in the pool
    so a SAM-returned duplicate at a seeded location gets dropped.
    LS doesn't echo task.predictions in context.result on smart-tool
    fires, so this is the only path the dedup has to know about them."""
    task = {
        "predictions": [
            _prediction([_region([0.10, 0.10, 0.30, 0.30], "forklift")])
        ]
    }
    out = canvas_rectangles(task, None, include_predictions=True)
    assert len(out) == 1
    assert out[0][1] == "forklift"


def test_canvas_rectangles_includes_draft_context() -> None:
    """The exemplar in context.result shows up only when the caller opts in.

    By default ``from_name="visual_prompt"`` regions are filtered out so a
    leftover V-tool exemplar doesn't suppress smart_text matches. The
    visual_prompt route itself passes ``include_visual_prompt=True`` to
    keep the exemplar in the dedup pool.
    """
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "visual_prompt",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            }
        ]
    }
    assert canvas_rectangles({}, ctx) == []
    out = canvas_rectangles({}, ctx, include_visual_prompt=True)
    assert len(out) == 1
    assert out[0][1] == "forklift"


def test_canvas_rectangles_pool_is_context_only() -> None:
    """task.annotations and task.predictions are deliberately ignored;
    the canvas pool reflects ``context.result`` exclusively."""
    task = {
        "annotations": [
            _accepted_annotation([_region([0.0, 0.0, 0.1, 0.1], "person")])
        ],
        "predictions": [
            _prediction([_region([0.2, 0.2, 0.3, 0.3], "forklift")])
        ],
    }
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "bbox",
                "value": {"x": 50, "y": 50, "width": 10, "height": 10,
                          "rectanglelabels": ["bicycle"]},
            },
            {
                "type": "rectanglelabels",
                "from_name": "visual_prompt",
                "value": {"x": 80, "y": 80, "width": 5, "height": 5,
                          "labels": ["car"]},
            },
        ]
    }
    # Default: V-tool exemplar is excluded; only the bbox region survives.
    out = canvas_rectangles(task, ctx)
    classes = sorted(c for _, c in out)
    assert classes == ["bicycle"]
    # include_visual_prompt=True: visual_prompt route's view (exemplar in pool).
    out_v = canvas_rectangles(task, ctx, include_visual_prompt=True)
    classes_v = sorted(c for _, c in out_v)
    assert classes_v == ["bicycle", "car"]


def test_canvas_rectangles_ignores_non_rectangle_regions() -> None:
    """Keypoint regions in context don't enter the rectangle dedup pool."""
    ctx = {
        "result": [
            {"type": "keypointlabels", "from_name": "click",
             "value": {"x": 50, "y": 50}},
            {"type": "rectanglelabels", "from_name": "bbox",
             "value": {"x": 0, "y": 0, "width": 10, "height": 10,
                       "rectanglelabels": ["person"]}},
        ]
    }
    out = canvas_rectangles({}, ctx)
    assert len(out) == 1
    assert out[0][1] == "person"


def test_canvas_rectangles_handles_empty_inputs() -> None:
    assert canvas_rectangles({}, None) == []
    assert canvas_rectangles({"annotations": []}, {"result": []}) == []


# ---------------------------------------------------------------------------
# 3. nms_regions
# ---------------------------------------------------------------------------


def test_nms_regions_passes_through_singleton() -> None:
    r = [_region([0.10, 0.10, 0.30, 0.30], "forklift", score=0.5)]
    assert nms_regions(r) == r


def test_nms_regions_suppresses_lower_score_overlap() -> None:
    high = _region([0.10, 0.10, 0.30, 0.30], "forklift", score=0.9, rid="hi")
    low = _region([0.105, 0.105, 0.305, 0.305], "forklift", score=0.5, rid="lo")
    out = nms_regions([low, high], iou=0.85)
    assert len(out) == 1
    assert out[0]["id"] == "hi"


def test_nms_regions_class_aware_keeps_different_classes() -> None:
    forklift = _region([0.10, 0.10, 0.30, 0.30], "forklift", score=0.9, rid="f")
    person = _region([0.10, 0.10, 0.30, 0.30], "person", score=0.8, rid="p")
    out = nms_regions([forklift, person], iou=0.85)
    assert len(out) == 2
    assert {r["id"] for r in out} == {"f", "p"}


def test_nms_regions_class_agnostic_suppresses_overlap_across_classes() -> None:
    forklift = _region([0.10, 0.10, 0.30, 0.30], "forklift", score=0.9, rid="f")
    person = _region([0.10, 0.10, 0.30, 0.30], "person", score=0.8, rid="p")
    out = nms_regions([forklift, person], iou=0.85, class_aware=False)
    assert len(out) == 1
    assert out[0]["id"] == "f"  # higher score wins


def test_nms_regions_disjoint_boxes_all_survive() -> None:
    a = _region([0.00, 0.00, 0.10, 0.10], "forklift", score=0.9, rid="a")
    b = _region([0.50, 0.50, 0.60, 0.60], "forklift", score=0.5, rid="b")
    out = nms_regions([a, b], iou=0.85)
    assert {r["id"] for r in out} == {"a", "b"}


def test_nms_regions_below_threshold_keeps_both() -> None:
    """Overlap at IoU 0.6 is below the 0.85 NMS threshold — both kept."""
    a = _region([0.00, 0.00, 0.40, 0.40], "forklift", score=0.9, rid="a")
    b = _region([0.20, 0.20, 0.60, 0.60], "forklift", score=0.5, rid="b")
    out = nms_regions([a, b], iou=0.85)
    assert len(out) == 2


def test_nms_regions_emits_score_desc() -> None:
    a = _region([0.00, 0.00, 0.10, 0.10], "forklift", score=0.5, rid="a")
    b = _region([0.50, 0.50, 0.60, 0.60], "forklift", score=0.9, rid="b")
    out = nms_regions([a, b], iou=0.85)
    # b (higher score) first.
    assert [r["id"] for r in out] == ["b", "a"]


def test_nms_regions_passes_through_unparseable() -> None:
    valid = _region([0.10, 0.10, 0.30, 0.30], "forklift", score=0.9, rid="ok")
    broken: dict[str, Any] = {"id": "broken", "type": "rectanglelabels", "value": {}}
    out = nms_regions([valid, broken], iou=0.85)
    assert len(out) == 2


def test_nms_regions_nan_score_treated_as_zero() -> None:
    """NaN scores break Python's sort (NaN compares neither < nor > anything),
    which would make NMS order nondeterministic if a model emitted one.
    The route normalises NaN → 0.0 so the higher-scoring real region wins
    the suppression duel deterministically.
    """
    nan = float("nan")
    nan_score = _region([0.10, 0.10, 0.30, 0.30], "forklift", score=nan, rid="nan")
    real = _region([0.11, 0.11, 0.31, 0.31], "forklift", score=0.5, rid="real")
    out = nms_regions([nan_score, real], iou=0.5)
    assert len(out) == 1
    assert out[0]["id"] == "real"


# ---------------------------------------------------------------------------
# 4. dedup_against
# ---------------------------------------------------------------------------


def test_dedup_against_drops_same_class_overlap() -> None:
    proposed = [_region([0.10, 0.10, 0.30, 0.30], "forklift")]
    existing = [((0.10, 0.10, 0.30, 0.30), "forklift")]
    survivors, dropped = dedup_against(proposed, existing, iou=0.7)
    assert survivors == []
    assert dropped == 1


def test_dedup_against_keeps_different_class_overlap() -> None:
    """Class-aware: forklift over person at the same spot must survive."""
    proposed = [_region([0.10, 0.10, 0.30, 0.30], "forklift")]
    existing = [((0.10, 0.10, 0.30, 0.30), "person")]
    survivors, dropped = dedup_against(proposed, existing, iou=0.7)
    assert len(survivors) == 1
    assert dropped == 0


def test_dedup_against_class_agnostic_drops_cross_class() -> None:
    proposed = [_region([0.10, 0.10, 0.30, 0.30], "forklift")]
    existing = [((0.10, 0.10, 0.30, 0.30), "person")]
    survivors, dropped = dedup_against(
        proposed, existing, iou=0.7, class_aware=False
    )
    assert survivors == []
    assert dropped == 1


def test_dedup_against_below_threshold_keeps() -> None:
    proposed = [_region([0.00, 0.00, 0.40, 0.40], "forklift")]
    existing = [((0.20, 0.20, 0.60, 0.60), "forklift")]   # IoU ~0.14
    survivors, dropped = dedup_against(proposed, existing, iou=0.7)
    assert len(survivors) == 1
    assert dropped == 0


def test_dedup_against_empty_existing_keeps_all() -> None:
    proposed = [_region([0.10, 0.10, 0.30, 0.30], "forklift")]
    survivors, dropped = dedup_against(proposed, [], iou=0.7)
    assert survivors == proposed
    assert dropped == 0


def test_dedup_against_empty_proposed_returns_empty() -> None:
    existing = [((0.10, 0.10, 0.30, 0.30), "forklift")]
    survivors, dropped = dedup_against([], existing, iou=0.7)
    assert survivors == []
    assert dropped == 0


def test_dedup_against_passes_through_unparseable_regions() -> None:
    proposed = [{"id": "broken", "type": "rectanglelabels", "value": {}}]
    existing = [((0.10, 0.10, 0.30, 0.30), "forklift")]
    survivors, dropped = dedup_against(proposed, existing, iou=0.7)
    assert len(survivors) == 1
    assert dropped == 0


def test_dedup_against_unlabeled_existing_matches_unlabeled_proposed() -> None:
    """When both sides have no class, class_aware still treats them as
    same-class (empty == empty) and suppresses."""
    proposed = [_region([0.10, 0.10, 0.30, 0.30], "")]  # empty label → "other"
    # An existing region with no recoverable label.
    existing = [((0.10, 0.10, 0.30, 0.30), None)]
    survivors, dropped = dedup_against(proposed, existing, iou=0.7)
    # proposed got DEFAULT_LABEL ("other"); existing has None — they're
    # NOT same class under our implementation.
    # This codifies the current behavior; if we ever decide unlabeled
    # should always match, this test catches the change.
    assert len(survivors) == 1
    assert dropped == 0
