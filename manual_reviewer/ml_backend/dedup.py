"""Shared dedup primitives for every auto-annotation route.

Two layers, applied uniformly:

  * **Internal NMS** — ``nms_regions(emitted, iou=0.85)`` mutually-suppress
    overlapping regions within a single route's output. Class-aware by
    default: a person box and a forklift box at the same location both
    survive (different classes).

  * **External dedup** — ``dedup_against(emitted, existing, iou=0.7)``
    drops emitted regions that overlap rectangles already on the canvas.
    The canvas pool is built by ``canvas_rectangles(task, context)`` and
    covers all three sources the reviewer can see at predict time:

       1. accepted ``task["annotations"][*].result`` (skipping cancelled),
       2. seeded-but-unaccepted ``task["predictions"][*].result``,
       3. the current draft ``context.result``.

Both layers use the **single** ``iou_xyxy`` IoU helper here — no per-route
re-implementations. ``routes.py`` and ``reconcile/propagate_static.py``
re-export it to keep the public surface stable for tests, but the
implementation lives here.

``batch_proposals`` is exempt from both layers — that route is the raw
per-detector evidence layer (aav4 already produced the deduped consensus
output as the green seeded predictions). Deduping it would erase the
"3 detectors agree at this spot" audit signal that's the only reason it
exists.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from manual_reviewer.ml_backend.ls_payload import ls_box_to_norm

logger = logging.getLogger(__name__)


__all__ = [
    "BboxNorm",
    "canvas_rectangles",
    "dedup_against",
    "iou_xyxy",
    "nms_regions",
]


BboxNorm = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def iou_xyxy(a: BboxNorm | list[float], b: BboxNorm | list[float]) -> float:
    """IoU between two normalized [x1, y1, x2, y2] boxes. 0.0 on empty union.

    Tolerates list and tuple inputs — every caller in the package uses
    one or the other and we don't want a wrapper layer.
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


# ---------------------------------------------------------------------------
# Region introspection
# ---------------------------------------------------------------------------


def _is_rectangle(region: dict[str, Any]) -> bool:
    rtype = (region.get("type") or "").lower()
    return rtype in {"rectanglelabels", "rectangle"}


def _region_bbox(region: dict[str, Any]) -> BboxNorm | None:
    """Extract a normalized [x1, y1, x2, y2] tuple from an LS region."""
    value = region.get("value") or {}
    bbox = ls_box_to_norm(value)
    if bbox is None:
        return None
    return (bbox[0], bbox[1], bbox[2], bbox[3])


def _region_label(region: dict[str, Any]) -> str | None:
    """Pull the class label off an LS region.

    Accepts every label-array key the XML may emit:

      * ``rectanglelabels`` — RectangleLabels (the bbox tool).
      * ``keypointlabels``  — KeyPointLabels (the smart_click tool).
      * ``labels``          — bare ``<Labels>`` block, kept for backward
        compat with older XML configs / external predictions.

    Returns ``None`` if no usable string is present, in which case
    class-aware dedup falls back to "no class" matching (any-class
    suppresses any-class).
    """
    value = region.get("value") or {}
    for key in ("rectanglelabels", "keypointlabels", "labels"):
        arr = value.get(key)
        if isinstance(arr, list) and arr:
            first = arr[0]
            if isinstance(first, str) and first:
                return first
    return None


def _region_score(region: dict[str, Any]) -> float:
    """Best-effort score read; missing/garbage/NaN → 0.0.

    NaN is normalised because Python's sort on NaN keys is implementation-
    defined (NaN compares neither less than nor greater than anything),
    which would make NMS order nondeterministic if a model emitted one.
    """
    raw = region.get("score", 0.0)
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return value


# ---------------------------------------------------------------------------
# Canvas pool
# ---------------------------------------------------------------------------


def canvas_rectangles(
    task: dict[str, Any],
    context: dict[str, Any] | None,
) -> list[tuple[BboxNorm, str | None]]:
    """All rectangle ``(bbox, class)`` pairs visible at predict time.

    Sources, in order:

      1. accepted annotations (``task["annotations"][*].result``) —
         cancelled or empty annotations are skipped;
      2. seeded predictions (``task["predictions"][*].result``) —
         covers the case where the reviewer re-fires a smart route on
         a task whose finalize boxes haven't been accepted yet;
      3. the current draft (``context.result``) — covers the
         exemplar-self duplicate for the V-tool and any accumulated
         in-flight regions.

    Order matters when a caller chooses to truncate the pool, but for
    IoU dedup the order is irrelevant.
    """
    out: list[tuple[BboxNorm, str | None]] = []

    annotations = task.get("annotations") or []
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get("was_cancelled") or ann.get("result_count") == 0:
                continue
            for region in ann.get("result") or []:
                if not isinstance(region, dict) or not _is_rectangle(region):
                    continue
                bbox = _region_bbox(region)
                if bbox is not None:
                    out.append((bbox, _region_label(region)))

    predictions = task.get("predictions") or []
    if isinstance(predictions, list):
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            for region in pred.get("result") or []:
                if not isinstance(region, dict) or not _is_rectangle(region):
                    continue
                bbox = _region_bbox(region)
                if bbox is not None:
                    out.append((bbox, _region_label(region)))

    if isinstance(context, dict):
        for region in context.get("result") or []:
            if not isinstance(region, dict) or not _is_rectangle(region):
                continue
            bbox = _region_bbox(region)
            if bbox is not None:
                out.append((bbox, _region_label(region)))

    return out


# ---------------------------------------------------------------------------
# NMS (within-route)
# ---------------------------------------------------------------------------


def nms_regions(
    regions: list[dict[str, Any]],
    *,
    iou: float = 0.85,
    class_aware: bool = True,
) -> list[dict[str, Any]]:
    """Greedy NMS over a list of LS rectangle regions.

    Sorts by ``score`` descending, then suppresses any later region with
    IoU > ``iou`` against an earlier survivor. With ``class_aware=True``
    (default), regions with different class labels never suppress each
    other — a person on a forklift produces two legitimate overlapping
    detections of different classes.

    Regions whose bbox can't be parsed are passed through unchanged
    (they're not suppressed, and they don't suppress others). Output
    order is score-descending across survivors, then unparseable
    regions in their original order.
    """
    if len(regions) <= 1:
        return list(regions)

    parsed: list[tuple[int, BboxNorm, str | None, float]] = []
    pass_through: list[dict[str, Any]] = []
    for i, region in enumerate(regions):
        bbox = _region_bbox(region)
        if bbox is None:
            pass_through.append(region)
            continue
        parsed.append((i, bbox, _region_label(region), _region_score(region)))

    # Score desc; break ties on original index for determinism.
    parsed.sort(key=lambda t: (-t[3], t[0]))

    suppressed: set[int] = set()
    keepers: list[int] = []
    for orig_idx, bbox, cls, _score in parsed:
        if orig_idx in suppressed:
            continue
        keepers.append(orig_idx)
        for other_idx, other_bbox, other_cls, _ in parsed:
            if other_idx == orig_idx or other_idx in suppressed:
                continue
            if class_aware and (cls or "") != (other_cls or ""):
                continue
            if iou_xyxy(bbox, other_bbox) > iou:
                suppressed.add(other_idx)

    survivors = [regions[i] for i in keepers]
    if suppressed:
        logger.info(
            "nms_regions: kept %d / %d (suppressed %d at iou>%.2f, class_aware=%s)",
            len(survivors),
            len(parsed),
            len(suppressed),
            iou,
            class_aware,
        )
    return survivors + pass_through


# ---------------------------------------------------------------------------
# External dedup (vs canvas)
# ---------------------------------------------------------------------------


def dedup_against(
    regions: list[dict[str, Any]],
    existing: list[tuple[BboxNorm, str | None]],
    *,
    iou: float = 0.7,
    class_aware: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Drop emitted regions that overlap existing canvas rectangles.

    With ``class_aware=True`` (default), an emitted region is dropped
    only if an existing rectangle of the **same class** covers it at
    IoU > ``iou``. Different-class overlap is preserved (person on
    forklift). Regions whose bbox can't be parsed are passed through
    unchanged.

    Returns ``(survivors, dropped_count)``. The count is logged to make
    "why are my SAM matches missing?" easier to diagnose.
    """
    if not regions or not existing:
        return list(regions), 0

    out: list[dict[str, Any]] = []
    dropped = 0
    for region in regions:
        bbox = _region_bbox(region)
        if bbox is None:
            out.append(region)
            continue
        cls = _region_label(region)
        is_dup = False
        for ex_bbox, ex_cls in existing:
            if class_aware and (cls or "") != (ex_cls or ""):
                continue
            if iou_xyxy(bbox, ex_bbox) > iou:
                is_dup = True
                break
        if is_dup:
            dropped += 1
        else:
            out.append(region)
    if dropped:
        logger.info(
            "dedup_against: kept %d / %d (dropped %d at iou>%.2f vs %d existing, class_aware=%s)",
            len(out),
            len(regions),
            dropped,
            iou,
            len(existing),
            class_aware,
        )
    return out, dropped
