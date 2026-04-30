"""Parse a Label Studio completion into a ``HumanReviewResult``.

LS exports completions with this rough shape (relevant fields only)::

    {
      "id": <completion_id>,
      "task": <task_id>,
      "result": [
        {"id": "<region_id>",
         "from_name": "bbox", "to_name": "image",
         "value": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0,
                   "rotation": 0, "rectanglelabels": ["forklift"]}},
        ...
      ],
      "completed_by": <user_id>,
      "lead_time": <seconds>,
      "created_at": "<iso>",
    }

The build_tasks predictions seed each region's ``id`` with the aav4
``candidate_id``, so the parser distinguishes:

- ``finalize`` — region matches a seeded id and bbox + class are unchanged.
- ``edited`` — region id matches but bbox geometry differs.
- ``relabeled`` — region id matches but rectanglelabels differ from the
  original final annotation.
- ``added`` — region id is missing from the seeded set (reviewer drew it).
- ``kept_dropped`` — region id matches a ``ghost_drops`` entry the reviewer
  promoted by toggling its label group on. (Reviewer must move ghost drops
  into the canonical bbox group; until they do, kept_dropped is unreached.)

Box coordinates are converted from LS's 0-100 percent space back to aav4's
normalized [0, 1] BoundingBox.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any, Iterable

from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    HumanCorrection,
    HumanReviewResult,
)

logger = logging.getLogger(__name__)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_ls_completion(
    completion: dict[str, Any],
    *,
    image_id: str,
    seeded_predictions: list[dict[str, Any]] | None = None,
    ghost_drop_ids: Iterable[str] | None = None,
    reviewer_id: str | None = None,
) -> HumanReviewResult:
    """Convert one LS completion to a ``HumanReviewResult``.

    ``seeded_predictions`` is the same list ``build_task`` placed in the
    ``predictions[0].result`` field — use it to compare reviewer output
    against the original finalize annotations.

    ``ghost_drop_ids`` is the set of candidate_ids surfaced as ghost drops in
    the task ``data`` blob. If the reviewer promoted one (drew a regular
    rectangle whose id matches), it is tagged ``kept_dropped``.
    """
    seeded_by_id: dict[str, dict[str, Any]] = {
        pred["id"]: pred
        for pred in (seeded_predictions or [])
        if isinstance(pred, dict) and pred.get("id")
    }
    ghost_ids = set(ghost_drop_ids or [])

    raw_results = completion.get("result") or []
    rectangles: list[dict[str, Any]] = [
        r for r in raw_results
        if isinstance(r, dict) and r.get("type") == "rectanglelabels"
    ]
    track_ids = _extract_track_ids(raw_results)
    seen_ids: set[str] = set()
    corrections: list[HumanCorrection] = []

    for region in rectangles:
        region_id = region.get("id") or ""
        value = region.get("value") or {}
        labels = value.get("rectanglelabels") or []
        class_name = labels[0] if labels else ""
        if not class_name:
            continue
        try:
            bbox = _ls_value_to_bbox(value)
        except ValueError:
            continue

        seed = seeded_by_id.get(region_id)
        original_class: str | None = None
        original_bbox: BoundingBox | None = None
        source: str

        if seed is None:
            source = "kept_dropped" if region_id in ghost_ids else "added"
        else:
            seen_ids.add(region_id)
            seed_value = seed.get("value") or {}
            seed_labels = seed_value.get("rectanglelabels") or []
            seed_class = seed_labels[0] if seed_labels else ""
            seed_bbox = _ls_value_to_bbox(seed_value)
            class_changed = bool(seed_class) and seed_class != class_name
            bbox_changed = not _bbox_equal(seed_bbox, bbox)
            if class_changed and bbox_changed:
                source = "relabeled"
                original_class = seed_class
                original_bbox = seed_bbox
            elif class_changed:
                source = "relabeled"
                original_class = seed_class
            elif bbox_changed:
                source = "edited"
                original_bbox = seed_bbox
            else:
                source = "finalize"

        corrections.append(
            HumanCorrection(
                candidate_id=region_id or None,
                class_name=class_name,
                bbox=bbox,
                track_id=track_ids.get(region_id) if region_id else None,
                source=source,
                original_class=original_class,
                original_bbox=original_bbox,
            )
        )

    deletions = sorted(seeded_by_id.keys() - seen_ids)

    raw_frame_state = _extract_choice(raw_results, "frame_state")
    frame_state = raw_frame_state or "clean"
    if frame_state not in {"clean", "needs_more_review", "ambiguous_skip"}:
        logger.warning(
            "unknown frame_state=%r for image_id=%s; falling back to needs_more_review",
            frame_state,
            image_id,
        )
        frame_state = "needs_more_review"
    notes = _extract_textarea(raw_results, "notes")

    reviewed_at = _completion_timestamp(completion)
    duration = float(completion.get("lead_time") or 0.0)
    rid = reviewer_id or _resolve_reviewer_id(completion)

    return HumanReviewResult(
        image_id=image_id,
        reviewer_id=rid,
        reviewed_at=reviewed_at,
        duration_seconds=duration,
        frame_state=frame_state,
        corrections=corrections,
        deletions=deletions,
        notes=notes,
        ml_modes_used=[],
        ls_completion_id=_safe_int(completion.get("id"), 0),
    )


def _ls_value_to_bbox(value: dict[str, Any]) -> BoundingBox:
    """LS percentage rectangle → normalized BoundingBox.

    Clamps to [0, 1] and re-normalises corners so x1<=x2 / y1<=y2. LS itself
    keeps width/height non-negative on commit, but ML-backend predictions or
    custom workflows can produce negative widths; without this guard,
    BoundingBox.width would silently return 0 and downstream IoU would break.
    """
    try:
        x = float(value["x"]) / 100.0
        y = float(value["y"]) / 100.0
        w = float(value["width"]) / 100.0
        h = float(value["height"]) / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing/invalid LS rectangle fields: {exc}") from exc
    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + w))
    y2 = max(0.0, min(1.0, y + h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _bbox_equal(
    a: BoundingBox,
    b: BoundingBox,
    tol: float | None = None,
    *,
    image_size: tuple[int, int] | list[int] | None = None,
) -> bool:
    if tol is None:
        if image_size and len(image_size) == 2:
            try:
                w_px = float(image_size[0])
                h_px = float(image_size[1])
            except (TypeError, ValueError):
                w_px = h_px = 0.0
            largest = max(w_px, h_px)
            tol = 0.5 / largest if largest > 0 else 1e-4
        else:
            tol = 1e-4
    return all(
        math.isclose(getattr(a, k), getattr(b, k), abs_tol=tol)
        for k in ("x1", "y1", "x2", "y2")
    )


def _extract_choice(results: list[dict[str, Any]], from_name: str) -> str | None:
    for r in results:
        if r.get("from_name") == from_name and r.get("type") == "choices":
            choices = (r.get("value") or {}).get("choices") or []
            if choices:
                return str(choices[0])
    return None


def _extract_track_ids(results: list[dict[str, Any]]) -> dict[str, str]:
    """Map ``rectangle_id → track_id`` for per-region ``track_id`` textareas.

    The XML defines ``<TextArea name="track_id" perRegion="true">`` so LS
    emits one entry per rectangle that the reviewer typed into. Each entry
    carries ``parentID`` pointing back at the rectangle id; we walk the
    raw result list once to build the lookup the rectangle pass uses.
    """
    out: dict[str, str] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("from_name") != "track_id" or r.get("type") != "textarea":
            continue
        parent = r.get("parentID") or r.get("parent_id")
        if not isinstance(parent, str) or not parent:
            continue
        text = (r.get("value") or {}).get("text") or []
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, list) or not text:
            continue
        joined = "\n".join(str(t) for t in text).strip()
        if joined:
            out[parent] = joined
    return out


def _extract_textarea(results: list[dict[str, Any]], from_name: str) -> str:
    """Pull a global TextArea value (no ``parentID``).

    Per-region textareas (e.g. ``track_id``) carry a parentID linking
    them to a rectangle and must NOT be returned here — the global
    ``notes`` field is the only textarea this helper is for.
    """
    for r in results:
        if r.get("from_name") != from_name or r.get("type") != "textarea":
            continue
        if r.get("parentID") or r.get("parent_id"):
            continue
        text = (r.get("value") or {}).get("text") or []
        if text:
            return "\n".join(str(t) for t in text)
    return ""


def _completion_timestamp(completion: dict[str, Any]) -> float:
    raw = completion.get("created_at") or completion.get("updated_at")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def _resolve_reviewer_id(completion: dict[str, Any]) -> str:
    completed_by = completion.get("completed_by")
    if isinstance(completed_by, dict):
        return str(completed_by.get("email") or completed_by.get("id") or "unknown")
    if completed_by is not None:
        return str(completed_by)
    return "unknown"
