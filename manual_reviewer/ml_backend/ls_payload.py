"""Translate between Label Studio JSON shapes and aav4 wire contracts.

LS represents bbox coordinates as pixel-percentages (0..100) on the canvas.
aav4 (and SAM 3.1) speak normalized [0..1]. This module is the only place
that knows about that scale factor — every LS↔aav4 boundary in the ML
backend goes through the helpers here.

No HTTP, no DB, no model imports.
"""

from __future__ import annotations

import uuid
from typing import Any

DEFAULT_LABEL = "other"
"""Fallback class when the reviewer hasn't picked one — ``other`` exists
in the labeling_config XML, so LS will accept it as a valid label."""


def ls_box_to_norm(value: dict[str, Any]) -> list[float] | None:
    """Convert an LS RectangleLabels ``value`` dict to normalized [x1,y1,x2,y2].

    LS values are pixel-percentages; we divide by 100 and clamp to [0, 1].
    Returns ``None`` for malformed input rather than raising — predict()
    must keep working on partial drafts.
    """
    try:
        x = float(value["x"]) / 100.0
        y = float(value["y"]) / 100.0
        w = float(value["width"]) / 100.0
        h = float(value["height"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + w))
    y2 = max(0.0, min(1.0, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def ls_keypoint_to_norm(value: dict[str, Any]) -> list[float] | None:
    """Convert an LS KeyPoint ``value`` dict to normalized [x, y]."""
    try:
        x = float(value["x"]) / 100.0
        y = float(value["y"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y))]


def ls_textarea_value_to_prompts(value: dict[str, Any]) -> list[str]:
    """Pull text prompts out of an LS TextArea region value.

    LS stores TextArea text as ``{"text": ["line1", "line2", ...]}``. We
    strip empties and return them as the prompt list.
    """
    raw = value.get("text") if isinstance(value, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def norm_box_to_ls_region(
    bbox_norm: list[float],
    label: str,
    *,
    score: float = 0.0,
    region_id: str | None = None,
    from_name: str = "bbox",
    to_name: str = "image",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a normalized bbox into an LS ``rectanglelabels`` result entry."""
    x1, y1, x2, y2 = bbox_norm
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))
    if x2 <= x1 or y2 <= y1:
        # Degenerate boxes are dropped by LS; emit them as zero-area at the
        # click anyway so the reviewer can see *something* rather than
        # nothing for an unconfident model output.
        x2, y2 = x1 + 1e-6, y1 + 1e-6
    width = x2 - x1
    height = y2 - y1
    region = {
        "id": region_id or uuid.uuid4().hex[:10],
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": to_name,
        "image_rotation": 0,
        "value": {
            "x": x1 * 100.0,
            "y": y1 * 100.0,
            "width": width * 100.0,
            "height": height * 100.0,
            "rotation": 0,
            "rectanglelabels": [label or DEFAULT_LABEL],
        },
        "score": float(score),
    }
    if extra_meta:
        region["meta"] = extra_meta
    return region


def predictions_envelope(
    regions: list[dict[str, Any]],
    *,
    model_version: str,
    score: float = 1.0,
) -> list[dict[str, Any]]:
    """Wrap a list of result regions into the LS predictions array shape.

    LS expects ``predictions = [{"model_version": ..., "result": [...], "score": ...}]``.
    Returning an empty list is fine — LS just skips seeding boxes.
    """
    if not regions:
        return []
    return [
        {
            "model_version": model_version,
            "result": regions,
            "score": float(score),
        }
    ]
