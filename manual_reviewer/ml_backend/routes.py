"""Pure routing for the LS ML backend predict() entry point.

Three modes — picked by inspecting the LS ``context`` payload:

  * No context (or no draft) → :func:`batch_proposals` (cached DB lookup).
  * Draft is a KeyPoint        → :func:`smart_click` (SAM 3.1 click→mask).
  * Draft is a TextArea        → :func:`smart_text` (SAM 3.1 text→detect).

Each route is independently unit-testable. The server (server.py) is just
a LabelStudioMLBase wrapper that calls :func:`dispatch`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from manual_reviewer.ml_backend.aav4_client import read_cached_proposals
from manual_reviewer.ml_backend.ls_payload import (
    DEFAULT_LABEL,
    ls_box_to_norm,
    ls_keypoint_to_norm,
    ls_textarea_value_to_prompts,
    norm_box_to_ls_region,
)

logger = logging.getLogger(__name__)


__all__ = [
    "Sam3LikeClient",
    "batch_proposals",
    "dispatch",
    "smart_click",
    "smart_text",
]


class Sam3LikeClient(Protocol):
    """Minimal Protocol the routes need from the SAM 3.1 client.

    Tests can supply any object with these two methods; the production
    backend wires :class:`Sam3OneHttpClient`.
    """

    def click_mask(
        self,
        *,
        image_path: str,
        point: list[float],
        point_label: int = 1,
        threshold: float = 0.5,
    ) -> Any: ...

    def text_detect(
        self,
        *,
        image_path: str,
        prompts: list[str],
        threshold: float | None = None,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_image_path(task: dict[str, Any]) -> str | None:
    """``data.image_path`` is the canonical key set by ``task_builder``."""
    data = task.get("data") or {}
    path = data.get("image_path")
    if isinstance(path, str) and path:
        return path
    return None


def _bbox_to_norm_list(bbox: Any) -> list[float] | None:
    """aav4 BoundingBox dict / list → normalized [x1, y1, x2, y2] list."""
    if isinstance(bbox, dict):
        try:
            return [
                float(bbox["x1"]),
                float(bbox["y1"]),
                float(bbox["x2"]),
                float(bbox["y2"]),
            ]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except (TypeError, ValueError):
            return None
    return None


def _picked_label_from_context(context: dict[str, Any] | None) -> str | None:
    """Look at the LS context for a label the reviewer pre-selected.

    Smart-tool drafts may carry ``rectanglelabels`` / ``keypointlabels`` /
    ``labels`` arrays. We prefer those over the default fallback so the
    seeded box uses the reviewer's intended class.
    """
    if not isinstance(context, dict):
        return None
    # ``keypointlabels`` carries the keypoint tool's "positive"/"negative"
    # control labels, not real class names — never use those for the output
    # region's class. ``rectanglelabels``/``labels`` come from the user's
    # actual class dropdown selection.
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        value = region.get("value") or {}
        for key in ("rectanglelabels", "labels"):
            arr = value.get(key)
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, str) and first:
                    return first
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def smart_click(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: Sam3LikeClient,
    *,
    threshold: float = 0.0,
    model_version: str = "sam3_1_click",
) -> list[dict[str, Any]]:
    """KeyPoint → mask. Returns at most one RectangleLabels region.

    The reviewer's pre-selected class (if any) is preserved on the response;
    otherwise it falls back to ``other`` so LS still accepts the region.

    Threshold defaults to 0.0: a click is an explicit ask for a region, so
    we always return the highest-scoring mask SAM 3.1 produces. The score
    rides along on the LS region for downstream review.
    """
    image_path = _get_image_path(task)
    if not image_path:
        return []
    if not isinstance(context, dict):
        return []
    point: list[float] | None = None
    point_label = 1
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        rtype = (region.get("type") or "").lower()
        if rtype not in {"keypointlabels", "keypoint"}:
            continue
        candidate = ls_keypoint_to_norm(region.get("value") or {})
        if candidate is not None:
            point = candidate
            label_value = (region.get("value") or {}).get("keypointlabels")
            if isinstance(label_value, list) and label_value:
                first = str(label_value[0]).lower()
                if first.startswith("neg"):
                    point_label = 0
            break
    if point is None:
        logger.info("smart_click: no keypoint found in context")
        return []

    logger.info(
        "smart_click: image=%s point=%s label=%d threshold=%.2f",
        image_path,
        point,
        point_label,
        threshold,
    )
    try:
        resp = sam3_client.click_mask(
            image_path=image_path,
            point=point,
            point_label=point_label,
            threshold=threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("smart_click: SAM 3.1 click_mask failed: %s", exc)
        return []

    bbox = getattr(resp, "bbox", None)
    score = float(getattr(resp, "score", 0.0) or 0.0)
    if bbox is None:
        logger.info("smart_click: SAM 3.1 returned no bbox (score=%.3f)", score)
        return []
    label = _picked_label_from_context(context) or DEFAULT_LABEL
    logger.info(
        "smart_click: bbox=%s score=%.3f label=%s",
        [round(b, 4) for b in bbox],
        score,
        label,
    )
    region = norm_box_to_ls_region(
        list(bbox),
        label,
        score=score,
        extra_meta={"source": "smart_click", "model_version": model_version},
    )
    return [region]


def smart_text(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: Sam3LikeClient,
    *,
    threshold: float | None = None,
    model_version: str = "sam3_1_text",
    max_regions: int = 50,
) -> list[dict[str, Any]]:
    """TextArea → detect. Returns one region per matched box.

    Caps output at ``max_regions`` so a verbose model doesn't flood the LS
    canvas. Each region is labeled with the prompt that produced it.
    """
    image_path = _get_image_path(task)
    if not image_path:
        return []
    if not isinstance(context, dict):
        return []
    prompts: list[str] = []
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        rtype = (region.get("type") or "").lower()
        if rtype != "textarea":
            continue
        prompts.extend(ls_textarea_value_to_prompts(region.get("value") or {}))
        break
    if not prompts:
        return []

    try:
        resp = sam3_client.text_detect(
            image_path=image_path,
            prompts=prompts,
            threshold=threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("smart_text: SAM 3.1 text_detect failed: %s", exc)
        return []

    boxes = list(getattr(resp, "boxes", []) or [])
    scores = list(getattr(resp, "scores", []) or [])
    labels = list(getattr(resp, "labels", []) or [])

    out: list[dict[str, Any]] = []
    for idx, bbox in enumerate(boxes):
        if idx >= max_regions:
            break
        bbox_norm = _bbox_to_norm_list(bbox)
        if bbox_norm is None:
            continue
        label = labels[idx] if idx < len(labels) and labels[idx] else DEFAULT_LABEL
        score = scores[idx] if idx < len(scores) else 0.0
        out.append(
            norm_box_to_ls_region(
                bbox_norm,
                label,
                score=float(score or 0.0),
                extra_meta={
                    "source": "smart_text",
                    "model_version": model_version,
                    "prompt": label,
                },
            )
        )
    return out


def batch_proposals(
    task: dict[str, Any],
    db_path: Path | str | None,
    *,
    model_version: str = "aav4_proposals",
    max_regions: int = 200,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Task open: return every cached proposal as an LS region.

    Pure DB read — no model call, so latency is dominated by the sqlite SELECT.
    Falls back to an empty list when ``db_path`` is missing or the image has
    no proposals (the LS predictions seed already covers finalize anyway).
    """
    if db_path is None:
        return []
    image_id = (task.get("data") or {}).get("image_id")
    if not isinstance(image_id, str) or not image_id:
        return []
    candidates = read_cached_proposals(db_path, image_id)
    out: list[dict[str, Any]] = []
    for cand in candidates:
        if len(out) >= max_regions:
            break
        confidence = float(cand.get("confidence") or 0.0)
        if confidence < min_confidence:
            continue
        bbox_norm = _bbox_to_norm_list(cand.get("bbox"))
        if bbox_norm is None:
            continue
        label = cand.get("class_name") or DEFAULT_LABEL
        out.append(
            norm_box_to_ls_region(
                bbox_norm,
                label,
                score=confidence,
                region_id=cand.get("candidate_id"),
                extra_meta={
                    "source": "batch_proposals",
                    "model_version": model_version,
                    "detector": cand.get("model"),
                },
            )
        )
    return out


def dispatch(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    *,
    sam3_client: Sam3LikeClient | None,
    db_path: Path | str | None,
) -> list[dict[str, Any]]:
    """Pick a route by sniffing the LS context payload.

    Returns the LS ``result`` list (not the wrapping ``predictions`` envelope —
    callers wrap that themselves so they can attach a per-mode ``model_version``).
    """
    if isinstance(context, dict) and context.get("result"):
        for region in context["result"]:
            if not isinstance(region, dict):
                continue
            rtype = (region.get("type") or "").lower()
            if rtype in {"keypointlabels", "keypoint"} and sam3_client is not None:
                return smart_click(task, context, sam3_client)
            if rtype == "textarea" and sam3_client is not None:
                return smart_text(task, context, sam3_client)
    return batch_proposals(task, db_path)
