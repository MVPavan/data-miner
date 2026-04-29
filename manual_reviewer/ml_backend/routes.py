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
from manual_reviewer.ml_backend.dedup import (
    canvas_rectangles,
    dedup_against,
    iou_xyxy,
    nms_regions,
)
from manual_reviewer.ml_backend.ls_payload import (
    DEFAULT_LABEL,
    ls_box_to_norm,
    ls_keypoint_to_norm,
    ls_textarea_value_to_prompts,
    norm_box_to_ls_region,
)

logger = logging.getLogger(__name__)


# IoU helper kept as a re-export so existing tests/callers still work.
# The implementation lives in :mod:`manual_reviewer.ml_backend.dedup`;
# every route uses the shared one.
_iou_xyxy = iou_xyxy


__all__ = [
    "Sam3LikeClient",
    "batch_proposals",
    "dispatch",
    "smart_click",
    "smart_text",
    "visual_prompt",
]


class Sam3LikeClient(Protocol):
    """Minimal Protocol the routes need from the SAM 3.1 client.

    Tests can supply any object with these methods; the production backend
    wires :class:`Sam3OneHttpClient`.
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

    def visual_prompt(
        self,
        *,
        image_path: str,
        exemplar_boxes_norm: list[list[float]],
        exemplar_labels: list[int] | None = None,
        threshold: float = 0.4,
        max_results: int = 50,
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

    Accepts every label-array key LS may emit, depending on which tool
    produced the draft:

      * ``rectanglelabels`` — RectangleLabels (the bbox tool, Phase A).
      * ``keypointlabels``  — KeyPointLabels (the smart_click tool).
      * ``labels``          — bare ``<Labels>`` block, kept for backward
        compat with older XML configs / seeded predictions.
    """
    if not isinstance(context, dict):
        return None
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        value = region.get("value") or {}
        for key in ("labels", "rectanglelabels", "keypointlabels"):
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
    dedup_iou: float = 0.7,
) -> list[dict[str, Any]]:
    """KeyPoint → mask. Returns at most one RectangleLabels region.

    The reviewer's pre-selected class (if any) is preserved on the response;
    otherwise it falls back to ``other`` so LS still accepts the region.

    Threshold defaults to 0.0: a click is an explicit ask for a region, so
    we always return the highest-scoring mask SAM 3.1 produces. The score
    rides along on the LS region for downstream review.

    The returned region is silently dropped if it duplicates a same-class
    rectangle already on the canvas at IoU > ``dedup_iou`` (default 0.7) —
    a click on an already-accepted box would otherwise stack a duplicate
    that the reviewer has to delete by hand.
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
    survivors, dropped = dedup_against(
        [region], canvas_rectangles(task, context), iou=dedup_iou
    )
    if dropped:
        logger.info("smart_click: dropped duplicate at IoU > %.2f", dedup_iou)
    return survivors


def _exemplars_from_context(
    context: dict[str, Any] | None,
) -> tuple[list[list[float]], str | None]:
    """Pull rectangle exemplar(s) + a class hint out of an LS context.

    Returns ``([], None)`` when no usable rectangle region is present. The
    class hint comes from the rectangle region's ``rectanglelabels`` /
    ``labels`` array (Phase A shape) and is preserved on the propagated
    output regions so they don't all get tagged ``other``.

    When dispatch identifies the exemplar via ``from_name="visual_prompt"``
    only those regions are returned; otherwise every rectangle is treated
    as a potential exemplar (covers test paths and direct route calls).
    """
    if not isinstance(context, dict):
        return [], None
    visual_only = any(
        isinstance(r, dict) and r.get("from_name") == "visual_prompt"
        for r in context.get("result") or []
    )
    boxes: list[list[float]] = []
    label_hint: str | None = None
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        rtype = (region.get("type") or "").lower()
        if rtype not in {"rectanglelabels", "rectangle"}:
            continue
        if visual_only and region.get("from_name") != "visual_prompt":
            continue
        value = region.get("value") or {}
        bbox = ls_box_to_norm(value)
        if bbox is None:
            continue
        boxes.append(bbox)
        if label_hint is None:
            for key in ("labels", "rectanglelabels", "keypointlabels"):
                arr = value.get(key)
                if isinstance(arr, list) and arr:
                    first = arr[0]
                    if isinstance(first, str) and first:
                        label_hint = first
                        break
    if label_hint is None:
        label_hint = _picked_label_from_context(context)
    return boxes, label_hint


def visual_prompt(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: Sam3LikeClient,
    *,
    threshold: float = 0.4,
    max_results: int = 50,
    dedup_iou: float = 0.7,
    nms_iou: float = 0.85,
    model_version: str = "sam3_1_visual",
) -> list[dict[str, Any]]:
    """Within-image visual prompting — exemplar bbox(es) → all matches.

    The reviewer draws an exemplar bbox with the V-tool (smart Rectangle),
    LS fires ``/predict``, SAM 3.1's geometric-prompt grounding head
    returns every matching instance in the same image. The exemplar's
    class label rides through onto every propagated region.

    Two dedup layers run on the SAM output:

      * **Internal NMS** at IoU > ``nms_iou`` (default 0.85, class-aware)
        suppresses near-duplicate matches the model emitted on the same
        instance. Score-desc; lower-scoring duplicate is dropped.
      * **External dedup** at IoU > ``dedup_iou`` (default 0.7, class-aware)
        against every rectangle on the canvas — accepted annotations,
        seeded predictions, and the current draft (the exemplar itself).
        That drops both the SAM-returned duplicate at the exemplar
        location AND any match that lands on a box the reviewer has
        already placed.

    Threshold defaults to 0.4 (wire default); ``max_results`` caps output
    so a verbose model doesn't flood LS.
    """
    image_path = _get_image_path(task)
    if not image_path:
        return []
    exemplars, label_hint = _exemplars_from_context(context)
    if not exemplars:
        logger.info("visual_prompt: no exemplar bbox found in context")
        return []
    # Drop degenerate (zero-area) exemplars — SAM grounding behavior on a
    # zero-w/h prompt is undefined.
    exemplars = [
        e for e in exemplars
        if (e[2] - e[0]) > 1e-3 and (e[3] - e[1]) > 1e-3
    ]
    if not exemplars:
        logger.info("visual_prompt: all exemplars degenerate (zero area)")
        return []
    existing = canvas_rectangles(task, context)

    logger.info(
        "visual_prompt: image=%s exemplars=%d label=%s threshold=%.2f existing=%d",
        image_path,
        len(exemplars),
        label_hint,
        threshold,
        len(existing),
    )
    try:
        resp = sam3_client.visual_prompt(
            image_path=image_path,
            exemplar_boxes_norm=exemplars,
            threshold=threshold,
            max_results=max_results,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("visual_prompt: SAM 3.1 visual_prompt failed: %s", exc)
        return []

    boxes = list(getattr(resp, "boxes_norm", []) or [])
    scores = list(getattr(resp, "scores", []) or [])
    label = label_hint or DEFAULT_LABEL

    proposed: list[dict[str, Any]] = []
    for idx, bbox in enumerate(boxes):
        bbox_norm = _bbox_to_norm_list(bbox)
        if bbox_norm is None:
            continue
        score = scores[idx] if idx < len(scores) else 0.0
        proposed.append(
            norm_box_to_ls_region(
                bbox_norm,
                label,
                score=float(score or 0.0),
                extra_meta={
                    "source": "visual_prompt",
                    "model_version": model_version,
                },
            )
        )

    # Internal NMS first: collapse near-duplicate matches the model
    # emitted on the same instance. External dedup second: drop matches
    # that overlap an existing canvas rectangle (exemplar, accepted box,
    # seeded prediction).
    nmsed = nms_regions(proposed, iou=nms_iou)
    survivors, dropped = dedup_against(nmsed, existing, iou=dedup_iou)
    out = survivors[:max_results]
    logger.info(
        "visual_prompt: SAM=%d → NMS=%d → canvas-dedup=%d → cap=%d (label=%s)",
        len(proposed),
        len(nmsed),
        len(survivors),
        len(out),
        label,
    )
    if dropped:
        logger.debug("visual_prompt: dropped %d duplicate(s) vs canvas", dropped)
    return out


def smart_text(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: Sam3LikeClient,
    *,
    threshold: float | None = None,
    model_version: str = "sam3_1_text",
    max_regions: int = 50,
    dedup_iou: float = 0.7,
    nms_iou: float = 0.85,
) -> list[dict[str, Any]]:
    """TextArea → detect. Returns one region per matched box.

    Two dedup layers, same shape as :func:`visual_prompt`:

      * **Internal NMS** at IoU > ``nms_iou`` (class-aware) collapses
        near-duplicate detections the grounding model emitted on the
        same instance.
      * **External dedup** at IoU > ``dedup_iou`` (class-aware) against
        the canvas pool (accepted + predictions + draft).

    Caps survivors at ``max_regions`` so a verbose model doesn't flood
    the LS canvas. Each region is labeled with the prompt that produced
    it.
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

    proposed: list[dict[str, Any]] = []
    for idx, bbox in enumerate(boxes):
        bbox_norm = _bbox_to_norm_list(bbox)
        if bbox_norm is None:
            continue
        label = labels[idx] if idx < len(labels) and labels[idx] else DEFAULT_LABEL
        score = scores[idx] if idx < len(scores) else 0.0
        proposed.append(
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

    nmsed = nms_regions(proposed, iou=nms_iou)
    survivors, dropped = dedup_against(
        nmsed, canvas_rectangles(task, context), iou=dedup_iou
    )
    out = survivors[:max_regions]
    logger.info(
        "smart_text: SAM=%d → NMS=%d → canvas-dedup=%d → cap=%d",
        len(proposed),
        len(nmsed),
        len(survivors),
        len(out),
    )
    if dropped:
        logger.debug("smart_text: dropped %d duplicate(s) vs canvas", dropped)
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

    Mirrors :meth:`ManualReviewerMLBackend._predict_one` so that callers
    using this pure function (tests, scripts that bypass the LS server)
    behave the same as the production server: a V-tool draft routes to
    :func:`visual_prompt`; otherwise the per-region rectype fans out to
    :func:`smart_click` / :func:`smart_text`. Drafts that don't match
    any smart route return ``[]`` rather than falling through to
    :func:`batch_proposals` — seeding cached proposals on top of an
    intentional draft is never the reviewer's intent.

    Returns the LS ``result`` list (not the wrapping ``predictions`` envelope —
    callers wrap that themselves so they can attach a per-mode ``model_version``).
    """
    if isinstance(context, dict) and context.get("result"):
        regions = context["result"]
        if sam3_client is not None and any(
            isinstance(r, dict) and r.get("from_name") == "visual_prompt"
            for r in regions
        ):
            return visual_prompt(task, context, sam3_client)
        handled = False
        for region in regions:
            if not isinstance(region, dict):
                continue
            if region.get("from_name") == "visual_prompt":
                handled = True
                continue
            rtype = (region.get("type") or "").lower()
            if rtype in {"keypointlabels", "keypoint"} and sam3_client is not None:
                return smart_click(task, context, sam3_client)
            if rtype == "textarea" and sam3_client is not None:
                return smart_text(task, context, sam3_client)
            handled = True
        if handled:
            return []
    return batch_proposals(task, db_path)
