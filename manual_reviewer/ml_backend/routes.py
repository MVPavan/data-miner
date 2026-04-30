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
    snap_label,
)
from manual_reviewer.ml_backend.ls_rest import LSRestClient
from manual_reviewer.ml_backend.smart_track_lib import (
    MultiSeedPropagateResult,
    PropagateResult,
    SeedSpec,
    TrackerLikeClient,
    propagate_multi_via_tracker,
    propagate_via_tracker,
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
    "propagate_now",
    "smart_click",
    "smart_text",
    "smart_track",
    "track_similar",
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


def _get_image_dims(task: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` from ``task.data.image_size`` if present."""
    data = task.get("data") or {}
    size = data.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


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


def _picked_label_from_context(
    context: dict[str, Any] | None,
    triggering_region: dict[str, Any] | None = None,
) -> str | None:
    """Look at the LS context for a label the reviewer pre-selected.

    Accepts every label-array key LS may emit, depending on which tool
    produced the draft:

      * ``rectanglelabels`` — RectangleLabels (the bbox tool, Phase A).
      * ``keypointlabels``  — KeyPointLabels (the smart_click tool).
      * ``labels``          — bare ``<Labels>`` block, kept for backward
        compat with older XML configs / seeded predictions.

    When ``triggering_region`` is supplied (e.g. the keypoint that fired
    smart_click), its own labels are checked first so a paired keypoint
    class wins over an unrelated rectangle that happens to ride the
    same draft.
    """

    def _from_value(value: dict[str, Any]) -> str | None:
        for key in ("keypointlabels", "rectanglelabels", "labels"):
            arr = value.get(key)
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, str) and first:
                    return first
        return None

    if isinstance(triggering_region, dict):
        own = _from_value(triggering_region.get("value") or {})
        if own:
            return own
    if not isinstance(context, dict):
        return None
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        own = _from_value(region.get("value") or {})
        if own:
            return own
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

    Canvas-dedup is context-only — the live canvas state LS sends with
    every smart-tool fire (user-drawn rectangles + accepted predictions).
    Untouched yellow seeded predictions are NOT in this pool, so a click
    on a seeded box still produces a region (the refine/replace use
    case). A second click at the same spot whose result has already been
    accepted will be deduped against that accepted region.
    """
    image_path = _get_image_path(task)
    if not image_path:
        return []
    if not isinstance(context, dict):
        return []
    point: list[float] | None = None
    point_label = 1
    trigger_region: dict[str, Any] | None = None
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        rtype = (region.get("type") or "").lower()
        if rtype not in {"keypointlabels", "keypoint"}:
            continue
        candidate = ls_keypoint_to_norm(region.get("value") or {})
        if candidate is not None:
            point = candidate
            trigger_region = region
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
    label = _picked_label_from_context(context, trigger_region) or DEFAULT_LABEL
    logger.info(
        "smart_click: bbox=%s score=%.3f label=%s",
        [round(b, 4) for b in bbox],
        score,
        label,
    )
    width, height = _get_image_dims(task)
    region = norm_box_to_ls_region(
        list(bbox),
        label,
        score=score,
        extra_meta={"source": "smart_click", "model_version": model_version},
        original_width=width,
        original_height=height,
        original_rotation=0,
    )
    if region is None:
        return []
    survivors, dropped = dedup_against(
        [region], canvas_rectangles(task, context), iou=dedup_iou
    )
    if dropped:
        logger.info("smart_click: dropped duplicate vs live canvas at IoU > %.2f", dedup_iou)
    return survivors


def _exemplars_from_context(
    context: dict[str, Any] | None,
    *,
    from_name: str = "visual_prompt",
) -> tuple[list[list[float]], str | None]:
    """Pull rectangle exemplar(s) + a class hint out of an LS context.

    Returns ``([], None)`` when no usable rectangle region is present. The
    class hint comes from the rectangle region's ``rectanglelabels`` /
    ``labels`` array (Phase A shape) and is preserved on the propagated
    output regions so they don't all get tagged ``other``.

    ``from_name`` selects which smart Rectangle is treated as the
    exemplar source. Default ``visual_prompt`` matches the V-tool;
    pass ``track_similar`` for the smart_track_all (Option A) flow.
    When that smart-tool tag is found in the context, only its regions
    are accepted as exemplars; otherwise we fall back to any rectangle
    (covers test paths and direct route calls).
    """
    if not isinstance(context, dict):
        return [], None
    scoped_only = any(
        isinstance(r, dict) and r.get("from_name") == from_name
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
        if scoped_only and region.get("from_name") != from_name:
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
    existing = canvas_rectangles(
        task, context, include_visual_prompt=True, include_predictions=True
    )

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
    width, height = _get_image_dims(task)

    proposed: list[dict[str, Any]] = []
    for idx, bbox in enumerate(boxes):
        bbox_norm = _bbox_to_norm_list(bbox)
        if bbox_norm is None:
            continue
        score = scores[idx] if idx < len(scores) else 0.0
        region = norm_box_to_ls_region(
            bbox_norm,
            label,
            score=float(score or 0.0),
            extra_meta={
                "source": "visual_prompt",
                "model_version": model_version,
            },
            original_width=width,
            original_height=height,
            original_rotation=0,
        )
        if region is not None:
            proposed.append(region)

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
    width, height = _get_image_dims(task)

    proposed: list[dict[str, Any]] = []
    for idx, bbox in enumerate(boxes):
        bbox_norm = _bbox_to_norm_list(bbox)
        if bbox_norm is None:
            continue
        raw_label = labels[idx] if idx < len(labels) and labels[idx] else DEFAULT_LABEL
        label = snap_label(raw_label)
        score = scores[idx] if idx < len(scores) else 0.0
        region = norm_box_to_ls_region(
            bbox_norm,
            label,
            score=float(score or 0.0),
            extra_meta={
                "source": "smart_text",
                "model_version": model_version,
                "prompt": raw_label,
            },
            original_width=width,
            original_height=height,
            original_rotation=0,
        )
        if region is not None:
            proposed.append(region)

    nmsed = nms_regions(proposed, iou=nms_iou)
    survivors, dropped = dedup_against(
        nmsed,
        canvas_rectangles(task, context, include_predictions=True),
        iou=dedup_iou,
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


def _seed_from_smart_track_context(
    context: dict[str, Any] | None,
) -> tuple[list[float] | None, str | None]:
    """Pull the seed bbox + class hint from a smart_track LS context.

    Looks only at regions with ``from_name="smart_track"`` so a plain
    bbox draw or a stray rectangle from another tool can't accidentally
    fire propagation.
    """
    if not isinstance(context, dict):
        return None, None
    for region in context.get("result") or []:
        if not isinstance(region, dict):
            continue
        if region.get("from_name") != "smart_track":
            continue
        rtype = (region.get("type") or "").lower()
        if rtype not in {"rectanglelabels", "rectangle"}:
            continue
        value = region.get("value") or {}
        bbox = ls_box_to_norm(value)
        if bbox is None:
            continue
        label_hint: str | None = None
        for key in ("rectanglelabels", "labels"):
            arr = value.get(key)
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, str) and first:
                    label_hint = first
                    break
        return bbox, label_hint
    return None, None


def _project_id_from_task(task: dict[str, Any]) -> int | None:
    """LS predict() payloads put project at top level; build_tasks output
    can also stash it under ``data.project``. Accept either.
    """
    proj = task.get("project")
    if isinstance(proj, int):
        return proj
    proj_d = (task.get("data") or {}).get("project")
    if isinstance(proj_d, int):
        return proj_d
    if isinstance(proj_d, str) and proj_d.isdigit():
        return int(proj_d)
    return None


def smart_track(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: TrackerLikeClient,
    ls_rest: LSRestClient | None,
    *,
    score_thresh: float = 0.5,
    motion_thresh: float = 0.05,
    max_siblings: int = 100,
    model_version: str = "sam3_1_track",
) -> tuple[list[dict[str, Any]], PropagateResult | None]:
    """Static-object tracker propagation across sibling clip frames.

    The reviewer draws an exemplar rectangle with the smart_track tool;
    SAM 3.1's video tracker propagates that bbox across every sibling
    task in the project (matched by clip-prefix on ``image_id``). Only
    surviving propagations — high score + low spatial drift relative to
    the seed — get POSTed to LS as predictions on the matching tasks.

    Returns ``(regions, result)``:

      * ``regions`` is always ``[]`` — propagation lands on *sibling*
        tasks via REST, not on the current task. The reviewer sees
        their own seed rectangle on the current task (LS draws drafts
        natively); navigating to a sibling reveals the propagated
        prediction.
      * ``result`` is the :class:`PropagateResult` summary (or None if
        we early-aborted before calling the tracker). The server logs
        it; tests assert against it.

    Required env-derived state (caller's job):

      * ``sam3_client`` — a :class:`TrackerLikeClient` (live SAM 3.1
        client OR a mock).
      * ``ls_rest`` — a :class:`LSRestClient`. None disables the route
        (cross-task writes are mandatory for usefulness).

    Both ``score_thresh`` and ``motion_thresh`` enforce the static-only
    semantics — moving objects either drop in confidence or drift
    spatially across sparse frames; either way, they're rejected.
    """
    if ls_rest is None:
        logger.info("smart_track: LS REST client not configured; skipping")
        return [], None

    seed_bbox, label_hint = _seed_from_smart_track_context(context)
    if seed_bbox is None:
        logger.info("smart_track: no seed rectangle in context")
        return [], None
    if (seed_bbox[2] - seed_bbox[0]) < 1e-3 or (seed_bbox[3] - seed_bbox[1]) < 1e-3:
        logger.info("smart_track: seed bbox is degenerate (zero area)")
        return [], None

    image_path = _get_image_path(task)
    image_id = (task.get("data") or {}).get("image_id")
    project_id = _project_id_from_task(task)
    if not image_path or not isinstance(image_id, str) or project_id is None:
        logger.warning(
            "smart_track: missing image_path/image_id/project (path=%r id=%r proj=%r)",
            image_path, image_id, project_id,
        )
        return [], None

    seed_label = snap_label(label_hint) if label_hint else DEFAULT_LABEL

    logger.info(
        "smart_track: seed image=%s bbox=%s label=%s project=%s",
        image_id,
        [round(b, 4) for b in seed_bbox],
        seed_label,
        project_id,
    )

    result = propagate_via_tracker(
        sam3_client=sam3_client,
        ls_rest=ls_rest,
        project_id=project_id,
        seed_image_path=image_path,
        seed_image_id=image_id,
        seed_bbox=seed_bbox,
        seed_label=seed_label,
        current_task_id=task.get("id"),
        score_thresh=score_thresh,
        motion_thresh=motion_thresh,
        max_siblings=max_siblings,
        model_version=model_version,
    )
    return [], result


# ---------------------------------------------------------------------------
# Option A: track_similar (Phase 1) + propagate_now (Phase 2)
# ---------------------------------------------------------------------------


def track_similar(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: Sam3LikeClient,
    ls_rest: LSRestClient | None = None,
    *,
    threshold: float = 0.4,
    max_results: int = 50,
    dedup_iou: float = 0.7,
    nms_iou: float = 0.85,
    model_version: str = "sam3_1_track_similar",
) -> list[dict[str, Any]]:
    """Phase 1 of Option A: in-frame find-similar with track_similar tag.

    Same SAM 3.1 visual_prompt grounding as :func:`visual_prompt`, but
    the exemplar is read from regions tagged ``from_name="track_similar"``
    and the returned regions also carry ``from_name="track_similar"``.
    The tag is the correlation key Phase 2 (:func:`propagate_now`) uses
    to know which rectangles on the canvas to propagate.

    **Persistence note:** smart-tool predict responses are ephemeral
    browser-side overlays — LS doesn't save them to ``task["predictions"]``
    automatically. For Phase 2 to find these matches later, we POST a
    persistent prediction record via LS REST (when ``ls_rest`` is
    configured). The same regions also ride back in the predict
    response so the reviewer sees immediate feedback; LS dedupes on
    region.id so they don't show twice.

    The reviewer drops or accepts the returned same-frame matches as
    they do today; only what survives review will get propagated when
    Phase 2 fires. False positives are isolated to this image — the
    cross-frame predictions never appear unless the reviewer asks for
    them, by design.
    """
    image_path = _get_image_path(task)
    if not image_path:
        return []
    exemplars, label_hint = _exemplars_from_context(context, from_name="track_similar")
    if not exemplars:
        logger.info("track_similar: no exemplar bbox in context")
        return []
    exemplars = [
        e for e in exemplars
        if (e[2] - e[0]) > 1e-3 and (e[3] - e[1]) > 1e-3
    ]
    if not exemplars:
        logger.info("track_similar: all exemplars degenerate")
        return []
    existing = canvas_rectangles(
        task, context, include_visual_prompt=True, include_predictions=True
    )

    logger.info(
        "track_similar: image=%s exemplars=%d label=%s threshold=%.2f existing=%d",
        image_path, len(exemplars), label_hint, threshold, len(existing),
    )
    try:
        resp = sam3_client.visual_prompt(
            image_path=image_path,
            exemplar_boxes_norm=exemplars,
            threshold=threshold,
            max_results=max_results,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("track_similar: SAM 3.1 visual_prompt failed: %s", exc)
        return []

    boxes = list(getattr(resp, "boxes_norm", []) or [])
    scores = list(getattr(resp, "scores", []) or [])
    label = label_hint or DEFAULT_LABEL
    width, height = _get_image_dims(task)

    proposed: list[dict[str, Any]] = []
    for idx, bbox in enumerate(boxes):
        bbox_norm = _bbox_to_norm_list(bbox)
        if bbox_norm is None:
            continue
        score = scores[idx] if idx < len(scores) else 0.0
        region = norm_box_to_ls_region(
            bbox_norm,
            label,
            score=float(score or 0.0),
            from_name="track_similar",
            extra_meta={
                "source": "track_similar",
                "model_version": model_version,
            },
            original_width=width,
            original_height=height,
            original_rotation=0,
        )
        if region is not None:
            proposed.append(region)

    nmsed = nms_regions(proposed, iou=nms_iou)
    survivors, dropped = dedup_against(nmsed, existing, iou=dedup_iou)
    out = survivors[:max_results]
    logger.info(
        "track_similar: SAM=%d → NMS=%d → canvas-dedup=%d → cap=%d (label=%s)",
        len(proposed), len(nmsed), len(survivors), len(out), label,
    )
    if dropped:
        logger.debug("track_similar: dropped %d duplicate(s) vs canvas", dropped)

    # Persist matches as a real prediction so Phase 2 (propagate_now) can
    # read them from task["predictions"]. Smart-tool predict responses
    # are ephemeral; without this, hitting Shift+J immediately after
    # Phase 1 finds nothing on the server side.
    task_id = task.get("id")
    if out and ls_rest is not None and isinstance(task_id, int):
        max_score = max(
            (float(r.get("score", 0.0) or 0.0) for r in out), default=0.0,
        )
        pid = ls_rest.post_prediction(
            task_id=task_id,
            result=out,
            score=max_score,
            model_version=model_version,
        )
        if pid is not None:
            logger.info(
                "track_similar: persisted %d region(s) as prediction id=%s",
                len(out), pid,
            )
        else:
            logger.warning(
                "track_similar: failed to persist prediction; Phase 2 won't see these"
            )

    return out


def _collect_track_similar_seeds(
    task: dict[str, Any],
    context: dict[str, Any] | None,
) -> list[SeedSpec]:
    """Walk current task's annotations + context to find track_similar
    rectangles that survived review.

    Sources, dedup'd by rounded bbox key:

      1. ``task["annotations"][N]["result"]`` — accepted regions.
         The reviewer explicitly committed these, so they're the
         strongest signal of "propagate this".
      2. ``context["result"]`` — current draft (typically the trigger
         KeyPoint, but may include in-flight rectangles).
      3. ``task["predictions"][N]["result"]`` — Phase 1 output that's
         still on the canvas. LS Community keeps predictions in this
         array regardless of UI-side rejection, so reading from here
         means **all** Phase 1 matches propagate unless the reviewer
         has accepted them into annotations (then dedup wins) or
         explicitly DELETEd the prediction record via the LS REST API.

    The reviewer's natural workflow ("draw exemplar → review same-frame
    → Shift+J") results in matches sitting in ``predictions[]`` until
    they accept; (3) is the path that makes Phase 2 fire usefully
    without a separate per-match accept step.

    Each seed gets a fresh ``track_group_id`` (UUID) tagged on the
    sibling propagations so a future cascade route can correlate.
    """
    import uuid

    seeds: list[SeedSpec] = []

    def _from_region(region: dict[str, Any]) -> SeedSpec | None:
        if not isinstance(region, dict):
            return None
        if region.get("from_name") != "track_similar":
            return None
        rtype = (region.get("type") or "").lower()
        if rtype not in {"rectanglelabels", "rectangle"}:
            return None
        value = region.get("value") or {}
        bbox = ls_box_to_norm(value)
        if bbox is None:
            return None
        if (bbox[2] - bbox[0]) < 1e-3 or (bbox[3] - bbox[1]) < 1e-3:
            return None
        label: str | None = None
        for key in ("rectanglelabels", "labels"):
            arr = value.get(key)
            if isinstance(arr, list) and arr and isinstance(arr[0], str) and arr[0]:
                label = arr[0]
                break
        return SeedSpec(
            bbox=bbox,
            label=snap_label(label) if label else DEFAULT_LABEL,
            track_group_id=uuid.uuid4().hex,
        )

    seen: set[tuple[float, float, float, float]] = set()

    def _consider(region: Any) -> None:
        spec = _from_region(region)
        if spec is None:
            return
        key = tuple(round(b, 4) for b in spec.bbox)
        if key in seen:
            return
        seen.add(key)
        seeds.append(spec)

    for ann in task.get("annotations") or []:
        for region in (ann or {}).get("result") or []:
            _consider(region)

    if isinstance(context, dict):
        for region in context.get("result") or []:
            _consider(region)

    for pred in task.get("predictions") or []:
        for region in (pred or {}).get("result") or []:
            _consider(region)

    return seeds


def propagate_now(
    task: dict[str, Any],
    context: dict[str, Any] | None,
    sam3_client: TrackerLikeClient,
    ls_rest: LSRestClient | None,
    *,
    score_thresh: float = 0.5,
    motion_thresh: float = 0.05,
    max_siblings: int = 100,
    model_version: str = "sam3_1_track",
) -> tuple[list[dict[str, Any]], MultiSeedPropagateResult | None]:
    """Phase 2 of Option A: multi-seed propagate accepted track_similar regions.

    Reads the current task's annotations + context for rectangles tagged
    ``from_name="track_similar"`` (the survivors of Phase 1 review),
    builds a multi-seed tracker call, applies the static-only filter
    per seed, and POSTs each surviving propagation to the matching
    sibling task via LS REST.

    Returns ``([], MultiSeedPropagateResult)``: nothing seeds back into
    the current task — the trigger keypoint stays as a draft the user
    can delete. The result summary is logged so the user can see how
    many siblings got propagations.

    Preconditions: ``ls_rest`` configured (cross-task writes), at least
    one ``track_similar`` rectangle present, image_id/path/project all
    resolvable from the task. Failures degrade gracefully — return
    ``([], None)`` and log.
    """
    if ls_rest is None:
        logger.info("propagate_now: LS REST client not configured; skipping")
        return [], None

    seeds = _collect_track_similar_seeds(task, context)
    if not seeds:
        logger.info("propagate_now: no track_similar regions found on current task")
        return [], None

    image_path = _get_image_path(task)
    image_id = (task.get("data") or {}).get("image_id")
    project_id = _project_id_from_task(task)
    if not image_path or not isinstance(image_id, str) or project_id is None:
        logger.warning(
            "propagate_now: missing image_path/image_id/project (path=%r id=%r proj=%r)",
            image_path, image_id, project_id,
        )
        return [], None

    logger.info(
        "propagate_now: seed image=%s seeds=%d project=%s",
        image_id, len(seeds), project_id,
    )
    result = propagate_multi_via_tracker(
        sam3_client=sam3_client,
        ls_rest=ls_rest,
        project_id=project_id,
        seed_image_path=image_path,
        seed_image_id=image_id,
        seeds=seeds,
        current_task_id=task.get("id"),
        score_thresh=score_thresh,
        motion_thresh=motion_thresh,
        max_siblings=max_siblings,
        model_version=model_version,
    )
    return [], result


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
    width, height = _get_image_dims(task)
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
        region = norm_box_to_ls_region(
            bbox_norm,
            label,
            score=confidence,
            region_id=cand.get("candidate_id"),
            extra_meta={
                "source": "batch_proposals",
                "model_version": model_version,
                "detector": cand.get("model"),
            },
            original_width=width,
            original_height=height,
            original_rotation=0,
        )
        if region is not None:
            out.append(region)
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
