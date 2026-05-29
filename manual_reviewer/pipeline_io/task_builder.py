"""Assemble Label Studio task dicts from aa_v4 ``pipeline.db`` payloads.

A task carries (a) a ``data`` blob — what the reviewer sees as image + side
panels — and (b) ``predictions`` — the seeded boxes the reviewer edits.

Pre-annotations are seeded from the ``finalize`` stage (canonical truth).
Filter drops and finalize drops are surfaced in ``data.ghost_drops`` so the
reviewer can toggle "show what the pipeline rejected" without cluttering the
default canvas. VLM verdicts and per-model proposal summaries are folded into
``data`` so the reviewer can answer "why did the pipeline label this X?".
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


def build_task(
    image_payload: dict[str, Any],
    *,
    image_url_template: str = "/data/local-files/?d={path}",
    job_id: str = "",
    model_version: str = "aa_v4_finalize",
    include_ghost_drops: bool = True,
) -> dict[str, Any] | None:
    """Build one Label Studio task dict from a ``read_image_payload`` result.

    ``image_url_template`` is formatted with ``path=<image_path>``. The default
    targets Label Studio's local-files-serving (``LOCAL_FILES_SERVING_ENABLED``)
    so the image bytes are streamed by LS itself.

    Returns ``None`` when ``meta.image_path`` is empty/missing or contains a
    ``..`` path-traversal segment — caller should skip the row.
    """
    image_id = image_payload["image_id"]
    meta = image_payload["meta"]
    stages = image_payload.get("stages", {})
    proposals = image_payload.get("proposals", {})

    finalize = stages.get("finalize") or {}
    final_annotations: list[dict[str, Any]] = list(
        finalize.get("final_annotations") or []
    )
    review_items: list[dict[str, Any]] = list(finalize.get("review_items") or [])
    finalize_dropped: list[dict[str, Any]] = list(finalize.get("dropped") or [])

    filter_stage = stages.get("filter") or {}
    filter_drops: list[dict[str, Any]] = list(filter_stage.get("drops") or [])

    evaluate = stages.get("evaluate") or {}
    vlm_summary = _summarize_vlm(evaluate.get("verdicts") or [])

    reconcile = stages.get("reconcile") or {}
    reconcile_propagated: list[dict[str, Any]] = list(
        reconcile.get("propagated") or []
    )

    image_size = _resolve_image_size(stages)

    image_path = meta.get("image_path", "") or ""
    if not image_path:
        logger.warning(
            "build_task: skipping image_id=%s with empty image_path", image_id
        )
        return None
    if _has_path_traversal(image_path):
        logger.warning(
            "build_task: skipping image_id=%s with path traversal in image_path=%r",
            image_id,
            image_path,
        )
        return None
    image_url = image_url_template.format(path=quote(image_path, safe="/"))

    data: dict[str, Any] = {
        "image": image_url,
        "image_id": image_id,
        "job_id": job_id,
        "image_path": image_path,
        "image_size": image_size,
        "cluster_id": meta.get("dedup_cluster_id"),
        "pre_annotations_finalize": final_annotations,
        "review_items": review_items,
        "cross_frame_suggestions": reconcile_propagated,
        "vlm_summary": vlm_summary,
        "proposal_summary": _summarize_proposals(proposals),
        "trace_excerpt": image_payload.get("trace_excerpt") or [],
    }
    if include_ghost_drops:
        data["ghost_drops"] = _ghost_drops(filter_drops, finalize_dropped)

    width, height = image_size if image_size else (None, None)
    predictions = _build_predictions(
        final_annotations,
        review_items,
        width=width,
        height=height,
        model_version=model_version,
        cross_frame_suggestions=reconcile_propagated,
        image_id=image_id,
    )

    return {
        "data": data,
        "predictions": predictions,
        "meta": {"image_id": image_id, "job_id": job_id},
    }


def _has_path_traversal(image_path: str) -> bool:
    posix_parts = PurePosixPath(image_path).parts
    win_parts = PureWindowsPath(image_path).parts
    return ".." in posix_parts or ".." in win_parts


def _bbox_to_ls_value(
    bbox: dict[str, Any], *, image_id: str, candidate_id: str
) -> tuple[float, float, float, float] | None:
    try:
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))
    except (TypeError, ValueError):
        return None
    for name, val in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if val < -0.05 or val > 1.05:
            logger.warning(
                "bbox out of [0,1] for image_id=%s candidate_id=%s %s=%s; clamping",
                image_id,
                candidate_id,
                name,
                val,
            )
            break
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return x1, y1, x2, y2


def _resolve_image_size(stages: dict[str, Any]) -> list[int] | None:
    """Pull ``image_size`` from the first stage that recorded it."""
    for stage_name in ("detect", "filter"):
        stage_data = stages.get(stage_name)
        if isinstance(stage_data, dict):
            size = stage_data.get("image_size")
            if size and isinstance(size, list) and len(size) == 2:
                return [int(size[0]), int(size[1])]
    return None


def _summarize_vlm(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim VLM verdicts to fields the reviewer cares about."""
    summary: list[dict[str, Any]] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        summary.append(
            {
                "candidate_id": v.get("candidate_id"),
                "detected_class": v.get("detected_class"),
                "class_confidence": v.get("class_confidence"),
                "bbox_score": v.get("bbox_score"),
                "reasoning": v.get("reasoning"),
            }
        )
    return summary


def _summarize_proposals(proposals: dict[str, Any]) -> dict[str, Any]:
    """Per-model count + class breakdown without the full candidate blobs."""
    out: dict[str, Any] = {}
    for model_name, payload in proposals.items():
        if not isinstance(payload, dict):
            continue
        candidates = payload.get("candidates") or []
        classes: dict[str, int] = {}
        for c in candidates:
            if not isinstance(c, dict):
                continue
            classes[c.get("class_name", "?")] = classes.get(c.get("class_name", "?"), 0) + 1
        out[model_name] = {
            "count": len(candidates),
            "classes": classes,
            "latency_ms": payload.get("latency_ms"),
        }
    return out


def _ghost_drops(
    filter_drops: list[dict[str, Any]],
    finalize_dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten filter+finalize drops into a single reviewer-friendly list."""
    out: list[dict[str, Any]] = []
    for d in filter_drops:
        if not isinstance(d, dict):
            continue
        out.append(
            {
                "candidate_id": d.get("candidate_id"),
                "reason": d.get("reason"),
                "context": d.get("context"),
                "detail": d.get("detail"),
                "stage": "filter",
            }
        )
    for d in finalize_dropped:
        if not isinstance(d, dict):
            continue
        out.append(
            {
                "candidate_id": d.get("candidate_id"),
                "class_name": d.get("class_name"),
                "reason": d.get("reason"),
                "bbox": d.get("bbox"),
                "stage": "finalize",
            }
        )
    return out


def _build_predictions(
    final_annotations: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
    *,
    width: int | None,
    height: int | None,
    model_version: str,
    cross_frame_suggestions: list[dict[str, Any]] | None = None,
    image_id: str = "",
) -> list[dict[str, Any]]:
    """Convert finalize annotations into LS RectangleLabels predictions.

    LS expects pixel-percentage coordinates (0-100), so normalized aav4 boxes
    are scaled by 100. Width/height are passed through unused for boxes (LS
    only needs them when serving a non-image task) but are surfaced in the
    prediction so frontends that re-render know the canvas size.

    Each region's id is the aav4 ``candidate_id`` so the export parser can
    distinguish reviewer-edited ids from new draws.

    ``cross_frame_suggestions`` (Stage RECONCILE output) are added as a
    secondary prediction group tagged ``meta.source="cross_frame"``, so the
    LS XML can render them in a distinct color and the export parser can
    classify their fate (accepted → ``edited``/``finalize``, ignored →
    deletion).
    """
    review_ids = {
        item.get("candidate_id")
        for item in review_items
        if isinstance(item, dict) and item.get("candidate_id")
    }

    results: list[dict[str, Any]] = []
    dropped = 0
    for i, ann in enumerate(final_annotations):
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox") or {}
        candidate_id = ann.get("candidate_id") or ""
        class_name = ann.get("class_name") or ""
        if not candidate_id or not class_name:
            logger.warning(
                "dropping finalize annotation idx=%d image_id=%s missing candidate_id/class_name",
                i,
                image_id,
            )
            dropped += 1
            continue
        coords = _bbox_to_ls_value(bbox, image_id=image_id, candidate_id=candidate_id)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        results.append(
            {
                "id": candidate_id,
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": x1 * 100.0,
                    "y": y1 * 100.0,
                    "width": (x2 - x1) * 100.0,
                    "height": (y2 - y1) * 100.0,
                    "rotation": 0,
                    "rectanglelabels": [class_name],
                },
                "meta": {
                    "needs_review": candidate_id in review_ids,
                    "source_model": ann.get("source_model"),
                    "was_refined": ann.get("was_refined", False),
                    "confidence": ann.get("confidence"),
                },
            }
        )

    for j, sug in enumerate(cross_frame_suggestions or []):
        if not isinstance(sug, dict):
            continue
        bbox = sug.get("bbox") or {}
        candidate_id = sug.get("candidate_id") or ""
        class_name = sug.get("class_name") or ""
        if not candidate_id or not class_name:
            logger.warning(
                "dropping cross_frame suggestion idx=%d image_id=%s missing candidate_id/class_name",
                j,
                image_id,
            )
            continue
        coords = _bbox_to_ls_value(bbox, image_id=image_id, candidate_id=candidate_id)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        results.append(
            {
                "id": candidate_id,
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": x1 * 100.0,
                    "y": y1 * 100.0,
                    "width": (x2 - x1) * 100.0,
                    "height": (y2 - y1) * 100.0,
                    "rotation": 0,
                    "rectanglelabels": [class_name],
                },
                "meta": {
                    "source": "cross_frame",
                    "cluster_id": sug.get("cluster_id"),
                    "mask_score": sug.get("mask_score"),
                    "seed_iou": sug.get("seed_iou"),
                    "vote_count": len(sug.get("votes") or []),
                },
            }
        )

    if not results:
        return []
    return [{"model_version": model_version, "result": results, "score": 1.0}]
