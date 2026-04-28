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

from typing import Any


def build_task(
    image_payload: dict[str, Any],
    *,
    image_url_template: str = "/data/local-files/?d={path}",
    job_id: str = "",
    model_version: str = "aa_v4_finalize",
    include_ghost_drops: bool = True,
) -> dict[str, Any]:
    """Build one Label Studio task dict from a ``read_image_payload`` result.

    ``image_url_template`` is formatted with ``path=<image_path>``. The default
    targets Label Studio's local-files-serving (``LOCAL_FILES_SERVING_ENABLED``)
    so the image bytes are streamed by LS itself.
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

    image_size = _resolve_image_size(stages)

    image_path = meta.get("image_path", "")
    image_url = image_url_template.format(path=image_path) if image_path else ""

    data: dict[str, Any] = {
        "image": image_url,
        "image_id": image_id,
        "job_id": job_id,
        "image_path": image_path,
        "image_size": image_size,
        "cluster_id": meta.get("dedup_cluster_id"),
        "pre_annotations_finalize": final_annotations,
        "review_items": review_items,
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
    )

    return {
        "data": data,
        "predictions": predictions,
        "meta": {"image_id": image_id, "job_id": job_id},
    }


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
) -> list[dict[str, Any]]:
    """Convert finalize annotations into LS RectangleLabels predictions.

    LS expects pixel-percentage coordinates (0-100), so normalized aav4 boxes
    are scaled by 100. Width/height are passed through unused for boxes (LS
    only needs them when serving a non-image task) but are surfaced in the
    prediction so frontends that re-render know the canvas size.

    Each region's id is the aav4 ``candidate_id`` so the export parser can
    distinguish reviewer-edited ids from new draws.
    """
    review_ids = {
        item.get("candidate_id")
        for item in review_items
        if isinstance(item, dict) and item.get("candidate_id")
    }

    results: list[dict[str, Any]] = []
    for ann in final_annotations:
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox") or {}
        candidate_id = ann.get("candidate_id") or ""
        class_name = ann.get("class_name") or ""
        if not candidate_id or not class_name:
            continue
        try:
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
        except (TypeError, ValueError):
            continue
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

    if not results:
        return []
    return [{"model_version": model_version, "result": results, "score": 1.0}]
