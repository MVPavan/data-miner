"""Parse CVAT Datumaro exports into frontend-neutral review results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from data_miner.annotation_io import (
    FrontendName,
    ReviewBox,
    ReviewBoxSource,
    ReviewExchangeResult,
    ReviewRegionOrigin,
)
from data_miner.auto_annotation_v4.configs.contracts import BoundingBox

FrameState = Literal["clean", "needs_more_review", "ambiguous_skip"]


def parse_datumaro_review_results(
    document: Mapping[str, Any],
    *,
    reviewer_id: str,
    reviewed_at: float,
    source_task_id: str | None = None,
    source_job_id: str | None = None,
    default_frame_state: FrameState = "clean",
    class_list_version: str | None = None,
) -> list[ReviewExchangeResult]:
    """Convert a Datumaro 1.0 JSON document into exchange review results."""
    id_to_label = _label_lookup(document)
    results: list[ReviewExchangeResult] = []

    for item in document.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        item_attrs = _attrs(item)
        image = item.get("image") if isinstance(item.get("image"), Mapping) else {}
        image_width, image_height = _image_size(image)
        boxes = [
            _parse_bbox_annotation(annotation, id_to_label, image_width, image_height)
            for annotation in item.get("annotations") or []
            if isinstance(annotation, Mapping) and annotation.get("type") == "bbox"
        ]
        results.append(
            ReviewExchangeResult(
                image_id=_item_image_id(item, item_attrs),
                source_frontend=FrontendName.CVAT,
                source_task_id=_attr_str(item_attrs, "task_id") or source_task_id,
                source_job_id=_attr_str(item_attrs, "job_id") or source_job_id,
                reviewer_id=reviewer_id,
                reviewed_at=reviewed_at,
                duration_seconds=_attr_float(item_attrs, "duration_seconds"),
                frame_state=_frame_state(item_attrs, default_frame_state),
                media_uri=_media_uri(image),
                image_width=image_width,
                image_height=image_height,
                clip_id=_attr_str(item_attrs, "clip_id"),
                frame_index=_attr_int(item_attrs, "frame_index"),
                class_list_version=(
                    _attr_str(item_attrs, "class_list_version") or class_list_version
                ),
                boxes=boxes,
                deletions=_deletions(item_attrs),
                notes=_attr_str(item_attrs, "notes") or "",
                ml_modes_used=_string_list(item_attrs.get("ml_modes_used")),
                source_completion_id=_attr_str(item_attrs, "completion_id"),
                attributes=dict(item_attrs),
            )
        )

    return results


def _label_lookup(document: Mapping[str, Any]) -> dict[int, str]:
    """Build a Datumaro label-id to label-name lookup."""
    categories = document.get("categories") or {}
    label_category = categories.get("label") if isinstance(categories, Mapping) else []
    if isinstance(label_category, Mapping):
        labels = label_category.get("labels") or label_category.get("items") or []
    else:
        labels = label_category or []

    out: dict[int, str] = {}
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        label_id = label.get("id")
        label_name = label.get("name")
        if isinstance(label_id, int) and isinstance(label_name, str):
            out[label_id] = label_name
    return out


def _attrs(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the Datumaro attributes mapping for an item or annotation."""
    attrs = value.get("attributes") or {}
    return attrs if isinstance(attrs, Mapping) else {}


def _image_size(image: Mapping[str, Any]) -> tuple[int, int]:
    """Read a Datumaro image size as ``(width, height)``."""
    size = image.get("size") or []
    if not isinstance(size, list | tuple) or len(size) < 2:
        raise ValueError("Datumaro item image.size must contain width and height")
    width = _positive_int(size[0], "image width")
    height = _positive_int(size[1], "image height")
    return width, height


def _positive_int(value: Any, field_name: str) -> int:
    """Convert a numeric value into a positive integer."""
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if converted <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return converted


def _parse_bbox_annotation(
    annotation: Mapping[str, Any],
    id_to_label: Mapping[int, str],
    image_width: int,
    image_height: int,
) -> ReviewBox:
    """Convert one Datumaro bbox annotation into an exchange box."""
    attrs = _attrs(annotation)
    label_id = annotation.get("label")
    if not isinstance(label_id, int) or label_id not in id_to_label:
        raise ValueError(f"unknown Datumaro label id: {label_id!r}")

    return ReviewBox(
        region_id=_region_id(annotation, attrs),
        class_name=id_to_label[label_id],
        bbox=_bbox(annotation, image_width, image_height),
        source=_box_source(attrs),
        origin=_box_origin(attrs),
        original_class=_attr_str(attrs, "original_class"),
        original_bbox=_original_bbox(attrs),
        track_id=_attr_str(attrs, "track_id"),
        attributes=dict(attrs),
    )


def _region_id(annotation: Mapping[str, Any], attrs: Mapping[str, Any]) -> str | None:
    """Resolve the stable region id for a Datumaro annotation."""
    return (
        _attr_str(attrs, "candidate_id")
        or _attr_str(attrs, "region_id")
        or _optional_str(annotation.get("id"))
    )


def _bbox(annotation: Mapping[str, Any], image_width: int, image_height: int) -> BoundingBox:
    """Normalize one Datumaro absolute-pixel bbox into a v4 BoundingBox."""
    x = _number(annotation.get("x"), "bbox x")
    y = _number(annotation.get("y"), "bbox y")
    width = _number(annotation.get("w"), "bbox width")
    height = _number(annotation.get("h"), "bbox height")
    x1 = _clamp(x / image_width)
    y1 = _clamp(y / image_height)
    x2 = _clamp((x + width) / image_width)
    y2 = _clamp((y + height) / image_height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _number(value: Any, field_name: str) -> float:
    """Convert a Datumaro numeric field into a finite float."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _clamp(value: float) -> float:
    """Clamp a normalized coordinate to the v4 bbox range."""
    return max(0.0, min(1.0, value))


def _box_source(attrs: Mapping[str, Any]) -> ReviewBoxSource:
    """Map Datumaro annotation attributes to the v4 review source vocabulary."""
    raw_source = str(attrs.get("review_source") or attrs.get("source") or "").lower()
    for source in ReviewBoxSource:
        if raw_source == source.value:
            return source
    if raw_source in {"prediction", "model", "preannotation"}:
        return ReviewBoxSource.FINALIZE
    return ReviewBoxSource.ADDED


def _box_origin(attrs: Mapping[str, Any]) -> ReviewRegionOrigin:
    """Map Datumaro annotation attributes to the exchange origin vocabulary."""
    raw_origin = str(attrs.get("origin") or attrs.get("source") or "").lower()
    for origin in ReviewRegionOrigin:
        if raw_origin == origin.value:
            return origin
    if raw_origin in {"prediction", "model", "preannotation"}:
        return ReviewRegionOrigin.PREDICTION
    if raw_origin in {"manual", "human", "added", "edited", "relabeled"}:
        return ReviewRegionOrigin.HUMAN
    return ReviewRegionOrigin.HUMAN


def _original_bbox(attrs: Mapping[str, Any]) -> BoundingBox | None:
    """Parse an optional original bbox from Datumaro attributes."""
    raw_bbox = attrs.get("original_bbox")
    if isinstance(raw_bbox, Mapping):
        return BoundingBox(**raw_bbox)
    if isinstance(raw_bbox, list | tuple) and len(raw_bbox) == 4:
        return BoundingBox(
            x1=float(raw_bbox[0]),
            y1=float(raw_bbox[1]),
            x2=float(raw_bbox[2]),
            y2=float(raw_bbox[3]),
        )
    return None


def _item_image_id(item: Mapping[str, Any], attrs: Mapping[str, Any]) -> str:
    """Resolve the exchange image id for a Datumaro item."""
    image_id = _attr_str(attrs, "image_id") or _optional_str(item.get("id"))
    if not image_id:
        raise ValueError("Datumaro item is missing an id/image_id")
    return image_id


def _media_uri(image: Mapping[str, Any]) -> str | None:
    """Resolve a Datumaro image media path if present."""
    return _optional_str(image.get("path"))


def _frame_state(attrs: Mapping[str, Any], default_frame_state: FrameState) -> FrameState:
    """Resolve the frame state from item attributes."""
    raw_state = attrs.get("frame_state")
    if raw_state in {"clean", "needs_more_review", "ambiguous_skip"}:
        return raw_state
    if attrs.get("needs_more_review") is True:
        return "needs_more_review"
    return default_frame_state


def _deletions(attrs: Mapping[str, Any]) -> list[str]:
    """Read deleted candidate ids from item attributes."""
    raw_deletions = attrs.get("deletions") or attrs.get("deleted_candidate_ids") or []
    return _string_list(raw_deletions)


def _string_list(value: Any) -> list[str]:
    """Convert a string or sequence into a list of non-empty strings."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return []


def _attr_str(attrs: Mapping[str, Any], key: str) -> str | None:
    """Read one optional attribute as a string."""
    return _optional_str(attrs.get(key))


def _attr_int(attrs: Mapping[str, Any], key: str) -> int | None:
    """Read one optional attribute as an integer."""
    value = attrs.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _attr_float(attrs: Mapping[str, Any], key: str) -> float:
    """Read one optional attribute as a finite float."""
    value = attrs.get(key)
    if value is None or value == "":
        return 0.0
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0


def _optional_str(value: Any) -> str | None:
    """Convert optional values to strings while preserving missing values."""
    if value is None or value == "":
        return None
    return str(value)