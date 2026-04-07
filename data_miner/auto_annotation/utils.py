from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .config import ClassPackConfig
from .contracts import BoundingBox, Candidate, PipelineResult
from .log_utils import get_logger


logger = get_logger(__name__)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def bbox_area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(left) + bbox_area(right) - inter
    return 0.0 if union <= 0 else inter / union


def weighted_box_fusion(candidates: Iterable[Candidate]) -> BoundingBox:
    items = list(candidates)
    total = sum(max(candidate.score, 1e-6) for candidate in items)
    return BoundingBox(
        x1=sum(candidate.bbox.x1 * candidate.score for candidate in items) / total,
        y1=sum(candidate.bbox.y1 * candidate.score for candidate in items) / total,
        x2=sum(candidate.bbox.x2 * candidate.score for candidate in items) / total,
        y2=sum(candidate.bbox.y2 * candidate.score for candidate in items) / total,
    )


def bbox_to_pixels(box: BoundingBox, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        int(clamp(box.x1) * width),
        int(clamp(box.y1) * height),
        int(clamp(box.x2) * width),
        int(clamp(box.y2) * height),
    )


def pixels_to_bbox(coords: tuple[int, int, int, int], size: tuple[int, int]) -> BoundingBox:
    width, height = size
    x1, y1, x2, y2 = coords
    return BoundingBox(
        x1=clamp(x1 / width),
        y1=clamp(y1 / height),
        x2=clamp(x2 / width),
        y2=clamp(y2 / height),
    )


def draw_candidate(image: Image.Image, candidate: Candidate) -> Image.Image:
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(bbox_to_pixels(candidate.bbox, rendered.size), outline=(255, 0, 0), width=3)
    draw.text(bbox_to_pixels(candidate.bbox, rendered.size)[:2], candidate.label, fill=(255, 0, 0))
    return rendered


def crop_candidate(image: Image.Image, candidate: Candidate, padding: float = 0.08) -> Image.Image:
    width, height = image.size
    box = candidate.bbox
    px = int((box.x2 - box.x1) * width * padding)
    py = int((box.y2 - box.y1) * height * padding)
    x1, y1, x2, y2 = bbox_to_pixels(box, image.size)
    return image.crop((max(0, x1 - px), max(0, y1 - py), min(width, x2 + px), min(height, y2 + py)))


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def candidate_to_yolo_line(candidate: Candidate, class_index: int) -> str:
    box = candidate.bbox
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    x_center = box.x1 + width / 2
    y_center = box.y1 + height / 2
    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def normalize_class_alias(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def build_class_alias_map(class_packs: list[ClassPackConfig]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for class_pack in class_packs:
        canonical = class_pack.name
        for alias in class_pack.names():
            alias_map[normalize_class_alias(alias)] = canonical
    return alias_map


def resolve_canonical_class_name(value: str | None, class_packs: list[ClassPackConfig]) -> str | None:
    if not value:
        return None
    alias_map = build_class_alias_map(class_packs)
    return alias_map.get(normalize_class_alias(value))


def save_result(result: PipelineResult, class_names: list[str], output_dir: Path, output_cfg) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem
    label_map = {name: index for index, name in enumerate(class_names)}
    export_warnings: list[str] = []
    export_review_ids: list[str] = []

    missing_labels = [candidate for candidate in result.accepted if candidate.class_name not in label_map]
    for candidate in missing_labels:
        warning = (
            f"Accepted candidate {candidate.candidate_id} has unmapped class_name "
            f"'{candidate.class_name}' and was not written to YOLO output."
        )
        export_warnings.append(warning)
        export_review_ids.append(candidate.candidate_id)
        logger.warning("save_result.warning image=%s candidate_id=%s class_name=%s", result.image_path, candidate.candidate_id, candidate.class_name)

    if output_cfg.save_labels:
        label_dir = output_dir / output_cfg.label_dirname
        label_dir.mkdir(parents=True, exist_ok=True)
        lines = [candidate_to_yolo_line(candidate, label_map[candidate.class_name]) for candidate in result.accepted if candidate.class_name in label_map]
        (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    if output_cfg.save_sidecars:
        sidecar_dir = output_dir / output_cfg.sidecar_dirname
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        payload["export_warnings"] = export_warnings
        (sidecar_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    review_candidate_ids = [candidate.candidate_id for candidate in result.human_review]
    review_candidate_ids.extend(candidate_id for candidate_id in export_review_ids if candidate_id not in review_candidate_ids)
    failure_candidate_ids = [failure.candidate_id for failure in result.failures if failure.candidate_id]
    review_candidate_ids.extend(candidate_id for candidate_id in failure_candidate_ids if candidate_id not in review_candidate_ids)
    review_candidate_ids = list(dict.fromkeys(review_candidate_ids))

    if output_cfg.save_review_queue and (review_candidate_ids or result.partial or result.failures or export_warnings):
        review_dir = output_dir / output_cfg.review_dirname
        review_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_path": result.image_path,
            "candidate_ids": review_candidate_ids,
            "export_warnings": export_warnings,
            "partial": result.partial,
            "failures": [failure.model_dump(mode="json") for failure in result.failures],
        }
        (review_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")