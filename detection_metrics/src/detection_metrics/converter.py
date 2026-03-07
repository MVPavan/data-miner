"""
Evaluation-focused format converter — bridges between dataset formats and
the internal evaluation representation.

For full dataset format conversion (COCO, YOLO, Darknet, Roboflow, etc.)
see :mod:`detection_metrics.convert_dataset`.

This module delegates YOLO→COCO ground-truth conversion to
``convert_dataset`` and adds evaluation-specific helpers:
  - ``ensure_coco_gt``  — guarantee a COCO GT JSON exists for a split
  - ``predictions_to_coco`` — model output → COCO prediction format
  - ``load_coco_gt`` / ``load_coco_predictions`` — load JSON files
  - ``coco_gt_to_internal`` / ``coco_preds_to_internal`` — COCO ↔ flat list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from detection_metrics.convert_dataset import (
    build_coco_json,
    read_data_yaml_names,
    read_yolo_split,
)
from detection_metrics.logging import logger


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    output_path: Optional[Path] = None,
) -> dict:
    """
    Convert a YOLO-format split (images/ + labels/) to a COCO JSON dict.

    Delegates to :func:`convert_dataset.read_yolo_split` for parsing and
    :func:`convert_dataset.build_coco_json` for serialisation.

    Args:
        images_dir:  Directory containing images.
        labels_dir:  Directory containing YOLO label .txt files.
        class_names: Ordered list of class names (index = class id).
        output_path: Optional path to write the JSON file.

    Returns:
        COCO-format dict with images, annotations, categories.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    logger.info(f"Converting YOLO → COCO: {images_dir}")
    entries = read_yolo_split(images_dir, labels_dir, desc="YOLO → COCO")
    coco = build_coco_json(class_names, entries)

    logger.info(
        f"Converted: {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(coco['categories'])} categories"
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(coco))
        logger.info(f"Saved COCO JSON → {output_path}")

    return coco


def ensure_coco_gt(
    dataset_dir: Path,
    split: str = "test",
    class_names: Optional[List[str]] = None,
) -> Path:
    """
    Ensure a COCO JSON GT file exists for a YOLO-format dataset split.
    Returns the path to the file (existing or newly generated).

    Searches for:
      1. {split}/_annotations.coco.json  (Roboflow export)
      2. annotations/instances_{split}2017.json  (COCO-style)
      3. .eval_cache/gt_{split}_coco.json  (previously generated)
    If none found, generates from YOLO labels.

    Args:
        dataset_dir: Root of the YOLO dataset.
        split:       Split name (train/valid/test).
        class_names: Ordered class names. Auto-read from data.yaml if None.

    Returns:
        Path to the COCO JSON GT file.
    """
    dataset_dir = Path(dataset_dir)

    # Check existing files
    candidates = [
        dataset_dir / split / "_annotations.coco.json",
        dataset_dir / "annotations" / f"instances_{split}2017.json",
    ]
    for c in candidates:
        if c.exists():
            logger.debug(f"Found existing COCO GT: {c}")
            return c

    # Cached version
    cache_dir = dataset_dir / ".eval_cache"
    cached = cache_dir / f"gt_{split}_coco.json"
    if cached.exists():
        logger.debug(f"Found cached COCO GT: {cached}")
        return cached

    # Generate from YOLO
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    if class_names is None:
        class_names = read_data_yaml_names(dataset_dir)

    yolo_to_coco(images_dir, labels_dir, class_names, output_path=cached)
    return cached


def predictions_to_coco(
    predictions: List[Tuple[int, float, float, float, float, float, int]],
    output_path: Optional[Path] = None,
) -> list:
    """
    Convert a list of detection tuples to COCO prediction JSON format.

    Each tuple: (image_id, x1, y1, x2, y2, score, class_id)  (0-based class_id)

    Returns:
       List of COCO prediction dicts (image_id, bbox [xywh], score, category_id).
    """
    coco_preds = []
    for det in predictions:
        img_id, x1, y1, x2, y2, score, cls = det
        coco_preds.append({
            "image_id": int(img_id),
            "category_id": int(cls) + 1,
            "bbox": [round(float(x1), 2), round(float(y1), 2),
                     round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
            "score": round(float(score), 4),
        })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(coco_preds))
        logger.info(f"Saved {len(coco_preds)} predictions → {output_path}")

    return coco_preds


def load_coco_gt(json_path: Path) -> Tuple[dict, Dict[int, str]]:
    """
    Load a COCO GT JSON and extract category map.

    Returns:
        (coco_dict, {cat_id: cat_name})
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    return data, cat_map


def load_coco_predictions(json_path: Path) -> list:
    """Load a COCO prediction JSON (list of dicts)."""
    with open(json_path) as f:
        return json.load(f)


def coco_gt_to_internal(
    coco_dict: dict,
) -> Tuple[List[List[float]], Dict[int, str]]:
    """
    Convert COCO GT dict → internal flat format + category map.

    Returns:
        (data_list, categories)
        data_list: [[img_id, cat_id, 1.0, x1, y1, x2, y2], ...]
        categories: {cat_id: name}
    """
    categories = {c["id"]: c["name"] for c in coco_dict.get("categories", [])}
    data = []
    for ann in coco_dict.get("annotations", []):
        x, y, w, h = ann["bbox"]
        data.append([
            ann["image_id"], ann["category_id"], 1.0,
            x, y, x + w, y + h,
        ])
    return data, categories


def coco_preds_to_internal(
    preds_list: list,
) -> List[List[float]]:
    """
    Convert COCO prediction list → internal flat format.

    Returns:
        [[img_id, cat_id, score, x1, y1, x2, y2], ...]
    """
    data = []
    for p in preds_list:
        x, y, w, h = p["bbox"]
        data.append([
            p["image_id"], p["category_id"], p["score"],
            x, y, x + w, y + h,
        ])
    return data
