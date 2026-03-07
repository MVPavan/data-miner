"""
COCO-centric data loader for detection_metrics.

All dataset formats are first converted to COCO JSON (via converter module),
then loaded into a unified internal representation for the evaluators.

Internal format:  List of [img_id, cat_id, confidence, x1, y1, x2, y2]
  - Ground truth always has confidence = 1.0
  - Predictions have the model's confidence score
  - Bounding boxes are in absolute pixel coordinates (x1, y1, x2, y2)
  - Category IDs match the COCO JSON (1-based)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from detection_metrics.converter import (
    coco_gt_to_internal,
    coco_preds_to_internal,
    ensure_coco_gt,
    load_coco_gt,
    load_coco_predictions,
)
from detection_metrics.logging import logger
from detection_metrics.configs.config import DatasetConfig, PredictionEntry


class DataLoader:
    """
    COCO-centric data loader.

    All inputs — whether COCO JSON or YOLO directories — are resolved
    to COCO JSON first (on-disk), then parsed into the internal list format.
    """

    def __init__(self):
        self.categories: Dict[int, str] = {}

    # ── Public API ────────────────────────────────────────────────────

    def load_gt(
        self,
        gt_path: Path,
        class_names: Optional[List[str]] = None,
    ) -> Tuple[List[List[float]], Path]:
        """
        Load ground truth from any supported format → internal list.

        For YOLO directories, auto-converts to COCO JSON (cached).
        For COCO JSON files, loads directly.

        Args:
            gt_path:     Path to COCO JSON or YOLO dataset root directory.
            class_names: Class names for YOLO→COCO conversion (optional).

        Returns:
            (data, coco_json_path)
            data: [[img_id, cat_id, 1.0, x1, y1, x2, y2], ...]
        """
        gt_path = Path(gt_path)

        if gt_path.suffix == ".json":
            coco_json_path = gt_path
        elif gt_path.is_dir():
            # Auto-detect split: look for images/ subdir directly
            # or treat as dataset root with split dirs
            if (gt_path / "images").is_dir():
                # This IS the split dir (e.g., dataset/test/)
                coco_json_path = self._ensure_split_coco(gt_path, class_names)
            else:
                raise ValueError(
                    f"Directory {gt_path} has no images/ sub-dir. "
                    f"Provide a COCO JSON or a split directory with images/."
                )
        else:
            raise ValueError(f"Unsupported GT path: {gt_path}")

        coco_dict, cat_map = load_coco_gt(coco_json_path)
        self.categories = cat_map
        data, _ = coco_gt_to_internal(coco_dict)

        logger.info(
            f"Loaded GT: {len(data)} annotations, "
            f"{len(coco_dict.get('images', []))} images, "
            f"{len(cat_map)} categories"
        )
        return data, coco_json_path

    def load_predictions(
        self,
        predictions: List[PredictionEntry],
    ) -> Dict[str, Tuple[List[List[float]], Path]]:
        """
        Load predictions from COCO JSON files.

        Args:
            predictions: List of PredictionEntry (path + name).

        Returns:
            {name: (internal_data, json_path)}
        """
        result = {}
        for entry in predictions:
            pred_path = Path(entry.path)
            if pred_path.suffix == ".json":
                preds_list = load_coco_predictions(pred_path)
                internal = coco_preds_to_internal(preds_list)
                logger.info(f"Loaded predictions '{entry.name}': {len(internal)} detections")
                result[entry.name] = (internal, pred_path)
            elif pred_path.is_dir():
                # YOLO predictions directory — convert to COCO first
                coco_path = self._convert_yolo_preds(pred_path, entry.name)
                preds_list = load_coco_predictions(coco_path)
                internal = coco_preds_to_internal(preds_list)
                logger.info(f"Loaded predictions '{entry.name}': {len(internal)} detections")
                result[entry.name] = (internal, coco_path)
            else:
                raise ValueError(f"Unsupported prediction format: {pred_path}")
        return result

    def load_dataset(
        self,
        dataset_config: DatasetConfig,
        skip_models: Optional[List[str]] = None,
    ) -> Tuple[List[List[float]], Dict[str, List[List[float]]], Path]:
        """
        Load complete dataset (GT + predictions) → COCO-centric workflow.

        Args:
            dataset_config: Dataset configuration.
            skip_models:    Model names to skip.

        Returns:
            (gt_data, {model_name: pred_data}, gt_coco_json_path)
        """
        skip_models = set(skip_models or [])

        if not dataset_config.gt_path:
            raise ValueError("GT path not specified in config")

        predictions = [
            p for p in dataset_config.predictions
            if p.name not in skip_models
        ]

        gt_data, gt_json_path = self.load_gt(dataset_config.gt_path)

        if not predictions:
            return gt_data, {}, gt_json_path

        pred_result = self.load_predictions(predictions)
        pred_data = {name: data for name, (data, _) in pred_result.items()}
        return gt_data, pred_data, gt_json_path

    # ── Internal helpers ──────────────────────────────────────────────

    def _ensure_split_coco(
        self,
        split_dir: Path,
        class_names: Optional[List[str]] = None,
    ) -> Path:
        """Ensure COCO JSON exists for a single split directory."""
        # Check for existing annotations
        for candidate in [
            split_dir / "_annotations.coco.json",
            split_dir / "annotations.json",
        ]:
            if candidate.exists():
                return candidate

        # Need to generate — find dataset root (parent of the split)
        dataset_root = split_dir.parent
        split_name = split_dir.name
        return ensure_coco_gt(dataset_root, split_name, class_names)

    def _convert_yolo_preds(self, pred_dir: Path, name: str) -> Path:
        """Convert YOLO prediction txt files to COCO JSON predictions."""
        import numpy as np

        if not self.categories:
            raise ValueError("Load GT first before loading YOLO predictions")

        cache_dir = pred_dir.parent / ".eval_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"preds_{name}_coco.json"

        if out_path.exists():
            return out_path

        preds: list[dict] = []
        txt_files = sorted(pred_dir.glob("*.txt"))

        for txt_path in txt_files:
            img_id = abs(hash(txt_path.stem)) % (10**9)
            raw = np.loadtxt(txt_path, ndmin=2)
            if raw.size == 0:
                continue
            for row in raw:
                cls_id = int(row[0])
                cx, cy, w, h = row[1], row[2], row[3], row[4]
                conf = float(row[5]) if len(row) > 5 else 1.0
                preds.append({
                    "image_id": img_id,
                    "category_id": cls_id + 1,
                    "bbox": [cx, cy, w, h],
                    "score": conf,
                })

        out_path.write_text(json.dumps(preds))
        logger.info(f"Converted YOLO preds → {out_path}: {len(preds)} detections")
        return out_path
