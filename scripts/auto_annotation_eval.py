"""
Auto-annotation evaluation pipeline.

Validates auto-annotation model detections against manually annotated GT
by applying NMS at various IoU thresholds and computing detection metrics.

Usage:
    python helpers/auto_annotation_eval.py --config eval_results/auto_annotation/eval_config.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# Add project root so detection_metrics is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "detection_metrics" / "src"))

from detection_metrics import DetectionMetrics
from detection_metrics.configs.config import (
    AnalysisConfig,
    EvaluateConfig,
    OutputConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMS (pure NumPy, single-class)
# ---------------------------------------------------------------------------

def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    Greedy NMS on xyxy boxes.  Returns indices to keep.

    Args:
        boxes_xyxy: (N, 4) float array  [x1, y1, x2, y2]
        scores:     (N,) float array
        iou_thr:    IoU threshold – boxes with IoU > iou_thr are suppressed.

    Returns:
        np.ndarray of kept indices.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (N,4) centre-xy-wh to x1y1x2y2."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


# ---------------------------------------------------------------------------
# YOLO txt I/O helpers
# ---------------------------------------------------------------------------

def load_yolo_preds(
    pred_dir: Path, default_conf: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Load YOLO prediction txts from a directory.

    Returns:
        {stem: np.ndarray (N, 6)} where cols = [class_id, cx, cy, w, h, conf]
        Empty files → array of shape (0, 6).
    """
    preds: Dict[str, np.ndarray] = {}
    for txt in sorted(pred_dir.glob("*.txt")):
        stem = txt.stem
        lines = txt.read_text().strip()
        if not lines:
            preds[stem] = np.empty((0, 6), dtype=float)
            continue
        rows = []
        for line in lines.split("\n"):
            vals = line.split()
            cls_id = int(vals[0])
            cx, cy, w, h = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
            conf = float(vals[5]) if len(vals) > 5 else default_conf
            rows.append([cls_id, cx, cy, w, h, conf])
        preds[stem] = np.array(rows, dtype=float)
    return preds


def save_yolo_preds(preds: Dict[str, np.ndarray], out_dir: Path) -> None:
    """Write YOLO prediction txts (6-col: class_id cx cy w h conf)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, arr in preds.items():
        lines = []
        for row in arr:
            cls_id = int(row[0])
            cx, cy, w, h, conf = row[1], row[2], row[3], row[4], row[5]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Phase A+B: NMS
# ---------------------------------------------------------------------------

def apply_nms_to_preds(
    preds: Dict[str, np.ndarray], iou_thr: float
) -> Dict[str, np.ndarray]:
    """Apply per-image NMS and return filtered predictions."""
    filtered: Dict[str, np.ndarray] = {}
    for stem, arr in preds.items():
        if arr.shape[0] == 0:
            filtered[stem] = arr
            continue
        boxes_cxcywh = arr[:, 1:5]
        scores = arr[:, 5]
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
        keep = nms_numpy(boxes_xyxy, scores, iou_thr)
        filtered[stem] = arr[keep]
    return filtered


# ---------------------------------------------------------------------------
# Phase C: Build combined GT COCO JSON
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def _find_image_path(images_dir: Path, stem: str) -> Optional[Path]:
    """Find image file by stem, trying common extensions."""
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_combined_gt_coco(
    dataset_path: Path,
    splits: List[str],
    output_path: Path,
) -> Tuple[dict, Dict[str, int], Dict[str, Tuple[int, int]]]:
    """
    Merge YOLO GT from multiple splits into a single COCO JSON.

    Returns:
        (coco_dict, stem_to_image_id, stem_to_size)
        stem_to_size: {stem: (width, height)}
    """
    categories = [{"id": 1, "name": "door", "supercategory": "object"}]
    images_list = []
    annotations_list = []
    stem_to_id: Dict[str, int] = {}
    stem_to_size: Dict[str, Tuple[int, int]] = {}
    img_id = 0
    ann_id = 0

    for split in splits:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        if not labels_dir.exists():
            logger.warning(f"Labels dir not found: {labels_dir}")
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        logger.info(f"  GT split '{split}': {len(label_files)} label files")

        for lbl_file in label_files:
            stem = lbl_file.stem
            if stem in stem_to_id:
                continue  # skip duplicates across splits

            img_path = _find_image_path(images_dir, stem)
            if img_path is None:
                continue

            # Read image dimensions (follow symlinks)
            real_path = img_path.resolve()
            with Image.open(real_path) as im:
                w_img, h_img = im.size

            img_id += 1
            stem_to_id[stem] = img_id
            stem_to_size[stem] = (w_img, h_img)

            images_list.append({
                "id": img_id,
                "file_name": f"{stem}{img_path.suffix}",
                "width": w_img,
                "height": h_img,
            })

            # Parse YOLO labels
            text = lbl_file.read_text().strip()
            if not text:
                continue
            for line in text.split("\n"):
                vals = line.split()
                if len(vals) < 5:
                    continue
                cls_id = int(vals[0])
                cx, cy, bw, bh = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
                # Convert normalised cxcywh → absolute xywh (COCO format)
                x_abs = (cx - bw / 2) * w_img
                y_abs = (cy - bh / 2) * h_img
                w_abs = bw * w_img
                h_abs = bh * h_img

                ann_id += 1
                annotations_list.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id + 1,  # YOLO 0-based → COCO 1-based
                    "bbox": [round(x_abs, 2), round(y_abs, 2), round(w_abs, 2), round(h_abs, 2)],
                    "area": round(w_abs * h_abs, 2),
                    "iscrowd": 0,
                })

    coco_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco_dict))
    logger.info(
        f"Combined GT COCO: {len(images_list)} images, "
        f"{len(annotations_list)} annotations → {output_path}"
    )
    return coco_dict, stem_to_id, stem_to_size


# ---------------------------------------------------------------------------
# Phase C: Convert NMS'd preds to COCO JSON
# ---------------------------------------------------------------------------

def preds_to_coco_json(
    preds: Dict[str, np.ndarray],
    stem_to_id: Dict[str, int],
    stem_to_size: Dict[str, Tuple[int, int]],
    output_path: Path,
) -> Tuple[list, int, int]:
    """
    Convert YOLO normalised predictions to COCO prediction JSON.

    Returns:
        (coco_preds_list, n_matched_images, n_skipped_images)
    """
    coco_preds = []
    n_matched = 0
    n_skipped = 0

    for stem, arr in preds.items():
        if stem not in stem_to_id:
            n_skipped += 1
            continue
        n_matched += 1
        img_id = stem_to_id[stem]
        w_img, h_img = stem_to_size[stem]

        for row in arr:
            cls_id = int(row[0])
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            conf = float(row[5])
            # Denormalise: cxcywh → xywh absolute pixels
            x_abs = (cx - bw / 2) * w_img
            y_abs = (cy - bh / 2) * h_img
            w_abs = bw * w_img
            h_abs = bh * h_img

            coco_preds.append({
                "image_id": img_id,
                "category_id": cls_id + 1,  # YOLO 0-based → COCO 1-based
                "bbox": [round(x_abs, 2), round(y_abs, 2), round(w_abs, 2), round(h_abs, 2)],
                "score": round(conf, 6),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco_preds))
    logger.info(
        f"  COCO preds: {len(coco_preds)} detections from {n_matched} images "
        f"(skipped {n_skipped} non-GT) → {output_path}"
    )
    return coco_preds, n_matched, n_skipped


# ---------------------------------------------------------------------------
# Phase D: Evaluate with detection_metrics
# ---------------------------------------------------------------------------

def run_evaluation(
    gt_coco_path: Path,
    pred_coco_path: Path,
    model_name: str,
    output_dir: Path,
    eval_iou: float,
    eval_conf: float,
    class_ids: List[int],
) -> Optional[dict]:
    """
    Run detection_metrics pipeline for one model/threshold combination.

    Returns:
        Dict with summary metrics, or None on failure.
    """
    try:
        dm = DetectionMetrics(
            gt_path=str(gt_coco_path),
            eval_config=EvaluateConfig(
                iou_threshold=eval_iou,
                conf_threshold=eval_conf,
                class_ids=class_ids,
            ),
            output_config=OutputConfig(path=output_dir, overwrite=True),
            analysis_config=AnalysisConfig(),
        )
        dm.add_predictions(model_name, str(pred_coco_path))
        results = dm.evaluate()
        dm.analyze()

        # Try visualization, but don't fail if it errors
        try:
            dm.visualize()
        except Exception as e:
            logger.warning(f"  Visualization failed for {model_name}: {e}")

        # Extract summary metrics
        mr = results.get(model_name)
        if mr is None:
            return None

        summary = {"model": model_name}
        if mr.coco_metrics:
            cm = mr.coco_metrics
            summary.update({
                "mAP@50": round(cm.map_50 * 100, 2),
                "mAP@50:95": round(cm.map_50_95 * 100, 2),
                "AR@100": round(cm.ar_100 * 100, 2),
            })
        if mr.detailed_metrics:
            dm_res = mr.detailed_metrics
            # class_metrics is Dict[int, ClassMetrics]
            for cid in class_ids:
                cls_m = dm_res.class_metrics.get(cid)
                if cls_m is None:
                    continue
                summary["AP"] = round(cls_m.ap * 100, 2)
                summary["total_gt"] = cls_m.total_gt
                summary["total_dets"] = cls_m.total_dets
                summary["total_tp"] = cls_m.total_tp
                summary["total_fp"] = cls_m.total_fp
                # Best F1 from the PR curve
                if cls_m.f1:
                    best_idx = int(np.argmax(cls_m.f1))
                    summary["best_F1"] = round(cls_m.f1[best_idx], 4)
                    summary["best_F1_conf"] = round(cls_m.conf[best_idx], 4) if cls_m.conf else None
                break
        return summary
    except Exception as e:
        logger.error(f"  Evaluation failed for {model_name}: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-annotation evaluation pipeline")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to eval_config.yaml",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_base = Path(cfg["pred_base_dir"])
    gt_dataset_path = Path(cfg["gt"]["dataset_path"])
    gt_splits = cfg["gt"]["splits"]
    nms_ious = cfg["nms"]["iou_thresholds"]
    default_conf = cfg.get("default_confidence", 1.0)
    eval_iou = cfg["evaluate"]["iou_threshold"]
    eval_conf = cfg["evaluate"]["conf_threshold"]
    class_ids = cfg["evaluate"]["class_ids"]
    output_base = Path(cfg["output_dir"])

    t_start = time.time()

    # ── Phase C-1: Build combined GT COCO ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1: Building combined GT COCO JSON")
    logger.info("=" * 60)
    gt_coco_path = output_base / "gt" / "combined_all_splits_coco.json"
    if gt_coco_path.exists():
        logger.info(f"GT COCO already exists, loading: {gt_coco_path}")
        gt_coco = json.loads(gt_coco_path.read_text())
        # Rebuild mappings
        stem_to_id: Dict[str, int] = {}
        stem_to_size: Dict[str, Tuple[int, int]] = {}
        for img in gt_coco["images"]:
            stem = Path(img["file_name"]).stem
            stem_to_id[stem] = img["id"]
            stem_to_size[stem] = (img["width"], img["height"])
        logger.info(f"Loaded GT: {len(gt_coco['images'])} images, {len(gt_coco['annotations'])} annotations")
    else:
        gt_coco, stem_to_id, stem_to_size = build_combined_gt_coco(
            gt_dataset_path, gt_splits, gt_coco_path
        )

    # ── Phase A+B: Load preds, apply NMS, save ────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Loading predictions and applying NMS")
    logger.info("=" * 60)

    nms_stats = []  # For summary

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        pred_dir = pred_base / model_cfg["pred_dir"]

        logger.info(f"\n--- Model: {model_name} ---")
        logger.info(f"  Source: {pred_dir}")

        if not pred_dir.exists():
            logger.error(f"  Prediction dir not found: {pred_dir}")
            continue

        # Load raw predictions
        raw_preds = load_yolo_preds(pred_dir, default_conf=default_conf)
        total_raw = sum(arr.shape[0] for arr in raw_preds.values())
        n_files = len(raw_preds)
        logger.info(f"  Loaded: {n_files} files, {total_raw} total detections")

        for iou_thr in nms_ious:
            logger.info(f"  NMS IoU={iou_thr:.1f} ...")
            filtered = apply_nms_to_preds(raw_preds, iou_thr)
            total_after = sum(arr.shape[0] for arr in filtered.values())
            removed = total_raw - total_after
            pct = (removed / total_raw * 100) if total_raw > 0 else 0

            logger.info(
                f"    Before: {total_raw} → After: {total_after} "
                f"(removed {removed}, {pct:.1f}%)"
            )

            # Save NMS'd YOLO txts
            nms_out_dir = output_base / "nms" / model_name / f"iou_{iou_thr}" / "pred_txt"
            save_yolo_preds(filtered, nms_out_dir)
            logger.info(f"    Saved YOLO txts → {nms_out_dir}")

            # Convert to COCO JSON
            coco_json_path = output_base / "predictions_coco" / model_name / f"iou_{iou_thr}.json"
            _, n_matched, n_skipped = preds_to_coco_json(
                filtered, stem_to_id, stem_to_size, coco_json_path
            )

            nms_stats.append({
                "model": model_name,
                "nms_iou": iou_thr,
                "raw_dets": total_raw,
                "nms_dets": total_after,
                "removed": removed,
                "removed_pct": round(pct, 1),
                "matched_images": n_matched,
                "skipped_images": n_skipped,
            })

    # Save NMS stats
    nms_stats_df = pd.DataFrame(nms_stats)
    nms_stats_path = output_base / "nms_stats.csv"
    nms_stats_df.to_csv(nms_stats_path, index=False)
    logger.info(f"\nNMS stats → {nms_stats_path}")
    logger.info(f"\n{nms_stats_df.to_string(index=False)}")

    # ── Phase D: Evaluate with detection_metrics ───────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: Running detection metrics evaluation")
    logger.info("=" * 60)

    eval_summaries = []

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        logger.info(f"\n--- Evaluating: {model_name} ---")

        for iou_thr in nms_ious:
            coco_json_path = output_base / "predictions_coco" / model_name / f"iou_{iou_thr}.json"
            if not coco_json_path.exists():
                logger.warning(f"  Missing pred JSON: {coco_json_path}")
                continue

            eval_label = f"{model_name}_nms{iou_thr}"
            metrics_dir = output_base / "metrics" / model_name / f"nms_iou_{iou_thr}"

            logger.info(f"  NMS IoU={iou_thr:.1f} → {metrics_dir}")
            summary = run_evaluation(
                gt_coco_path=gt_coco_path,
                pred_coco_path=coco_json_path,
                model_name=eval_label,
                output_dir=metrics_dir,
                eval_iou=eval_iou,
                eval_conf=eval_conf,
                class_ids=class_ids,
            )
            if summary:
                summary["nms_iou"] = iou_thr
                summary["model"] = model_name
                eval_summaries.append(summary)
                logger.info(f"    mAP@50={summary.get('mAP@50', 'N/A')} | AP={summary.get('AP', 'N/A')}")

    # Save final summary
    if eval_summaries:
        summary_df = pd.DataFrame(eval_summaries)
        # Reorder columns
        col_order = [
            "model", "nms_iou", "mAP@50", "mAP@50:95", "AP", "AR@100",
            "best_F1", "best_F1_conf", "total_gt", "total_dets", "total_tp", "total_fp",
        ]
        cols = [c for c in col_order if c in summary_df.columns]
        cols += [c for c in summary_df.columns if c not in cols]
        summary_df = summary_df[cols]

        summary_path = output_base / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Final summary → {summary_path}")
        logger.info(f"{'=' * 60}")
        logger.info(f"\n{summary_df.to_string(index=False)}")
    else:
        logger.warning("No evaluation results produced.")

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
