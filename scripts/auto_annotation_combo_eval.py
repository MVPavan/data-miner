"""
Model combo merge evaluation pipeline.

Takes individually NMS'd + confidence-filtered predictions from multiple
auto-annotation models, merges them in all 2/3/4-model combinations,
applies cross-model NMS, and evaluates against GT.

Usage:
    python helpers/auto_annotation_combo_eval.py --config eval_results/auto_annotation/combo_config.yaml
"""

import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

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
# NMS (pure NumPy, single-class) — same as auto_annotation_eval.py
# ---------------------------------------------------------------------------

def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
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
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


# ---------------------------------------------------------------------------
# Load & filter NMS'd predictions
# ---------------------------------------------------------------------------

def load_and_filter(
    nms_dir: Path, model_name: str, nms_iou: float, conf_threshold: float
) -> Dict[str, np.ndarray]:
    """
    Load already-NMS'd YOLO predictions and filter by confidence threshold.
    Returns {stem: np.ndarray (N, 5)} where cols = [class_id, cx, cy, w, h].
    Confidence is dropped after filtering — all survivors treated equally.
    """
    pred_dir = nms_dir / model_name / f"iou_{nms_iou}" / "pred_txt"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")

    preds: Dict[str, np.ndarray] = {}
    total_before = 0
    total_after = 0

    for txt in sorted(pred_dir.glob("*.txt")):
        stem = txt.stem
        text = txt.read_text().strip()
        if not text:
            preds[stem] = np.empty((0, 5), dtype=float)
            continue

        rows = []
        for line in text.split("\n"):
            vals = line.split()
            cls_id = int(vals[0])
            cx, cy, w, h = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
            conf = float(vals[5]) if len(vals) > 5 else 1.0
            total_before += 1
            if conf >= conf_threshold:
                rows.append([cls_id, cx, cy, w, h])
                total_after += 1

        if rows:
            preds[stem] = np.array(rows, dtype=float)
        else:
            preds[stem] = np.empty((0, 5), dtype=float)

    logger.info(
        f"  {model_name}: loaded {len(preds)} files, "
        f"{total_before} dets → {total_after} after conf≥{conf_threshold:.3f} "
        f"(removed {total_before - total_after})"
    )
    return preds


# ---------------------------------------------------------------------------
# Merge predictions from multiple models
# ---------------------------------------------------------------------------

def merge_preds(
    model_preds: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Merge predictions from multiple models by image stem.
    All models' predictions for the same image are concatenated.
    Returns {stem: np.ndarray (N, 5)} where cols = [class_id, cx, cy, w, h].
    """
    all_stems = set()
    for preds in model_preds.values():
        all_stems.update(preds.keys())

    merged: Dict[str, np.ndarray] = {}
    for stem in sorted(all_stems):
        arrays = []
        for preds in model_preds.values():
            if stem in preds and preds[stem].shape[0] > 0:
                arrays.append(preds[stem])
        if arrays:
            merged[stem] = np.concatenate(arrays, axis=0)
        else:
            merged[stem] = np.empty((0, 5), dtype=float)

    return merged


# ---------------------------------------------------------------------------
# Apply NMS on merged predictions (no real confidence — use uniform scores)
# ---------------------------------------------------------------------------

def apply_merge_nms(
    preds: Dict[str, np.ndarray], iou_thr: float
) -> Dict[str, np.ndarray]:
    """Apply NMS on merged predictions. All detections have equal score."""
    filtered: Dict[str, np.ndarray] = {}
    for stem, arr in preds.items():
        if arr.shape[0] == 0:
            filtered[stem] = arr
            continue
        boxes_cxcywh = arr[:, 1:5]  # cx, cy, w, h
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
        # Uniform scores — NMS will keep whichever comes first (arbitrary but consistent)
        scores = np.ones(len(arr), dtype=float)
        keep = nms_numpy(boxes_xyxy, scores, iou_thr)
        filtered[stem] = arr[keep]
    return filtered


# ---------------------------------------------------------------------------
# Convert to COCO JSON for evaluation
# ---------------------------------------------------------------------------

def preds_to_coco_json(
    preds: Dict[str, np.ndarray],
    stem_to_id: Dict[str, int],
    stem_to_size: Dict[str, Tuple[int, int]],
    output_path: Path,
) -> int:
    """Convert merged YOLO predictions → COCO prediction JSON. Returns det count."""
    coco_preds = []
    for stem, arr in preds.items():
        if stem not in stem_to_id:
            continue
        img_id = stem_to_id[stem]
        w_img, h_img = stem_to_size[stem]

        for row in arr:
            cls_id = int(row[0])
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            x_abs = (cx - bw / 2) * w_img
            y_abs = (cy - bh / 2) * h_img
            w_abs = bw * w_img
            h_abs = bh * h_img
            coco_preds.append({
                "image_id": img_id,
                "category_id": cls_id + 1,
                "bbox": [round(x_abs, 2), round(y_abs, 2), round(w_abs, 2), round(h_abs, 2)],
                "score": 1.0,  # no confidence differentiation after merge
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco_preds))
    return len(coco_preds)


# ---------------------------------------------------------------------------
# Run detection_metrics evaluation
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
        try:
            dm.visualize()
        except Exception as e:
            logger.warning(f"  Visualization failed: {e}")

        mr = results.get(model_name)
        if mr is None:
            return None

        summary = {"combo": model_name}
        if mr.coco_metrics:
            cm = mr.coco_metrics
            summary.update({
                "mAP@50": round(cm.map_50 * 100, 2),
                "mAP@50:95": round(cm.map_50_95 * 100, 2),
                "AR@100": round(cm.ar_100 * 100, 2),
            })
        if mr.detailed_metrics:
            dm_res = mr.detailed_metrics
            for cid in class_ids:
                cls_m = dm_res.class_metrics.get(cid)
                if cls_m is None:
                    continue
                summary["AP"] = round(cls_m.ap * 100, 2)
                summary["total_gt"] = cls_m.total_gt
                summary["total_dets"] = cls_m.total_dets
                summary["total_tp"] = cls_m.total_tp
                summary["total_fp"] = cls_m.total_fp
                summary["total_fn"] = cls_m.total_gt - cls_m.total_tp
                summary["precision"] = round(cls_m.total_tp / max(cls_m.total_dets, 1), 4)
                summary["recall"] = round(cls_m.total_tp / max(cls_m.total_gt, 1), 4)
                p, r = summary["precision"], summary["recall"]
                summary["F1"] = round(2 * p * r / max(p + r, 1e-9), 4)
                # Best F1 from curve
                if cls_m.f1:
                    best_idx = int(np.argmax(cls_m.f1))
                    summary["best_F1_curve"] = round(cls_m.f1[best_idx], 4)
                break
        return summary
    except Exception as e:
        logger.error(f"  Evaluation failed for {model_name}: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combo merge evaluation pipeline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    nms_dir = Path(cfg["source_nms_dir"])
    gt_coco_path = Path(cfg["gt_coco_path"])
    merge_nms_ious = cfg["merge_nms_iou_thresholds"]
    eval_iou = cfg["evaluate"]["iou_threshold"]
    eval_conf = cfg["evaluate"]["conf_threshold"]
    class_ids = cfg["evaluate"]["class_ids"]
    output_base = Path(cfg["output_dir"])

    t_start = time.time()

    # ── Load GT metadata for COCO conversion ──────────────────────────
    logger.info("Loading GT metadata...")
    gt_coco = json.loads(gt_coco_path.read_text())
    stem_to_id: Dict[str, int] = {}
    stem_to_size: Dict[str, Tuple[int, int]] = {}
    for img in gt_coco["images"]:
        stem = Path(img["file_name"]).stem
        stem_to_id[stem] = img["id"]
        stem_to_size[stem] = (img["width"], img["height"])
    total_gt_ann = len(gt_coco["annotations"])
    logger.info(f"GT: {len(stem_to_id)} images, {total_gt_ann} annotations")

    # ── Phase 1: Load & filter each model's predictions ───────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1: Loading & filtering individual model predictions")
    logger.info("=" * 60)

    model_names = list(cfg["models"].keys())
    model_filtered: Dict[str, Dict[str, np.ndarray]] = {}

    for mname in model_names:
        mcfg = cfg["models"][mname]
        nms_iou = mcfg["nms_iou"]
        conf_thr = mcfg["conf_threshold"]
        model_filtered[mname] = load_and_filter(nms_dir, mname, nms_iou, conf_thr)

    # Log per-model stats
    logger.info("")
    for mname, preds in model_filtered.items():
        n_files = len(preds)
        n_dets = sum(arr.shape[0] for arr in preds.values())
        n_with_dets = sum(1 for arr in preds.values() if arr.shape[0] > 0)
        logger.info(f"  {mname}: {n_files} files, {n_with_dets} with detections, {n_dets} total dets")

    # ── Phase 2: Generate all model combinations ──────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Generating model combinations")
    logger.info("=" * 60)

    all_combos = []
    # Singles (for baseline comparison)
    for mname in model_names:
        all_combos.append((mname,))
    # 2-model combos
    for combo in combinations(model_names, 2):
        all_combos.append(combo)
    # 3-model combos
    for combo in combinations(model_names, 3):
        all_combos.append(combo)
    # 4-model combo
    if len(model_names) >= 4:
        all_combos.append(tuple(model_names))

    logger.info(f"Total combinations: {len(all_combos)}")
    for combo in all_combos:
        logger.info(f"  {' + '.join(combo)}")

    # ── Phase 3: Merge, NMS, convert, evaluate ────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: Merge → NMS → Evaluate")
    logger.info("=" * 60)

    all_summaries = []

    for combo in all_combos:
        combo_label = "+".join(combo)
        combo_preds = {m: model_filtered[m] for m in combo}

        # Merge
        merged = merge_preds(combo_preds)
        n_merged = sum(arr.shape[0] for arr in merged.values())

        for merge_nms_iou in merge_nms_ious:
            run_label = f"{combo_label}_nms{merge_nms_iou}"
            logger.info(f"\n--- {run_label} ---")

            # Apply cross-model NMS
            nms_result = apply_merge_nms(merged, merge_nms_iou)
            n_after_nms = sum(arr.shape[0] for arr in nms_result.values())
            n_removed = n_merged - n_after_nms
            pct = (n_removed / n_merged * 100) if n_merged > 0 else 0

            logger.info(
                f"  Merged: {n_merged} → NMS(IoU={merge_nms_iou}): {n_after_nms} "
                f"(removed {n_removed}, {pct:.1f}%)"
            )

            # Save NMS'd YOLO txts
            nms_out_dir = output_base / "nms" / combo_label / f"merge_nms_{merge_nms_iou}" / "pred_txt"
            nms_out_dir.mkdir(parents=True, exist_ok=True)
            for stem, arr in nms_result.items():
                lines = []
                for row in arr:
                    cls_id = int(row[0])
                    cx, cy, w, h = row[1], row[2], row[3], row[4]
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                (nms_out_dir / f"{stem}.txt").write_text("\n".join(lines))

            # Convert to COCO JSON
            coco_path = output_base / "predictions_coco" / f"{combo_label}_nms{merge_nms_iou}.json"
            n_coco = preds_to_coco_json(nms_result, stem_to_id, stem_to_size, coco_path)
            logger.info(f"  COCO preds: {n_coco} → {coco_path}")

            # Evaluate
            metrics_dir = output_base / "metrics" / combo_label / f"merge_nms_{merge_nms_iou}"
            summary = run_evaluation(
                gt_coco_path, coco_path, run_label, metrics_dir,
                eval_iou, eval_conf, class_ids,
            )

            if summary:
                summary["models"] = combo_label
                summary["n_models"] = len(combo)
                summary["merge_nms_iou"] = merge_nms_iou
                summary["merged_dets"] = n_merged
                summary["after_nms_dets"] = n_after_nms
                all_summaries.append(summary)
                logger.info(
                    f"  mAP@50={summary.get('mAP@50','N/A')} | "
                    f"prec={summary.get('precision','N/A')} | "
                    f"recall={summary.get('recall','N/A')} | "
                    f"F1={summary.get('F1','N/A')}"
                )

    # ── Phase 4: Summary ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 4: Summary")
    logger.info("=" * 60)

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        col_order = [
            "models", "n_models", "merge_nms_iou",
            "mAP@50", "mAP@50:95", "AP", "AR@100",
            "precision", "recall", "F1", "best_F1_curve",
            "total_gt", "total_dets", "total_tp", "total_fp", "total_fn",
            "merged_dets", "after_nms_dets",
        ]
        cols = [c for c in col_order if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

        summary_path = output_base / "combo_summary.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"\nCombo summary → {summary_path}")

        # Print a nice sorted view
        display_cols = [
            "models", "n_models", "merge_nms_iou",
            "precision", "recall", "F1",
            "total_tp", "total_fp", "total_fn",
        ]
        dc = [c for c in display_cols if c in df.columns]
        df_sorted = df[dc].sort_values("F1", ascending=False)
        logger.info(f"\n{df_sorted.to_string(index=False)}")

        # Best combo summary
        logger.info("\n--- Best combos by F1 per merge NMS IoU ---")
        for nms_iou in merge_nms_ious:
            subset = df[df["merge_nms_iou"] == nms_iou]
            if subset.empty:
                continue
            best = subset.loc[subset["F1"].idxmax()]
            logger.info(
                f"  NMS {nms_iou}: {best['models']} → "
                f"prec={best['precision']:.3f} recall={best['recall']:.3f} "
                f"F1={best['F1']:.4f} TP={int(best['total_tp'])} FP={int(best['total_fp'])} FN={int(best['total_fn'])}"
            )

        # Best combo with precision >= 0.6
        logger.info("\n--- Best combos with precision ≥ 0.60 ---")
        for nms_iou in merge_nms_ious:
            subset = df[(df["merge_nms_iou"] == nms_iou) & (df["precision"] >= 0.6)]
            if subset.empty:
                logger.info(f"  NMS {nms_iou}: none achieve precision ≥ 0.60")
                continue
            best = subset.loc[subset["recall"].idxmax()]
            logger.info(
                f"  NMS {nms_iou}: {best['models']} → "
                f"prec={best['precision']:.3f} recall={best['recall']:.3f} "
                f"F1={best['F1']:.4f} TP={int(best['total_tp'])} FP={int(best['total_fp'])} FN={int(best['total_fn'])}"
            )
    else:
        logger.warning("No evaluation results produced.")

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
