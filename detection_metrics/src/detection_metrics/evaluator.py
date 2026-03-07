"""
Core evaluation logic with two-stage evaluation:
1. PyCocoEvaluator - Official COCO metrics (mAP50, mAP50-95)
2. DetailedEvaluator - Custom detailed analysis (PR curves, F1, confusion matrix)
"""

import io
import contextlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detection_metrics.logging import logger, create_progress


# Constants
FP_PRECISION_EPSILON = 1e-6


# --- Pydantic Models for Results ---

class ClassMetrics(BaseModel):
    """Per-class evaluation metrics."""
    class_id: int
    ap: float
    precision: List[float] = Field(default_factory=list)
    recall: List[float] = Field(default_factory=list)
    f1: List[float] = Field(default_factory=list)
    conf: List[float] = Field(default_factory=list)
    tp: List[float] = Field(default_factory=list)
    fp: List[float] = Field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_gt: int = 0
    total_dets: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validator(self):
        if self.total_dets > 0 and self.total_tp + self.total_fp != self.total_dets:
            raise ValueError("Total TP + FP != Total Dets")
        return self


class EvaluationResult(BaseModel):
    """Aggregated evaluation results."""
    class_metrics: Dict[int, ClassMetrics]
    confusion_data: Dict[Tuple[int, int], List[float]] = Field(default_factory=dict)
    total_gt_counts: Dict[int, int] = Field(default_factory=dict)
    conf_threshold: float = 0.001

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PyCocoResult(BaseModel):
    """Results from PyCocoTools evaluation."""
    map_50: float
    map_50_95: float
    map_small: float
    map_medium: float
    map_large: float
    ar_1: float
    ar_10: float
    ar_100: float
    per_class_ap: Dict[int, float] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- Stage 1: PyCocoTools Evaluator ---

class PyCocoEvaluator:
    """
    Stage 1: Standard COCO evaluation using pycocotools.
    
    Provides official mAP50, mAP50-95 metrics.
    """
    
    def __init__(self, gt_path: Path, pred_path: Path, cat_ids: Optional[List[int]] = None):
        """
        Initialize PyCocoEvaluator.
        
        Args:
            gt_path: Path to ground truth annotations (COCO format).
            pred_path: Path to predictions (COCO format).
            cat_ids: Optional list of category IDs to evaluate.
        """
        self.gt_path = Path(gt_path)
        self.pred_path = Path(pred_path)
        self.cat_ids = cat_ids
        
        # Suppress pycocotools output during loading
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.coco_gt = COCO(str(self.gt_path))
        
    def evaluate(self) -> PyCocoResult:
        """
        Run COCO evaluation.
        
        Returns:
            PyCocoResult with mAP metrics.
        """
        logger.info("Running PyCocoTools evaluation...")
        
        # Load predictions
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            coco_dt = self.coco_gt.loadRes(str(self.pred_path))
        
        # Create evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        
        if self.cat_ids:
            coco_eval.params.catIds = self.cat_ids
            logger.info(f"Evaluating {len(self.cat_ids)} categories")
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Capture summary output
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            coco_eval.summarize()
            summary = buf.getvalue()
        
        logger.info(f"PyCocoTools Summary:\n{summary}")
        
        # Extract stats
        stats = coco_eval.stats
        
        # Per-class AP at IoU=0.5
        per_class_ap = {}
        if self.cat_ids:
            for cat_id in self.cat_ids:
                coco_eval_cls = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
                coco_eval_cls.params.catIds = [cat_id]
                coco_eval_cls.evaluate()
                coco_eval_cls.accumulate()
                per_class_ap[cat_id] = float(coco_eval_cls.stats[1])  # AP@0.5
        
        return PyCocoResult(
            map_50_95=float(stats[0]),
            map_50=float(stats[1]),
            map_small=float(stats[3]),
            map_medium=float(stats[4]),
            map_large=float(stats[5]),
            ar_1=float(stats[6]),
            ar_10=float(stats[7]),
            ar_100=float(stats[8]),
            per_class_ap=per_class_ap,
        )


# --- Stage 2: Detailed Evaluator ---

class DetailedEvaluator:
    """
    Stage 2: Custom detailed evaluation.
    
    Calculates AP, mAP, F1, Confusion Matrix at all confidence levels.
    Preserved from original detection_metrics.py logic.
    """
    
    def __init__(
        self, 
        ground_truths: List[List[float]], 
        predictions: List[List[float]],
        iou_threshold: float = 0.5, 
        conf_threshold: float = 0.001,
        eval_classids: Optional[List[int]] = None
    ):
        """
        Initialize DetailedEvaluator.
        
        Args:
            ground_truths: List of [img_id, class_id, conf, x1, y1, x2, y2]
            predictions: List of [img_id, class_id, conf, x1, y1, x2, y2]
            iou_threshold: IoU threshold for matching.
            conf_threshold: Confidence threshold for filtering predictions.
            eval_classids: Optional list of class IDs to evaluate.
        """
        self.ground_truths = np.array(ground_truths) if ground_truths else np.empty((0, 7))
        self.predictions = np.array(predictions) if predictions else np.empty((0, 7))
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.eval_classids = set(eval_classids) if eval_classids else None
        
        # Determine classes present
        gt_classes = set(self.ground_truths[:, 1].astype(int)) if len(self.ground_truths) > 0 else set()
        pred_classes = set(self.predictions[:, 1].astype(int)) if len(self.predictions) > 0 else set()
        self.all_classes = sorted(list(gt_classes.union(pred_classes)))

    def _compute_iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Vectorized IoU calculation."""
        # boxes: [x1, y1, x2, y2]
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)))

        # Expand dims: (N, 1, 4) vs (1, M, 4)
        a = boxes_a[:, np.newaxis, :]
        b = boxes_b[np.newaxis, :, :]

        inter_x1 = np.maximum(a[..., 0], b[..., 0])
        inter_y1 = np.maximum(a[..., 1], b[..., 1])
        inter_x2 = np.minimum(a[..., 2], b[..., 2])
        inter_y2 = np.minimum(a[..., 3], b[..., 3])

        w = np.maximum(0, inter_x2 - inter_x1)
        h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = w * h

        area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
        area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        union_area = area_a + area_b - inter_area

        return np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area != 0)

    def run(self) -> EvaluationResult:
        """Runs the evaluation pipeline."""
        
        # 0. Filter Data
        gt_all = self.ground_truths
        pred_all = self.predictions

        if self.eval_classids:
            if len(gt_all) > 0:
                gt_all = gt_all[np.isin(gt_all[:, 1], list(self.eval_classids))]
            if len(pred_all) > 0:
                pred_all = pred_all[np.isin(pred_all[:, 1], list(self.eval_classids))]
            target_classes = sorted(list(self.eval_classids))
        else:
            target_classes = self.all_classes
        
        if len(pred_all) > 0:
            pred_all = pred_all[pred_all[:, 2] >= self.conf_threshold]

        # 1. Pre-process GT for fast access by Image ID
        gt_by_img = defaultdict(list)
        for row in gt_all:
            gt_by_img[row[0]].append(row)
        for k in gt_by_img:
            gt_by_img[k] = np.array(gt_by_img[k])

        class_metrics_map = {}
        confusion_data = defaultdict(list)
        total_gt_counts = defaultdict(int)

        logger.info(f"Evaluating {len(target_classes)} classes...")
        
        with create_progress() as progress:
            main_task = progress.add_task("Detailed Evaluation", total=len(target_classes))
            
            for cls_id in target_classes:
                gt_cls = gt_all[gt_all[:, 1] == cls_id] if len(gt_all) > 0 else np.empty((0, 7))
                pred_cls = pred_all[pred_all[:, 1] == cls_id] if len(pred_all) > 0 else np.empty((0, 7))
                
                total_gt_counts[int(cls_id)] = len(gt_cls)

                if len(pred_cls) > 0:
                    pred_cls = pred_cls[np.argsort(-pred_cls[:, 2])]

                tp = np.zeros(len(pred_cls))
                fp = np.zeros(len(pred_cls))

                def check_confusion(img_id, pred_box):
                    pred_conf = pred_box[2]
                    img_gts = gt_by_img.get(img_id, np.empty((0, 7)))
                    if len(img_gts) == 0:
                        confusion_data[(-1, int(cls_id))].append(pred_conf)
                        return

                    iou_scores = self._compute_iou_matrix(pred_box[np.newaxis, 3:], img_gts[:, 3:])
                    max_iou = np.max(iou_scores)
                    if max_iou >= self.iou_threshold - FP_PRECISION_EPSILON:
                        idx = np.argmax(iou_scores)
                        matched_gt_cls = int(img_gts[idx, 1])
                        confusion_data[(matched_gt_cls, int(cls_id))].append(pred_conf)
                    else:
                        confusion_data[(-1, int(cls_id))].append(pred_conf)

                # Get unique images for this class
                if len(gt_cls) > 0 and len(pred_cls) > 0:
                    unique_imgs = np.unique(np.concatenate((gt_cls[:, 0], pred_cls[:, 0])))
                elif len(gt_cls) > 0:
                    unique_imgs = np.unique(gt_cls[:, 0])
                elif len(pred_cls) > 0:
                    unique_imgs = np.unique(pred_cls[:, 0])
                else:
                    unique_imgs = np.array([])
                
                for img_id in unique_imgs:
                    gts_img = gt_cls[gt_cls[:, 0] == img_id]
                    pred_idxs = np.where(pred_cls[:, 0] == img_id)[0]
                    preds_img = pred_cls[pred_idxs]

                    if len(preds_img) == 0:
                        continue

                    if len(gts_img) == 0:
                        fp[pred_idxs] = 1
                        for p in preds_img:
                            check_confusion(img_id, p)
                        continue

                    iou_mat = self._compute_iou_matrix(preds_img[:, 3:], gts_img[:, 3:])
                    gt_matched = np.zeros(len(gts_img), dtype=bool)

                    for i, pred_row_idx in enumerate(pred_idxs):
                        iou_row = iou_mat[i]
                        max_idx = np.argmax(iou_row) if len(iou_row) > 0 else -1
                        max_iou = iou_row[max_idx] if max_idx != -1 else 0

                        if max_idx != -1 and max_iou >= (self.iou_threshold - FP_PRECISION_EPSILON) and not gt_matched[max_idx]:
                            tp[pred_row_idx] = 1
                            gt_matched[max_idx] = True
                            confusion_data[(int(cls_id), int(cls_id))].append(preds_img[i, 2])
                        else:
                            fp[pred_row_idx] = 1
                            check_confusion(img_id, preds_img[i])

                # Calculate Metrics
                if len(tp) == 0:
                    class_metrics_map[cls_id] = ClassMetrics(
                        class_id=cls_id,
                        ap=0.0,
                        total_gt=len(gt_cls)
                    )
                else:
                    acc_tp = np.cumsum(tp)
                    acc_fp = np.cumsum(fp)
                    eps = FP_PRECISION_EPSILON
                    prec = acc_tp / (acc_tp + acc_fp + eps)
                    rec = acc_tp / (len(gt_cls) + eps)
                    f1 = 2 * (prec * rec) / (prec + rec + eps)

                    # AP (All points interpolation)
                    m_rec = np.concatenate(([0.0], rec, [1.0]))
                    m_pre = np.concatenate(([0.0], prec, [0.0]))
                    for i in range(len(m_pre) - 1, 0, -1):
                        m_pre[i - 1] = np.maximum(m_pre[i - 1], m_pre[i])
                    i_indices = np.where(m_rec[1:] != m_rec[:-1])[0]
                    ap = np.sum((m_rec[i_indices + 1] - m_rec[i_indices]) * m_pre[i_indices + 1])

                    class_metrics_map[cls_id] = ClassMetrics(
                        class_id=cls_id,
                        ap=float(ap),
                        precision=prec.tolist(),
                        recall=rec.tolist(),
                        f1=f1.tolist(),
                        conf=pred_cls[:, 2].tolist(),
                        tp=tp.tolist(),
                        fp=fp.tolist(),
                        total_tp=int(np.sum(tp)),
                        total_fp=int(np.sum(fp)),
                        total_gt=len(gt_cls),
                        total_dets=len(pred_cls)
                    )
                
                progress.update(main_task, advance=1)

        return EvaluationResult(
            class_metrics=class_metrics_map,
            confusion_data=dict(confusion_data),
            total_gt_counts=dict(total_gt_counts),
            conf_threshold=self.conf_threshold,
        )
