"""Visualization module for detection metrics."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from detection_metrics.evaluator import EvaluationResult
from detection_metrics.logging import logger

# Use non-interactive backend for server environments
matplotlib.use('Agg')


class Visualizer:
    """Handles plotting for evaluation results."""
    
    def __init__(
        self, 
        output_folder: Path, 
        labels_map: Optional[Dict[int, str]] = None,
        overwrite: bool = True
    ):
        """
        Initialize Visualizer.
        
        Args:
            output_folder: Directory to save plots.
            labels_map: Optional dict mapping class IDs to names.
            overwrite: Whether to overwrite existing files.
        """
        self.labels_map = labels_map or {}
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        
        self.pr_pdf_file = self.output_folder / "pr_curves.pdf"
        self.f1_pdf_file = self.output_folder / "f1_curves.pdf"
        self.cm_pdf_file = self.output_folder / "confusion_matrices.pdf"

    def _get_label(self, cls_id: int) -> str:
        """Get label for class ID."""
        label = self.labels_map.get(cls_id, str(cls_id))
        return (label[:17] + '..') if len(label) > 19 else label

    def plot_pr_curves(
        self, 
        results: Dict[str, EvaluationResult], 
        target_class_ids: Optional[List[int]] = None
    ):
        """
        Plot Precision-Recall curves for each class.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            target_class_ids: Optional list of class IDs to plot.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        if not self.overwrite and self.pr_pdf_file.exists():
            raise FileExistsError(f"File exists: {self.pr_pdf_file}")
        
        # Get all class IDs if not specified
        if target_class_ids is None:
            target_class_ids = sorted(set().union(*[r.class_metrics.keys() for r in results.values()]))
        
        with PdfPages(self.pr_pdf_file) as pdf:
            for cls_id in target_class_ids:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for pred_name, result in results.items():
                    if cls_id not in result.class_metrics:
                        continue
                    
                    data = result.class_metrics[cls_id]
                    if len(data.recall) == 0:
                        continue
                    
                    ax.plot(
                        data.recall, 
                        data.precision, 
                        label=f"{pred_name} (AP={data.ap:.3f})"
                    )
                
                label = self._get_label(cls_id)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve: {label} (Class {cls_id})')
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                
                pdf.savefig(fig)
                plt.close(fig)
        
        logger.info(f"PR curves saved to {self.pr_pdf_file}")

    def plot_f1_curves(
        self, 
        results: Dict[str, EvaluationResult], 
        target_class_ids: Optional[List[int]] = None
    ):
        """
        Plot F1 vs Confidence curves for each class.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            target_class_ids: Optional list of class IDs to plot.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        if not self.overwrite and self.f1_pdf_file.exists():
            raise FileExistsError(f"File exists: {self.f1_pdf_file}")
        
        if target_class_ids is None:
            target_class_ids = sorted(set().union(*[r.class_metrics.keys() for r in results.values()]))
        
        with PdfPages(self.f1_pdf_file) as pdf:
            for cls_id in target_class_ids:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for pred_name, result in results.items():
                    if cls_id not in result.class_metrics:
                        continue
                    
                    data = result.class_metrics[cls_id]
                    if len(data.f1) == 0:
                        continue
                    
                    max_f1 = np.max(data.f1)
                    max_idx = np.argmax(data.f1)
                    conf_at_max = data.conf[max_idx]
                    
                    ax.plot(
                        data.conf, 
                        data.f1, 
                        label=f"{pred_name} (max={max_f1:.3f} @ {conf_at_max:.3f})"
                    )
                
                label = self._get_label(cls_id)
                ax.set_xlabel('Confidence')
                ax.set_ylabel('F1 Score')
                ax.set_title(f'F1 vs Confidence: {label} (Class {cls_id})')
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                
                pdf.savefig(fig)
                plt.close(fig)
        
        logger.info(f"F1 curves saved to {self.f1_pdf_file}")

    def plot_confusion_matrix(
        self, 
        results: Dict[str, EvaluationResult], 
        cm_conf_choices: Dict[str, List[float]],
        target_class_ids: Optional[List[int]] = None
    ):
        """
        Plot confusion matrices at specified confidence thresholds.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            cm_conf_choices: Dict mapping model name to list of conf thresholds.
            target_class_ids: Optional list of class IDs to include.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns
        
        if not self.overwrite and self.cm_pdf_file.exists():
            raise FileExistsError(f"File exists: {self.cm_pdf_file}")
        
        with PdfPages(self.cm_pdf_file) as pdf:
            for pred_name, result in results.items():
                if pred_name not in cm_conf_choices or not cm_conf_choices[pred_name]:
                    continue
                
                for conf in cm_conf_choices[pred_name]:
                    confusion_data = result.confusion_data
                    
                    # Get class IDs
                    class_ids = set()
                    for (gt_cls, pred_cls) in confusion_data.keys():
                        class_ids.add(gt_cls)
                        class_ids.add(pred_cls)
                    
                    if target_class_ids:
                        class_ids = class_ids.intersection(set(target_class_ids))
                    
                    class_ids = sorted(list(class_ids))
                    
                    # Build matrix
                    matrix = np.zeros((len(class_ids), len(class_ids)))
                    for i, gt_cls in enumerate(class_ids):
                        for j, pred_cls in enumerate(class_ids):
                            confs = confusion_data.get((gt_cls, pred_cls), [])
                            matrix[i, j] = sum(1 for c in confs if c >= conf)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    labels = [self._get_label(c) for c in class_ids]
                    
                    sns.heatmap(
                        matrix, 
                        annot=True, 
                        fmt='.0f', 
                        cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax
                    )
                    
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Ground Truth')
                    ax.set_title(f'Confusion Matrix: {pred_name} @ conf={conf:.2f}')
                    plt.tight_layout()
                    
                    pdf.savefig(fig)
                    plt.close(fig)
        
        logger.info(f"Confusion matrices saved to {self.cm_pdf_file}")

    def plot_metrics_comparison(
        self, 
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None
    ):
        """
        Plot comparison of mAP across models.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            target_class_ids: Optional list of class IDs to include.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = list(results.keys())
        
        if target_class_ids is None:
            target_class_ids = sorted(set().union(*[r.class_metrics.keys() for r in results.values()]))
        
        x = np.arange(len(target_class_ids))
        width = 0.8 / len(model_names)
        
        for i, pred_name in enumerate(model_names):
            result = results[pred_name]
            aps = [result.class_metrics.get(cid, type('', (), {'ap': 0})()).ap for cid in target_class_ids]
            ax.bar(x + i * width, aps, width, label=pred_name)
        
        labels = [self._get_label(c) for c in target_class_ids]
        ax.set_xlabel('Class')
        ax.set_ylabel('AP')
        ax.set_title('Per-Class AP Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        comparison_file = self.output_folder / "ap_comparison.png"
        fig.savefig(comparison_file, dpi=150)
        plt.close(fig)
        
        logger.info(f"AP comparison saved to {comparison_file}")
