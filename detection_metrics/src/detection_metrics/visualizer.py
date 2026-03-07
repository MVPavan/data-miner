"""Visualization module for detection metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from detection_metrics.evaluator import EvaluationResult
from detection_metrics.logging import logger

# Use non-interactive backend for server environments
matplotlib.use('Agg')


def _resolve_class_ids(
    results: Dict[str, EvaluationResult],
    target_class_ids: Optional[List[int]] = None,
) -> List[int]:
    """Return sorted class IDs — provided list or union across all results."""
    if target_class_ids is not None:
        return target_class_ids
    return sorted(set().union(*[r.class_metrics.keys() for r in results.values()]))


class Visualizer:
    """Handles plotting for evaluation results."""

    def __init__(
        self,
        output_folder: Path,
        labels_map: Optional[Dict[int, str]] = None,
        overwrite: bool = True,
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
        self.prec_conf_pdf_file = self.output_folder / "precision_vs_confidence.pdf"
        self.rec_conf_pdf_file = self.output_folder / "recall_vs_confidence.pdf"
        self.cm_pdf_file = self.output_folder / "confusion_matrices.pdf"
        self.report_pdf_file = self.output_folder / "report.pdf"

    def _get_label(self, cls_id: int) -> str:
        """Get label for class ID."""
        label = self.labels_map.get(cls_id, str(cls_id))
        return (label[:17] + '..') if len(label) > 19 else label

    def _check_overwrite(self, path: Path) -> None:
        if not self.overwrite and path.exists():
            raise FileExistsError(f"File exists: {path}")

    # -- helpers to save a figure to the right destination ---------------

    @staticmethod
    def _save_fig(fig, pdf: Optional[PdfPages]) -> None:
        """Save *fig* to *pdf* (if provided) then close it."""
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

    # -- Individual plot methods ----------------------------------------
    # Each method accepts an optional *pdf* parameter.
    # * When *pdf* is given the figures are appended there and no
    #   standalone file is created.
    # * When *pdf* is ``None`` (default) a standalone file is written.

    def plot_pr_curves(
        self,
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None,
        *,
        pdf: Optional[PdfPages] = None,
    ) -> None:
        """Plot Precision-Recall curves for each class."""
        own_pdf = None
        if pdf is None:
            self._check_overwrite(self.pr_pdf_file)
            own_pdf = PdfPages(self.pr_pdf_file)
            pdf = own_pdf

        try:
            for cls_id in _resolve_class_ids(results, target_class_ids):
                fig, ax = plt.subplots(figsize=(10, 8))
                for pred_name, result in results.items():
                    if cls_id not in result.class_metrics:
                        continue
                    data = result.class_metrics[cls_id]
                    if len(data.recall) == 0:
                        continue
                    ax.plot(data.recall, data.precision,
                            label=f"{pred_name} (AP={data.ap:.3f})")

                label = self._get_label(cls_id)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve: {label} (Class {cls_id})')
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                self._save_fig(fig, pdf)
        finally:
            if own_pdf is not None:
                own_pdf.close()
                logger.info(f"PR curves saved to {self.pr_pdf_file}")

    def plot_f1_curves(
        self,
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None,
        *,
        pdf: Optional[PdfPages] = None,
    ) -> None:
        """Plot F1 vs Confidence curves for each class."""
        own_pdf = None
        if pdf is None:
            self._check_overwrite(self.f1_pdf_file)
            own_pdf = PdfPages(self.f1_pdf_file)
            pdf = own_pdf

        try:
            for cls_id in _resolve_class_ids(results, target_class_ids):
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
                    ax.plot(data.conf, data.f1,
                            label=f"{pred_name} (max={max_f1:.3f} @ {conf_at_max:.3f})")

                label = self._get_label(cls_id)
                ax.set_xlabel('Confidence')
                ax.set_ylabel('F1 Score')
                ax.set_title(f'F1 vs Confidence: {label} (Class {cls_id})')
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                self._save_fig(fig, pdf)
        finally:
            if own_pdf is not None:
                own_pdf.close()
                logger.info(f"F1 curves saved to {self.f1_pdf_file}")

    def plot_precision_curves(
        self,
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None,
        *,
        pdf: Optional[PdfPages] = None,
    ) -> None:
        """Plot Precision vs Confidence curves for each class."""
        own_pdf = None
        if pdf is None:
            self._check_overwrite(self.prec_conf_pdf_file)
            own_pdf = PdfPages(self.prec_conf_pdf_file)
            pdf = own_pdf

        try:
            for cls_id in _resolve_class_ids(results, target_class_ids):
                fig, ax = plt.subplots(figsize=(10, 8))
                for pred_name, result in results.items():
                    if cls_id not in result.class_metrics:
                        continue
                    data = result.class_metrics[cls_id]
                    if len(data.precision) == 0:
                        continue
                    ax.plot(data.conf, data.precision,
                            label=f"{pred_name} (AP={data.ap:.3f})")

                label = self._get_label(cls_id)
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision vs Confidence: {label} (Class {cls_id})')
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                self._save_fig(fig, pdf)
        finally:
            if own_pdf is not None:
                own_pdf.close()
                logger.info(f"Precision curves saved to {self.prec_conf_pdf_file}")

    def plot_recall_curves(
        self,
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None,
        *,
        pdf: Optional[PdfPages] = None,
    ) -> None:
        """Plot Recall vs Confidence curves for each class."""
        own_pdf = None
        if pdf is None:
            self._check_overwrite(self.rec_conf_pdf_file)
            own_pdf = PdfPages(self.rec_conf_pdf_file)
            pdf = own_pdf

        try:
            for cls_id in _resolve_class_ids(results, target_class_ids):
                fig, ax = plt.subplots(figsize=(10, 8))
                for pred_name, result in results.items():
                    if cls_id not in result.class_metrics:
                        continue
                    data = result.class_metrics[cls_id]
                    if len(data.recall) == 0:
                        continue
                    ax.plot(data.conf, data.recall,
                            label=f"{pred_name} (AP={data.ap:.3f})")

                label = self._get_label(cls_id)
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Recall')
                ax.set_title(f'Recall vs Confidence: {label} (Class {cls_id})')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                self._save_fig(fig, pdf)
        finally:
            if own_pdf is not None:
                own_pdf.close()
                logger.info(f"Recall curves saved to {self.rec_conf_pdf_file}")

    def plot_confusion_matrix(
        self,
        results: Dict[str, EvaluationResult],
        cm_conf_choices: Dict[str, List[float]],
        target_class_ids: Optional[List[int]] = None,
        *,
        pdf: Optional[PdfPages] = None,
    ) -> None:
        """Plot confusion matrices at specified confidence thresholds."""
        import seaborn as sns

        own_pdf = None
        if pdf is None:
            self._check_overwrite(self.cm_pdf_file)
            own_pdf = PdfPages(self.cm_pdf_file)
            pdf = own_pdf

        try:
            for pred_name, result in results.items():
                if pred_name not in cm_conf_choices or not cm_conf_choices[pred_name]:
                    continue

                for conf in cm_conf_choices[pred_name]:
                    confusion_data = result.confusion_data

                    class_ids = set()
                    for (gt_cls, pred_cls) in confusion_data.keys():
                        class_ids.add(gt_cls)
                        class_ids.add(pred_cls)

                    if target_class_ids:
                        class_ids = class_ids.intersection(set(target_class_ids))

                    class_ids = sorted(list(class_ids))

                    if not class_ids:
                        logger.warning(
                            f"Skipping confusion matrix for {pred_name} @ conf={conf:.2f}: no classes with data"
                        )
                        continue

                    matrix = np.zeros((len(class_ids), len(class_ids)))
                    for i, gt_cls in enumerate(class_ids):
                        for j, pred_cls in enumerate(class_ids):
                            confs = confusion_data.get((gt_cls, pred_cls), [])
                            matrix[i, j] = sum(1 for c in confs if c >= conf)

                    fig, ax = plt.subplots(figsize=(12, 10))
                    labels = [self._get_label(c) for c in class_ids]

                    sns.heatmap(
                        matrix,
                        annot=True,
                        fmt='.0f',
                        cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax,
                    )

                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Ground Truth')
                    ax.set_title(f'Confusion Matrix: {pred_name} @ conf={conf:.2f}')
                    plt.tight_layout()
                    self._save_fig(fig, pdf)
        finally:
            if own_pdf is not None:
                own_pdf.close()
                logger.info(f"Confusion matrices saved to {self.cm_pdf_file}")

    def plot_metrics_comparison(
        self,
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None,
        *,
        pdf: Optional[PdfPages] = None,
    ) -> None:
        """Plot comparison of AP across models."""
        target_class_ids = _resolve_class_ids(results, target_class_ids)
        model_names = list(results.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(target_class_ids))
        width = 0.8 / max(len(model_names), 1)

        for i, pred_name in enumerate(model_names):
            result = results[pred_name]
            aps = [
                result.class_metrics.get(cid, type('', (), {'ap': 0})()).ap
                for cid in target_class_ids
            ]
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

        if pdf is not None:
            self._save_fig(fig, pdf)
        else:
            comparison_file = self.output_folder / "ap_comparison.png"
            fig.savefig(comparison_file, dpi=150)
            plt.close(fig)
            logger.info(f"AP comparison saved to {comparison_file}")

    # -- Consolidated report -------------------------------------------

    def generate_report(
        self,
        results: Dict[str, EvaluationResult],
        target_class_ids: Optional[List[int]] = None,
        cm_conf_choices: Optional[Dict[str, List[float]]] = None,
    ) -> Path:
        """
        Generate a single consolidated PDF containing all plots.

        Args:
            results:          Dict mapping model name to EvaluationResult.
            target_class_ids: Optional list of class IDs to plot.
            cm_conf_choices:  Confidence thresholds per model for confusion matrices.

        Returns:
            Path to the generated report PDF.
        """
        self._check_overwrite(self.report_pdf_file)

        with PdfPages(self.report_pdf_file) as pdf:
            self.plot_pr_curves(results, target_class_ids, pdf=pdf)
            self.plot_f1_curves(results, target_class_ids, pdf=pdf)
            self.plot_precision_curves(results, target_class_ids, pdf=pdf)
            self.plot_recall_curves(results, target_class_ids, pdf=pdf)
            if cm_conf_choices:
                self.plot_confusion_matrix(results, cm_conf_choices, target_class_ids, pdf=pdf)
            self.plot_metrics_comparison(results, target_class_ids, pdf=pdf)

        logger.info(f"Consolidated report saved to {self.report_pdf_file}")
        return self.report_pdf_file
