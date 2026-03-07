"""
High-level orchestrator for the detection_metrics evaluation pipeline.

Provides a single entry point that:
  1. Converts any input format -> COCO JSON
  2. Runs PyCocoEvaluator for standard mAP metrics
  3. Runs DetailedEvaluator for PR curves, F1, confusion matrix
  4. Generates CSV reports (cached)
  5. Generates PDF/PNG visualisations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from detection_metrics.cache import CacheManager
from detection_metrics.configs.config import (
    AnalysisConfig,
    EvaluateConfig,
    OutputConfig,
)
from detection_metrics.converter import coco_preds_to_internal
from detection_metrics.data_loader import DataLoader
from detection_metrics.evaluator import (
    DetailedEvaluator,
    EvaluationResult,
    PyCocoEvaluator,
    PyCocoResult,
)
from detection_metrics.logging import logger
from detection_metrics.report import ReportGenerator
from detection_metrics.visualizer import Visualizer


class MetricsResult(BaseModel):
    """Combined result from both evaluation stages for a single model."""

    model_name: str
    coco_metrics: Optional[PyCocoResult] = None
    detailed_metrics: Optional[EvaluationResult] = None


class DetectionMetrics:
    """
    Orchestrator: evaluate one or many models against a ground truth dataset.

    Example usage::

        from detection_metrics import DetectionMetrics
        from detection_metrics.configs.config import EvaluateConfig, OutputConfig

        dm = DetectionMetrics(
            gt_path="/data/datasets/doors/test",
            eval_config=EvaluateConfig(iou_threshold=0.5),
            output_config=OutputConfig(path="./eval_output"),
        )

        dm.add_predictions("RF-DETR-S", "/path/to/preds_rfdetr_s.json")
        dm.add_predictions("YOLOX-M",   "/path/to/preds_yolox_m.json")

        dm.evaluate()
        dm.analyze()
        dm.visualize()
    """

    def __init__(
        self,
        gt_path: str | Path,
        eval_config: EvaluateConfig | None = None,
        output_config: OutputConfig | None = None,
        analysis_config: AnalysisConfig | None = None,
    ):
        """
        Args:
            gt_path:         COCO JSON file **or** YOLO dataset directory.
            eval_config:     Evaluation thresholds and class filtering.
            output_config:   Output directory, caching and overwrite settings.
            analysis_config: Analysis report / visualisation settings.
        """
        self.gt_path = Path(gt_path)
        self.eval_cfg = eval_config or EvaluateConfig()
        self.output_cfg = output_config or OutputConfig()
        self.analysis_cfg = analysis_config or AnalysisConfig()

        self.output_dir = self.output_cfg.path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._loader = DataLoader()
        self._predictions: Dict[str, Path] = {}  # name -> COCO JSON path
        self._gt_data: Optional[list] = None
        self._gt_coco_path: Optional[Path] = None
        self._categories: Dict[int, str] = {}

        # Results
        self.results: Dict[str, MetricsResult] = {}
        self._detailed_results: Dict[str, EvaluationResult] = {}

        # Cache
        self._cache = (
            CacheManager(self.output_dir / ".cache")
            if self.output_cfg.use_cache
            else None
        )

    # -- Setup ---------------------------------------------------------

    def add_predictions(self, name: str, pred_path: str | Path) -> None:
        """Register a prediction file (COCO JSON) for evaluation."""
        self._predictions[name] = Path(pred_path)

    # -- Stage 1: Load & Convert ---------------------------------------

    def _ensure_gt_loaded(self) -> None:
        """Load GT data and resolve to COCO JSON."""
        if self._gt_data is not None:
            return

        self._gt_data, self._gt_coco_path = self._loader.load_gt(
            self.gt_path, class_names=self.eval_cfg.class_names
        )
        self._categories = self._loader.categories

    # -- Stage 2: Evaluate ---------------------------------------------

    def evaluate(self) -> Dict[str, MetricsResult]:
        """
        Run full evaluation (pycocotools + detailed) for all registered models.

        Returns:
            ``{model_name: MetricsResult}``
        """
        self._ensure_gt_loaded()

        if not self._predictions:
            raise ValueError("No predictions registered. Call add_predictions() first.")

        ecfg = self.eval_cfg

        for name, pred_path in self._predictions.items():
            logger.info(f"Evaluating: {name}")
            result = MetricsResult(model_name=name)

            # -- PyCocoTools (Stage 1) ---------------------------------
            try:
                pce = PyCocoEvaluator(
                    gt_path=self._gt_coco_path,
                    pred_path=pred_path,
                    cat_ids=ecfg.class_ids,
                )
                result.coco_metrics = pce.evaluate()
                logger.info(
                    f"  COCO: mAP@50={result.coco_metrics.map_50:.3f}  "
                    f"mAP@50:95={result.coco_metrics.map_50_95:.3f}"
                )
            except Exception as e:
                logger.warning(f"  PyCocoEvaluator failed for {name}: {e}")

            # -- DetailedEvaluator (Stage 2) ---------------------------
            cache_key = None
            cached = None
            if self._cache:
                cache_key = CacheManager.generate_cache_key(
                    name, self._gt_coco_path,
                    ecfg.iou_threshold, ecfg.conf_threshold,
                    ecfg.class_ids,
                )
                cached = self._cache.load(cache_key)

            if cached is not None:
                logger.info("  Detailed: loaded from cache")
                result.detailed_metrics = cached
            else:
                preds_list = json.loads(pred_path.read_text())
                pred_internal = coco_preds_to_internal(preds_list)

                evaluator = DetailedEvaluator(
                    ground_truths=self._gt_data,
                    predictions=pred_internal,
                    iou_threshold=ecfg.iou_threshold,
                    conf_threshold=ecfg.conf_threshold,
                    eval_classids=ecfg.class_ids,
                )
                result.detailed_metrics = evaluator.run()

                if self._cache and cache_key:
                    self._cache.save(cache_key, result.detailed_metrics)

            self.results[name] = result
            self._detailed_results[name] = result.detailed_metrics

            if result.detailed_metrics:
                self._report_gen.print_summary(result.detailed_metrics)

        return self.results

    @property
    def _report_gen(self) -> ReportGenerator:
        return ReportGenerator(
            output_folder=self.output_dir,
            labels_map=self._categories,
            overwrite=self.output_cfg.overwrite,
        )

    # -- Stage 3: Analysis ---------------------------------------------

    def analyze(self, config: AnalysisConfig | None = None) -> None:
        """
        Generate CSV analysis reports.

        Args:
            config: Override analysis settings. Falls back to ``self.analysis_cfg``.
        """
        if not self._detailed_results:
            logger.warning("No detailed results -- run evaluate() first.")
            return

        acfg = config or self.analysis_cfg
        target_class_ids = (
            sorted(self.eval_cfg.class_ids)
            if self.eval_cfg.class_ids
            else sorted(self._categories.keys())
        )
        rg = self._report_gen

        rg.map_analysis(self._detailed_results)
        rg.tpfp_analysis(self._detailed_results, target_class_ids, acfg.conf_thresholds)
        rg.precision_choices_analysis(
            self._detailed_results, target_class_ids, acfg.precision_targets,
        )

        cm_conf = {name: acfg.conf_thresholds for name in self._detailed_results}
        rg.confusion_matrix_report(self._detailed_results, cm_conf)

        self._save_coco_summary_csv()
        logger.info(f"Analysis CSVs saved to {self.output_dir}")

    def _save_coco_summary_csv(self) -> None:
        """Save a summary CSV of COCO metrics for all models."""
        import pandas as pd

        rows = []
        for name, mr in self.results.items():
            row = {"model": name}
            if mr.coco_metrics:
                cm = mr.coco_metrics
                row.update({
                    "mAP@50": round(cm.map_50 * 100, 2),
                    "mAP@50:95": round(cm.map_50_95 * 100, 2),
                    "mAP_small": round(cm.map_small * 100, 2),
                    "mAP_medium": round(cm.map_medium * 100, 2),
                    "mAP_large": round(cm.map_large * 100, 2),
                    "AR@1": round(cm.ar_1 * 100, 2),
                    "AR@10": round(cm.ar_10 * 100, 2),
                    "AR@100": round(cm.ar_100 * 100, 2),
                })
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            path = self.output_dir / "coco_metrics_summary.csv"
            df.to_csv(path, index=False)
            logger.info(f"COCO summary -> {path}")

    # -- Stage 4: Visualise --------------------------------------------

    def visualize(self, config: AnalysisConfig | None = None) -> None:
        """
        Generate PDF/PNG visualisations.

        Args:
            config: Override analysis settings. Falls back to ``self.analysis_cfg``.
        """
        if not self._detailed_results:
            logger.warning("No detailed results -- run evaluate() first.")
            return

        acfg = config or self.analysis_cfg
        target_class_ids = (
            sorted(self.eval_cfg.class_ids)
            if self.eval_cfg.class_ids
            else sorted(self._categories.keys())
        )
        viz = Visualizer(
            output_folder=self.output_dir,
            labels_map=self._categories,
            overwrite=self.output_cfg.overwrite,
        )

        cm_conf = {name: acfg.conf_thresholds for name in self._detailed_results}

        if acfg.single_pdf:
            viz.generate_report(
                self._detailed_results, target_class_ids, cm_conf,
            )
        else:
            viz.plot_pr_curves(self._detailed_results, target_class_ids)
            viz.plot_f1_curves(self._detailed_results, target_class_ids)
            viz.plot_precision_curves(self._detailed_results, target_class_ids)
            viz.plot_recall_curves(self._detailed_results, target_class_ids)
            viz.plot_confusion_matrix(self._detailed_results, cm_conf, target_class_ids)
            viz.plot_metrics_comparison(self._detailed_results, target_class_ids)

        logger.info(f"Visualisations saved to {self.output_dir}")
