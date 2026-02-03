"""Report generation for detection metrics evaluation results."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from detection_metrics.evaluator import ClassMetrics, EvaluationResult
from detection_metrics.logging import logger, console, create_results_table


# Constants
FP_PRECISION_EPSILON = 1e-6


class ReportGenerator:
    """Generates CSV reports and text summaries."""

    def __init__(
        self, 
        output_folder: Path, 
        labels_map: Optional[Dict[int, str]] = None, 
        overwrite: bool = True
    ):
        """
        Initialize ReportGenerator.
        
        Args:
            output_folder: Directory to save reports.
            labels_map: Optional dict mapping class IDs to names.
            overwrite: Whether to overwrite existing files.
        """
        self.labels_map = labels_map or {}
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.tpfp_analysis_file = self.output_folder / "tpfp_analysis.csv"
        self.map_analysis_file = self.output_folder / "map_analysis.csv"
        self.precision_choices_file = self.output_folder / "precision_choices_analysis.csv"
        
        if not overwrite:
            existing = [f for f in [self.tpfp_analysis_file, self.map_analysis_file, self.precision_choices_file] if f.exists()]
            if existing:
                raise FileExistsError(f"Files exist: {existing}. Set overwrite=True.")

    def print_summary(self, result: EvaluationResult):
        """Print a Rich table summary of evaluation results."""
        table = create_results_table(
            "Evaluation Summary",
            ["Class", "AP", "Max F1, Prec @ conf", "TP", "FP", "FN", "Total GT"]
        )
        
        mAPs = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for cls_id in sorted(result.class_metrics.keys()):
            data = result.class_metrics[cls_id]
            mAPs.append(data.ap)
            
            f1s = np.array(data.f1)
            if len(f1s) > 0:
                max_f1 = np.max(f1s)
                max_idx = np.argmax(f1s)
                conf_at_max_f1 = data.conf[max_idx]
                prec_at_max_f1 = data.precision[max_idx]
            else:
                max_f1 = conf_at_max_f1 = prec_at_max_f1 = 0.0
            
            tp = data.total_tp
            fp = data.total_fp
            fn = data.total_gt - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            label = self.labels_map.get(cls_id, str(cls_id))
            label = (label[:17] + '..') if len(label) > 19 else label
            
            table.add_row(
                label,
                f"{data.ap:.3f}",
                f"{max_f1:.3f}, {prec_at_max_f1:.3f} @ {conf_at_max_f1:.3f}",
                str(tp),
                str(fp),
                str(fn),
                str(data.total_gt)
            )
        
        # Add summary row
        mAP = np.mean(mAPs) if mAPs else 0.0
        table.add_row(
            "[bold]Mean/Total[/bold]",
            f"[bold]{mAP:.3f}[/bold]",
            "-",
            f"[bold]{total_tp}[/bold]",
            f"[bold]{total_fp}[/bold]",
            f"[bold]{total_fn}[/bold]",
            f"[bold]{total_tp + total_fn}[/bold]"
        )
        
        console.print(table)

    def _eval_analysis_cls(self, data: ClassMetrics, conf_thresholds: List[float]) -> pd.DataFrame:
        """Analyze metrics at different confidence thresholds for a single class."""
        df = pd.DataFrame(columns=['Conf', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'])
        
        total_gt = data.total_gt
        conf_arr = np.array(data.conf)
        tp_arr = np.array(data.tp)
        fp_arr = np.array(data.fp)
        
        for th in conf_thresholds:
            mask = conf_arr >= th
            tp = np.sum(tp_arr[mask])
            fp = np.sum(fp_arr[mask])
            fn = total_gt - tp
            
            prec = tp / (tp + fp + FP_PRECISION_EPSILON)
            rec = tp / (total_gt + FP_PRECISION_EPSILON)
            f1 = 2 * (prec * rec) / (prec + rec + FP_PRECISION_EPSILON)
            
            df.loc[len(df)] = [th, tp, fp, fn, prec, rec, f1]

        return df.sort_values(by='Conf', ascending=True)

    def tpfp_analysis(
        self, 
        results: Dict[str, EvaluationResult], 
        target_class_ids: List[int], 
        conf_thresholds: Optional[List[float]] = None
    ):
        """
        Generate TP/FP analysis across confidence thresholds.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            target_class_ids: Class IDs to analyze.
            conf_thresholds: Confidence thresholds to evaluate.
        """
        if conf_thresholds is None:
            conf_thresholds = [round(float(x), 1) for x in np.arange(0.1, 1.0, 0.1)]
        
        all_dfs = []
        for pred_name, result in results.items():
            for cid in target_class_ids:
                if cid not in result.class_metrics:
                    logger.warning(f"Class {cid} not found in {pred_name}")
                    continue
                    
                data = result.class_metrics[cid]
                df = self._eval_analysis_cls(data, conf_thresholds)
                df['eval_name'] = pred_name
                df['class_id'] = cid
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("No data for TP/FP analysis")
            return
        
        full_df = pd.concat(all_dfs)
        wide_df = full_df.pivot_table(
            index=['class_id', 'Conf'], 
            columns='eval_name', 
            values=['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        )
        
        metric_order = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        metric_priority = {m: i for i, m in enumerate(metric_order)}
        wide_df = wide_df.reindex(columns=sorted(wide_df.columns, key=lambda x: (x[1], metric_priority.get(x[0], 100))))
        wide_df.columns.names = [None, None]
        wide_df = wide_df.reset_index().round(3)
        
        wide_df.to_csv(self.tpfp_analysis_file, index=False)
        logger.info(f"TP/FP Analysis saved to {self.tpfp_analysis_file}")
    
    def map_analysis(self, results: Dict[str, EvaluationResult]):
        """Generate mAP analysis report."""
        pred_names = list(results.keys())
        df = pd.DataFrame(columns=['Class ID', 'AP', 'Max F1', 'Prec', 'conf', 'eval_name'])
        
        for pred_name, result in results.items():
            for cid, data in result.class_metrics.items():
                f1s = np.array(data.f1)
                if len(f1s) > 0:
                    max_f1 = np.max(f1s)
                    max_idx = np.argmax(f1s)
                    conf_at_max_f1 = data.conf[max_idx]
                    prec_at_max_f1 = data.precision[max_idx]
                else:
                    max_f1 = conf_at_max_f1 = prec_at_max_f1 = 0.0
                
                df.loc[len(df)] = [cid, data.ap, max_f1, prec_at_max_f1, conf_at_max_f1, pred_name]
        
        # Add mAP rows
        for pred_name in pred_names:
            pred_data = df[df['eval_name'] == pred_name]
            mAP = np.mean(pred_data['AP']) if len(pred_data) > 0 else 0.0
            df.loc[len(df)] = ['mAP', mAP, np.nan, np.nan, np.nan, pred_name]
        
        df_pivot = df.pivot_table(
            index='Class ID',
            columns='eval_name',
            values=['AP', 'Max F1', 'Prec', 'conf']
        )
        
        metric_order = ['AP', 'Max F1', 'Prec', 'conf']
        metric_priority = {m: i for i, m in enumerate(metric_order)}
        df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (x[1], metric_priority.get(x[0], 100))))
        df_pivot.columns.names = [None, None]
        df_pivot = df_pivot.reset_index().round(3)
        
        df_pivot.to_csv(self.map_analysis_file, index=False)
        logger.info(f"mAP Analysis saved to {self.map_analysis_file}")
    
    def precision_choices_analysis(
        self, 
        results: Dict[str, EvaluationResult], 
        target_class_ids: List[int], 
        precision_values: List[float]
    ) -> pd.DataFrame:
        """
        Find confidence thresholds that achieve target precision values.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            target_class_ids: Class IDs to analyze.
            precision_values: Target precision values (e.g., [0.8, 0.85, 0.9]).
        
        Returns:
            DataFrame with analysis results.
        """
        target_class_ids = sorted(target_class_ids)
        df = pd.DataFrame(columns=['eval_name', 'class_id', 'Precision', 'Conf', 'TP', 'FP', 'FN', 'Recall', 'F1'])
        
        for pred_name, result in results.items():
            for cid in target_class_ids:
                if cid not in result.class_metrics:
                    continue
                    
                data = result.class_metrics[cid]
                prec_arr = np.array(data.precision)
                sorted_indices = np.argsort(prec_arr)
                prec_arr = prec_arr[sorted_indices]
                conf_arr = np.array(data.conf)[sorted_indices]
                tp_arr = np.array(data.tp)[sorted_indices]
                fp_arr = np.array(data.fp)[sorted_indices]
                rec_arr = np.array(data.recall)[sorted_indices]
                f1_arr = np.array(data.f1)[sorted_indices]
                total_gt = data.total_gt
                
                for prec_target in precision_values:
                    indices = np.where(prec_arr >= prec_target)[0]
                    if len(indices) == 0:
                        logger.warning(f"Class {cid}: No threshold for precision >= {prec_target}")
                        continue
                    
                    idx = indices[0]
                    conf_thr = conf_arr[idx]
                    mask = conf_arr >= conf_thr
                    tp = np.sum(tp_arr[mask])
                    fp = np.sum(fp_arr[mask])
                    fn = total_gt - tp
                    rec = rec_arr[idx]
                    f1 = f1_arr[idx]
                    
                    df.loc[len(df)] = [pred_name, cid, prec_target, conf_thr, tp, fp, fn, rec, f1]
        
        df = df.round(3)
        df.to_csv(self.precision_choices_file, index=False)
        logger.info(f"Precision Choices saved to {self.precision_choices_file}")
        
        return df

    def confusion_matrix_report(
        self, 
        results: Dict[str, EvaluationResult], 
        cm_conf_choices: Dict[str, List[float]]
    ):
        """
        Generate confusion matrix reports at specified confidence thresholds.
        
        Args:
            results: Dict mapping model name to EvaluationResult.
            cm_conf_choices: Dict mapping model name to list of conf thresholds.
        """
        all_dfs = []
        
        for pred_name, result in results.items():
            if pred_name not in cm_conf_choices or not cm_conf_choices[pred_name]:
                continue
            
            for conf in cm_conf_choices[pred_name]:
                confusion_data = result.confusion_data
                total_gt_counts = result.total_gt_counts
                
                class_ids = set()
                for (gt_cls, pred_cls) in confusion_data.keys():
                    class_ids.add(gt_cls)
                    class_ids.add(pred_cls)
                class_ids = sorted(list(class_ids))
                
                df = pd.DataFrame(0, index=class_ids, columns=class_ids)
                for (gt_cls, pred_cls), confs in confusion_data.items():
                    confs_thr = [c for c in confs if c >= conf]
                    df.at[gt_cls, pred_cls] = len(confs_thr)
                
                df.index.name = 'GT \\ Pred'
                df['eval_name'] = pred_name
                df['conf'] = conf
                all_dfs.append(df)
        
        if all_dfs:
            full_df = pd.concat(all_dfs)
            cm_file = self.output_folder / "confusion_matrix.csv"
            full_df.to_csv(cm_file)
            logger.info(f"Confusion Matrix saved to {cm_file}")
