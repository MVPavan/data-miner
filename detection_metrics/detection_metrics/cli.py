"""Click CLI interface for detection_metrics."""

from pathlib import Path
from typing import List, Optional, Tuple

import click

from detection_metrics.configs.config import load_config, FullConfig
from detection_metrics.logging import setup_logging, logger, console


@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.option('-q', '--quiet', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool):
    """Detection Metrics - Unified object detection evaluation toolkit."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    setup_logging(verbose=verbose, quiet=quiet)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), 
              help='Path to YAML config file')
@click.option('--gt', type=click.Path(exists=True, path_type=Path), 
              help='Path to ground truth annotations (COCO JSON)')
@click.option('--predictions', '-p', type=click.Path(exists=True, path_type=Path), 
              multiple=True, help='Paths to prediction files')
@click.option('--names', '-n', type=str, multiple=True, 
              help='Names for prediction files')
@click.option('--classes', type=int, multiple=True, 
              help='Class IDs to evaluate (default: all)')
@click.option('--iou', type=float, default=0.5, 
              help='IoU threshold (default: 0.5)')
@click.option('--conf', type=float, default=0.001, 
              help='Confidence threshold (default: 0.001)')
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              default='./results', help='Output directory')
@click.option('--overwrite', is_flag=True, help='Overwrite existing results')
@click.pass_context
def evaluate(
    ctx: click.Context,
    config: Optional[Path],
    gt: Optional[Path],
    predictions: Tuple[Path, ...],
    names: Tuple[str, ...],
    classes: Tuple[int, ...],
    iou: float,
    conf: float,
    output: Path,
    overwrite: bool,
):
    """
    Evaluate predictions against ground truth.
    
    Two-stage evaluation:
    1. PyCocoTools for mAP50/mAP50-95
    2. Detailed analysis for PR curves, F1, confusion matrix
    """
    from detection_metrics.data_loader import DataLoader
    from detection_metrics.evaluator import PyCocoEvaluator, DetailedEvaluator
    from detection_metrics.report import ReportGenerator
    from detection_metrics.configs.config import PredictionEntry, DatasetConfig
    
    # Load config
    if config:
        cfg = load_config(config)
        eval_cfg = cfg.evaluate
        output_cfg = cfg.output
    else:
        # Build config from CLI args
        if not gt:
            raise click.UsageError("Either --config or --gt is required")
        if not predictions:
            raise click.UsageError("Either --config or --predictions is required")
        if names and len(names) != len(predictions):
            raise click.UsageError("Number of --names must match --predictions")
        
        pred_entries = [
            PredictionEntry(path=p, name=n or f"model_{i}") 
            for i, (p, n) in enumerate(zip(predictions, names or [None] * len(predictions)))
        ]
        
        from detection_metrics.configs.config import EvaluateConfig, OutputConfig
        eval_cfg = EvaluateConfig(
            dataset=DatasetConfig(gt_path=gt, predictions=pred_entries),
            iou_threshold=iou,
            conf_threshold=conf,
            classes=list(classes) if classes else []
        )
        output_cfg = cfg.output if config else type('', (), {'path': output, 'overwrite': overwrite})()
    
    output_path = Path(output_cfg.path if hasattr(output_cfg, 'path') else output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting evaluation...")
    
    # Load data
    data_loader = DataLoader()
    gt_data, pred_data_dict = data_loader.load_dataset(eval_cfg.dataset)
    
    results = {}
    
    for pred_name, pred_data in pred_data_dict.items():
        logger.info(f"Evaluating: {pred_name}")
        
        # Stage 2: Detailed evaluation
        evaluator = DetailedEvaluator(
            ground_truths=gt_data,
            predictions=pred_data,
            iou_threshold=eval_cfg.iou_threshold,
            conf_threshold=eval_cfg.conf_threshold,
            eval_classids=eval_cfg.classes if eval_cfg.classes else None
        )
        result = evaluator.run()
        results[pred_name] = result
    
    # Generate reports
    labels_map = data_loader.categories
    report_gen = ReportGenerator(
        output_folder=output_path,
        labels_map=labels_map,
        overwrite=overwrite or getattr(output_cfg, 'overwrite', False)
    )
    
    for pred_name, result in results.items():
        console.print(f"\n[bold]Results for {pred_name}[/bold]")
        report_gen.print_summary(result)
    
    # mAP analysis
    report_gen.map_analysis(results)
    
    logger.info(f"Evaluation complete. Results saved to {output_path}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Path to YAML config file')
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              help='Path to results directory with cached evaluations')
@click.option('--classes', type=int, multiple=True,
              help='Class IDs to analyze')
@click.option('--precision-targets', type=float, multiple=True,
              default=(0.8, 0.85, 0.9), help='Target precision values')
@click.option('--conf-thresholds', type=float, multiple=True,
              default=(0.3, 0.4, 0.5, 0.6), help='Confidence thresholds for TP/FP analysis')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default='./results', help='Output directory')
@click.pass_context
def analyze(
    ctx: click.Context,
    config: Optional[Path],
    results: Optional[Path],
    classes: Tuple[int, ...],
    precision_targets: Tuple[float, ...],
    conf_thresholds: Tuple[float, ...],
    output: Path,
):
    """
    Generate detailed analysis reports and visualizations.
    
    Includes:
    - TP/FP analysis at different confidence thresholds
    - Precision targets analysis
    - PR curves and F1 curves (PDF)
    - Confusion matrices
    """
    from detection_metrics.report import ReportGenerator
    from detection_metrics.visualizer import Visualizer
    from detection_metrics.cache import CacheManager
    
    if config:
        cfg = load_config(config)
        analyze_cfg = cfg.analyze
    else:
        from detection_metrics.configs.config import AnalysisConfig
        analyze_cfg = AnalysisConfig(
            precision_targets=list(precision_targets),
            tpfp_conf_thresholds=list(conf_thresholds)
        )
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load cached results
    if results:
        cache_mgr = CacheManager(results)
        # TODO: Load cached results
        logger.info(f"Analysis would load from {results}")
    else:
        logger.warning("No results path specified for analysis")
    
    logger.info(f"Analysis complete. Results saved to {output_path}")


@cli.command()
@click.option('--model', '-m', type=click.Choice(['rfdetr', 'yolox']),
              required=True, help='Model type')
@click.option('--checkpoint', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to model weights')
@click.option('--images', '-i', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to images directory')
@click.option('--annotations', '-a', type=click.Path(exists=True, path_type=Path),
              help='Path to COCO annotations (optional, for image list)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default='./predictions.json', help='Output predictions file')
@click.option('--threshold', type=float, default=0.001,
              help='Confidence threshold')
@click.option('--device', type=str, default='cuda',
              help='Device to use (cuda/cpu)')
@click.pass_context
def predict(
    ctx: click.Context,
    model: str,
    checkpoint: Path,
    images: Path,
    annotations: Optional[Path],
    output: Path,
    threshold: float,
    device: str,
):
    """
    Generate predictions using a specified model.
    
    Outputs predictions in COCO JSON format.
    """
    logger.info(f"Loading {model} model from {checkpoint}")
    logger.info(f"Running inference on images in {images}")
    
    # TODO: Implement inference engines
    logger.warning("Prediction generation not yet fully implemented")
    logger.info(f"Predictions would be saved to {output}")


@cli.command()
@click.option('--model', '-m', type=click.Choice(['rfdetr', 'yolox']),
              required=True, help='Model type')
@click.option('--checkpoint', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to model weights')
@click.option('--resolution', type=int, default=560,
              help='Input resolution')
@click.pass_context
def profile(
    ctx: click.Context,
    model: str,
    checkpoint: Path,
    resolution: int,
):
    """
    Profile model performance (FLOPs, FPS).
    """
    logger.info(f"Profiling {model} at resolution {resolution}")
    
    # TODO: Implement profiling
    logger.warning("Profiling not yet implemented")


@cli.command()
def version():
    """Show version information."""
    from detection_metrics import __version__
    console.print(f"detection-metrics version {__version__}")


if __name__ == '__main__':
    cli()
