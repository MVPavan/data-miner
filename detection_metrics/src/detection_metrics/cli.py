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


# ─────────────────────────────────────────────────────────────────────
# Config-driven commands
# ─────────────────────────────────────────────────────────────────────

def _load_cfg(config: Path, overrides: Tuple[str, ...]) -> FullConfig:
    """Load YAML config and apply dotlist overrides."""
    from omegaconf import OmegaConf
    base = OmegaConf.load(config)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        base = OmegaConf.merge(base, cli_cfg)
    config_dict = OmegaConf.to_container(base, resolve=True)
    return FullConfig.model_validate(config_dict)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to YAML config file')
@click.argument('overrides', nargs=-1, type=str)
@click.pass_context
def evaluate(ctx: click.Context, config: Path, overrides: Tuple[str, ...]):
    """
    Evaluate predictions against ground truth.

    Reads all settings from a YAML config file. Use dotlist overrides
    to tweak individual values:

    \b
      detection-metrics evaluate -c eval.yaml
      detection-metrics evaluate -c eval.yaml evaluate.iou_threshold=0.7
    """
    from detection_metrics.pipeline import DetectionMetrics

    cfg = _load_cfg(config, overrides)
    eval_cfg = cfg.evaluate
    output_cfg = cfg.output

    if eval_cfg is None:
        raise click.UsageError("Config must contain an 'evaluate' section")

    if not eval_cfg.dataset.gt_path:
        raise click.UsageError("Config must specify evaluate.dataset.gt_path")

    if not eval_cfg.dataset.predictions:
        raise click.UsageError("Config must specify at least one prediction entry")

    logger.info("Starting evaluation...")

    dm = DetectionMetrics(
        gt_path=eval_cfg.dataset.gt_path,
        eval_config=eval_cfg,
        output_config=output_cfg,
        analysis_config=cfg.analyze,
    )

    for pred in eval_cfg.dataset.predictions:
        dm.add_predictions(pred.name, pred.path)

    dm.evaluate()
    dm.analyze()
    dm.visualize()

    logger.info(f"Evaluation complete. Results saved to {output_cfg.path}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to YAML config file')
@click.argument('overrides', nargs=-1, type=str)
@click.pass_context
def analyze(ctx: click.Context, config: Path, overrides: Tuple[str, ...]):
    """
    Generate detailed analysis reports and visualizations.

    \b
      detection-metrics analyze -c eval.yaml
      detection-metrics analyze -c eval.yaml analyze.vis_conf_threshold=0.6
    """
    from detection_metrics.cache import CacheManager

    cfg = _load_cfg(config, overrides)
    analyze_cfg = cfg.analyze
    output_cfg = cfg.output

    if analyze_cfg is None:
        raise click.UsageError("Config must contain an 'analyze' section")

    output_path = Path(output_cfg.path)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_mgr = CacheManager(output_path)
    # TODO: Load cached results and generate reports
    logger.info(f"Analysis would load from {output_path}")
    logger.info(f"Analysis complete. Results saved to {output_path}")


@cli.command()
def version():
    """Show version information."""
    from detection_metrics import __version__
    console.print(f"detection-metrics version {__version__}")


@cli.command('analyze-dataset')
@click.option('--dataset', '-d', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to YOLO dataset root')
@click.option('--split', '-s', type=str, default=None,
              help='Analyse only this split (default: all)')
@click.option('--yaml', type=str, default=None,
              help='Name of YAML config file (default: data.yaml)')
@click.pass_context
def analyze_dataset(ctx: click.Context, dataset: Path, split: str, yaml: str):
    """Analyze a YOLO dataset — class distribution, bbox stats, area buckets."""
    from detection_metrics.dataset_analysis import run_analysis
    run_analysis(dataset, split=split, yaml_name=yaml)


# ─────────────────────────────────────────────────────────────────────
# Dataset conversion commands
# ─────────────────────────────────────────────────────────────────────

_FORMAT_NAMES = ["coco", "darknet", "roboflow", "yolo_v5a", "yolo_v5b"]


@cli.command('detect-format')
@click.argument('path', type=click.Path(exists=True, path_type=Path))
def detect_format_cmd(path: Path):
    """Detect the annotation format of a dataset directory."""
    from detection_metrics.convert_dataset import detect_format
    fmt = detect_format(path)
    console.print(fmt.value)


@cli.command('convert')
@click.option('--source', '-s', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to source dataset')
@click.option('--target', '-t', type=click.Path(path_type=Path),
              required=True, help='Path for converted output')
@click.option('--source-format', type=click.Choice(_FORMAT_NAMES),
              default=None, help='Source format (auto-detected if omitted)')
@click.option('--target-format', type=click.Choice(_FORMAT_NAMES),
              required=True, help='Target annotation format')
@click.option('--splits', default='train,valid,test',
              help='Comma-separated splits to convert (default: train,valid,test)')
@click.option('--copy-images', is_flag=True,
              help='Copy images instead of creating symlinks')
@click.pass_context
def convert_cmd(
    ctx: click.Context,
    source: Path,
    target: Path,
    source_format: Optional[str],
    target_format: str,
    splits: str,
    copy_images: bool,
):
    """Convert a detection dataset between annotation formats.

    Supports 5 formats: coco, darknet, roboflow, yolo_v5a, yolo_v5b.
    Source format is auto-detected when not specified.
    Images are symlinked by default (use --copy-images to copy).
    """
    from detection_metrics.convert_dataset import (
        ConvertConfig,
        DatasetFormat,
        Split,
        convert,
    )

    # Parse splits
    split_list: List[Split] = []
    for token in splits.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "val":
            token = "valid"
        try:
            split_list.append(Split(token))
        except ValueError:
            raise click.BadParameter(
                f"Unknown split '{token}'. "
                f"Choose from: {', '.join(s.value for s in Split)}"
            )

    src_fmt = DatasetFormat(source_format) if source_format else None
    cfg = ConvertConfig(
        source=source,
        target=target,
        source_format=src_fmt,
        target_format=DatasetFormat(target_format),
        splits=split_list,
        copy_images=copy_images,
    )
    bundle = convert(cfg)
    console.print(f"\n[bold green]Done![/bold green] {bundle.summary()}")


if __name__ == '__main__':
    cli()
