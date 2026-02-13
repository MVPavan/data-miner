#!/usr/bin/env python3
"""
CVAT Annotation CLI - Upload and download YOLO datasets.

Usage:
    python -m scripts.annotation.annotate upload config.yaml
    python -m scripts.annotation.annotate download config.yaml
    python -m scripts.annotation.annotate status config.yaml
    python -m scripts.annotation.annotate init job_config.yaml
    python -m scripts.annotation.annotate cleanup config.yaml
    python -m scripts.annotation.annotate list config.yaml
"""

import functools
import time
from pathlib import Path
from typing import Callable, TypeVar

import fiftyone as fo
import typer
from tqdm import tqdm

from data_miner.logging import get_logger

from .config import AnnotationConfig, create_sample_config, load_config

logger = get_logger(__name__)
app = typer.Typer(
    name="annotate", help="CVAT annotation workflow", add_completion=False
)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # Exponential backoff

T = TypeVar("T")


def retry_with_backoff(
    retries: int = MAX_RETRIES,
    delays: list[int] = RETRY_DELAYS,
    exceptions: tuple = (ConnectionError, TimeoutError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < retries:
                        delay = delays[min(attempt, len(delays) - 1)]
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {retries + 1} attempts failed")
            raise last_exception

        return wrapper

    return decorator


def _load_dataset(cfg: AnnotationConfig) -> fo.Dataset:
    """Load or create FiftyOne dataset (follows remote_view.py pattern)."""
    dc = cfg.dataset

    if fo.dataset_exists(dc.name):
        if dc.overwrite:
            logger.info(f"Deleting existing dataset: {dc.name}")
            fo.delete_dataset(dc.name)
        else:
            logger.info(f"Loading existing dataset: {dc.name}")
            return fo.load_dataset(dc.name)

    logger.info(f"Creating dataset: {dc.name}")

    # Create dataset from images directory
    if dc.labels_dir:
        # Load with YOLO labels
        # Pass classes from config to map index 0 -> 'door', etc.
        classes = cfg.annotate.classes
        
        ds = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv4Dataset,
            data_path=str(dc.images_dir),
            labels_path=str(dc.labels_dir),
            label_field=dc.label_field,
            name=dc.name,
            classes=classes,  # Maps class indices to names
        )
    else:
        # Images only (no labels)
        ds = fo.Dataset.from_dir(
            dataset_type=fo.types.ImageDirectory,
            dataset_dir=str(dc.images_dir),
            name=dc.name,
        )

    ds.persistent = True
    logger.info(f"Loaded {len(ds)} samples")
    return ds


def _apply_quality_attributes(ds: fo.Dataset, cfg: AnnotationConfig) -> fo.Dataset:
    """Map confidence scores to quality class labels.
    
    Instead of attributes, changes the detection label to the quality value
    (bad/partial/loose/good) based on confidence thresholds.
    """
    qm = cfg.annotate.quality_mapping
    if not qm.enabled:
        return ds
    
    label_field = cfg.dataset.label_field
    
    logger.info("Mapping confidence to quality class labels")
    logger.info(f"  bad < {qm.bad_threshold}, partial < {qm.partial_threshold}, loose < {qm.loose_threshold}")
    
    counts = {v: 0 for v in qm.values}
    
    for sample in ds:
        det_field = sample.get_field(label_field)
        if not det_field or not det_field.detections:
            continue
        
        for det in det_field.detections:
            confidence = getattr(det, "confidence", None)
            quality = qm.confidence_to_quality(confidence)
            det.label = quality  # Change the class label to quality value
            counts[quality] += 1
        
        sample.save()
    
    logger.info(f"Quality distribution: {counts}")
    return ds


def _export_yolo(ds: fo.Dataset, cfg: AnnotationConfig):
    """Export dataset to YOLO format with confidence."""
    out = cfg.export.output_dir
    out.mkdir(parents=True, exist_ok=True)
    label_field = cfg.dataset.label_field
    classes = ds.distinct(f"{label_field}.detections.label")

    logger.info(f"Exporting to: {out}")
    ds.export(
        str(out),
        fo.types.YOLOv5Dataset,
        split=cfg.export.split,
        label_field=label_field,
        classes=classes,
    )

    if cfg.export.include_confidence:
        labels_dir = out / "labels" / cfg.export.split
        labels_dir.mkdir(parents=True, exist_ok=True)
        class_idx = {c: i for i, c in enumerate(classes)}

        for sample in tqdm(ds, desc="Writing confidence scores", unit="sample"):
            det_field = sample.get_field(label_field)
            if not det_field:
                continue
            lines = []
            for det in det_field.detections:
                idx = class_idx.get(det.label, -1)
                if idx < 0:
                    continue
                x, y, w, h = det.bounding_box
                conf = getattr(det, "confidence", None)
                line = f"{idx} {x+w/2:.6f} {y+h/2:.6f} {w:.6f} {h:.6f}"
                if conf is not None:
                    line += f" {conf:.6f}"
                lines.append(line)
            (labels_dir / f"{Path(sample.filepath).stem}.txt").write_text(
                "\n".join(lines)
            )

    logger.info("Export complete")


@app.command()
def init(config_path: Path = typer.Argument(..., help="Path for new config file")):
    """Create sample configuration file."""
    create_sample_config(config_path)


@app.command()
def upload(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    launch: bool = typer.Option(False, "--launch", "-l", help="Open CVAT in browser"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only"),
):
    """Upload dataset to CVAT."""
    cfg = load_config(config_path)
    cfg.cvat.validate_credentials()

    # Test connection before proceeding
    logger.info(f"Testing connection to CVAT: {cfg.cvat.url}")
    cfg.cvat.test_connection()
    logger.info("CVAT connection successful")

    if dry_run:
        typer.echo(f"Config valid: {cfg.anno_key}")
        typer.echo(f"  Dataset: {cfg.dataset.images_dir}")
        typer.echo(f"  CVAT: {cfg.cvat.url}")
        return

    ds = _load_dataset(cfg)
    
    # Apply quality attributes if enabled
    _apply_quality_attributes(ds, cfg)

    # Build kwargs using Pydantic model method
    kwargs = cfg.annotate.to_annotate_kwargs(
        cvat=cfg.cvat,
        label_field=cfg.dataset.label_field,
        launch=launch,
    )

    logger.info(f"Uploading to CVAT: {cfg.cvat.url}")
    try:
        _upload_with_retry(ds, cfg.anno_key, kwargs)
        logger.info("Upload complete")
        typer.echo(f"Upload complete: {cfg.anno_key}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        typer.echo(
            "Upload failed. You may need to run 'cleanup' to remove partial tasks.",
            err=True,
        )
        raise typer.Exit(1)


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _upload_with_retry(ds: fo.Dataset, anno_key: str, kwargs: dict):
    """Upload to CVAT with retry logic."""
    ds.annotate(anno_key, **kwargs)


@app.command()
def download(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only"),
):
    """Download annotations from CVAT and export to YOLO."""
    cfg = load_config(config_path)
    cfg.cvat.validate_credentials()

    # Test connection
    logger.info(f"Testing connection to CVAT: {cfg.cvat.url}")
    cfg.cvat.test_connection()

    if not fo.dataset_exists(cfg.dataset.name):
        logger.error(f"Dataset not found: {cfg.dataset.name}")
        raise typer.Exit(1)

    ds = fo.load_dataset(cfg.dataset.name)

    if cfg.anno_key not in ds.list_annotation_runs():
        logger.error(f"Annotation run not found: {cfg.anno_key}")
        raise typer.Exit(1)

    if dry_run:
        typer.echo(f"Would download: {cfg.anno_key}")
        typer.echo(f"  Export to: {cfg.export.output_dir}")
        return

    logger.info("Downloading annotations from CVAT...")
    _download_with_retry(
        ds,
        cfg.anno_key,
        cfg.cvat.to_credentials(),
        cfg.export.unexpected.value,
        cfg.export.cleanup,
    )

    _export_yolo(ds, cfg)
    logger.info("All done")
    typer.echo("Download and export complete")


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _download_with_retry(
    ds: fo.Dataset, anno_key: str, cvat_kwargs: dict, unexpected: str, cleanup: bool
):
    """Download from CVAT with retry logic."""
    ds.load_annotations(
        anno_key,
        **cvat_kwargs,
        unexpected=unexpected,
        cleanup=cleanup,
    )


@app.command()
def status(config_path: Path = typer.Argument(..., help="Path to config YAML")):
    """Show annotation task status."""
    cfg = load_config(config_path)
    cfg.cvat.validate_credentials()

    if not fo.dataset_exists(cfg.dataset.name):
        logger.error(f"Dataset not found: {cfg.dataset.name}")
        raise typer.Exit(1)

    ds = fo.load_dataset(cfg.dataset.name)

    if cfg.anno_key not in ds.list_annotation_runs():
        typer.echo(f"No annotation run: {cfg.anno_key}")
        return

    # Pass CVAT credentials to load results from custom CVAT server
    results = ds.load_annotation_results(cfg.anno_key, **cfg.cvat.to_credentials())
    results.print_status()


@app.command()
def cleanup(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete CVAT tasks without downloading annotations."""
    cfg = load_config(config_path)
    cfg.cvat.validate_credentials()

    if not fo.dataset_exists(cfg.dataset.name):
        logger.error(f"Dataset not found: {cfg.dataset.name}")
        raise typer.Exit(1)

    ds = fo.load_dataset(cfg.dataset.name)

    if cfg.anno_key not in ds.list_annotation_runs():
        typer.echo(f"No annotation run: {cfg.anno_key}")
        return

    if not force:
        confirm = typer.confirm(
            f"Delete CVAT tasks for '{cfg.anno_key}'? Annotations will be lost."
        )
        if not confirm:
            typer.echo("Aborted")
            return

    logger.info(f"Cleaning up CVAT tasks for: {cfg.anno_key}")
    try:
        # Pass CVAT credentials to load results from custom CVAT server
        results = ds.load_annotation_results(cfg.anno_key, **cfg.cvat.to_credentials())
        results.cleanup()
        ds.delete_annotation_run(cfg.anno_key)
        logger.info("Cleanup complete")
        typer.echo(f"Cleaned up: {cfg.anno_key}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise typer.Exit(1)


@app.command("list")
def list_runs(config_path: Path = typer.Argument(..., help="Path to config YAML")):
    """List all annotation runs on the dataset."""
    cfg = load_config(config_path)

    if not fo.dataset_exists(cfg.dataset.name):
        logger.error(f"Dataset not found: {cfg.dataset.name}")
        raise typer.Exit(1)

    ds = fo.load_dataset(cfg.dataset.name)
    runs = ds.list_annotation_runs()

    if not runs:
        typer.echo("No annotation runs found")
        return

    typer.echo(f"Annotation runs on '{cfg.dataset.name}':")
    for run in runs:
        marker = " (current)" if run == cfg.anno_key else ""
        typer.echo(f"  - {run}{marker}")


@app.command()
def info(config_path: Path = typer.Argument(..., help="Path to config YAML")):
    """Show detailed annotation run info."""
    cfg = load_config(config_path)

    if not fo.dataset_exists(cfg.dataset.name):
        logger.error(f"Dataset not found: {cfg.dataset.name}")
        raise typer.Exit(1)

    ds = fo.load_dataset(cfg.dataset.name)

    if cfg.anno_key not in ds.list_annotation_runs():
        typer.echo(f"No annotation run: {cfg.anno_key}")
        return

    info = ds.get_annotation_info(cfg.anno_key)
    typer.echo(f"Annotation run: {cfg.anno_key}")
    typer.echo(f"  Backend: {info.config.get('backend', 'unknown')}")
    typer.echo(f"  Label fields: {info.config.get('label_schema', {}).keys()}")
    typer.echo(f"  Created: {info.timestamp}")


def main():
    app()


if __name__ == "__main__":
    main()
