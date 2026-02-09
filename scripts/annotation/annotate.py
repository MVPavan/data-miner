#!/usr/bin/env python3
"""
CVAT Annotation CLI - Upload and download YOLO datasets.

Usage:
    python -m scripts.annotation.annotate upload config.yaml
    python -m scripts.annotation.annotate download config.yaml
    python -m scripts.annotation.annotate status config.yaml
    python -m scripts.annotation.annotate init job_config.yaml
"""

from pathlib import Path

import fiftyone as fo
import typer

from .config import AnnotationConfig, create_sample_config, load_config

app = typer.Typer(name="annotate", help="CVAT annotation workflow", add_completion=False)


def _load_dataset(cfg: AnnotationConfig) -> fo.Dataset:
    """Load or create FiftyOne dataset from YOLO format."""
    if fo.dataset_exists(cfg.dataset_name):
        typer.echo(f"Loading: {cfg.dataset_name}")
        return fo.load_dataset(cfg.dataset_name)
    
    typer.echo(f"Creating: {cfg.dataset_name} from {cfg.dataset.dir}")
    ds = fo.Dataset(name=cfg.dataset_name, persistent=True)
    
    splits = [cfg.dataset.split] if cfg.dataset.split else ["train", "val", "test"]
    for split in splits:
        try:
            ds.add_dir(
                dataset_dir=str(cfg.dataset.dir),
                dataset_type=fo.types.YOLOv5Dataset,
                split=split,
                label_field=cfg.dataset.label_field,
                tags=[split],
            )
            typer.echo(f"  Loaded: {split}")
        except Exception:
            pass
    
    typer.echo(f"Total: {len(ds)} samples")
    return ds


def _export_yolo(ds: fo.Dataset, cfg: AnnotationConfig):
    """Export dataset to YOLO format with confidence."""
    out = cfg.export.output_dir
    out.mkdir(parents=True, exist_ok=True)
    label_field = cfg.dataset.label_field
    classes = ds.distinct(f"{label_field}.detections.label")
    
    typer.echo(f"Exporting to: {out}")
    ds.export(str(out), fo.types.YOLOv5Dataset, split=cfg.export.split, label_field=label_field, classes=classes)
    
    if cfg.export.include_confidence:
        labels_dir = out / "labels" / cfg.export.split
        labels_dir.mkdir(parents=True, exist_ok=True)
        class_idx = {c: i for i, c in enumerate(classes)}
        
        for sample in ds:
            det_field = sample.get_field(label_field)
            if not det_field:
                continue
            lines = []
            for det in det_field.detections:
                idx = class_idx.get(det.label, -1)
                if idx < 0:
                    continue
                x, y, w, h = det.bounding_box
                conf = getattr(det, 'confidence', None)
                line = f"{idx} {x+w/2:.6f} {y+h/2:.6f} {w:.6f} {h:.6f}"
                if conf is not None:
                    line += f" {conf:.6f}"
                lines.append(line)
            (labels_dir / f"{Path(sample.filepath).stem}.txt").write_text('\n'.join(lines))
    
    typer.echo("✓ Export complete")


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
    """Upload YOLO dataset to CVAT."""
    cfg = load_config(config_path)
    cfg.cvat.validate_credentials()
    
    if dry_run:
        typer.echo(f"Config valid: {cfg.anno_key}")
        typer.echo(f"  Dataset: {cfg.dataset.dir}")
        typer.echo(f"  CVAT: {cfg.cvat.url}")
        return
    
    ds = _load_dataset(cfg)
    kwargs = cfg.cvat.to_kwargs()
    kwargs.update({
        "label_field": cfg.dataset.label_field,
        "launch_editor": launch,
        "task_size": cfg.task.task_size,
    })
    if cfg.task.project_name:
        kwargs["project_name"] = cfg.task.project_name
    if cfg.task.segment_size:
        kwargs["segment_size"] = cfg.task.segment_size
    if cfg.task.task_assignee:
        kwargs["task_assignee"] = cfg.task.task_assignee
    if cfg.task.job_assignees:
        kwargs["job_assignees"] = cfg.task.job_assignees
    if cfg.task.classes:
        kwargs["classes"] = cfg.task.classes
    
    typer.echo(f"Uploading to CVAT: {cfg.cvat.url}")
    ds.annotate(cfg.anno_key, **kwargs)
    typer.echo("✓ Upload complete")


@app.command()
def download(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only"),
):
    """Download annotations from CVAT and export to YOLO."""
    cfg = load_config(config_path)
    cfg.cvat.validate_credentials()
    
    if not fo.dataset_exists(cfg.dataset_name):
        typer.echo(f"Dataset not found: {cfg.dataset_name}", err=True)
        raise typer.Exit(1)
    
    ds = fo.load_dataset(cfg.dataset_name)
    
    if cfg.anno_key not in ds.list_annotation_runs():
        typer.echo(f"Annotation run not found: {cfg.anno_key}", err=True)
        raise typer.Exit(1)
    
    if dry_run:
        typer.echo(f"Would download: {cfg.anno_key}")
        typer.echo(f"  Export to: {cfg.export.output_dir}")
        return
    
    typer.echo(f"Downloading from CVAT...")
    ds.load_annotations(
        cfg.anno_key,
        **cfg.cvat.to_kwargs(),
        unexpected=cfg.export.unexpected.value,
        cleanup=cfg.export.cleanup,
    )
    
    _export_yolo(ds, cfg)
    typer.echo("✓ All done")


@app.command()
def status(config_path: Path = typer.Argument(..., help="Path to config YAML")):
    """Show annotation task status."""
    cfg = load_config(config_path)
    
    if not fo.dataset_exists(cfg.dataset_name):
        typer.echo(f"Dataset not found: {cfg.dataset_name}", err=True)
        raise typer.Exit(1)
    
    ds = fo.load_dataset(cfg.dataset_name)
    
    if cfg.anno_key not in ds.list_annotation_runs():
        typer.echo(f"No annotation run: {cfg.anno_key}")
        return
    
    results = ds.load_annotation_results(cfg.anno_key)
    results.print_status()


def main():
    app()


if __name__ == "__main__":
    main()
