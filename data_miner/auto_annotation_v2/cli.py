"""CLI entry point for auto_annotation_v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .contracts import StageName
from .log_utils import configure_logging, get_logger
from .pipeline import STAGE_ORDER, AutoAnnotationPipelineV2

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _collect_images(image: str | None, image_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    if image:
        paths.append(Path(image))
    if image_dir:
        d = Path(image_dir)
        if not d.is_dir():
            logger.error("Image directory does not exist: %s", d)
            sys.exit(1)
        paths.extend(
            sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
        )
    if not paths:
        logger.error(
            "No images provided. Use --image or --image-dir, or set runtime.image / runtime.image_dir in YAML."
        )
        sys.exit(1)
    return paths


def _build_runtime_overrides(args: argparse.Namespace) -> list[str]:
    """Build OmegaConf dotlist overrides for non-None CLI runtime args."""
    mapping = {
        "output_dir": "runtime.output_dir",
        "log_level": "runtime.log_level",
        "image": "runtime.image",
        "image_dir": "runtime.image_dir",
        "force_redo": "runtime.force_redo",
    }
    overrides: list[str] = []
    for attr, dotpath in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides.append(f"{dotpath}={val}")
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto Annotation V2 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to custom YAML config"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image (overrides runtime.image in YAML)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory of images (overrides runtime.image_dir in YAML)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides runtime.output_dir in YAML)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (overrides runtime.log_level in YAML)",
    )
    parser.add_argument(
        "--force-redo",
        type=str,
        default=None,
        help=(
            "Force re-run stages even if checkpoints exist. "
            "Comma-separated stage names or 'all'. "
            "Downstream stages are also cleared. "
            "e.g. --force-redo filtering  or  --force-redo vlm_reasoning,vlm_refinement  or  --force-redo all"
        ),
    )

    args, unknown = parser.parse_known_args(argv)

    # Build overrides: CLI runtime args + extra dotpath overrides
    runtime_overrides = _build_runtime_overrides(args)
    extra_overrides = [arg for arg in unknown if "=" in arg]
    all_overrides = runtime_overrides + extra_overrides

    config = load_config(args.config, overrides=all_overrides or None)

    # Resolve runtime values from config (YAML defaults merged with CLI overrides)
    rt = config.runtime
    configure_logging(rt.log_level, output_dir=rt.output_dir)

    logger.info(
        "Config loaded: %d classes, %d detection models",
        len(config.classes),
        len(config.detection_models),
    )

    images = _collect_images(rt.image, rt.image_dir)
    logger.info("Processing %d images", len(images))

    output_dir = Path(rt.output_dir)

    # Parse force_redo
    force_redo_stages: set[StageName] = set()
    if rt.force_redo:
        raw = rt.force_redo.strip().lower()
        if raw == "all":
            force_redo_stages = set(STAGE_ORDER)
        else:
            valid_names = {s.value for s in StageName}
            for name in raw.split(","):
                name = name.strip()
                if name not in valid_names:
                    logger.error(
                        "Unknown stage '%s'. Valid: %s",
                        name,
                        ", ".join(sorted(valid_names)),
                    )
                    sys.exit(1)
                force_redo_stages.add(StageName(name))
        logger.info("Force redo stages: %s", [s.value for s in force_redo_stages])

    pipeline = AutoAnnotationPipelineV2(
        config, output_dir, force_redo_stages=force_redo_stages
    )

    results = pipeline.run_batch_sync(images)

    total_accepted = sum(len(r.accepted) for r in results)
    total_rejected = sum(len(r.rejected) for r in results)
    total_review = sum(len(r.human_review) for r in results)
    total_partial = sum(1 for r in results if r.partial)

    logger.info(
        "Done: %d images, %d accepted, %d rejected, %d review, %d partial",
        len(results),
        total_accepted,
        total_rejected,
        total_review,
        total_partial,
    )


if __name__ == "__main__":
    main()
