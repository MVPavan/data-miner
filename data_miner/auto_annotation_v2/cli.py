"""CLI entry point for auto_annotation_v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .log_utils import configure_logging, get_logger
from .pipeline import AutoAnnotationPipelineV2

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
        logger.error("No images provided. Use --image or --image-dir.")
        sys.exit(1)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto Annotation V2 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom YAML config")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images")
    parser.add_argument(
        "--output-dir", type=str, default="output/auto_annotation_v2",
        help="Output directory (default: output/auto_annotation_v2)",
    )
    parser.add_argument("--log-level", type=str, default=None, help="Logging level")

    args, unknown = parser.parse_known_args(argv)

    configure_logging(args.log_level)

    # Unknown args become OmegaConf overrides (e.g. vlm.temperature=0.1)
    overrides = [arg for arg in unknown if "=" in arg]

    config = load_config(args.config, overrides=overrides or None)
    logger.info("Config loaded: %d classes, %d detection models", len(config.classes), len(config.detection_models))

    images = _collect_images(args.image, args.image_dir)
    logger.info("Processing %d images", len(images))

    output_dir = Path(args.output_dir)
    pipeline = AutoAnnotationPipelineV2(config, output_dir)

    results = pipeline.run_batch_sync(images)

    total_accepted = sum(len(r.accepted) for r in results)
    total_rejected = sum(len(r.rejected) for r in results)
    total_review = sum(len(r.human_review) for r in results)
    total_partial = sum(1 for r in results if r.partial)

    logger.info(
        "Done: %d images, %d accepted, %d rejected, %d review, %d partial",
        len(results), total_accepted, total_rejected, total_review, total_partial,
    )


if __name__ == "__main__":
    main()
