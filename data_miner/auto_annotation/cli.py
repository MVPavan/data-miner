from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .log_utils import configure_logging
from .pipeline import AutoAnnotationPipeline
from .utils import save_result

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the auto-annotation pipeline.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser.parse_args()


def iter_images(image: str | None, image_dir: str | None) -> list[Path]:
    if image:
        return [Path(image)]
    if not image_dir:
        raise ValueError("Provide --image or --image-dir")
    root = Path(image_dir)
    return sorted(
        path for path in root.iterdir() if path.suffix.lower() in IMG_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config, args.overrides)
    pipeline = AutoAnnotationPipeline(config)
    if args.output_dir is None:
        args.output_dir = "aa_output"
    output_dir = Path(args.output_dir)
    class_names = [class_pack.name for class_pack in config.classes]
    for image_path in iter_images(args.image, args.image_dir):
        result = pipeline.run_image(image_path)
        save_result(result, class_names, output_dir, config.output)


if __name__ == "__main__":
    main()
