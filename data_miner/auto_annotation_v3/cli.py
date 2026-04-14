"""CLI for auto_annotation_v3 pipeline."""

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto Annotation V3 Pipeline",
        prog="python -m data_miner.auto_annotation_v3",
    )

    # Input — exactly one of --image-dir or --image must be supplied
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image-dir",
        type=str,
        help="Directory of images to process",
    )
    input_group.add_argument(
        "--image",
        type=str,
        nargs="+",
        help="Individual image path(s)",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML (shallow-merged over defaults)",
    )

    # Job identity
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID (auto-generated if not provided)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        image_dir=args.image_dir,
        image_paths=args.image,
        job_id=args.job_id,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
