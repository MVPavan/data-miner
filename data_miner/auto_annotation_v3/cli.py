"""CLI for the auto_annotation_v3 pipeline.

The CLI takes a user config YAML and merges it over the packaged defaults at
``configs/default.yaml``. All previous CLI flags (``--image-dir``, ``--image``,
``--job-id``, ``--log-level``) now live under the ``runtime:`` section of the
YAML. Ad-hoc overrides can still be passed as OmegaConf dotlist arguments.

Usage:
    # Minimum — user provides input via their YAML
    python -m data_miner.auto_annotation_v3 --config my_job.yaml

    # Same, plus one-off override
    python -m data_miner.auto_annotation_v3 --config my_job.yaml \\
        runtime.log_level=DEBUG workers.detect_count=2

    # Run with defaults only (rare — requires image_dir/image_paths override)
    python -m data_miner.auto_annotation_v3 runtime.image_dir=/data/images
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto Annotation V3 Pipeline",
        prog="python -m data_miner.auto_annotation_v3",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to user YAML config. Deep-merged over the packaged "
            "configs/default.yaml. All runtime options (image_dir, job_id, "
            "log_level) go under a `runtime:` section."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "OmegaConf dotlist overrides, e.g. "
            "`runtime.log_level=DEBUG workers.detect_count=2`"
        ),
    )

    args = parser.parse_args()

    if args.config is None and not args.overrides:
        parser.error(
            "either --config <path> or at least one dotlist override "
            "(e.g. runtime.image_dir=/path/to/images) is required"
        )

    run_pipeline(
        user_config=args.config,
        overrides=args.overrides or None,
    )


if __name__ == "__main__":
    main()
