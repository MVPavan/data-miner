"""CLI for the auto_annotation_v4 pipeline.

The CLI takes a user config YAML and merges it over the packaged defaults
(split across ``servers.yaml``, ``class_config.yaml``, ``database.yaml``,
``runtime.yaml``, ``default.yaml``).  All previous CLI flags
(``--image-dir``, ``--image``, ``--job-id``, ``--log-level``) now live
under the ``runtime:`` section of the YAML.  Ad-hoc overrides can still be
passed as OmegaConf dotlist arguments.

Usage:
    # Minimum -- user provides input via their YAML
    python -m data_miner.auto_annotation_v4 --config my_job.yaml

    # Same, plus one-off override
    python -m data_miner.auto_annotation_v4 --config my_job.yaml \\
        runtime.log_level=DEBUG workers.detect_per_model=4

    # Run with defaults only (rare -- requires image_dir/image_paths override)
    python -m data_miner.auto_annotation_v4 runtime.image_dir=/data/images
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse CLI arguments and launch the v4 annotation pipeline.

    Accepts a ``--config`` path to a user YAML and any number of OmegaConf
    dotlist overrides.  At least one of ``--config`` or an override must be
    provided.
    """
    parser = argparse.ArgumentParser(
        description="Auto Annotation V4 Pipeline",
        prog="python -m data_miner.auto_annotation_v4",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to user YAML config. Deep-merged over the packaged "
            "base configs (servers.yaml, class_config.yaml, database.yaml, "
            "runtime.yaml, default.yaml). All runtime options (image_dir, "
            "job_id, log_level) go under a `runtime:` section."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "OmegaConf dotlist overrides, e.g. "
            "`runtime.log_level=DEBUG workers.detect_per_model=4`"
        ),
    )

    args = parser.parse_args()

    if args.config is None and not args.overrides:
        parser.error(
            "either --config <path> or at least one dotlist override "
            "(e.g. runtime.image_dir=/path/to/images) is required"
        )

    from .pipeline import run_pipeline

    run_pipeline(
        user_config=args.config,
        overrides=args.overrides or None,
    )


if __name__ == "__main__":
    main()
