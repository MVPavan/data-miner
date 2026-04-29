"""Cross-frame static-object propagation against an aa_v4 ``pipeline.db``.

For each frame group (clip_id by default), this script:
  1. Reads finalize annotations across the group.
  2. Clusters cross-frame detections of the same class at IoU ≥ threshold.
  3. For each cluster missing from some frame, queries SAM3-DART's /refine
     endpoint on that frame at the cluster's canonical box.
  4. If SAM3-DART confirms (mask_score ≥ accept_score AND seed_iou ≥
     accept_iou), writes a propagated detection to ``Stage.RECONCILE``.

Usage::

    python -m manual_reviewer.scripts.run_reconcile \\
        --db /jobs/run_42/pipeline.db \\
        --sam3-url http://localhost:3013/refine \\
        --grouping clip_id

Idempotent: re-running overwrites prior reconcile rows for the same images.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from data_miner.auto_annotation_v4.configs.contracts import FinalAnnotation

from manual_reviewer.pipeline_io import (
    iter_survivor_images,
    read_image_payload,
    write_reconcile_results,
)
from manual_reviewer.reconcile import (
    DEFAULT_CLIP_REGEX,
    DEFAULT_SAM3_1_REFINE_URL,
    DEFAULT_SAM3_DART_REFINE_URL,
    ImageContext,
    PropagationConfig,
    Sam3HttpClient,
    Sam3OneHttpClient,
    group_images,
    reconcile_group,
)

logger = logging.getLogger("run_reconcile")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-frame static-object propagation for aa_v4 pipeline.db",
    )
    parser.add_argument("--db", required=True, type=Path, help="Path to pipeline.db")
    parser.add_argument(
        "--backend",
        choices=["sam3_1", "sam3_dart"],
        default="sam3_1",
        help="Which SAM server to call (default: sam3_1, the manual_reviewer default)",
    )
    parser.add_argument(
        "--sam3-url",
        default=None,
        help=(
            "Override the SAM server URL. Defaults: "
            f"sam3_1 → {DEFAULT_SAM3_1_REFINE_URL}; "
            f"sam3_dart → {DEFAULT_SAM3_DART_REFINE_URL}."
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=["clip_id", "all", "per_image"],
        default="clip_id",
        help="How to group survivor frames into reconciliation cohorts",
    )
    parser.add_argument(
        "--clip-regex",
        default=None,
        help=f"Override the clip_id regex (default: {DEFAULT_CLIP_REGEX.pattern!r})",
    )
    parser.add_argument(
        "--cluster-iou",
        type=float,
        default=0.5,
        help="IoU threshold for cross-frame same-class clustering",
    )
    parser.add_argument(
        "--min-positive-frames",
        type=int,
        default=2,
        help="Cluster must appear in ≥ N frames before propagating",
    )
    parser.add_argument(
        "--accept-score",
        type=float,
        default=0.5,
        help="SAM3-DART mask score floor to accept propagation",
    )
    parser.add_argument(
        "--accept-iou",
        type=float,
        default=0.7,
        help="IoU(seed, refined) floor to accept propagation",
    )
    parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0.5,
        help="Threshold passed to SAM3-DART /refine",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for SAM3-DART calls (seconds)",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Process at most N survivor images (debugging)",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Persist reconcile rows even when no propagations occurred",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return parser


def _build_image_contexts(db_path: Path, *, limit: int | None) -> list[ImageContext]:
    """Pull every survivor image's finalize annotations into ImageContexts."""
    contexts: list[ImageContext] = []
    for record in iter_survivor_images(db_path, limit=limit, require_finalize=True):
        image_id = record["image_id"]
        image_path = record.get("image_path", "")
        payload = read_image_payload(db_path, image_id)
        finalize = (payload.get("stages") or {}).get("finalize") or {}
        anns_raw = finalize.get("final_annotations") or []
        annotations: list[FinalAnnotation] = []
        for raw in anns_raw:
            if not isinstance(raw, dict):
                continue
            try:
                annotations.append(FinalAnnotation.model_validate(raw))
            except Exception as exc:
                logger.debug("skipping malformed finalize annotation on %s: %s", image_id, exc)
        contexts.append(
            ImageContext(
                image_id=image_id,
                image_path=image_path,
                final_annotations=annotations,
            )
        )
    return contexts


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db.exists():
        logger.error("DB not found: %s", args.db)
        return 2

    contexts = _build_image_contexts(args.db, limit=args.limit_images)
    if not contexts:
        logger.warning("No survivor images with finalize stage found — nothing to do.")
        return 0

    clip_regex = re.compile(args.clip_regex) if args.clip_regex else None
    grouped = group_images(
        [(c.image_id, c.image_path) for c in contexts],
        strategy=args.grouping,
        clip_regex=clip_regex,
    )
    logger.info(
        "Grouping: strategy=%s -> %d groups across %d images",
        args.grouping,
        len(grouped),
        len(contexts),
    )

    by_id = {c.image_id: c for c in contexts}
    config = PropagationConfig(
        cluster_iou_threshold=args.cluster_iou,
        min_positive_frames=args.min_positive_frames,
        accept_score=args.accept_score,
        accept_iou=args.accept_iou,
        refine_threshold=args.refine_threshold,
    )

    if args.backend == "sam3_1":
        client = Sam3OneHttpClient(url=args.sam3_url, timeout=args.http_timeout)
    else:
        client = Sam3HttpClient(url=args.sam3_url, timeout=args.http_timeout)

    all_results = []
    started = time.time()
    propagated_count = 0
    rejected_count = 0

    failed_groups = 0
    for group_id, image_ids in grouped.items():
        members = [by_id[i] for i in image_ids if i in by_id]
        if len(members) < 2:
            logger.debug("group %s has only %d image(s); skipping", group_id, len(members))
            continue

        group_started = time.time()
        try:
            results = reconcile_group(
                group_id,
                members,
                client=client,
                config=config,
            )
        except Exception:
            # One bad group must not kill a multi-hour batch — partial
            # output (every group that ran cleanly so far) is still useful.
            logger.exception(
                "group=%s reconcile failed; continuing with remaining groups",
                group_id,
            )
            failed_groups += 1
            continue
        group_elapsed = (time.time() - group_started) * 1000.0
        for res in results.values():
            res = res.model_copy(update={"stage_timing_ms": group_elapsed})
            all_results.append(res)
            propagated_count += len(res.propagated)
            rejected_count += len(res.rejected)

        logger.info(
            "group=%s images=%d propagated=%d rejected=%d in %.0fms",
            group_id,
            len(members),
            sum(len(r.propagated) for r in results.values()),
            sum(len(r.rejected) for r in results.values()),
            group_elapsed,
        )

    written = write_reconcile_results(
        args.db,
        all_results,
        skip_empty=not args.keep_empty,
    )
    total_elapsed = time.time() - started
    logger.info(
        "Done: %d images touched (rows written=%d), %d propagated, %d rejected, "
        "%d group(s) failed, %.1fs",
        len(all_results),
        written,
        propagated_count,
        rejected_count,
        failed_groups,
        total_elapsed,
    )
    # Non-zero exit on partial failure so CI / wrapping cron jobs notice;
    # data already written for clean groups is preserved.
    return 0 if failed_groups == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
