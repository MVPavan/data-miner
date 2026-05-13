"""Round-trip CVAT annotations back into pipeline.db as Stage.HUMAN_REVIEW.

Replaces manual_reviewer/scripts/export_to_aa_v4.py for the CVAT path.
Same DB contract — same `HumanReviewResult` Pydantic, same trace append,
same YOLO label rewrite. Only the source format changes (CVAT Datumaro
JSON instead of LS REST exports).

Why Datumaro and not COCO/CVAT-XML:
  - Datumaro 1.0 is JSON-only, self-describing (categories + items + media).
  - Carries CVAT-specific attributes verbatim (assignee, source tag, frame
    state) which COCO drops.
  - One file per task; trivial streaming parse.

Algorithm:
  1. For each task in --cvat-project:
       a. GET /api/tasks/{id}/dataset?format=Datumaro%201.0 → zip → extract
          annotations/default.json.
       b. Read job metadata (assignee, state, updated_date) via SDK.
  2. For each image in the Datumaro file:
       a. Map CVAT label name → aa_v4 class name (1:1 if seed_tasks_from_yolo
          set them up correctly).
       b. Build HumanCorrection rows. source = "edited" / "added" / "relabeled"
          inferred by diffing against the YOLO pre-seed (read from pipeline.db
          proposals or from the dataset's yolo/labels/<stem>.txt).
       c. Build HumanReviewResult with:
            - reviewer_id    = job.assignee.email
            - reviewed_at    = job.updated_date (epoch)
            - frame_state    = "clean" unless reviewer set a "needs_more_review"
                              tag attribute on the frame
            - ml_modes_used  = []  (no smart tools in this CVAT pass)
            - duration_seconds = best-effort from CVAT annotation timestamps
       d. CheckpointDB.save_stage(image_id, Stage.HUMAN_REVIEW, result, config_hash)
       e. Append to traces/{image_id}.json.
       f. Rewrite labels/{image_id}.txt from corrected boxes.
  3. --since <iso> filters to tasks updated after that timestamp (idempotent
     re-exports skip unchanged work).

Offline usage:
  python -m manual_reviewer_cvat.scripts.export_to_aa_v4 \
    --datumaro-json /tmp/cvat_task/annotations/default.json \
    --pipeline-db /tmp/datatang_review/pipeline.db \
    --reviewer-id reviewer@example.com \
    --reviewed-at 2026-05-04T12:00:00Z

Future live CVAT usage:
    python -m manual_reviewer_cvat.scripts.export_to_aa_v4 \\
        --cvat-url http://127.0.0.1:8081 \\
        --cvat-user admin --cvat-pass <pw> \\
        --cvat-project 1 \\
        --pipeline-db /tmp/datatang_review/pipeline.db \\
        --since 2026-05-04T00:00:00
Live CVAT fetching is not implemented yet. Offline Datumaro JSON writeback is
implemented so exported tasks can already round-trip through the shared
exchange model into ``Stage.HUMAN_REVIEW``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from data_miner.annotation_io import ReviewExchangeResult
from manual_reviewer.pipeline_io import write_human_review
from manual_reviewer_cvat.pipeline_io import parse_datumaro_review_results

logger = logging.getLogger("manual_reviewer_cvat.export_to_aa_v4")


def main(argv: list[str] | None = None) -> int:
    """Write CVAT review exports into aa_v4 ``Stage.HUMAN_REVIEW`` rows."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db_path = args.pipeline_db.resolve()
    if not db_path.exists():
        logger.error("pipeline.db not found: %s", db_path)
        return 2

    if args.cvat_url:
        logger.error("live CVAT API export is not implemented yet; use --datumaro-json")
        return 2

    try:
        exchange_results = _load_datumaro_results(args)
    except Exception:  # noqa: BLE001
        logger.exception("failed to parse Datumaro export: %s", args.datumaro_json)
        return 1

    if not exchange_results:
        logger.warning("no Datumaro review results to process")
        return 1

    written = 0
    errors = 0
    for exchange_result in exchange_results:
        human_review = exchange_result.to_human_review_result()
        if args.dry_run:
            logger.info(
                "[dry-run] would write human_review for image_id=%s boxes=%d deletions=%d",
                human_review.image_id,
                len(human_review.corrections),
                len(human_review.deletions),
            )
            written += 1
            continue
        try:
            write_human_review(
                db_path,
                human_review,
                config_hash=args.config_hash or "",
            )
        except Exception:  # noqa: BLE001
            logger.exception("write_human_review failed for %s", human_review.image_id)
            errors += 1
            continue
        written += 1

    logger.info("wrote %d CVAT human_review rows (errors=%d)", written, errors)
    if errors:
        return 1
    return 0 if written else 1


def _load_datumaro_results(args: argparse.Namespace) -> list[ReviewExchangeResult]:
    """Load local Datumaro JSON and parse exchange review results."""
    document = json.loads(args.datumaro_json.read_text(encoding="utf-8"))
    reviewed_at = args.reviewed_at if args.reviewed_at is not None else time.time()
    return parse_datumaro_review_results(
        document,
        reviewer_id=args.reviewer_id,
        reviewed_at=reviewed_at,
        source_task_id=args.source_task_id,
        source_job_id=args.source_job_id,
        default_frame_state=args.default_frame_state,
        class_list_version=args.class_list_version,
    )


def _parse_timestamp(value: str) -> float:
    """Parse a Unix timestamp or ISO-8601 timestamp for CLI arguments."""
    try:
        return float(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected a Unix timestamp or ISO-8601 datetime"
            ) from exc


def _require_live_cvat_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate arguments needed for the future live CVAT source."""
    if args.cvat_url and (
        not args.cvat_user or not args.cvat_pass or args.cvat_project is None
    ):
        p.error("--cvat-url requires --cvat-user, --cvat-pass, and --cvat-project")


def _require_datumaro_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate arguments needed for local Datumaro JSON source."""
    if args.datumaro_json and not args.reviewer_id:
        p.error("--datumaro-json requires --reviewer-id")
    if args.datumaro_json and args.since is not None:
        p.error("--since is only supported for future live CVAT API export")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--datumaro-json", type=Path, default=None)
    source.add_argument("--cvat-url", default=None)

    p.add_argument("--pipeline-db", type=Path, required=True)
    p.add_argument("--reviewer-id", default=None)
    p.add_argument("--reviewed-at", type=_parse_timestamp, default=None)
    p.add_argument("--source-task-id", default=None)
    p.add_argument("--source-job-id", default=None)
    p.add_argument(
        "--default-frame-state",
        choices=["clean", "needs_more_review", "ambiguous_skip"],
        default="clean",
    )
    p.add_argument("--class-list-version", default=None)
    p.add_argument("--config-hash", default="")
    p.add_argument("--cvat-user", default=None)
    p.add_argument("--cvat-pass", default=None)
    p.add_argument("--cvat-project", type=int, default=None)
    p.add_argument("--since", help="ISO-8601 timestamp; only export tasks updated after")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    _require_datumaro_args(p, args)
    _require_live_cvat_args(p, args)
    return args


if __name__ == "__main__":
    sys.exit(main())
