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

Usage:
    python -m manual_reviewer_cvat.scripts.export_to_aa_v4 \\
        --cvat-url http://127.0.0.1:8081 \\
        --cvat-user admin --cvat-pass <pw> \\
        --cvat-project 1 \\
        --pipeline-db /tmp/datatang_review/pipeline.db \\
        --since 2026-05-04T00:00:00

NOT YET IMPLEMENTED — stub.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Stub — see README.md step 8 and the plan in this docstring.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--cvat-url", required=True)
    p.add_argument("--cvat-user", required=True)
    p.add_argument("--cvat-pass", required=True)
    p.add_argument("--cvat-project", type=int, required=True)
    p.add_argument("--pipeline-db", required=True)
    p.add_argument("--since", help="ISO-8601 timestamp; only export tasks updated after")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
