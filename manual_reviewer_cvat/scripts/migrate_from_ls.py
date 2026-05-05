"""Resume in-flight LS work in CVAT.

The point of this script: reviewers should NOT have to redo any annotation
work they already submitted in Label Studio when we cut over to CVAT. We
read every completion in the LS project, translate to CVAT shapes, and
upload them onto the matching CVAT task — keyed on `image_id` (which is
present in both stacks because it's the canonical aa_v4 identifier).

Algorithm (matches README step 5):

  1. Read every LS task in --ls-project. For each task:
        image_id        from task.data.image_id
        completions[]   each is one (annotator_email, [region, ...], state)
        skipped         was-cancelled flag
        was_cancelled   reviewer hit Skip
  2. Build CVAT image_id → task_id map by GETting /api/tasks?project_id=...
     and inspecting each task's frame metadata (Task.get_frames_info).
  3. For each LS image_id:
        a. Find matching CVAT task_id. If none, log and skip.
        b. Translate LS RectangleLabels → CVAT shapes:
             - LS gives % of image; CVAT wants pixels.
             - Label name is preserved verbatim (label_id resolved against
               CVAT project labels).
             - Region IDs sanitised (no colons; LS bug we hit on task 348).
        c. POST /api/tasks/{id}/annotations with action=create. CVAT merges
           with anything already on the task (which should be the YOLO
           pre-seed from seed_tasks_from_yolo.py).
        d. PATCH the job: assignee = the LS annotator who took it last,
           state = "in progress" (NOT completed — let them re-submit so
           we get a fresh CVAT timestamp + reviewer in the audit trail).
  4. Emit a CSV report: image_id, ls_task_id, cvat_task_id,
     ls_annotator, cvat_assignee, regions_migrated, status.

--dry-run prints the report without POSTing anything. Run it first.

Edge cases handled:
  - LS task with multiple completions from different annotators: keep the
    latest by `updated_at`, log the others.
  - LS image_id present in CVAT under a different filename: matched on
    image_id (basename strip), not full path.
  - LS task with no completions: skipped silently (no work to migrate).
  - LS task in `was_cancelled` state: imports as empty annotation,
    Job marked `state=in progress` so the new reviewer can re-evaluate.
  - Image IDs containing colons: replaced with underscores in CVAT region
    IDs as a defensive measure (LSF-specific bug, but cheap to mirror).

Usage:
    python -m manual_reviewer_cvat.scripts.migrate_from_ls \\
        --ls-url http://127.0.0.1:8080 \\
        --ls-token "$LS_TOKEN" \\
        --ls-project 9 \\
        --cvat-url http://127.0.0.1:8081 \\
        --cvat-user admin --cvat-pass <pw> \\
        --cvat-project 1 \\
        --dry-run

NOT YET IMPLEMENTED — stub.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "Stub — see README.md step 5 and docs/migration_from_ls.md for the full spec."
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ls-url", required=True)
    p.add_argument("--ls-token", required=True)
    p.add_argument("--ls-project", type=int, required=True)
    p.add_argument("--cvat-url", required=True)
    p.add_argument("--cvat-user", required=True)
    p.add_argument("--cvat-pass", required=True)
    p.add_argument("--cvat-project", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", default="-", help="CSV output path; '-' for stdout")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
