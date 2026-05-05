"""Create one CVAT Task per video clip, pre-seeded with YOLO predictions.

Mirrors manual_reviewer/scripts/build_tasks_from_yolo.py but targets CVAT.
Key differences from the LS version:

  - One Task per CLIP (not one per image). `segment_size` ≥ clip frame count
    so one Task = one Job. Job.assignee = the reviewer for that clip.
  - YOLO predictions are uploaded as the task's INITIAL annotations
    (CVAT lacks LS's "predictions" channel — predictions and annotations
    are the same field; reviewer edits in place).
  - class_id mapping: aa_v4's YOLO export uses GLOBAL class_registry IDs
    (e.g. head=35, not classes.txt position). Pass --aav4-config so we
    decode IDs the same way build_tasks_from_yolo.py does.
  - Image hosting: --image-mount-mode shared-folder (recommended) references
    images by path under CVAT's /home/django/share volume; --image-mount-mode
    upload pushes bytes per-task (slower, doubles disk).

Usage:
    python -m manual_reviewer_cvat.migrations_from_LS.scripts.seed_tasks_from_yolo \\
        --cvat-url http://127.0.0.1:8081 \\
        --admin-user admin --admin-pass <pw> \\
        --project-id 1 \\
        --dataset output/dataset_selection/datatang_diverse_1000 \\
        --aav4-config data_miner/auto_annotation_v4/configs/default.yaml \\
        --reviewers pavan,sree,raj,sathish,deepak \\
        --strategy frame-count-rr \\
        --image-mount-mode shared-folder

NOT YET IMPLEMENTED — stub.

Plan (matches README step 4):
  1. Read classes.txt and class_registry from --aav4-config; build id_to_name.
  2. Group YOLO label files by clip prefix (re-use clip_prefix from
     manual_reviewer/pipeline_io/clip_id.py) — read clips.txt as the canonical
     clip list.
  3. Sort clips by descending frame count, round-robin assign to --reviewers
     so pavan gets the longest clip, sree the second-longest, etc.
     (Same algorithm as manual_reviewer.scripts.build_tasks.assign_by_frame_count_rr.)
  4. Per clip: create CVAT task via cvat_sdk Task.create + Task.upload_data.
     - shared-folder mode: pass server_files=[paths under /home/django/share]
     - upload mode: pass client_files=[local paths]
     - segment_size = len(clip_frames)
     - assignee on the (single) job
  5. Build Datumaro JSON with the YOLO rows mapped to CVAT label IDs and POST
     to /api/tasks/{id}/annotations?format=Datumaro%201.0.
  6. Print a CSV summary: clip, frame_count, reviewer, task_id, status.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Stub — see README.md step 4 and the plan in this docstring.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--cvat-url", required=True)
    p.add_argument("--admin-user", required=True)
    p.add_argument("--admin-pass", required=True)
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--dataset", required=True, help="dataset root (containing yolo/, stem_to_path.json)")
    p.add_argument("--aav4-config", help="aa_v4 yaml config for class_registry mapping")
    p.add_argument("--reviewers", required=True)
    p.add_argument("--strategy", choices=["frame-count-rr", "hash"], default="frame-count-rr")
    p.add_argument("--image-mount-mode", choices=["shared-folder", "upload"], default="shared-folder")
    p.add_argument("--limit", type=int, help="cap clip count for smoke-testing")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
