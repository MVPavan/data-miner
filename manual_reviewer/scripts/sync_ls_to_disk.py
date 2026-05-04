"""Pull-based LS annotation backup to disk (cron-driven companion to
the push-based webhook handler).

Runs idempotently against an LS project: fetches the current annotation
state, diffs against ``.ls_backup/project_<id>/annotations/``, and feeds
synthesized events into the same :func:`process_event` pipeline the
webhook handler uses. The on-disk schema is identical regardless of
source, so a future webhook switchover is a no-op.

Diff plan per run:

  * Fetch ``GET /api/projects/<id>/tasks?fields=all`` (paginated).
    LS returns each task with its full ``annotations`` list inline.
  * For each LS annotation, compare against the on-disk snapshot:
      - missing on disk      → emit synthetic ``ANNOTATION_CREATED``
      - on disk but body diff → emit synthetic ``ANNOTATION_UPDATED``
      - identical body        → no-op (skip; nothing to write)
  * For each on-disk annotation NOT seen in LS → emit synthetic
    ``ANNOTATIONS_DELETED`` (single-id payload, matches LS shape).

Re-runs are cheap: when nothing changed, the run still appends nothing
to ``events.jsonl`` and writes no snapshots.

Schedule via cron::

    */5 * * * * cd /media/data_2/vlm/code/data_miner && \\
        LS_TOKEN="$LS_TOKEN" \\
        .venv/bin/python -m manual_reviewer.scripts.sync_ls_to_disk \\
            --ls-url http://localhost:8080 --project 3 \\
            >> /tmp/ls_sync.log 2>&1

5-minute cadence is a safe default for a single reviewer; the worst-
case data loss between sync runs is the run interval.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from manual_reviewer.ml_backend.lswebhook import (
    default_backup_dir,
    process_event,
)

logger = logging.getLogger("manual_reviewer.sync_ls_to_disk")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    token = args.ls_token or os.environ.get("LS_TOKEN")
    if not token:
        logger.error("--ls-token or LS_TOKEN env var required")
        return 2

    backup_dir = args.backup_dir or default_backup_dir()
    base_url = args.ls_url.rstrip("/")
    headers = {"Authorization": f"Token {token}"}

    project_id = int(args.project)
    project_dir = backup_dir / f"project_{project_id}"

    try:
        live = _fetch_live_annotations(
            base_url, headers, project_id, timeout=args.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch from LS failed: %s", exc)
        return 3

    on_disk = _read_disk_snapshots(project_dir)

    created = 0
    updated = 0
    deleted = 0
    unchanged = 0

    seen_ids: set[int] = set()
    for ann in live:
        aid = int(ann["id"])
        seen_ids.add(aid)
        prev = on_disk.get(aid)
        if prev is None:
            process_event(
                {"action": "ANNOTATION_CREATED",
                 "project": {"id": project_id},
                 "annotation": ann},
                backup_dir=backup_dir,
            )
            created += 1
        elif _normalized(prev) != _normalized(ann):
            process_event(
                {"action": "ANNOTATION_UPDATED",
                 "project": {"id": project_id},
                 "annotation": ann},
                backup_dir=backup_dir,
            )
            updated += 1
        else:
            unchanged += 1

    # Anything on disk but not on LS = deleted upstream.
    for aid in sorted(on_disk.keys() - seen_ids):
        process_event(
            {"action": "ANNOTATIONS_DELETED",
             "project": {"id": project_id},
             "annotations": [{"id": aid}]},
            backup_dir=backup_dir,
        )
        deleted += 1

    logger.info(
        "sync complete project=%d created=%d updated=%d deleted=%d unchanged=%d",
        project_id, created, updated, deleted, unchanged,
    )
    return 0


def _fetch_live_annotations(
    base_url: str,
    headers: dict[str, str],
    project_id: int,
    *,
    timeout: float,
    page_size: int = 200,
) -> list[dict[str, Any]]:
    """Fetch every annotation on every task in the project, flattened."""
    import httpx

    out: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout, headers=headers) as client:
        page = 1
        while True:
            url = f"{base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": page_size, "fields": "all"}
            resp = client.get(url, params=params)
            # LS returns 404 (not an empty list) when paginating past the
            # last page on totals divisible by page_size — treat as EOF.
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            tasks = data if isinstance(data, list) else (
                data.get("tasks") or data.get("results") or []
            )
            if not tasks:
                break
            for task in tasks:
                anns = task.get("annotations") or []
                for ann in anns:
                    if not isinstance(ann, dict) or "id" not in ann:
                        continue
                    # Normalize the task field to int if LS embedded a dict.
                    if "task" not in ann:
                        ann["task"] = task.get("id")
                    out.append(ann)
            if len(tasks) < page_size:
                break
            page += 1
    return out


def _read_disk_snapshots(project_dir: Path) -> dict[int, dict[str, Any]]:
    """Load all live annotation snapshots (skipping deleted/) keyed by id."""
    out: dict[int, dict[str, Any]] = {}
    ann_dir = project_dir / "annotations"
    if not ann_dir.exists():
        return out
    for path in ann_dir.glob("*.json"):
        try:
            ann = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(ann, dict) and "id" in ann:
            try:
                out[int(ann["id"])] = ann
            except (TypeError, ValueError):
                continue
    return out


def _normalized(ann: dict[str, Any]) -> str:
    """Stable JSON key-sort for body comparison.

    LS sometimes adds/changes ``updated_at`` server-side without a real
    edit (e.g. when serialization differs between requests). We strip
    those volatile fields so the diff only fires on substantive changes.
    """
    volatile = {"updated_at", "draft_created_at", "lead_time"}
    body = {k: v for k, v in ann.items() if k not in volatile}
    return json.dumps(body, sort_keys=True, ensure_ascii=False)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ls-url", default="http://localhost:8080",
                   help="Label Studio base URL")
    p.add_argument("--ls-token", default=None,
                   help="LS API token (falls back to LS_TOKEN env var)")
    p.add_argument("--project", required=True, type=int,
                   help="LS project ID to sync")
    p.add_argument("--backup-dir", default=None, type=Path,
                   help="root for .ls_backup tree; defaults to "
                        "manual_reviewer/.ls_backup or LS_BACKUP_DIR env")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="HTTP timeout in seconds (default 30)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
