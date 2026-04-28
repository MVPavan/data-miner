"""Build Label Studio tasks from an aa_v4 ``pipeline.db``.

Two output modes:

- ``--out-file <path.json>``: write tasks as a JSON array to disk. Suitable
  for LS's ``Import → Upload Files`` UI or for inspection.
- ``--ls-url <url> --ls-token <token> --ls-project <id>``: POST tasks
  directly to the LS REST API. Tasks already imported (matched by
  ``data.image_id``) are skipped.

Usage::

    python -m manual_reviewer.scripts.build_tasks \\
        --db /jobs/run_42/pipeline.db \\
        --traces-dir /jobs/run_42/traces \\
        --out-file /tmp/tasks.json \\
        --limit 50

The image URL template defaults to LS's local-files-serving form. Override
with ``--image-url-template`` when serving images via S3/HTTPS instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from manual_reviewer.pipeline_io import (
    build_task,
    iter_survivor_images,
    read_image_payload,
    read_job_info,
)

logger = logging.getLogger("manual_reviewer.build_tasks")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = args.db.resolve()
    if not db_path.exists():
        logger.error("pipeline.db not found: %s", db_path)
        return 2

    job_info = read_job_info(db_path) or {}
    job_id = job_info.get("job_id", "")
    logger.info("job_id=%s db=%s", job_id, db_path)

    traces_dir = args.traces_dir.resolve() if args.traces_dir else None

    tasks: list[dict[str, Any]] = []
    skipped_no_finalize = 0
    for row in iter_survivor_images(db_path, limit=args.limit, require_finalize=True):
        image_id = row["image_id"]
        try:
            payload = read_image_payload(db_path, image_id, traces_dir=traces_dir)
        except KeyError:
            logger.warning("image_id missing from image_meta: %s", image_id)
            continue
        if "finalize" not in payload["stages"]:
            skipped_no_finalize += 1
            continue
        task = build_task(
            payload,
            image_url_template=args.image_url_template,
            job_id=job_id,
        )
        tasks.append(task)

    logger.info("built %d tasks (skipped %d without finalize)", len(tasks), skipped_no_finalize)
    if not tasks:
        logger.warning("no tasks to write/post")
        return 1

    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out_file)

    if args.ls_url:
        if not args.ls_token or args.ls_project is None:
            logger.error("--ls-url requires --ls-token and --ls-project")
            return 2
        posted = _post_to_ls(
            tasks,
            base_url=args.ls_url,
            token=args.ls_token,
            project_id=args.ls_project,
            skip_existing=args.skip_existing,
            timeout=args.ls_timeout,
        )
        logger.info("posted %d tasks to LS project %s", posted, args.ls_project)

    return 0


def _post_to_ls(
    tasks: list[dict[str, Any]],
    *,
    base_url: str,
    token: str,
    project_id: int,
    skip_existing: bool,
    timeout: float,
) -> int:
    """POST tasks via Label Studio's bulk import endpoint.

    ``skip_existing`` does a per-image pre-flight to LS's task list filtered
    by ``data.image_id`` so re-runs of build_tasks are idempotent. Trades one
    extra HTTP roundtrip per task for safety; pass ``False`` when bulk
    importing into a clean project.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required for --ls-url mode; install with `pip install httpx`."
        ) from exc

    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    base_url = base_url.rstrip("/")

    if skip_existing:
        existing = _fetch_existing_image_ids(httpx, base_url, headers, project_id, timeout)
        before = len(tasks)
        tasks = [t for t in tasks if t["data"].get("image_id") not in existing]
        logger.info("skip_existing: %d already imported, %d remain", before - len(tasks), len(tasks))
        if not tasks:
            return 0

    url = f"{base_url}/api/projects/{project_id}/import"
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.post(url, json=tasks)
        if resp.status_code >= 300:
            raise RuntimeError(f"LS import failed {resp.status_code}: {resp.text[:500]}")
    return len(tasks)


def _fetch_existing_image_ids(
    httpx_module: Any,
    base_url: str,
    headers: dict[str, str],
    project_id: int,
    timeout: float,
) -> set[str]:
    """Return image_ids already in the LS project. Best-effort, non-fatal."""
    url = f"{base_url}/api/projects/{project_id}/tasks"
    out: set[str] = set()
    page = 1
    with httpx_module.Client(timeout=timeout, headers=headers) as client:
        while True:
            resp = client.get(url, params={"page": page, "page_size": 200})
            if resp.status_code >= 300:
                logger.warning("LS task list failed %s: %s", resp.status_code, resp.text[:200])
                return out
            payload = resp.json() or {}
            tasks = payload if isinstance(payload, list) else payload.get("tasks") or []
            if not tasks:
                break
            for t in tasks:
                data = t.get("data") if isinstance(t, dict) else None
                if isinstance(data, dict) and data.get("image_id"):
                    out.add(data["image_id"])
            if len(tasks) < 200:
                break
            page += 1
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Label Studio tasks from aa_v4 pipeline.db")
    p.add_argument("--db", type=Path, required=True, help="Path to pipeline.db")
    p.add_argument("--traces-dir", type=Path, default=None,
                   help="Optional path to traces/ dir (for trace_excerpt in task data)")
    p.add_argument("--out-file", type=Path, default=None, help="Write tasks JSON to this path")
    p.add_argument("--limit", type=int, default=None, help="Cap number of tasks")
    p.add_argument(
        "--image-url-template",
        default="/data/local-files/?d={path}",
        help="URL template; {path} is the image_meta.image_path value",
    )
    p.add_argument("--ls-url", default=None, help="LS base URL, e.g. http://localhost:8080")
    p.add_argument("--ls-token", default=None, help="LS API token")
    p.add_argument("--ls-project", type=int, default=None, help="LS project id")
    p.add_argument("--ls-timeout", type=float, default=30.0)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tasks whose image_id already exists in the LS project")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
