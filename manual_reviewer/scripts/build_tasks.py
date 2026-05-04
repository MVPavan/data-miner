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
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from manual_reviewer.pipeline_io.clip_id import clip_prefix as _clip_prefix
from manual_reviewer.pipeline_io import (
    build_task,
    iter_survivor_images,
    read_image_payload,
    read_job_info,
)


def _stable_assignee(clip: str, assignees: list[str]) -> str:
    """SHA1-based deterministic assignment so re-runs are stable per clip.

    Whole clips go to one user — this keeps smart_track useful within
    each user's slice (all sibling frames available) and the per-user
    smart_track containment filter then prevents cross-user leakage.
    """
    h = int(hashlib.sha1(clip.encode("utf-8")).hexdigest(), 16)
    return assignees[h % len(assignees)]


def _parse_assignees(raw: str | None) -> list[str]:
    if not raw:
        return []
    out = [u.strip() for u in raw.split(",") if u.strip()]
    if len(out) != len(set(out)):
        raise ValueError(f"--assignees must be unique: got {out}")
    return out


def assign_by_frame_count_rr(
    clip_counts: dict[str, int], assignees: list[str],
) -> dict[str, str]:
    """Round-robin assign clips to ``assignees`` in descending frame-count order.

    The first user listed in ``assignees`` gets the largest clip, the
    second gets the next largest, etc., wrapping around. With a long
    tail of singleton clips the load evens out across users; the lead
    reviewer (typically ``assignees[0]``) ends up with the most
    valuable single clip for smart_track propagation.

    Returns a ``clip → assignee`` map. Caller stamps it onto each task.
    """
    # Descending by frame count; alphabetical on tie for reproducibility.
    ordered = sorted(clip_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {clip: assignees[i % len(assignees)] for i, (clip, _) in enumerate(ordered)}

logger = logging.getLogger("manual_reviewer.build_tasks")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = args.db.resolve()
    if not db_path.exists():
        logger.error("pipeline.db not found: %s", db_path)
        return 2

    try:
        assignees = _parse_assignees(args.assignees)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if assignees:
        logger.info("clip-level assignment across %d users: %s", len(assignees), assignees)

    job_info = read_job_info(db_path) or {}
    job_id = job_info.get("job_id", "")
    logger.info("job_id=%s db=%s", job_id, db_path)

    traces_dir = args.traces_dir.resolve() if args.traces_dir else None

    tasks: list[dict[str, Any]] = []
    skipped_no_finalize = 0
    skipped_invalid = 0

    # Default: flat iteration (sql LIMIT in db_reader). With
    # --per-clip-limit, we widen the SQL fetch and round-robin pick N
    # per clip ourselves so the reviewer sees frames from many videos
    # instead of all-from-one (clips appear contiguously in the DB
    # because the pipeline ingested them sequentially).
    if args.per_clip_limit is not None:
        rows_iter = _select_per_clip(
            db_path,
            per_clip_limit=args.per_clip_limit,
            total_limit=args.limit,
        )
    else:
        rows_iter = iter_survivor_images(
            db_path, limit=args.limit, require_finalize=True,
        )

    for row in rows_iter:
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
        if task is None:
            skipped_invalid += 1
            continue
        if assignees and args.assignment_strategy == "hash":
            clip = _clip_prefix(image_id)
            task["data"]["assigned_to"] = _stable_assignee(clip, assignees)
        tasks.append(task)

    logger.info(
        "built %d tasks (skipped %d without finalize, %d invalid)",
        len(tasks),
        skipped_no_finalize,
        skipped_invalid,
    )
    if assignees and args.assignment_strategy == "frame-count-rr":
        clip_counts = Counter(_clip_prefix(t["data"]["image_id"]) for t in tasks)
        clip_to_user = assign_by_frame_count_rr(dict(clip_counts), assignees)
        for t in tasks:
            t["data"]["assigned_to"] = clip_to_user[_clip_prefix(t["data"]["image_id"])]
    if assignees:
        counts = Counter(t["data"].get("assigned_to") for t in tasks)
        summary = ", ".join(f"{u}={counts.get(u, 0)}" for u in assignees)
        logger.info("assignment counts: %s", summary)
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
            batch_size=args.ls_batch_size,
        )
        logger.info("posted %d tasks to LS project %s", posted, args.ls_project)

    return 0


def _redact_token(s: str, token: str | None) -> str:
    if not token or not s:
        return s
    return s.replace(token, "<redacted>")


def _post_to_ls(
    tasks: list[dict[str, Any]],
    *,
    base_url: str,
    token: str,
    project_id: int,
    skip_existing: bool,
    timeout: float,
    batch_size: int = 100,
) -> int:
    """POST tasks via Label Studio's bulk import endpoint.

    ``skip_existing`` does a per-image pre-flight to LS's task list filtered
    by ``data.image_id`` so re-runs of build_tasks are idempotent. Trades one
    extra HTTP roundtrip per task for safety; pass ``False`` when bulk
    importing into a clean project.

    Sends in chunks of ``batch_size`` with up to 3 retries (1s/2s/4s
    exponential backoff) on 5xx so a transient LS hiccup doesn't drop the
    whole push.
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
        existing = _fetch_existing_image_ids(
            httpx, base_url, headers, project_id, timeout, strict_existing=True
        )
        before = len(tasks)
        tasks = [t for t in tasks if t["data"].get("image_id") not in existing]
        logger.info("skip_existing: %d already imported, %d remain", before - len(tasks), len(tasks))
        if not tasks:
            return 0

    url = f"{base_url}/api/projects/{project_id}/import"
    posted = 0
    with httpx.Client(timeout=timeout, headers=headers) as client:
        for start in range(0, len(tasks), max(1, batch_size)):
            chunk = tasks[start:start + batch_size]
            chunk_ids = [t["data"].get("image_id") for t in chunk]
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    resp = client.post(url, json=chunk)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    resp = None
                if resp is not None and resp.status_code < 300:
                    posted += len(chunk)
                    last_exc = None
                    break
                if resp is not None and 400 <= resp.status_code < 500:
                    body = _redact_token(resp.text[:500], token)
                    raise RuntimeError(
                        f"LS import failed {resp.status_code}: {body}"
                    )
                if attempt < 2:
                    time.sleep(2 ** attempt)
            else:
                status = getattr(resp, "status_code", "n/a") if resp is not None else "n/a"
                body = _redact_token(getattr(resp, "text", "")[:500], token) if resp is not None else str(last_exc)
                logger.error(
                    "LS import chunk failed after 3 attempts (status=%s) image_ids=%s: %s",
                    status, chunk_ids, body,
                )
    return posted


def _fetch_existing_image_ids(
    httpx_module: Any,
    base_url: str,
    headers: dict[str, str],
    project_id: int,
    timeout: float,
    *,
    strict_existing: bool = False,
) -> set[str]:
    """Return image_ids already in the LS project.

    When ``strict_existing`` is True, raises on transport/HTTP failure so the
    caller knows idempotency cannot be guaranteed. Otherwise falls back to
    an empty set.
    """
    url = f"{base_url}/api/projects/{project_id}/tasks"
    out: set[str] = set()
    page = 1
    token = headers.get("Authorization", "").removeprefix("Token ").strip() or None
    with httpx_module.Client(timeout=timeout, headers=headers) as client:
        while True:
            resp = client.get(url, params={"page": page, "page_size": 200})
            if resp.status_code >= 300:
                body = _redact_token(resp.text[:200], token)
                msg = f"LS task list failed {resp.status_code}: {body}"
                if strict_existing:
                    raise RuntimeError(msg)
                logger.warning("%s", msg)
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


def _select_per_clip(
    db_path: Path,
    *,
    per_clip_limit: int,
    total_limit: int | None,
) -> list[dict[str, Any]]:
    """Round-robin pick at most ``per_clip_limit`` survivors per clip.

    Strategy:
      1. Walk every survivor with finalize (no SQL LIMIT) — natural DB
         order clusters by clip because ingest is per-video.
      2. Group by ``_clip_prefix``; track full per-clip frame counts so
         we can rank clips by size.
      3. Sort clips by **descending frame count** (ties broken
         alphabetically). Real multi-frame videos dominate the result;
         loose single-image entries get pushed to the tail.
      4. Round-robin the top clips: row 0 of every clip first, then
         row 1, etc., until ``total_limit``.
    """
    by_clip: dict[str, list[dict[str, Any]]] = {}
    full_counts: dict[str, int] = {}
    for row in iter_survivor_images(db_path, limit=None, require_finalize=True):
        clip = _clip_prefix(row["image_id"])
        full_counts[clip] = full_counts.get(clip, 0) + 1
        bucket = by_clip.setdefault(clip, [])
        if len(bucket) < per_clip_limit:
            bucket.append(row)

    # Bigger clips first; alphabetic on tie for reproducibility.
    clips = sorted(by_clip, key=lambda c: (-full_counts[c], c))

    # Cap to the smallest clip set that can still hit total_limit at
    # per_clip_limit depth. Without this cap, the first round-robin pass
    # (one frame per clip) would fill total_limit before any clip got a
    # second frame — defeating the "frames per video" intent.
    if total_limit is not None and per_clip_limit > 0:
        max_clips_needed = -(-total_limit // per_clip_limit)  # ceil div
        clips = clips[:max_clips_needed]

    out: list[dict[str, Any]] = []
    for round_idx in range(per_clip_limit):
        for clip in clips:
            bucket = by_clip[clip]
            if round_idx < len(bucket):
                out.append(bucket[round_idx])
                if total_limit is not None and len(out) >= total_limit:
                    return out
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
    p.add_argument(
        "--ls-token",
        default=os.environ.get("LS_TOKEN"),
        help="LS API token; defaults to $LS_TOKEN env var (preferred — avoids exposure in `ps`)",
    )
    p.add_argument("--ls-project", type=int, default=None, help="LS project id")
    p.add_argument("--ls-timeout", type=float, default=30.0)
    p.add_argument("--ls-batch-size", type=int, default=100,
                   help="Tasks per LS bulk-import POST; chunked to bound retry blast radius")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tasks whose image_id already exists in the LS project")
    p.add_argument(
        "--per-clip-limit",
        type=int,
        default=None,
        help="Spread picks across clips: take at most N frames per clip "
             "(round-robin) until --limit is reached. Without this flag, "
             "tasks are pulled in DB insertion order which often clusters "
             "into one clip. Clip prefix = image_id with the trailing "
             "'_f<digits>' suffix stripped.",
    )
    p.add_argument(
        "--assignees",
        default=None,
        help="Comma-separated reviewer names. Sets data.assigned_to on "
             "every task. Whole clips go to one user so smart_track "
             "propagation stays useful inside each user's slice. "
             "Example: --assignees pavan,sree,raj,sathish,deepak",
    )
    p.add_argument(
        "--assignment-strategy",
        choices=["hash", "frame-count-rr"],
        default="hash",
        help="hash (default): SHA1 of clip prefix → stable round-robin "
             "across runs. frame-count-rr: clips sorted by descending "
             "frame count then round-robined; the first listed assignee "
             "gets the largest clip — useful when the lead reviewer "
             "should own the highest-leverage video.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
