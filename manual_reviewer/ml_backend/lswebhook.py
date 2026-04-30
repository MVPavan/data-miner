"""Push-backup of LS annotation events to disk.

Receives the LS webhook payload (configured per project at LS Project
Settings → Webhooks) and persists every event to a project-scoped
backup tree:

    .ls_backup/project_<id>/
      events.jsonl                          ← append-only audit log
      annotations/<ann_id>.json             ← latest live snapshot
      annotations/deleted/<ann_id>-<ts>.json ← preserved on delete
      tasks/<task_id>.json                  ← live aggregate per task

``events.jsonl`` is the source of truth — every webhook payload (with a
receipt timestamp) is appended on a single line, fsync'd. The snapshots
under ``annotations/`` and ``tasks/`` are convenience indexes derived
from the event stream; if they corrupt or drift, ``events.jsonl`` can
rebuild them.

Idempotency: re-delivery of the same event by LS (which retries on 5xx
or timeout) is safe — the event line is appended again (audit shows the
duplicate), the snapshot write is a no-op overwrite, and the aggregate
refresh re-derives the same state. Move-to-deleted is no-op when the
file is already gone.

The handler always returns 200 once ``events.jsonl`` has been fsync'd —
snapshot/aggregate write failures are logged but not raised, since the
event log is the recoverable record. This avoids forcing LS into a
retry loop when the cosmetic indexes have a transient issue.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default backup root: ``manual_reviewer/.ls_backup`` (gitignored). Override
# per-process via the ``LS_BACKUP_DIR`` env var or the ``backup_dir`` kwarg
# (tests pass tmp_path here).
def default_backup_dir() -> Path:
    env = os.environ.get("LS_BACKUP_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / ".ls_backup"


_HANDLED_ACTIONS = {
    "ANNOTATION_CREATED",      # singular create (one annotation)
    "ANNOTATION_UPDATED",      # singular update
    "ANNOTATIONS_CREATED",     # bulk create (LS 1.23 ships this for batch ops)
    "ANNOTATIONS_DELETED",     # delete (always plural in LS 1.23, even for one)
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _now_compact() -> str:
    """Filename-safe UTC timestamp with microseconds — used for the
    deleted-snapshot suffix so a fresh delete never collides with an
    older one for the same annotation_id."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")


def _atomic_write(path: Path, data: str) -> None:
    """Tmp-write + fsync + atomic replace. ``os.replace`` is atomic on
    POSIX as long as src and dst are on the same filesystem, which is
    always the case here (sibling tmpfile)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_event(project_dir: Path, payload: dict[str, Any]) -> None:
    """Append one JSONL line to events.jsonl, fsync.

    The wrapping object adds ``received_at`` so reading the log doesn't
    require the LS-supplied timestamp (which lives at varying paths
    depending on event type). The original payload is preserved
    untouched under ``payload``.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / "events.jsonl"
    line = json.dumps(
        {"received_at": _now_iso(), "payload": payload},
        ensure_ascii=False,
    ) + "\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _project_dir(backup_dir: Path, project_id: int) -> Path:
    return backup_dir / f"project_{project_id}"


def _annotation_path(project_dir: Path, annotation_id: int) -> Path:
    return project_dir / "annotations" / f"{annotation_id}.json"


def _deleted_path(project_dir: Path, annotation_id: int) -> Path:
    return project_dir / "annotations" / "deleted" / f"{annotation_id}-{_now_compact()}.json"


def _task_path(project_dir: Path, task_id: int) -> Path:
    return project_dir / "tasks" / f"{task_id}.json"


# ---------------------------------------------------------------------------
# Per-action handlers
# ---------------------------------------------------------------------------


def _write_snapshot(project_dir: Path, annotation: dict[str, Any]) -> None:
    aid = int(annotation["id"])
    _atomic_write(
        _annotation_path(project_dir, aid),
        json.dumps(annotation, ensure_ascii=False, indent=2),
    )


def _move_to_deleted(project_dir: Path, annotation_id: int) -> bool:
    """Move the live snapshot to ``deleted/`` with a timestamp suffix.

    Returns ``True`` if a file was moved, ``False`` if the snapshot was
    already gone (already-deleted or never created — both are idempotent
    no-ops).
    """
    src = _annotation_path(project_dir, annotation_id)
    if not src.exists():
        return False
    dst = _deleted_path(project_dir, annotation_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    return True


def _annotation_task_id(annotation: dict[str, Any]) -> int | None:
    """Extract the task id from an annotation payload — LS uses 'task'
    most commonly, sometimes 'task_id'."""
    for key in ("task", "task_id"):
        v = annotation.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            inner = v.get("id")
            if isinstance(inner, int):
                return inner
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def _refresh_task_aggregate(project_dir: Path, task_id: int) -> None:
    """Rewrite ``tasks/<task_id>.json`` with the live annotation set.

    Walks ``annotations/*.json`` (excluding the ``deleted/`` subdir,
    which has its own bucket) and filters by ``annotation["task"] ==
    task_id``. O(N_live) per call; fine for the volumes a single
    reviewer produces.
    """
    ann_dir = project_dir / "annotations"
    live: list[dict[str, Any]] = []
    if ann_dir.exists():
        for path in ann_dir.glob("*.json"):
            # ``deleted`` is a subdir, ``glob("*.json")`` skips it.
            try:
                ann = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "skipping unreadable snapshot %s: %s", path.name, exc
                )
                continue
            if _annotation_task_id(ann) == task_id:
                live.append(ann)

    aggregate = {
        "task_id": task_id,
        "refreshed_at": _now_iso(),
        "annotation_ids": sorted(int(a["id"]) for a in live if "id" in a),
        "annotations": live,
    }
    _atomic_write(
        _task_path(project_dir, task_id),
        json.dumps(aggregate, ensure_ascii=False, indent=2),
    )


# ---------------------------------------------------------------------------
# Project id extraction
# ---------------------------------------------------------------------------


def _extract_project_id(payload: dict[str, Any]) -> int | None:
    """LS payloads embed the project id at varying paths.

    For annotation events: ``payload.annotation.project`` (int) or
    ``payload.task.project`` (int) or top-level ``payload.project``
    (object or int). For task-bulk events: ``payload.project.id``.
    Walk the common positions in order and coerce to int.
    """
    candidates: list[Any] = []
    project = payload.get("project")
    candidates.append(project)
    ann = payload.get("annotation")
    if isinstance(ann, dict):
        candidates.append(ann.get("project"))
    task = payload.get("task")
    if isinstance(task, dict):
        candidates.append(task.get("project"))

    for c in candidates:
        if isinstance(c, int):
            return c
        if isinstance(c, dict):
            inner = c.get("id")
            if isinstance(inner, int):
                return inner
            if isinstance(inner, str) and inner.isdigit():
                return int(inner)
        if isinstance(c, str) and c.isdigit():
            return int(c)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def process_event(
    payload: dict[str, Any],
    *,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist one LS webhook payload to disk; return a summary dict.

    Always appends to ``events.jsonl`` first (the source of truth), then
    dispatches on ``payload["action"]``. Snapshot/aggregate write
    failures are logged via ``logger.exception`` but not raised — the
    HTTP layer keeps returning 200 because the event log already
    captured the data.
    """
    if backup_dir is None:
        backup_dir = default_backup_dir()
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "payload_not_dict"}

    action = (payload.get("action") or "").upper()
    project_id = _extract_project_id(payload)
    if project_id is None:
        logger.warning("lswebhook: no project_id in payload (action=%s)", action)
        # Still record the event so we don't lose anything — fall back to
        # a "project_unknown" bucket so the on-disk log isn't lost.
        project_dir = backup_dir / "project_unknown"
        try:
            _append_event(project_dir, payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("lswebhook: failed appending unknown-project event: %s", exc)
        return {"ok": False, "reason": "missing_project_id", "action": action}

    project_dir = _project_dir(backup_dir, project_id)
    try:
        _append_event(project_dir, payload)
    except Exception as exc:  # noqa: BLE001
        # If we can't even fsync the audit line, that's the only case we
        # propagate — the caller (HTTP route) will return 5xx so LS
        # retries. Without the event line, the snapshot doesn't matter.
        logger.exception("lswebhook: failed appending event: %s", exc)
        raise

    summary: dict[str, Any] = {"ok": True, "action": action, "project_id": project_id}

    if action in {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}:
        # Singular variant — payload.annotation is a full annotation dict.
        ann = payload.get("annotation") or {}
        if isinstance(ann, dict) and "id" in ann:
            try:
                _write_snapshot(project_dir, ann)
                summary["annotation_id"] = int(ann["id"])
            except Exception as exc:  # noqa: BLE001
                logger.exception("lswebhook: snapshot write failed: %s", exc)
            tid = _annotation_task_id(ann)
            if tid is not None:
                try:
                    _refresh_task_aggregate(project_dir, tid)
                    summary["task_id"] = tid
                except Exception as exc:  # noqa: BLE001
                    logger.exception("lswebhook: aggregate refresh failed: %s", exc)

    elif action == "ANNOTATIONS_CREATED":
        # Bulk variant — payload.annotations is a list of full annotations.
        created = payload.get("annotations") or []
        affected_tasks: set[int] = set()
        written_ids: list[int] = []
        for ann in created:
            if not isinstance(ann, dict) or "id" not in ann:
                continue
            try:
                _write_snapshot(project_dir, ann)
                written_ids.append(int(ann["id"]))
            except Exception as exc:  # noqa: BLE001
                logger.exception("lswebhook: snapshot write failed: %s", exc)
            tid = _annotation_task_id(ann)
            if tid is not None:
                affected_tasks.add(tid)
        for tid in affected_tasks:
            try:
                _refresh_task_aggregate(project_dir, tid)
            except Exception as exc:  # noqa: BLE001
                logger.exception("lswebhook: aggregate refresh failed: %s", exc)
        summary["annotation_ids"] = written_ids
        summary["affected_tasks"] = sorted(affected_tasks)

    elif action == "ANNOTATIONS_DELETED":
        # LS 1.23 sends ID-only payload here (the annotations are gone, so
        # only the IDs survive). To refresh the right task aggregates we
        # peek at our live snapshot for each deleted id BEFORE moving it
        # to deleted/.
        deleted = payload.get("annotations") or []
        moved_count = 0
        affected_tasks_d: set[int] = set()
        moved_ids: list[int] = []
        for item in deleted:
            if not isinstance(item, dict):
                continue
            ann_id = item.get("id")
            if not isinstance(ann_id, int):
                continue
            live_path = _annotation_path(project_dir, ann_id)
            tid: int | None = None
            if live_path.exists():
                try:
                    live = json.loads(live_path.read_text(encoding="utf-8"))
                    tid = _annotation_task_id(live)
                except (OSError, json.JSONDecodeError):
                    pass
            try:
                if _move_to_deleted(project_dir, ann_id):
                    moved_count += 1
                    moved_ids.append(ann_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("lswebhook: move-to-deleted failed: %s", exc)
            if tid is not None:
                affected_tasks_d.add(tid)
        for tid in affected_tasks_d:
            try:
                _refresh_task_aggregate(project_dir, tid)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "lswebhook: aggregate refresh after delete failed: %s", exc
                )
        summary["moved_count"] = moved_count
        summary["moved_ids"] = moved_ids
        summary["affected_tasks"] = sorted(affected_tasks_d)

    elif action and action not in _HANDLED_ACTIONS:
        # Already in events.jsonl. Nothing else to do — the user may
        # have enabled an event we don't index (e.g. PROJECT_UPDATED).
        summary["unhandled"] = True

    return summary


# ---------------------------------------------------------------------------
# Flask wiring
# ---------------------------------------------------------------------------


def register_lswebhook_routes(app: Any, *, route: str = "/lswebhook/annotations") -> None:
    """Register the POST handler on a Flask app.

    Imported lazily from the server module so this module stays
    importable in tests without Flask. Returns 200 on every well-formed
    request unless the events.jsonl append itself failed.
    """
    from flask import jsonify, request

    @app.route(route, methods=["POST"])  # type: ignore[misc]
    def lswebhook_annotations():  # noqa: D401
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "reason": "invalid_json"}), 400
        try:
            summary = process_event(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("lswebhook: unrecoverable failure: %s", exc)
            return jsonify({"ok": False, "reason": "internal_error"}), 500
        return jsonify(summary), 200

    logger.info("lswebhook: registered route %s", route)
