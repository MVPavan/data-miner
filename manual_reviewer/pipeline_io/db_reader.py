"""Sync sqlite3 reader for the aa_v4 ``pipeline.db``.

Mirrors the connection pattern in ``data_miner/auto_annotation_v4/viewer/app.py``
— PRAGMA query_only, fresh connection per call. Safe alongside an async
pipeline writer because WAL allows concurrent reads.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = TRUE")
    return conn


def _has_dedup_columns(db_path: Path) -> bool:
    with _connect(db_path) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(image_meta)").fetchall()}
    return "dedup_status" in cols and "dedup_cluster_id" in cols


def read_job_info(db_path: Path) -> dict[str, Any] | None:
    """Return the singleton job_info row as a dict, or None if absent."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM job_info LIMIT 1").fetchone()
    return dict(row) if row else None


def iter_survivor_images(
    db_path: Path,
    *,
    limit: int | None = None,
    require_finalize: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield image_meta rows for dedup-survivor images.

    ``require_finalize=True`` restricts to images whose stages_completed JSON
    array contains "finalize" — only fully-pipelined images are reviewable.

    Falls back to "every image is a survivor" when ``dedup_status`` doesn't
    exist on the table — happens with pre-migration aa_v4 DBs that haven't
    been reopened by ``CheckpointDB.connect()`` since this PR. The CLI surface
    stays usable without a forced migration roundtrip.
    """
    if not _has_dedup_columns(db_path):
        sql = (
            "SELECT image_id, image_path, status, stages_completed, "
            "NULL AS dedup_status, NULL AS dedup_cluster_id, total_timing_ms "
            "FROM image_meta"
        )
    else:
        sql = (
            "SELECT image_id, image_path, status, stages_completed, "
            "dedup_status, dedup_cluster_id, total_timing_ms "
            "FROM image_meta WHERE dedup_status='survivor'"
        )
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    for row in rows:
        record = dict(row)
        if require_finalize:
            try:
                completed = json.loads(record.get("stages_completed") or "[]")
            except (TypeError, ValueError):
                completed = []
            if "finalize" not in completed:
                continue
        yield record


def read_image_payload(
    db_path: Path,
    image_id: str,
    *,
    traces_dir: Path | None = None,
    trace_excerpt_limit: int = 20,
) -> dict[str, Any]:
    """Return the full per-image payload used by ``task_builder.build_task``.

    Bundles ``image_meta`` + every row from ``stages`` + every row from
    ``proposals`` + the tail of ``traces/{image_id}.json`` (when *traces_dir*
    is supplied). All JSON columns are parsed eagerly so callers receive plain
    dicts/lists — failures fall back to the raw string with a ``_parse_error``
    marker rather than raising.
    """
    with _connect(db_path) as conn:
        meta_row = conn.execute(
            "SELECT * FROM image_meta WHERE image_id = ?", (image_id,)
        ).fetchone()
        if meta_row is None:
            raise KeyError(f"image_id not found in image_meta: {image_id}")
        stage_rows = conn.execute(
            "SELECT stage, data FROM stages WHERE image_id = ?", (image_id,)
        ).fetchall()
        proposal_rows = conn.execute(
            "SELECT model, data FROM proposals WHERE image_id = ?", (image_id,)
        ).fetchall()

    meta = dict(meta_row)
    try:
        meta["stages_completed"] = json.loads(meta.get("stages_completed") or "[]")
    except (TypeError, ValueError):
        meta["stages_completed"] = []

    stages: dict[str, Any] = {}
    for row in stage_rows:
        stages[row["stage"]] = _safe_load_json(row["data"])

    proposals: dict[str, Any] = {}
    for row in proposal_rows:
        proposals[row["model"]] = _safe_load_json(row["data"])

    trace_excerpt: list[Any] = []
    if traces_dir is not None:
        trace_path = traces_dir / f"{image_id}.json"
        if trace_path.exists():
            try:
                full_trace = json.loads(trace_path.read_text(encoding="utf-8"))
                if isinstance(full_trace, list):
                    trace_excerpt = full_trace[-trace_excerpt_limit:]
                else:
                    trace_excerpt = [full_trace]
            except (OSError, ValueError):
                trace_excerpt = []

    return {
        "image_id": image_id,
        "meta": meta,
        "stages": stages,
        "proposals": proposals,
        "trace_excerpt": trace_excerpt,
    }


def _safe_load_json(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return {"_parse_error": True, "_raw": raw[:500]}
