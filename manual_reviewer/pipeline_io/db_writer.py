"""Sync sqlite3 writer for ``Stage.HUMAN_REVIEW`` rows and dedup metadata.

Uses a sync connection (with WAL busy_timeout) rather than aiosqlite because
the export script is a one-shot CLI, not part of the live pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Iterable

from data_miner.auto_annotation_v4.configs.contracts import (
    HumanReviewResult,
    ReconcileResult,
)
from data_miner.auto_annotation_v4.configs.enums import Stage

logger = logging.getLogger(__name__)

__all__ = [
    "write_human_review",
    "write_dedup_assignments",
    "write_reconcile_results",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def write_human_review(
    db_path: Path,
    result: HumanReviewResult,
    *,
    config_hash: str = "",
) -> None:
    """Persist a ``HumanReviewResult`` and update ``image_meta`` audit fields.

    Three writes inside one transaction:

    1. INSERT OR REPLACE into ``stages`` keyed on (image_id, 'human_review').
       Idempotent on re-export of the same completion.
    2. Append "human_review" to ``image_meta.stages_completed`` if not already
       present, so the viewer's stage filter and any downstream consumers
       discover the new audit row.
    3. Bump ``image_meta.updated_at`` so live viewers see the change.
    """
    payload = result.model_dump_json()
    now = time.time()

    with closing(_connect(db_path)) as conn, conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO stages (image_id, stage, data, config_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (result.image_id, Stage.HUMAN_REVIEW.value, payload, config_hash, now),
        )

        meta_row = cur.execute(
            "SELECT stages_completed FROM image_meta WHERE image_id = ?",
            (result.image_id,),
        ).fetchone()
        if meta_row is not None:
            try:
                completed = json.loads(meta_row["stages_completed"] or "[]")
            except (TypeError, ValueError):
                completed = []
            if Stage.HUMAN_REVIEW.value not in completed:
                completed.append(Stage.HUMAN_REVIEW.value)
            cur.execute(
                "UPDATE image_meta SET stages_completed = ?, updated_at = ?"
                " WHERE image_id = ?",
                (json.dumps(completed), now, result.image_id),
            )


def write_reconcile_results(
    db_path: Path,
    results: Iterable[ReconcileResult],
    *,
    config_hash: str = "",
    skip_empty: bool = True,
) -> int:
    """Persist ``ReconcileResult`` rows under ``stage='reconcile'``.

    Idempotent on re-run (INSERT OR REPLACE keyed on (image_id, stage)) so the
    reconciler can be invoked multiple times. Also appends "reconcile" to
    ``image_meta.stages_completed`` for each touched image so the viewer's
    stage filter discovers the new audit row.

    ``skip_empty=True`` (default) drops rows with no propagated and no rejected
    detections — those add no information. Set False to write a row for every
    image (useful when callers want to know "reconcile ran on this image").
    Returns the number of rows actually written.
    """
    now = time.time()
    written = 0
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.cursor()
        for result in results:
            if skip_empty and not result.propagated and not result.rejected:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO stages (image_id, stage, data, config_hash, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    result.image_id,
                    Stage.RECONCILE.value,
                    result.model_dump_json(),
                    config_hash,
                    now,
                ),
            )
            meta_row = cur.execute(
                "SELECT stages_completed FROM image_meta WHERE image_id = ?",
                (result.image_id,),
            ).fetchone()
            if meta_row is not None:
                try:
                    completed = json.loads(meta_row["stages_completed"] or "[]")
                except (TypeError, ValueError):
                    completed = []
                if Stage.RECONCILE.value not in completed:
                    completed.append(Stage.RECONCILE.value)
                cur.execute(
                    "UPDATE image_meta SET stages_completed = ?, updated_at = ?"
                    " WHERE image_id = ?",
                    (json.dumps(completed), now, result.image_id),
                )
            written += 1
    return written


def write_dedup_assignments(
    db_path: Path,
    assignments: Iterable[tuple[str, str | None, bool]],
) -> int:
    """Set ``dedup_status`` and ``dedup_cluster_id`` for the given image_ids.

    *assignments* yields ``(image_id, cluster_id, is_survivor)``. Rows whose
    image_id is missing from ``image_meta`` are silently skipped — caller is
    responsible for ensuring the IDs exist (typically because aa_v4 already
    processed them). Returns the number of rows actually updated.
    """
    now = time.time()
    rows = [
        (
            "survivor" if is_survivor else "dropped",
            cluster_id,
            now,
            image_id,
        )
        for image_id, cluster_id, is_survivor in assignments
    ]
    if not rows:
        return 0
    image_ids = [r[3] for r in rows]
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in image_ids)
        existing = {
            r["image_id"]
            for r in cur.execute(
                f"SELECT image_id FROM image_meta WHERE image_id IN ({placeholders})",
                image_ids,
            ).fetchall()
        }
        missing = [iid for iid in image_ids if iid not in existing]
        if missing:
            logger.warning(
                "write_dedup_assignments: %d image_ids missing from image_meta (sample=%s)",
                len(missing),
                missing[:5],
            )
        cur.executemany(
            "UPDATE image_meta SET dedup_status = ?, dedup_cluster_id = ?, updated_at = ?"
            " WHERE image_id = ?",
            rows,
        )
        return cur.rowcount
