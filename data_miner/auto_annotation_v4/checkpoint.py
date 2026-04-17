"""SQLite-backed checkpoint and work distribution for auto_annotation_v4.

Replaces BOTH the v3 file-based CheckpointManager and the Redis Streams
RedisMessageBroker with a single per-job SQLite database.

Layout::

    {job_dir}/pipeline.db          -- one DB per pipeline run

Tables::

    job_info     -- singleton row with job-level metadata
    image_meta   -- per-image lifecycle tracking
    proposals    -- raw per-model detection proposals (Phase 2 barrier)
    stages       -- per-image, per-stage checkpoint data (Pydantic JSON)
    work_queue   -- priority queue replacing Redis Streams
    failures     -- dead-letter rows after max retries exhausted
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TypeVar

import aiosqlite
from pydantic import BaseModel

from .configs.enums import (
    STAGE_ORDER,
    ImageStatus,
    Stage,
    WorkStatus,
    DetectorName,
)

logger = logging.getLogger("data_miner.auto_annotation_v4.checkpoint")

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS job_info (
    job_id          TEXT NOT NULL,
    image_dir       TEXT,
    config_hash     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    created_at      REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS image_meta (
    image_id         TEXT PRIMARY KEY,
    image_path       TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    stages_completed TEXT NOT NULL DEFAULT '[]',
    config_hash      TEXT NOT NULL DEFAULT '',
    prompt_version   TEXT NOT NULL DEFAULT '',
    total_timing_ms  REAL NOT NULL DEFAULT 0.0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    image_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, model)
);
CREATE INDEX IF NOT EXISTS idx_proposals_model ON proposals(model);

CREATE TABLE IF NOT EXISTS stages (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_stages_stage ON stages(stage);

CREATE TABLE IF NOT EXISTS work_queue (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    worker_id   TEXT,
    score       REAL NOT NULL,
    claimed_at  REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_wq_claim ON work_queue(stage, status, score);
CREATE INDEX IF NOT EXISTS idx_wq_stale ON work_queue(status, claimed_at);

CREATE TABLE IF NOT EXISTS failures (
    image_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 1,
    last_error      TEXT,
    last_attempt_at REAL,
    PRIMARY KEY (image_id, stage)
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(data: BaseModel) -> str:
    """Serialize a Pydantic model to a JSON string for DB storage."""
    return data.model_dump_json()


# ---------------------------------------------------------------------------
# CheckpointDB
# ---------------------------------------------------------------------------


class CheckpointDB:
    """SQLite-backed checkpoint + work distribution.

    One instance per pipeline run.  All state for a single job lives in one
    DB file, eliminating cross-job contamination and external dependencies
    (Redis, filesystem locks).

    Replaces both :class:`~auto_annotation_v3.checkpoint.CheckpointManager`
    (file-based) and :class:`~auto_annotation_v3.workers.messaging.RedisMessageBroker`
    (Redis Streams).

    Args:
        db_path: Path to SQLite database file (auto-created if missing).
        lock_ttl: Seconds before stale 'processing' claims are recovered.
        max_retries: Max attempts before an image/stage moves to the failures table.
    """

    def __init__(
        self,
        db_path: str | Path,
        lock_ttl: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.db_path = Path(db_path)
        self.lock_ttl = lock_ttl
        self.max_retries = max_retries
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the aiosqlite connection, apply PRAGMAs, and create tables.

        The database file and its parent directory are created if they do not
        exist.  WAL journal mode is enabled for concurrent-read performance.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        # Row factory gives us dict-like access via sqlite3.Row
        self._db.row_factory = aiosqlite.Row
        # Execute schema (PRAGMAs + CREATE TABLE IF NOT EXISTS).
        # executescript commits any open transaction then runs in autocommit,
        # so we use executescript for the DDL block.
        await self._db.executescript(_SCHEMA)
        logger.info("CheckpointDB connected: %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection gracefully.

        Safe to call multiple times or when the connection is already closed.
        """
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.debug("CheckpointDB closed: %s", self.db_path)

    async def __aenter__(self) -> "CheckpointDB":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    def _require_db(self) -> aiosqlite.Connection:
        """Return the live connection or raise if not connected."""
        if self._db is None:
            raise RuntimeError(
                "CheckpointDB is not connected — call connect() first."
            )
        return self._db

    # ------------------------------------------------------------------
    # Job info
    # ------------------------------------------------------------------

    async def save_job_info(
        self,
        job_id: str,
        image_dir: str | None,
        config_hash: str,
        prompt_version: str,
    ) -> None:
        """Persist job-level metadata as a singleton row.

        Deletes any existing row first so there is always at most one.

        Args:
            job_id: Unique identifier for this pipeline run.
            image_dir: Root directory containing the source images (may be None).
            config_hash: Hash of the pipeline configuration for cache invalidation.
            prompt_version: Version string for the prompt templates in use.
        """
        db = self._require_db()
        await db.execute("DELETE FROM job_info")
        await db.execute(
            "INSERT INTO job_info (job_id, image_dir, config_hash, prompt_version, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (job_id, image_dir, config_hash, prompt_version, time.time()),
        )
        await db.commit()
        logger.debug("Saved job_info: job_id=%s", job_id)

    async def get_job_info(self) -> dict | None:
        """Return the singleton job_info row as a dict, or None if empty.

        Returns:
            Dict with keys: job_id, image_dir, config_hash, prompt_version,
            created_at, status — or None if no job has been registered.
        """
        db = self._require_db()
        cursor = await db.execute("SELECT * FROM job_info LIMIT 1")
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Image registration
    # ------------------------------------------------------------------

    async def register_image(self, image_id: str, image_path: str) -> None:
        """Register an image for pipeline processing.

        Uses INSERT OR IGNORE so re-registering an already-known image is a
        no-op.  This makes the method idempotent for restarts.

        Args:
            image_id: Unique identifier (typically the filename stem).
            image_path: Absolute path to the image file on disk.
        """
        db = self._require_db()
        now = time.time()
        await db.execute(
            "INSERT OR IGNORE INTO image_meta"
            " (image_id, image_path, status, stages_completed, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, image_path, ImageStatus.PENDING, "[]", now, now),
        )
        await db.commit()

    async def resolve_image_path(self, image_id: str) -> str:
        """Look up the filesystem path for *image_id*.

        Args:
            image_id: The image identifier to resolve.

        Returns:
            The stored image_path string.

        Raises:
            FileNotFoundError: If *image_id* is not registered.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT image_path FROM image_meta WHERE image_id = ?",
            (image_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise FileNotFoundError(
                f"Image '{image_id}' not found in checkpoint database"
            )
        return row["image_path"]

    # ------------------------------------------------------------------
    # Stage checkpoints
    # ------------------------------------------------------------------

    async def save_stage(
        self,
        image_id: str,
        stage: Stage,
        data: BaseModel,
        config_hash: str,
    ) -> None:
        """Write a stage checkpoint, overwriting any previous result.

        The Pydantic model is serialized to JSON via ``model_dump_json()``
        for compact, schema-aware storage.

        Args:
            image_id: Target image.
            stage: Pipeline stage (from the Stage enum).
            data: Pydantic model instance containing the stage result.
            config_hash: Configuration hash to track invalidation.
        """
        db = self._require_db()
        await db.execute(
            "INSERT OR REPLACE INTO stages (image_id, stage, data, config_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (image_id, stage.value, _serialize(data), config_hash, time.time()),
        )
        await db.commit()
        logger.debug("Saved stage checkpoint %s/%s", image_id, stage.value)

    async def load_stage(
        self,
        image_id: str,
        stage: Stage,
        model_class: type[T],
    ) -> T | None:
        """Load and deserialize a stage checkpoint.

        Args:
            image_id: Target image.
            stage: Pipeline stage to load.
            model_class: Pydantic model class to validate/deserialize into.

        Returns:
            A validated instance of *model_class*, or None if no checkpoint exists.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT data FROM stages WHERE image_id = ? AND stage = ?",
            (image_id, stage.value),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        try:
            return model_class.model_validate_json(row["data"])
        except Exception:
            logger.exception(
                "Failed to deserialize stage checkpoint %s/%s", image_id, stage.value
            )
            return None

    async def stage_exists(self, image_id: str, stage: Stage) -> bool:
        """Check whether a checkpoint exists for the given image and stage.

        Args:
            image_id: Target image.
            stage: Pipeline stage to check.

        Returns:
            True if a row exists in the stages table.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT 1 FROM stages WHERE image_id = ? AND stage = ?",
            (image_id, stage.value),
        )
        return (await cursor.fetchone()) is not None

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------

    async def save_proposal(
        self,
        image_id: str,
        model: DetectorName,
        data: BaseModel,
    ) -> None:
        """Save a per-model detection proposal.

        Uses INSERT OR REPLACE so re-running a detector overwrites cleanly.

        Args:
            image_id: Target image.
            model: Detector that produced this proposal.
            data: Pydantic model with the raw proposal output.
        """
        db = self._require_db()
        await db.execute(
            "INSERT OR REPLACE INTO proposals (image_id, model, data, config_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (image_id, model.value, _serialize(data), "", time.time()),
        )
        await db.commit()
        logger.debug("Saved proposal %s/%s", image_id, model.value)

    async def load_proposal(
        self,
        image_id: str,
        model: DetectorName,
        model_class: type[T],
    ) -> T | None:
        """Load and deserialize a per-model proposal.

        Args:
            image_id: Target image.
            model: Detector whose proposal to load.
            model_class: Pydantic model class for deserialization.

        Returns:
            Validated model instance, or None if no proposal exists.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT data FROM proposals WHERE image_id = ? AND model = ?",
            (image_id, model.value),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        try:
            return model_class.model_validate_json(row["data"])
        except Exception:
            logger.exception(
                "Failed to deserialize proposal %s/%s", image_id, model.value
            )
            return None

    async def proposal_exists(self, image_id: str, model: DetectorName) -> bool:
        """Check whether a proposal exists for the given image and detector.

        Args:
            image_id: Target image.
            model: Detector to check.

        Returns:
            True if a row exists in the proposals table.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT 1 FROM proposals WHERE image_id = ? AND model = ?",
            (image_id, model.value),
        )
        return (await cursor.fetchone()) is not None

    async def load_all_proposals(
        self,
        image_id: str,
        model_class: type[T],
    ) -> dict[str, T]:
        """Load all per-model proposals for *image_id*.

        Used by the Phase 2 DetectMergeWorker to gather proposals from
        every detector before running the filtering pipeline.

        Args:
            image_id: Target image.
            model_class: Pydantic model class for deserialization.

        Returns:
            Dict mapping model name (str) to validated model instance.
            Models whose proposals fail deserialization are logged and skipped.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT model, data FROM proposals WHERE image_id = ?",
            (image_id,),
        )
        results: dict[str, T] = {}
        for row in await cursor.fetchall():
            try:
                results[row["model"]] = model_class.model_validate_json(row["data"])
            except Exception:
                logger.exception(
                    "Failed to deserialize proposal %s/%s", image_id, row["model"]
                )
        return results

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------

    async def should_run_stage(
        self,
        image_id: str,
        stage: Stage,
        config_hash: str,
    ) -> bool:
        """Decide whether *stage* needs to execute for *image_id*.

        Decision logic:
        - No existing checkpoint for this stage -> must run.
        - Checkpoint exists but config_hash differs -> invalidate downstream,
          then re-run.
        - Checkpoint exists and hash matches -> skip (already done).

        Args:
            image_id: Target image.
            stage: Pipeline stage being considered.
            config_hash: Current configuration hash.

        Returns:
            True if the stage should run; False to skip.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT config_hash FROM stages WHERE image_id = ? AND stage = ?",
            (image_id, stage.value),
        )
        row = await cursor.fetchone()
        if row is None:
            return True
        if row["config_hash"] != config_hash:
            logger.info(
                "Config hash changed for %s/%s (was %s, now %s); invalidating downstream",
                image_id,
                stage.value,
                row["config_hash"],
                config_hash,
            )
            await self.clear_downstream(image_id, stage)
            return True
        return False

    async def clear_downstream(self, image_id: str, from_stage: Stage) -> None:
        """Delete checkpoints and queue entries for *from_stage* and all later stages.

        Uses STAGE_ORDER to determine which stages are downstream.  Also
        updates image_meta to reflect the truncated completion list and
        resets the image status to RUNNING.

        Args:
            image_id: Target image.
            from_stage: The stage from which to start clearing (inclusive).
        """
        try:
            idx = STAGE_ORDER.index(from_stage)
        except ValueError:
            logger.warning(
                "Unknown stage '%s'; skipping clear_downstream.", from_stage
            )
            return

        downstream = [s.value for s in STAGE_ORDER[idx:]]
        db = self._require_db()

        # Build placeholders for the IN clause
        placeholders = ",".join("?" for _ in downstream)

        # Delete stage checkpoints for downstream stages
        await db.execute(
            f"DELETE FROM stages WHERE image_id = ? AND stage IN ({placeholders})",
            [image_id, *downstream],
        )

        # Delete work_queue entries for downstream stages.
        # Exact match for canonical stage names (e.g. "evaluate").
        await db.execute(
            f"DELETE FROM work_queue WHERE image_id = ? AND stage IN ({placeholders})",
            [image_id, *downstream],
        )

        # Also delete compound stage entries (e.g. "detect:grounding_dino",
        # "detect:merge") whose parent stage is in the downstream set.
        # Phase 2 uses "parent:sub" format for per-model detect workers.
        for stage_val in downstream:
            await db.execute(
                "DELETE FROM work_queue WHERE image_id = ? AND stage LIKE ?",
                (image_id, f"{stage_val}:%"),
            )

        # Update image_meta: keep only stages that come *before* from_stage
        keep_stages = [s.value for s in STAGE_ORDER[:idx]]
        cursor = await db.execute(
            "SELECT stages_completed FROM image_meta WHERE image_id = ?",
            (image_id,),
        )
        row = await cursor.fetchone()
        if row is not None:
            current = json.loads(row["stages_completed"])
            filtered = [s for s in current if s in keep_stages]
            await db.execute(
                "UPDATE image_meta SET stages_completed = ?, status = ?, updated_at = ?"
                " WHERE image_id = ?",
                (json.dumps(filtered), ImageStatus.RUNNING, time.time(), image_id),
            )

        await db.commit()
        logger.info(
            "Cleared downstream from %s for %s: %s",
            from_stage.value,
            image_id,
            downstream,
        )

    async def all_stages_complete(self, image_id: str) -> bool:
        """Check whether all pipeline stages have completed for *image_id*.

        Args:
            image_id: Target image.

        Returns:
            True if image_meta.status equals 'complete'.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT status FROM image_meta WHERE image_id = ?",
            (image_id,),
        )
        row = await cursor.fetchone()
        return row is not None and row["status"] == ImageStatus.COMPLETE

    async def clear_image(self, image_id: str) -> None:
        """Remove all data for *image_id* from every table.

        Used for hard resets or removing corrupted images.

        Args:
            image_id: The image to purge.
        """
        db = self._require_db()
        for table in ("image_meta", "proposals", "stages", "work_queue", "failures"):
            await db.execute(
                f"DELETE FROM {table} WHERE image_id = ?",  # noqa: S608
                (image_id,),
            )
        await db.commit()
        logger.info("Cleared all data for image %s", image_id)

    # ------------------------------------------------------------------
    # Work distribution
    # ------------------------------------------------------------------

    async def add_work(
        self,
        stage: str,
        image_id: str,
        score: float | None = None,
    ) -> None:
        """Enqueue a single work item.

        Uses INSERT OR IGNORE so duplicate submissions (e.g. on restart) are
        harmless.  The *stage* parameter is a plain string (not a Stage enum)
        because Phase 2 uses compound stages like ``"detect:grounding_dino"``.

        Args:
            stage: Queue name / stage identifier.
            image_id: Image to process.
            score: Priority score (lower = higher priority).
                   Defaults to ``time.time()`` for FIFO ordering.
        """
        if score is None:
            score = time.time()
        db = self._require_db()
        await db.execute(
            "INSERT OR IGNORE INTO work_queue (image_id, stage, status, score)"
            " VALUES (?, ?, ?, ?)",
            (image_id, stage, WorkStatus.PENDING, score),
        )
        await db.commit()

    async def add_work_batch(self, stage: str, image_ids: list[str]) -> None:
        """Enqueue multiple work items in a single transaction.

        All items get the current timestamp as their score for FIFO ordering.

        Args:
            stage: Queue name / stage identifier.
            image_ids: List of image identifiers to enqueue.
        """
        db = self._require_db()
        now = time.time()
        await db.executemany(
            "INSERT OR IGNORE INTO work_queue (image_id, stage, status, score)"
            " VALUES (?, ?, ?, ?)",
            [(img_id, stage, WorkStatus.PENDING, now) for img_id in image_ids],
        )
        await db.commit()

    async def claim_work(self, stage: str, worker_id: str) -> str | None:
        """Atomically claim the highest-priority pending work item for *stage*.

        Uses a single UPDATE...WHERE rowid = (SELECT ... LIMIT 1) RETURNING
        pattern to avoid TOCTOU races between concurrent workers.

        Args:
            stage: Queue name to claim from.
            worker_id: Identifier of the worker claiming the item.

        Returns:
            The image_id of the claimed item, or None if the queue is empty.
        """
        db = self._require_db()
        now = time.time()
        # Atomic claim: the subquery selects the rowid of the best candidate
        # and the outer UPDATE sets it to processing in one statement.
        cursor = await db.execute(
            "UPDATE work_queue"
            " SET status = ?, worker_id = ?, claimed_at = ?"
            " WHERE rowid = ("
            "   SELECT rowid FROM work_queue"
            "   WHERE stage = ? AND status = ?"
            "   ORDER BY score ASC"
            "   LIMIT 1"
            " )"
            " RETURNING image_id",
            (WorkStatus.PROCESSING, worker_id, now, stage, WorkStatus.PENDING),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        await db.commit()
        image_id: str = row["image_id"]
        logger.debug("Worker %s claimed %s/%s", worker_id, stage, image_id)
        return image_id

    async def release_work(self, image_id: str, stage: str) -> None:
        """Release a claimed work item back to pending.

        Used on graceful shutdown or task cancellation so the item can be
        picked up by another worker.

        Args:
            image_id: Image whose work item to release.
            stage: Stage/queue the item belongs to.
        """
        db = self._require_db()
        await db.execute(
            "UPDATE work_queue SET status = ?, worker_id = NULL, claimed_at = NULL"
            " WHERE image_id = ? AND stage = ?",
            (WorkStatus.PENDING, image_id, stage),
        )
        await db.commit()

    async def complete_work(self, image_id: str, stage: str) -> None:
        """Mark a work item as done.

        Args:
            image_id: Image whose work item completed.
            stage: Stage/queue the item belongs to.
        """
        db = self._require_db()
        await db.execute(
            "UPDATE work_queue SET status = ? WHERE image_id = ? AND stage = ?",
            (WorkStatus.DONE, image_id, stage),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Atomic save + forward  (the key method)
    # ------------------------------------------------------------------

    async def save_and_forward(
        self,
        image_id: str,
        stage: Stage,
        data: BaseModel,
        config_hash: str,
        next_stage: str | None,
        timing_ms: float,
        work_stage: str | None = None,
    ) -> None:
        """Atomically checkpoint a stage result and enqueue the next stage.

        This is the critical method that replaces both the file-based
        checkpoint write AND the Redis XADD in a single SQLite transaction.
        If the process crashes before ``commit()``, nothing is persisted —
        the image is retried from the previous stage on restart.

        Steps (all within one transaction):
          1. INSERT OR REPLACE the stage checkpoint into the stages table.
          2. Update image_meta: append the stage to stages_completed,
             accumulate timing_ms, update config_hash/prompt_version,
             and set status to 'complete' if all STAGE_ORDER stages are done.
          3. Mark the current work_queue entry as done.
          4. If *next_stage* is set and is not ``"done"``, insert a new
             pending work_queue entry for the next stage.

        Args:
            image_id: Target image.
            stage: The stage that just completed (used for checkpoint storage).
            data: Pydantic model with the stage result.
            config_hash: Configuration hash for invalidation tracking.
            next_stage: Queue name for the next stage, or None / ``"done"``
                        if the pipeline is finished for this image.
            timing_ms: Wall-clock time this stage took, in milliseconds.
            work_stage: Work queue stage key to mark as done.  Defaults to
                        ``stage.value`` for simple stages, but compound stages
                        (e.g. ``"detect:merge"``) pass their actual queue key
                        here because it differs from the checkpoint stage.
        """
        db = self._require_db()
        now = time.time()

        # 1. Save stage checkpoint
        await db.execute(
            "INSERT OR REPLACE INTO stages (image_id, stage, data, config_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (image_id, stage.value, _serialize(data), config_hash, now),
        )

        # 2. Update image_meta
        cursor = await db.execute(
            "SELECT stages_completed, total_timing_ms FROM image_meta WHERE image_id = ?",
            (image_id,),
        )
        row = await cursor.fetchone()
        if row is not None:
            completed: list[str] = json.loads(row["stages_completed"])
            if stage.value not in completed:
                completed.append(stage.value)
            total_timing = row["total_timing_ms"] + timing_ms

            # Check if all canonical stages are done
            all_done = all(s.value in completed for s in STAGE_ORDER)
            new_status = ImageStatus.COMPLETE if all_done else ImageStatus.RUNNING

            await db.execute(
                "UPDATE image_meta"
                " SET stages_completed = ?, total_timing_ms = ?, config_hash = ?,"
                "     status = ?, updated_at = ?"
                " WHERE image_id = ?",
                (json.dumps(completed), total_timing, config_hash, new_status, now, image_id),
            )

        # 3. Mark current work_queue entry as done.
        #    work_stage may differ from stage.value for compound stages
        #    (e.g. work_stage="detect:merge" but stage=Stage.DETECT).
        wq_stage = work_stage if work_stage is not None else stage.value
        await db.execute(
            "UPDATE work_queue SET status = ? WHERE image_id = ? AND stage = ?",
            (WorkStatus.DONE, image_id, wq_stage),
        )

        # 4. Enqueue next stage (if applicable)
        if next_stage and next_stage != "done":
            await db.execute(
                "INSERT OR IGNORE INTO work_queue (image_id, stage, status, score)"
                " VALUES (?, ?, ?, ?)",
                (image_id, next_stage, WorkStatus.PENDING, now),
            )

        # Single commit — atomic: everything above either persists together
        # or rolls back together on crash.
        await db.commit()
        logger.debug(
            "save_and_forward %s/%s -> %s (%.1f ms)",
            image_id,
            stage.value,
            next_stage or "done",
            timing_ms,
        )

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    async def fail_work(self, image_id: str, stage: str, error: str) -> None:
        """Record a processing failure for a work item.

        Increments the attempt counter on the work_queue row.  If attempts
        reach *max_retries*, the item is moved to the failures table and
        removed from the queue (dead-lettered).  Otherwise the item is
        reset to pending for automatic retry.

        Args:
            image_id: Image that failed.
            stage: Stage/queue where the failure occurred.
            error: Human-readable error description.
        """
        db = self._require_db()
        now = time.time()

        # Increment attempts on the work_queue row
        cursor = await db.execute(
            "UPDATE work_queue SET attempts = attempts + 1, status = ?, claimed_at = NULL, worker_id = NULL"
            " WHERE image_id = ? AND stage = ?"
            " RETURNING attempts",
            (WorkStatus.PENDING, image_id, stage),
        )
        row = await cursor.fetchone()
        attempts = row["attempts"] if row else 1

        if attempts >= self.max_retries:
            # Move to failures table (dead letter)
            await db.execute(
                "INSERT OR REPLACE INTO failures (image_id, stage, attempts, last_error, last_attempt_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (image_id, stage, attempts, error, now),
            )
            await db.execute(
                "DELETE FROM work_queue WHERE image_id = ? AND stage = ?",
                (image_id, stage),
            )
            logger.warning(
                "Image %s/%s exhausted %d retries — moved to failures: %s",
                image_id,
                stage,
                attempts,
                error,
            )
        else:
            logger.info(
                "Image %s/%s failed (attempt %d/%d), returning to queue: %s",
                image_id,
                stage,
                attempts,
                self.max_retries,
                error,
            )

        await db.commit()

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def recover_stale(
        self,
        lock_ttl: int | None = None,
        max_retries: int | None = None,
    ) -> int:
        """Recover work items that have been in 'processing' state too long.

        A claim becomes stale when ``claimed_at`` is older than
        ``now - lock_ttl``.  Stale items have their attempt counter
        incremented; if the counter reaches *max_retries* they are moved
        to the failures table, otherwise they are reset to pending.

        Args:
            lock_ttl: Override the instance lock_ttl for this call.
            max_retries: Override the instance max_retries for this call.

        Returns:
            Number of stale items recovered.
        """
        ttl = lock_ttl if lock_ttl is not None else self.lock_ttl
        retries = max_retries if max_retries is not None else self.max_retries
        db = self._require_db()
        cutoff = time.time() - ttl

        cursor = await db.execute(
            "SELECT rowid, image_id, stage, attempts FROM work_queue"
            " WHERE status = ? AND claimed_at < ?",
            (WorkStatus.PROCESSING, cutoff),
        )
        stale_rows = await cursor.fetchall()
        recovered = 0

        for row in stale_rows:
            new_attempts = row["attempts"] + 1
            if new_attempts >= retries:
                # Dead-letter
                await db.execute(
                    "INSERT OR REPLACE INTO failures"
                    " (image_id, stage, attempts, last_error, last_attempt_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        row["image_id"],
                        row["stage"],
                        new_attempts,
                        "stale claim recovered — max retries exceeded",
                        time.time(),
                    ),
                )
                await db.execute(
                    "DELETE FROM work_queue WHERE rowid = ?",
                    (row["rowid"],),
                )
                logger.warning(
                    "Stale item %s/%s dead-lettered after %d attempts",
                    row["image_id"],
                    row["stage"],
                    new_attempts,
                )
            else:
                # Reset to pending for retry
                await db.execute(
                    "UPDATE work_queue"
                    " SET status = ?, worker_id = NULL, claimed_at = NULL, attempts = ?"
                    " WHERE rowid = ?",
                    (WorkStatus.PENDING, new_attempts, row["rowid"]),
                )
                logger.info(
                    "Recovered stale item %s/%s (attempt %d/%d)",
                    row["image_id"],
                    row["stage"],
                    new_attempts,
                    retries,
                )
            recovered += 1

        if recovered:
            await db.commit()
        return recovered

    async def progress_summary(self) -> dict:
        """Build a comprehensive progress snapshot for monitoring UIs.

        Returns:
            Dict with keys:
            - ``proposals``: ``{model_name: count}`` — proposals per detector.
            - ``stages``: ``{stage_name: count}`` — completed checkpoints per stage.
            - ``queue``: ``{stage: {status: count}}`` — work queue breakdown.
            - ``failed``: total count of dead-lettered items.
        """
        db = self._require_db()
        summary: dict = {
            "proposals": {},
            "stages": {},
            "queue": {},
            "failed": 0,
        }

        # Proposals per model
        cursor = await db.execute(
            "SELECT model, COUNT(*) as cnt FROM proposals GROUP BY model"
        )
        for row in await cursor.fetchall():
            summary["proposals"][row["model"]] = row["cnt"]

        # Completed stages
        cursor = await db.execute(
            "SELECT stage, COUNT(*) as cnt FROM stages GROUP BY stage"
        )
        for row in await cursor.fetchall():
            summary["stages"][row["stage"]] = row["cnt"]

        # Work queue by stage and status
        cursor = await db.execute(
            "SELECT stage, status, COUNT(*) as cnt FROM work_queue GROUP BY stage, status"
        )
        for row in await cursor.fetchall():
            stage_key = row["stage"]
            if stage_key not in summary["queue"]:
                summary["queue"][stage_key] = {}
            summary["queue"][stage_key][row["status"]] = row["cnt"]

        # Total failures
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM failures")
        row = await cursor.fetchone()
        summary["failed"] = row["cnt"] if row else 0

        return summary

    async def queue_counts(self) -> dict[str, dict[str, int]]:
        """Return per-stage work queue counts grouped by status.

        Returns:
            ``{stage: {status: count}}`` dict — e.g.
            ``{"detect": {"pending": 5, "processing": 2, "done": 10}}``.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT stage, status, COUNT(*) as cnt FROM work_queue GROUP BY stage, status"
        )
        counts: dict[str, dict[str, int]] = {}
        for row in await cursor.fetchall():
            stage_key = row["stage"]
            if stage_key not in counts:
                counts[stage_key] = {}
            counts[stage_key][row["status"]] = row["cnt"]
        return counts

    # ------------------------------------------------------------------
    # Phase 2 support (barrier pattern for multi-model detect)
    # ------------------------------------------------------------------

    async def barrier_ready(self, image_id: str, models: list[str]) -> bool:
        """Check whether all required model proposals have arrived.

        Used by the Phase 2 detect barrier to gate the merge step until
        every detector has submitted its proposals.

        Args:
            image_id: Target image.
            models: List of model names that must all have proposals.

        Returns:
            True if the number of distinct proposals >= len(models).
        """
        db = self._require_db()
        placeholders = ",".join("?" for _ in models)
        cursor = await db.execute(
            f"SELECT COUNT(DISTINCT model) as cnt FROM proposals"
            f" WHERE image_id = ? AND model IN ({placeholders})",
            [image_id, *models],
        )
        row = await cursor.fetchone()
        return row is not None and row["cnt"] >= len(models)

    async def delete_proposal(self, image_id: str, model: str) -> None:
        """Delete a single proposal row.

        Args:
            image_id: Target image.
            model: Detector name (plain string for Phase 2 flexibility).
        """
        db = self._require_db()
        await db.execute(
            "DELETE FROM proposals WHERE image_id = ? AND model = ?",
            (image_id, model),
        )
        await db.commit()

    async def delete_stage(self, image_id: str, stage: str) -> None:
        """Delete a single stage checkpoint row.

        Args:
            image_id: Target image.
            stage: Stage name (plain string for Phase 2 flexibility).
        """
        db = self._require_db()
        await db.execute(
            "DELETE FROM stages WHERE image_id = ? AND stage = ?",
            (image_id, stage),
        )
        await db.commit()
