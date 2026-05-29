"""Tier 1 unit tests for CheckpointDB — no GPU, tmp SQLite, in-process."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

from pydantic import BaseModel

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.enums import (
    STAGE_ORDER,
    ImageStatus,
    Stage,
    WorkStatus,
)
from data_miner.auto_annotation_v4.configs.loader import (
    compute_config_hash,
    load_config,
)


class _Payload(BaseModel):
    """Minimal Pydantic payload for stage serialization tests."""

    image_id: str
    value: int = 0


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. register → add_work_batch → claim → save_and_forward
# ---------------------------------------------------------------------------


def test_register_claim_forward(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            await db.register_image("img1", "/tmp/img1.jpg")
            await db.add_work_batch(Stage.DETECT.value, ["img1"])

            claimed = await db.claim_work(Stage.DETECT.value, "worker-A")
            assert claimed == "img1"

            payload = _Payload(image_id="img1", value=42)
            await db.save_and_forward(
                image_id="img1",
                stage=Stage.DETECT,
                data=payload,
                config_hash="v1",
                next_stage=Stage.FILTER.value,
                timing_ms=12.5,
                work_stage=Stage.DETECT.value,
            )

            conn = db._require_db()

            cur = await conn.execute(
                "SELECT data, config_hash FROM stages WHERE image_id=? AND stage=?",
                ("img1", Stage.DETECT.value),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row["config_hash"] == "v1"
            reloaded = _Payload.model_validate_json(row["data"])
            assert reloaded.value == 42

            cur = await conn.execute(
                "SELECT status FROM work_queue WHERE image_id=? AND stage=?",
                ("img1", Stage.DETECT.value),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row["status"] == WorkStatus.DONE

            cur = await conn.execute(
                "SELECT status FROM work_queue WHERE image_id=? AND stage=?",
                ("img1", Stage.FILTER.value),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row["status"] == WorkStatus.PENDING

            cur = await conn.execute(
                "SELECT stages_completed FROM image_meta WHERE image_id=?",
                ("img1",),
            )
            row = await cur.fetchone()
            completed = json.loads(row["stages_completed"])
            assert Stage.DETECT.value in completed
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 2. Concurrent claim_work returns unique image_ids
# ---------------------------------------------------------------------------


def test_concurrent_claim(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(10)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            await db.add_work_batch(Stage.DETECT.value, image_ids)

            claims = await asyncio.gather(
                *[
                    db.claim_work(Stage.DETECT.value, f"worker-{i}")
                    for i in range(10)
                ]
            )
            assert None not in claims, f"Some claims returned None: {claims}"
            assert len(set(claims)) == 10, f"Duplicate claims: {claims}"
            assert set(claims) == set(image_ids)
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 3. Stale claim recovery and eventual dead-letter
# ---------------------------------------------------------------------------


def test_stale_recovery(tmp_path):
    async def scenario():
        db_path = tmp_path / "pipeline.db"
        db = CheckpointDB(db_path, lock_ttl=60, max_retries=3)
        await db.connect()
        try:
            await db.register_image("img-stale", "/tmp/img-stale.jpg")
            await db.add_work_batch(Stage.DETECT.value, ["img-stale"])
            claimed = await db.claim_work(Stage.DETECT.value, "worker-X")
            assert claimed == "img-stale"
        finally:
            await db.close()

        # Force a stale claimed_at via raw sqlite3 — simulate a dead worker.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE work_queue SET claimed_at = ? WHERE image_id = ?",
            (time.time() - 999, "img-stale"),
        )
        conn.commit()
        conn.close()

        db = CheckpointDB(db_path, lock_ttl=60, max_retries=3)
        await db.connect()
        try:
            # First recovery → attempts 1 → back to pending.
            recovered = await db.recover_stale(lock_ttl=60, max_retries=3)
            assert recovered == 1
            conn = db._require_db()
            cur = await conn.execute(
                "SELECT status, attempts FROM work_queue WHERE image_id=?",
                ("img-stale",),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row["status"] == WorkStatus.PENDING
            assert row["attempts"] == 1

            # Repeat claim→stale→recover cycles until the row is dead-lettered.
            # recover_stale does: new_attempts = attempts + 1; if >= retries ->
            # move to failures. Starting at attempts=1 with retries=3, this
            # dead-letters on the second cycle (attempts goes 1→2 then 2→3).
            for _ in range(3):
                c = await db.claim_work(Stage.DETECT.value, "worker-Y")
                if c is None:
                    # Already dead-lettered; no more pending rows to claim.
                    break
                await conn.execute(
                    "UPDATE work_queue SET claimed_at = ? WHERE image_id = ?",
                    (time.time() - 999, "img-stale"),
                )
                await conn.commit()
                await db.recover_stale(lock_ttl=60, max_retries=3)

            # After enough retries the row should be in failures table and
            # removed from the queue.
            cur = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM failures WHERE image_id=?",
                ("img-stale",),
            )
            row = await cur.fetchone()
            assert row["cnt"] >= 1, "Expected entry in failures table"

            cur = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM work_queue WHERE image_id=?",
                ("img-stale",),
            )
            row = await cur.fetchone()
            assert row["cnt"] == 0, "Dead-lettered item still in work_queue"
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 4. clear_downstream deletes rows from FILTER onward but preserves DETECT
# ---------------------------------------------------------------------------


def test_clear_downstream(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            await db.register_image("imgD", "/tmp/imgD.jpg")
            # Mark all canonical stages as completed in image_meta.
            conn = db._require_db()
            all_stages = [s.value for s in STAGE_ORDER]
            await conn.execute(
                "UPDATE image_meta SET stages_completed = ? WHERE image_id = ?",
                (json.dumps(all_stages), "imgD"),
            )
            now = time.time()
            for s in [Stage.DETECT, Stage.FILTER, Stage.EVALUATE, Stage.REFINE, Stage.FINALIZE]:
                await conn.execute(
                    "INSERT INTO stages (image_id, stage, data, config_hash, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    ("imgD", s.value, "{}", "v1", now),
                )
                await conn.execute(
                    "INSERT INTO work_queue (image_id, stage, status, score)"
                    " VALUES (?, ?, ?, ?)",
                    ("imgD", s.value, WorkStatus.DONE, now),
                )
            await conn.commit()

            await db.clear_downstream("imgD", Stage.FILTER)

            cur = await conn.execute(
                "SELECT stage FROM stages WHERE image_id=?", ("imgD",)
            )
            remaining = {r["stage"] for r in await cur.fetchall()}
            assert remaining == {Stage.DETECT.value}, (
                f"Expected only detect to remain, got {remaining}"
            )

            cur = await conn.execute(
                "SELECT stage FROM work_queue WHERE image_id=?", ("imgD",)
            )
            remaining_wq = {r["stage"] for r in await cur.fetchall()}
            assert remaining_wq == {Stage.DETECT.value}, (
                f"Expected only detect work_queue row to remain, got {remaining_wq}"
            )

            cur = await conn.execute(
                "SELECT stages_completed FROM image_meta WHERE image_id=?",
                ("imgD",),
            )
            row = await cur.fetchone()
            completed = json.loads(row["stages_completed"])
            assert completed == [Stage.DETECT.value], (
                f"image_meta.stages_completed not truncated: {completed}"
            )
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 5. wal_checkpoint(TRUNCATE) runs, WAL file shrinks
# ---------------------------------------------------------------------------


def test_wal_checkpoint(tmp_path):
    async def scenario():
        db_path = tmp_path / "pipeline.db"
        db = CheckpointDB(db_path)
        await db.connect()
        try:
            # Write many small stage rows to force WAL growth.
            conn = db._require_db()
            now = time.time()
            rows = [
                (f"img{i}", Stage.DETECT.value, "{}", "v1", now)
                for i in range(1000)
            ]
            await conn.executemany(
                "INSERT OR REPLACE INTO stages"
                " (image_id, stage, data, config_hash, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            await conn.commit()

            wal_path = db_path.parent / (db_path.name + "-wal")
            size_before = wal_path.stat().st_size if wal_path.exists() else 0

            # This should not raise.
            await db.wal_checkpoint()

            size_after = wal_path.stat().st_size if wal_path.exists() else 0
            assert size_after <= size_before, (
                f"WAL did not shrink: before={size_before} after={size_after}"
            )
        finally:
            await db.close()

    _run(scenario())


def test_wal_checkpoint_concurrent_with_writer(tmp_path):
    """wal_checkpoint() must not deadlock or raise while another coroutine
    holds an uncommitted transaction on the shared connection.
    """

    async def scenario():
        db_path = tmp_path / "pipeline.db"
        db = CheckpointDB(db_path)
        await db.connect()
        try:
            await db.register_image("imgW", "/tmp/imgW.jpg")
            conn = db._require_db()

            writer_started = asyncio.Event()
            writer_release = asyncio.Event()

            async def long_writer():
                # Begin and hold a transaction (an executed INSERT keeps a
                # reserved write lock until commit).
                now = time.time()
                await conn.execute(
                    "INSERT OR REPLACE INTO stages"
                    " (image_id, stage, data, config_hash, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    ("imgW", Stage.DETECT.value, "{}", "v1", now),
                )
                writer_started.set()
                await writer_release.wait()
                await conn.commit()

            writer_task = asyncio.create_task(long_writer())
            await writer_started.wait()

            # Checkpoint runs on a *separate* connection. Must not raise,
            # must not deadlock. It may log BUSY internally — that's fine.
            await asyncio.wait_for(db.wal_checkpoint(), timeout=15.0)

            writer_release.set()
            await writer_task
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 6. should_run_stage on hash mismatch clears downstream
# ---------------------------------------------------------------------------


def test_config_hash_invalidation(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            await db.register_image("imgH", "/tmp/imgH.jpg")
            conn = db._require_db()
            now = time.time()
            # Seed detect + filter + evaluate stages with config_hash="v1".
            for s in [Stage.DETECT, Stage.FILTER, Stage.EVALUATE]:
                await conn.execute(
                    "INSERT INTO stages (image_id, stage, data, config_hash, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    ("imgH", s.value, "{}", "v1", now),
                )
            # Also populate work_queue for filter+evaluate+refine+finalize.
            for s in [Stage.FILTER, Stage.EVALUATE, Stage.REFINE, Stage.FINALIZE]:
                await conn.execute(
                    "INSERT INTO work_queue (image_id, stage, status, score)"
                    " VALUES (?, ?, ?, ?)",
                    ("imgH", s.value, WorkStatus.DONE, now),
                )
            await conn.commit()

            should = await db.should_run_stage("imgH", Stage.FILTER, config_hash="v2")
            assert should is True

            cur = await conn.execute(
                "SELECT stage FROM stages WHERE image_id=?", ("imgH",)
            )
            remaining = {r["stage"] for r in await cur.fetchall()}
            assert Stage.FILTER.value not in remaining
            assert Stage.EVALUATE.value not in remaining
            assert Stage.DETECT.value in remaining

            cur = await conn.execute(
                "SELECT stage FROM work_queue WHERE image_id=?", ("imgH",)
            )
            remaining_wq = {r["stage"] for r in await cur.fetchall()}
            assert Stage.FILTER.value not in remaining_wq
            assert Stage.EVALUATE.value not in remaining_wq
            assert Stage.REFINE.value not in remaining_wq
            assert Stage.FINALIZE.value not in remaining_wq
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 7. save_and_forward with compound work_stage (detect:merge)
# ---------------------------------------------------------------------------


def test_compound_stage_routing(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            await db.register_image("imgC", "/tmp/imgC.jpg")
            # Seed a compound work_queue row.
            conn = db._require_db()
            now = time.time()
            await conn.execute(
                "INSERT INTO work_queue (image_id, stage, status, score)"
                " VALUES (?, ?, ?, ?)",
                ("imgC", "detect:merge", WorkStatus.PROCESSING, now),
            )
            await conn.commit()

            payload = _Payload(image_id="imgC", value=7)
            await db.save_and_forward(
                image_id="imgC",
                stage=Stage.DETECT,
                data=payload,
                config_hash="v1",
                next_stage=Stage.FILTER.value,
                timing_ms=1.0,
                work_stage="detect:merge",
            )

            cur = await conn.execute(
                "SELECT status FROM work_queue WHERE image_id=? AND stage=?",
                ("imgC", "detect:merge"),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row["status"] == WorkStatus.DONE

            cur = await conn.execute(
                "SELECT stage FROM stages WHERE image_id=?", ("imgC",)
            )
            stages_saved = {r["stage"] for r in await cur.fetchall()}
            assert stages_saved == {Stage.DETECT.value}, (
                f"Expected stages rows to use 'detect' (not 'detect:merge'), "
                f"got {stages_saved}"
            )

            cur = await conn.execute(
                "SELECT status FROM work_queue WHERE image_id=? AND stage=?",
                ("imgC", Stage.FILTER.value),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row["status"] == WorkStatus.PENDING
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# 8. compute_config_hash — scoped to invalidation-relevant fields
# ---------------------------------------------------------------------------


def test_config_hash_detects_filter_changes():
    """Changes to filtering.* must change the hash; runtime/workers/output
    changes must not (those don't affect cached stage outputs)."""
    base = load_config()
    hash_base = compute_config_hash(base, base.prompts_dir)

    # 1. Real filtering change must flip the hash (the live S9 bug).
    changed = load_config(overrides=["filtering.min_area=0.123"])
    hash_changed = compute_config_hash(changed, changed.prompts_dir)
    assert hash_changed != hash_base, (
        "filtering.min_area change did not propagate into config_hash"
    )

    # Other filter-subtree fields also drive invalidation.
    for override in (
        "filtering.max_per_class=7",
        "filtering.iou_dedup.threshold=0.42",
        "evaluate.reject_below=0.25",
        "auto_accept.min_model_agreement=5",
    ):
        cfg = load_config(overrides=[override])
        assert compute_config_hash(cfg, cfg.prompts_dir) != hash_base, (
            f"{override} did not change config_hash"
        )

    # 2. Unrelated fields must NOT change the hash (belt-and-suspenders
    #    against over-broad hashing).
    for override in (
        "runtime.log_level=DEBUG",
        "runtime.job_id=some_other_job",
        "runtime.image_dir=/tmp/elsewhere",
        "workers.filter_count=8",
        "output.labels_dir=labels_v2",
        "database.lock_ttl=999",
    ):
        cfg = load_config(overrides=[override])
        assert compute_config_hash(cfg, cfg.prompts_dir) == hash_base, (
            f"{override} unexpectedly changed config_hash"
        )
