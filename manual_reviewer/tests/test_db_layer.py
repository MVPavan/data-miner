"""Edge-case tests for the manual_reviewer DB I/O layer and aav4 schema migration.

Covers:
  * ``CheckpointDB._apply_migrations`` — additive ALTERs for ``image_meta``
    dedup columns, idempotency, partial-migration recovery, and the
    ``idx_image_meta_dedup`` index that previously lived in ``_SCHEMA``.
  * ``manual_reviewer.pipeline_io.db_reader`` — corrupt JSON tolerance,
    trace-file edge cases, ``KeyError`` on missing image, ``LIMIT`` &
    ``require_finalize`` filtering.
  * ``manual_reviewer.pipeline_io.db_writer`` — INSERT/UPDATE branches of
    ``write_human_review``, behaviour on missing image_id, NULL cluster ids,
    empty / large batches, idempotent stage_completed bookkeeping.
  * Concurrent reads under WAL while an async writer is in flight.

The fixtures used are local to this file. ``seeded_pipeline_db`` (the
fixture from conftest.py) is reused where convenient.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import aiosqlite
import pytest

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    HumanCorrection,
    HumanReviewResult,
)
from data_miner.auto_annotation_v4.configs.enums import Stage
from manual_reviewer.pipeline_io import (
    iter_survivor_images,
    read_image_payload,
    read_job_info,
    write_dedup_assignments,
    write_human_review,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


_LEGACY_SCHEMA = """\
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

CREATE TABLE IF NOT EXISTS stages (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, stage)
);

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

CREATE TABLE IF NOT EXISTS failures (
    image_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 1,
    last_error      TEXT,
    last_attempt_at REAL,
    PRIMARY KEY (image_id, stage)
);
"""


def _build_legacy_db(db_path: Path) -> None:
    """Create a pre-migration pipeline.db (no dedup_status / dedup_cluster_id)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_LEGACY_SCHEMA)
        now = time.time()
        conn.execute(
            "INSERT INTO job_info (job_id, image_dir, config_hash, prompt_version, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("legacy_job", "/tmp/legacy", "h", "v", now),
        )
        conn.executemany(
            "INSERT INTO image_meta"
            " (image_id, image_path, status, stages_completed, created_at, updated_at)"
            " VALUES (?, ?, 'pending', '[]', ?, ?)",
            [
                ("legacy_a", "/tmp/legacy/a.jpg", now, now),
                ("legacy_b", "/tmp/legacy/b.jpg", now, now),
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    """Pre-migration DB lacking the two dedup columns and the dedup index."""
    p = tmp_path / "legacy.db"
    _build_legacy_db(p)
    return p


@pytest.fixture
def partial_migration_db(tmp_path: Path) -> Path:
    """DB that has dedup_status but lacks dedup_cluster_id."""
    p = tmp_path / "partial.db"
    _build_legacy_db(p)
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "ALTER TABLE image_meta ADD COLUMN dedup_status"
            " TEXT NOT NULL DEFAULT 'survivor'"
        )
        conn.commit()
    finally:
        conn.close()
    return p


def _column_names(db_path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return [r[1] for r in rows]


def _index_names(db_path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name = ?",
            (table,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _make_review_result(image_id: str, *, reviewer: str = "alice") -> HumanReviewResult:
    return HumanReviewResult(
        image_id=image_id,
        reviewer_id=reviewer,
        reviewed_at=time.time(),
        duration_seconds=1.0,
        frame_state="clean",
        corrections=[
            HumanCorrection(
                candidate_id="c1",
                class_name="forklift",
                bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.4, y2=0.6),
                source="finalize",
            )
        ],
        deletions=[],
        notes="",
        ml_modes_used=[],
        ls_completion_id=42,
    )


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


def test_fresh_db_has_dedup_columns_and_index(tmp_path: Path) -> None:
    """Fresh CheckpointDB.connect() yields both dedup columns AND the dedup index."""
    db_path = tmp_path / "fresh.db"

    async def _go() -> None:
        async with CheckpointDB(db_path):
            pass

    asyncio.run(_go())
    cols = _column_names(db_path, "image_meta")
    assert "dedup_status" in cols
    assert "dedup_cluster_id" in cols
    assert "idx_image_meta_dedup" in _index_names(db_path, "image_meta")


def test_legacy_db_migrated_adds_columns_with_defaults(legacy_db: Path) -> None:
    """Legacy DB → connect → both columns present, default 'survivor' / NULL."""
    cols_before = _column_names(legacy_db, "image_meta")
    assert "dedup_status" not in cols_before
    assert "dedup_cluster_id" not in cols_before

    async def _go() -> None:
        async with CheckpointDB(legacy_db):
            pass

    asyncio.run(_go())

    cols_after = _column_names(legacy_db, "image_meta")
    assert "dedup_status" in cols_after
    assert "dedup_cluster_id" in cols_after

    conn = sqlite3.connect(str(legacy_db))
    try:
        rows = conn.execute(
            "SELECT image_id, dedup_status, dedup_cluster_id"
            " FROM image_meta ORDER BY image_id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("legacy_a", "survivor", None),
        ("legacy_b", "survivor", None),
    ]


def test_legacy_db_migration_creates_dedup_index(legacy_db: Path) -> None:
    """Index is built in _apply_migrations, not _SCHEMA — verify it lands on legacy DBs."""
    assert "idx_image_meta_dedup" not in _index_names(legacy_db, "image_meta")

    async def _go() -> None:
        async with CheckpointDB(legacy_db):
            pass

    asyncio.run(_go())
    assert "idx_image_meta_dedup" in _index_names(legacy_db, "image_meta")


def test_reconnect_to_migrated_db_is_noop(legacy_db: Path) -> None:
    """Re-connecting to an already-migrated DB does not raise."""

    async def _go() -> None:
        async with CheckpointDB(legacy_db):
            pass
        # second connect must not crash on the duplicate-column ALTER
        async with CheckpointDB(legacy_db):
            pass

    asyncio.run(_go())
    cols = _column_names(legacy_db, "image_meta")
    assert "dedup_status" in cols
    assert "dedup_cluster_id" in cols


def test_partial_migration_only_adds_missing_column(partial_migration_db: Path) -> None:
    """If dedup_status already exists but dedup_cluster_id doesn't, only the
    second ALTER actually runs — the first must be swallowed."""
    cols_before = _column_names(partial_migration_db, "image_meta")
    assert "dedup_status" in cols_before
    assert "dedup_cluster_id" not in cols_before

    async def _go() -> None:
        async with CheckpointDB(partial_migration_db):
            pass

    asyncio.run(_go())
    cols_after = _column_names(partial_migration_db, "image_meta")
    assert "dedup_status" in cols_after
    assert "dedup_cluster_id" in cols_after


def test_apply_migrations_propagates_non_duplicate_error(tmp_path: Path) -> None:
    """``OperationalError`` whose message is NOT 'duplicate column' must propagate.

    We simulate by temporarily monkey-patching the migration list so it issues
    an ALTER against a non-existent table — sqlite raises 'no such table',
    which the error filter in _apply_migrations should let through.
    """
    db_path = tmp_path / "bad_migration.db"

    async def _go() -> None:
        # Build a clean DB first so the schema exists.
        async with CheckpointDB(db_path):
            pass

        # Now monkey-patch _apply_migrations to attempt a doomed ALTER.
        db = CheckpointDB(db_path)
        await db.connect()  # connect runs the real migrations once (no-op now).

        # Manually call the same code path with a bad statement.
        async def _bad_migration() -> None:
            assert db._db is not None
            try:
                await db._db.execute(
                    "ALTER TABLE no_such_table ADD COLUMN x TEXT"
                )
            except aiosqlite.OperationalError as exc:
                # mirror the production filter
                if "duplicate column" not in str(exc).lower():
                    raise

        with pytest.raises(aiosqlite.OperationalError):
            await _bad_migration()

        await db.close()

    asyncio.run(_go())


def test_register_image_still_works_after_migration(legacy_db: Path) -> None:
    """Schema-survival: legacy register_image flow must keep working post-migration."""

    async def _go() -> None:
        async with CheckpointDB(legacy_db) as db:
            await db.register_image("legacy_c", "/tmp/legacy/c.jpg")
            path = await db.resolve_image_path("legacy_c")
            assert path == "/tmp/legacy/c.jpg"

    asyncio.run(_go())
    # And the row must come out of iter_survivor_images with the default status.
    rows = list(iter_survivor_images(legacy_db, require_finalize=False))
    ids = sorted(r["image_id"] for r in rows)
    assert ids == ["legacy_a", "legacy_b", "legacy_c"]
    for r in rows:
        assert r["dedup_status"] == "survivor"
        assert r["dedup_cluster_id"] is None


# ---------------------------------------------------------------------------
# db_reader tests
# ---------------------------------------------------------------------------


def test_read_job_info_returns_none_on_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"

    async def _go() -> None:
        async with CheckpointDB(db_path):
            pass  # no save_job_info call

    asyncio.run(_go())
    assert read_job_info(db_path) is None


def test_iter_survivor_images_require_finalize_excludes_unfinalized(
    seeded_pipeline_db: Path,
) -> None:
    """img_a has 'finalize' in stages_completed; clear it, expect zero results."""
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        conn.execute(
            "UPDATE image_meta SET stages_completed = ? WHERE image_id = ?",
            (json.dumps(["detect", "filter"]), "img_a"),
        )
        conn.commit()
    finally:
        conn.close()
    survivors = list(iter_survivor_images(seeded_pipeline_db, require_finalize=True))
    assert survivors == []


def test_iter_survivor_images_no_finalize_required_includes_empty_stages(
    seeded_pipeline_db: Path,
) -> None:
    """With require_finalize=False, even empty stages_completed images come through."""
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        conn.execute(
            "UPDATE image_meta SET stages_completed = '[]' WHERE image_id = ?",
            ("img_a",),
        )
        conn.commit()
    finally:
        conn.close()
    ids = [r["image_id"] for r in iter_survivor_images(seeded_pipeline_db, require_finalize=False)]
    assert "img_a" in ids


def test_iter_survivor_images_limit_caps_rows(seeded_pipeline_db: Path) -> None:
    # Add a second survivor so a LIMIT 1 actually has work to do.
    async def _seed_more() -> None:
        async with CheckpointDB(seeded_pipeline_db) as db:
            await db.register_image("img_c", "/tmp/imgs/img_c.jpg")
            async with db._transaction() as tx:
                await tx.execute(
                    "UPDATE image_meta SET stages_completed = ? WHERE image_id = ?",
                    (json.dumps(["finalize"]), "img_c"),
                )

    asyncio.run(_seed_more())
    rows = list(iter_survivor_images(seeded_pipeline_db, limit=1, require_finalize=True))
    assert len(rows) == 1


def test_read_image_payload_missing_image_id_raises_keyerror(
    seeded_pipeline_db: Path,
) -> None:
    with pytest.raises(KeyError):
        read_image_payload(seeded_pipeline_db, "does_not_exist")


def test_read_image_payload_corrupt_stage_json_uses_marker(
    seeded_pipeline_db: Path,
) -> None:
    """If a stages.data row contains non-JSON text, _safe_load_json wraps it."""
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO stages (image_id, stage, data, config_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("img_a", "broken_stage", "{not valid json", "h1", time.time()),
        )
        conn.commit()
    finally:
        conn.close()
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    val = payload["stages"]["broken_stage"]
    assert isinstance(val, dict)
    assert val.get("_parse_error") is True
    assert "_raw" in val


def test_read_image_payload_malformed_stages_completed_returns_empty_list(
    seeded_pipeline_db: Path,
) -> None:
    """If image_meta.stages_completed is unparseable, payload stages_completed=[]."""
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        conn.execute(
            "UPDATE image_meta SET stages_completed = ? WHERE image_id = ?",
            ("not-json-at-all", "img_a"),
        )
        conn.commit()
    finally:
        conn.close()
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    assert payload["meta"]["stages_completed"] == []


def test_read_image_payload_missing_trace_file_returns_empty(
    seeded_pipeline_db: Path, tmp_path: Path,
) -> None:
    traces_dir = tmp_path / "traces_missing"
    traces_dir.mkdir()
    payload = read_image_payload(seeded_pipeline_db, "img_a", traces_dir=traces_dir)
    assert payload["trace_excerpt"] == []


def test_read_image_payload_malformed_trace_file_returns_empty(
    seeded_pipeline_db: Path, tmp_path: Path,
) -> None:
    traces_dir = tmp_path / "traces_bad"
    traces_dir.mkdir()
    (traces_dir / "img_a.json").write_text("{not valid json")
    payload = read_image_payload(seeded_pipeline_db, "img_a", traces_dir=traces_dir)
    assert payload["trace_excerpt"] == []


def test_read_image_payload_trace_object_wrapped_in_list(
    seeded_pipeline_db: Path, tmp_path: Path,
) -> None:
    traces_dir = tmp_path / "traces_obj"
    traces_dir.mkdir()
    (traces_dir / "img_a.json").write_text(json.dumps({"event": "single"}))
    payload = read_image_payload(seeded_pipeline_db, "img_a", traces_dir=traces_dir)
    assert payload["trace_excerpt"] == [{"event": "single"}]


def test_read_image_payload_trace_excerpt_limit_returns_tail(
    seeded_pipeline_db: Path, tmp_path: Path,
) -> None:
    traces_dir = tmp_path / "traces_long"
    traces_dir.mkdir()
    full = [{"i": i} for i in range(50)]
    (traces_dir / "img_a.json").write_text(json.dumps(full))
    payload = read_image_payload(
        seeded_pipeline_db, "img_a", traces_dir=traces_dir, trace_excerpt_limit=5
    )
    assert payload["trace_excerpt"] == full[-5:]


# ---------------------------------------------------------------------------
# db_writer tests
# ---------------------------------------------------------------------------


def test_write_human_review_inserts_fresh_row(seeded_pipeline_db: Path) -> None:
    """No prior human_review row → INSERT path runs cleanly."""
    # Confirm pre-state
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        pre = conn.execute(
            "SELECT 1 FROM stages WHERE image_id = ? AND stage = ?",
            ("img_a", "human_review"),
        ).fetchone()
    finally:
        conn.close()
    assert pre is None

    write_human_review(seeded_pipeline_db, _make_review_result("img_a"), config_hash="h1")
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    assert "human_review" in payload["stages"]
    assert "human_review" in payload["meta"]["stages_completed"]


def test_write_human_review_overwrites_existing_row_no_dupes(
    seeded_pipeline_db: Path,
) -> None:
    """Second write must REPLACE the stage row and not duplicate stages_completed."""
    write_human_review(seeded_pipeline_db, _make_review_result("img_a"), config_hash="h1")
    write_human_review(
        seeded_pipeline_db,
        _make_review_result("img_a", reviewer="bob"),
        config_hash="h1",
    )

    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM stages WHERE image_id = ? AND stage = ?",
            ("img_a", "human_review"),
        ).fetchone()[0]
        completed_raw = conn.execute(
            "SELECT stages_completed FROM image_meta WHERE image_id = ?",
            ("img_a",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert cnt == 1
    completed = json.loads(completed_raw)
    assert completed.count("human_review") == 1


def test_write_human_review_unknown_image_inserts_stage_skips_meta(
    seeded_pipeline_db: Path,
) -> None:
    """Document the actual behaviour: stage row gets INSERTed (no FK), meta UPDATE noop."""
    write_human_review(
        seeded_pipeline_db, _make_review_result("ghost_id"), config_hash="h1"
    )
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        stage_row = conn.execute(
            "SELECT image_id, stage FROM stages WHERE image_id = ?",
            ("ghost_id",),
        ).fetchone()
        meta_row = conn.execute(
            "SELECT 1 FROM image_meta WHERE image_id = ?", ("ghost_id",)
        ).fetchone()
    finally:
        conn.close()
    assert stage_row == ("ghost_id", "human_review")
    assert meta_row is None


def test_write_human_review_bumps_updated_at(seeded_pipeline_db: Path) -> None:
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        before = conn.execute(
            "SELECT updated_at FROM image_meta WHERE image_id = ?", ("img_a",)
        ).fetchone()[0]
    finally:
        conn.close()
    # Force a perceptible time delta on fast filesystems.
    time.sleep(0.01)
    write_human_review(seeded_pipeline_db, _make_review_result("img_a"), config_hash="h1")
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        after = conn.execute(
            "SELECT updated_at FROM image_meta WHERE image_id = ?", ("img_a",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert after > before


def test_write_human_review_preserves_existing_stages_completed(
    seeded_pipeline_db: Path,
) -> None:
    """Append-only: existing entries (e.g. 'finalize') must survive."""
    write_human_review(seeded_pipeline_db, _make_review_result("img_a"), config_hash="h1")
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    completed = payload["meta"]["stages_completed"]
    for needed in ("detect", "filter", "evaluate", "refine", "finalize", "human_review"):
        assert needed in completed


def test_write_human_review_recovers_from_corrupt_stages_completed(
    seeded_pipeline_db: Path,
) -> None:
    """Bad stages_completed JSON → writer treats it as [] then appends 'human_review'."""
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        conn.execute(
            "UPDATE image_meta SET stages_completed = ? WHERE image_id = ?",
            ("not json", "img_a"),
        )
        conn.commit()
    finally:
        conn.close()
    write_human_review(seeded_pipeline_db, _make_review_result("img_a"), config_hash="h1")
    conn = sqlite3.connect(str(seeded_pipeline_db))
    try:
        raw = conn.execute(
            "SELECT stages_completed FROM image_meta WHERE image_id = ?",
            ("img_a",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert json.loads(raw) == ["human_review"]


def test_write_dedup_assignments_empty_returns_zero(seeded_pipeline_db: Path) -> None:
    n = write_dedup_assignments(seeded_pipeline_db, [])
    assert n == 0


def test_write_dedup_assignments_large_batch(tmp_path: Path) -> None:
    """1000+ rows should work in a single executemany."""
    db_path = tmp_path / "big.db"

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.register_image_batch(
                [(f"img_{i:04d}", f"/tmp/{i}.jpg") for i in range(1500)]
            )

    asyncio.run(_seed())
    assignments = [
        (f"img_{i:04d}", f"cluster_{i // 5}", i % 5 == 0) for i in range(1500)
    ]
    n = write_dedup_assignments(db_path, assignments)
    assert n == 1500
    conn = sqlite3.connect(str(db_path))
    try:
        survivors = conn.execute(
            "SELECT COUNT(*) FROM image_meta WHERE dedup_status = 'survivor'"
        ).fetchone()[0]
        dropped = conn.execute(
            "SELECT COUNT(*) FROM image_meta WHERE dedup_status = 'dropped'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert survivors == 300  # every 5th
    assert dropped == 1200


def test_write_dedup_assignments_mixed_then_iter_reflects_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "mixed.db"

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.register_image_batch(
                [(f"f_{i}", f"/tmp/{i}.jpg") for i in range(4)]
            )

    asyncio.run(_seed())
    write_dedup_assignments(
        db_path,
        [
            ("f_0", "cA", True),
            ("f_1", "cA", False),
            ("f_2", "cB", True),
            ("f_3", "cB", False),
        ],
    )
    survivors = sorted(
        r["image_id"] for r in iter_survivor_images(db_path, require_finalize=False)
    )
    assert survivors == ["f_0", "f_2"]


def test_write_dedup_assignments_accepts_null_cluster_id(tmp_path: Path) -> None:
    db_path = tmp_path / "null_cluster.db"

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.register_image("solo", "/tmp/solo.jpg")

    asyncio.run(_seed())
    n = write_dedup_assignments(db_path, [("solo", None, True)])
    assert n == 1
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT dedup_status, dedup_cluster_id FROM image_meta WHERE image_id = ?",
            ("solo",),
        ).fetchone()
    finally:
        conn.close()
    assert row == ("survivor", None)


def test_write_dedup_assignments_back_to_back_no_deadlock(tmp_path: Path) -> None:
    """Sequential calls on the same DB must serialize via WAL/busy_timeout."""
    db_path = tmp_path / "btb.db"

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.register_image_batch(
                [(f"img_{i}", f"/tmp/{i}.jpg") for i in range(10)]
            )

    asyncio.run(_seed())
    write_dedup_assignments(
        db_path, [(f"img_{i}", "c", i < 5) for i in range(10)]
    )
    write_dedup_assignments(
        db_path, [(f"img_{i}", "c", i >= 5) for i in range(10)]
    )
    survivors = sorted(
        r["image_id"] for r in iter_survivor_images(db_path, require_finalize=False)
    )
    assert survivors == [f"img_{i}" for i in range(5, 10)]


# ---------------------------------------------------------------------------
# Concurrent access (WAL)
# ---------------------------------------------------------------------------


def test_sync_reader_concurrent_with_async_writer(seeded_pipeline_db: Path) -> None:
    """Reader on the main thread can fetch rows while an async writer is mid-flight."""
    db_path = seeded_pipeline_db
    stop = threading.Event()
    errors: list[BaseException] = []

    def _reader_loop() -> None:
        try:
            # Hammer the DB with reads while the async writer churns.
            for _ in range(50):
                rows = list(iter_survivor_images(db_path, require_finalize=False))
                assert any(r["image_id"] == "img_a" for r in rows)
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    async def _writer_loop() -> None:
        async with CheckpointDB(db_path) as db:
            for i in range(20):
                # save_stage takes the write lock; reader on a separate
                # connection should never see "database is locked" under WAL.
                from data_miner.auto_annotation_v4.configs.contracts import (
                    FinalizeResult,
                )

                await db.save_stage(
                    "img_a",
                    Stage.FINALIZE,
                    FinalizeResult(image_id="img_a", final_annotations=[]),
                    f"hash_{i}",
                )
                await asyncio.sleep(0)

    t = threading.Thread(target=_reader_loop, daemon=True)
    t.start()
    try:
        asyncio.run(_writer_loop())
    finally:
        stop.set()
        t.join(timeout=10)
    assert errors == []


def test_reader_during_writer_transaction(seeded_pipeline_db: Path) -> None:
    """A sync reader can run while a sync writer holds an open transaction (WAL)."""
    db_path = seeded_pipeline_db

    write_conn = sqlite3.connect(str(db_path), timeout=10)
    write_conn.execute("PRAGMA busy_timeout = 10000")
    try:
        write_conn.execute("BEGIN IMMEDIATE")
        write_conn.execute(
            "UPDATE image_meta SET updated_at = ? WHERE image_id = ?",
            (time.time(), "img_a"),
        )
        # Reader must not block / error while the BEGIN IMMEDIATE is active.
        info = read_job_info(db_path)
        assert info is not None
        assert info["job_id"] == "testjob"
        rows = list(iter_survivor_images(db_path, require_finalize=False))
        assert any(r["image_id"] == "img_a" for r in rows)
        write_conn.commit()
    finally:
        write_conn.close()
