"""Regression tests for the issues raised by the subagent review.

Each test corresponds to a fix in db_reader / ls_export_parser / viewer.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from data_miner.auto_annotation_v4.configs.contracts import BoundingBox
from manual_reviewer.pipeline_io import iter_survivor_images
from manual_reviewer.pipeline_io.ls_export_parser import _ls_value_to_bbox


# ---------------------------------------------------------------------------
# BLOCKER 1: defensive bbox normalisation
# ---------------------------------------------------------------------------


def test_negative_width_renormalises_corners() -> None:
    """A reviewer/ML-backend rectangle drawn 'backwards' must come out as a
    valid BoundingBox (width > 0) after parsing — not a zero-width artefact.
    """
    bbox = _ls_value_to_bbox(
        {"x": 60.0, "y": 70.0, "width": -30.0, "height": -40.0}
    )
    assert isinstance(bbox, BoundingBox)
    assert bbox.x1 == pytest.approx(0.30)
    assert bbox.x2 == pytest.approx(0.60)
    assert bbox.y1 == pytest.approx(0.30)
    assert bbox.y2 == pytest.approx(0.70)
    assert bbox.width > 0
    assert bbox.height > 0


def test_inverted_already_clamped_box_renormalises() -> None:
    """Backward drag near the right edge: x=110, width=-30 → clamped to
    [0,1] would invert without the swap. Swap restores ordering.
    """
    bbox = _ls_value_to_bbox(
        {"x": 110.0, "y": 50.0, "width": -30.0, "height": 20.0}
    )
    assert bbox.x1 <= bbox.x2
    assert bbox.y1 <= bbox.y2


# ---------------------------------------------------------------------------
# BUG 5: iter_survivor_images falls back when migration hasn't run
# ---------------------------------------------------------------------------


def _make_legacy_db(path: Path) -> None:
    """Build an aa_v4-shaped DB without the dedup columns (pre-migration)."""

    async def _seed() -> None:
        async with aiosqlite.connect(str(path)) as db:
            await db.executescript(
                """
                CREATE TABLE image_meta (
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
                INSERT INTO image_meta (image_id, image_path, stages_completed, created_at, updated_at)
                VALUES
                  ('legacy_a', '/tmp/a.jpg', '["detect","filter","evaluate","refine","finalize"]', 1.0, 1.0),
                  ('legacy_b', '/tmp/b.jpg', '[]',                                                  1.0, 1.0);
                """
            )
            await db.commit()

    asyncio.run(_seed())


def test_iter_survivor_falls_back_on_unmigrated_db(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    # require_finalize=True keeps only legacy_a
    finalized = list(iter_survivor_images(db_path, require_finalize=True))
    assert [r["image_id"] for r in finalized] == ["legacy_a"]
    # The fallback synthesises NULL columns rather than crashing.
    assert finalized[0]["dedup_status"] is None
    assert finalized[0]["dedup_cluster_id"] is None


def test_iter_survivor_unmigrated_includes_all_with_finalize_off(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)
    rows = list(iter_survivor_images(db_path, require_finalize=False))
    assert sorted(r["image_id"] for r in rows) == ["legacy_a", "legacy_b"]


# ---------------------------------------------------------------------------
# BLOCKER 3: viewer surfaces human_review in PIPELINE_STAGES + /api/data
# ---------------------------------------------------------------------------


def test_viewer_pipeline_stages_includes_human_review() -> None:
    from data_miner.auto_annotation_v4.viewer import app as viewer_app

    assert "human_review" in viewer_app.PIPELINE_STAGES


def test_viewer_data_endpoint_exposes_human_review_key(seeded_pipeline_db: Path) -> None:
    """The viewer's /api/data/{image_id} response must include the
    ``human_review`` field so the frontend can render reviewer corrections.
    Tests the response shape via the FastAPI app; we don't need a live
    server, just a ``TestClient``.
    """
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult
    from data_miner.auto_annotation_v4.viewer.app import create_app
    from manual_reviewer.pipeline_io import write_human_review

    write_human_review(
        seeded_pipeline_db,
        HumanReviewResult(
            image_id="img_a", reviewer_id="reviewer", reviewed_at=1.0,
        ),
    )

    job_dir = seeded_pipeline_db.parent
    app = create_app(job_dir)
    client = TestClient(app)
    resp = client.get("/api/data/img_a")
    assert resp.status_code == 200
    body = resp.json()
    assert "human_review" in body
    assert body["human_review"] is not None
    assert body["human_review"]["reviewer_id"] == "reviewer"
