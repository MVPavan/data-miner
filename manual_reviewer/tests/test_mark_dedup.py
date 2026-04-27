"""Cluster + flat manifest → image_meta dedup status round-trip."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from manual_reviewer.pipeline_io import iter_survivor_images, write_dedup_assignments


@pytest.fixture
def empty_db_with_images(tmp_path: Path) -> Path:
    db_path = tmp_path / "pipeline.db"

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.save_job_info(
                job_id="dedup_test",
                image_dir="/tmp",
                config_hash="h",
                prompt_version="v",
            )
            await db.register_image_batch(
                [
                    ("frame_0001", "/tmp/frame_0001.jpg"),
                    ("frame_0002", "/tmp/frame_0002.jpg"),
                    ("frame_0003", "/tmp/frame_0003.jpg"),
                    ("frame_0004", "/tmp/frame_0004.jpg"),
                ]
            )

    asyncio.run(_seed())
    return db_path


def test_write_dedup_assignments_marks_survivors_and_drops(empty_db_with_images: Path) -> None:
    write_dedup_assignments(
        empty_db_with_images,
        [
            ("frame_0001", "c001", True),
            ("frame_0002", "c001", False),
            ("frame_0003", "c001", False),
            ("frame_0004", "c002", True),
        ],
    )
    conn = sqlite3.connect(str(empty_db_with_images))
    rows = conn.execute(
        "SELECT image_id, dedup_status, dedup_cluster_id FROM image_meta ORDER BY image_id"
    ).fetchall()
    conn.close()
    assert rows == [
        ("frame_0001", "survivor", "c001"),
        ("frame_0002", "dropped", "c001"),
        ("frame_0003", "dropped", "c001"),
        ("frame_0004", "survivor", "c002"),
    ]

    survivors = sorted(s["image_id"] for s in iter_survivor_images(empty_db_with_images, require_finalize=False))
    assert survivors == ["frame_0001", "frame_0004"]


def test_write_dedup_assignments_skips_unknown_image_ids(empty_db_with_images: Path) -> None:
    updated = write_dedup_assignments(
        empty_db_with_images,
        [
            ("frame_0001", "c001", True),
            ("nonexistent", "c001", False),
        ],
    )
    assert updated == 1


def test_mark_dedup_cli_with_cluster_manifest(empty_db_with_images: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "clusters.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "cluster_id": "c001",
                    "survivor": "frame_0001",
                    "dropped": ["frame_0002", "frame_0003"],
                },
                {"cluster_id": "c002", "survivor": "frame_0004", "dropped": []},
            ]
        )
    )
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "manual_reviewer.scripts.mark_dedup",
            "--db",
            str(empty_db_with_images),
            "--manifest-clusters",
            str(manifest),
        ]
    )
    assert rc == 0

    survivors = sorted(s["image_id"] for s in iter_survivor_images(empty_db_with_images, require_finalize=False))
    assert survivors == ["frame_0001", "frame_0004"]


def test_mark_dedup_cli_with_flat_manifest(empty_db_with_images: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "flat.json"
    manifest.write_text(
        json.dumps(
            [
                {"image_id": "frame_0001", "cluster_id": "c1", "is_survivor": True},
                {"image_id": "frame_0002", "cluster_id": "c1", "is_survivor": False},
                {"image_id": "frame_0003", "cluster_id": "c2", "is_survivor": True},
                {"image_id": "frame_0004", "cluster_id": "c2", "is_survivor": False},
            ]
        )
    )
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "manual_reviewer.scripts.mark_dedup",
            "--db",
            str(empty_db_with_images),
            "--manifest-flat",
            str(manifest),
        ]
    )
    assert rc == 0

    survivors = sorted(s["image_id"] for s in iter_survivor_images(empty_db_with_images, require_finalize=False))
    assert survivors == ["frame_0001", "frame_0003"]
