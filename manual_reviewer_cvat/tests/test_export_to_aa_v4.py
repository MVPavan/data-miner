"""Tests for the CVAT export-to-aa_v4 command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from manual_reviewer_cvat.scripts import export_to_aa_v4 as mod


def _datumaro_doc() -> dict[str, Any]:
    """Build a minimal Datumaro document for exporter tests."""
    return {
        "version": "1.0",
        "categories": {"label": [{"id": 0, "name": "forklift"}]},
        "items": [
            {
                "id": "img_a",
                "image": {"path": "images/img_a.jpg", "size": [640, 480]},
                "attributes": {"image_id": "img_a", "task_id": 12, "job_id": 34},
                "annotations": [
                    {
                        "id": 1,
                        "type": "bbox",
                        "x": 64,
                        "y": 48,
                        "w": 128,
                        "h": 96,
                        "label": 0,
                        "attributes": {"candidate_id": "cand_1", "source": "edited"},
                    }
                ],
            }
        ],
    }


def test_main_writes_local_datumaro_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local Datumaro JSON should parse and write v4 human-review rows."""
    datumaro_path = tmp_path / "default.json"
    datumaro_path.write_text(json.dumps(_datumaro_doc()), encoding="utf-8")
    db_path = tmp_path / "pipeline.db"
    db_path.write_text("", encoding="utf-8")
    captured: list[tuple[Path, Any, str]] = []

    def _fake_write(path: Path, result: Any, *, config_hash: str = "") -> None:
        captured.append((path, result, config_hash))

    monkeypatch.setattr(mod, "write_human_review", _fake_write)

    rc = mod.main(
        [
            "--datumaro-json",
            str(datumaro_path),
            "--pipeline-db",
            str(db_path),
            "--reviewer-id",
            "reviewer@example.com",
            "--reviewed-at",
            "2026-05-04T12:00:00Z",
            "--config-hash",
            "h1",
        ]
    )

    assert rc == 0
    assert len(captured) == 1
    path, result, config_hash = captured[0]
    assert path == db_path.resolve()
    assert config_hash == "h1"
    assert result.image_id == "img_a"
    assert result.reviewer_id == "reviewer@example.com"
    assert result.corrections[0].candidate_id == "cand_1"
    assert result.corrections[0].source == "edited"


def test_main_dry_run_skips_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run should parse Datumaro but avoid DB writes."""
    datumaro_path = tmp_path / "default.json"
    datumaro_path.write_text(json.dumps(_datumaro_doc()), encoding="utf-8")
    db_path = tmp_path / "pipeline.db"
    db_path.write_text("", encoding="utf-8")

    def _fail_write(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("write_human_review should not be called")

    monkeypatch.setattr(mod, "write_human_review", _fail_write)

    rc = mod.main(
        [
            "--datumaro-json",
            str(datumaro_path),
            "--pipeline-db",
            str(db_path),
            "--reviewer-id",
            "reviewer@example.com",
            "--dry-run",
        ]
    )

    assert rc == 0


def test_argparse_requires_reviewer_for_datumaro(tmp_path: Path) -> None:
    """Offline Datumaro mode needs an explicit reviewer id."""
    datumaro_path = tmp_path / "default.json"
    datumaro_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        mod._parse_args(
            [
                "--datumaro-json",
                str(datumaro_path),
                "--pipeline-db",
                str(tmp_path / "pipeline.db"),
            ]
        )


def test_live_cvat_mode_is_explicitly_pending(tmp_path: Path) -> None:
    """Live CVAT API mode should fail clearly until implemented."""
    db_path = tmp_path / "pipeline.db"
    db_path.write_text("", encoding="utf-8")

    rc = mod.main(
        [
            "--cvat-url",
            "http://127.0.0.1:8081",
            "--cvat-user",
            "admin",
            "--cvat-pass",
            "pw",
            "--cvat-project",
            "1",
            "--pipeline-db",
            str(db_path),
        ]
    )

    assert rc == 2