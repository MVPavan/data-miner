"""Tests for the cron-driven LS → disk pull sync.

Mocks ``httpx`` so each test runs offline against a synthetic LS
project; asserts that the same on-disk schema produced by the webhook
handler comes out, regardless of source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from manual_reviewer.scripts import sync_ls_to_disk as mod


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    """Tiny stand-in for ``httpx.Client`` that returns one canned page
    of tasks then an empty page. Each ``__init__`` resets state so test
    ordering doesn't matter."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        self._page = 0
        # Class-level pages set by tests via ``_FakeClient._next_pages``.
        self._pages = list(_FakeClient._next_pages)

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, *exc) -> None:  # noqa: D401
        return None

    def get(self, url: str, params: dict[str, Any] | None = None):
        if self._page >= len(self._pages):
            return _FakeResp([])
        page_data = self._pages[self._page]
        self._page += 1
        return _FakeResp(page_data)


# Set per-test by populating before invocation.
_FakeClient._next_pages: list[Any] = []  # type: ignore[attr-defined]


def _set_pages(*pages: list[dict[str, Any]]) -> None:
    _FakeClient._next_pages = list(pages)  # type: ignore[attr-defined]


def _ann(annotation_id: int, task_id: int, value_x: float = 10.0,
         labels: tuple[str, ...] = ("car",)) -> dict[str, Any]:
    return {
        "id": annotation_id,
        "task": task_id,
        "project": 3,
        "result": [{
            "id": f"r_{annotation_id}",
            "type": "rectanglelabels",
            "from_name": "bbox", "to_name": "image",
            "value": {
                "x": value_x, "y": 10, "width": 5, "height": 5,
                "rectanglelabels": list(labels),
            },
        }],
    }


def _task(task_id: int, *anns: dict[str, Any]) -> dict[str, Any]:
    return {"id": task_id, "annotations": list(anns)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_httpx():
    with patch("httpx.Client", _FakeClient):
        yield


def _run(project_id: int, tmp_path: Path) -> int:
    return mod.main([
        "--ls-url", "http://x",
        "--ls-token", "t",
        "--project", str(project_id),
        "--backup-dir", str(tmp_path),
    ])


def test_first_run_creates_snapshots_and_aggregates(tmp_path: Path) -> None:
    _set_pages([_task(14, _ann(100, 14))])
    rc = _run(3, tmp_path)
    assert rc == 0
    snap = json.loads(
        (tmp_path / "project_3" / "annotations" / "100.json").read_text()
    )
    assert snap["id"] == 100
    agg = json.loads(
        (tmp_path / "project_3" / "tasks" / "14.json").read_text()
    )
    assert agg["annotation_ids"] == [100]


def test_re_run_with_no_changes_appends_no_events(tmp_path: Path) -> None:
    _set_pages([_task(14, _ann(100, 14))])
    _run(3, tmp_path)
    log_path = tmp_path / "project_3" / "events.jsonl"
    size_before = log_path.stat().st_size
    _set_pages([_task(14, _ann(100, 14))])  # identical second run
    _run(3, tmp_path)
    assert log_path.stat().st_size == size_before


def test_change_to_annotation_emits_update_event(tmp_path: Path) -> None:
    _set_pages([_task(14, _ann(100, 14, value_x=10.0))])
    _run(3, tmp_path)
    _set_pages([_task(14, _ann(100, 14, value_x=99.0))])
    _run(3, tmp_path)
    log_lines = [
        json.loads(line) for line in
        (tmp_path / "project_3" / "events.jsonl").read_text().splitlines() if line
    ]
    actions = [e["payload"]["action"] for e in log_lines]
    assert actions == ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]


def test_disappeared_annotation_emits_delete(tmp_path: Path) -> None:
    _set_pages([_task(14, _ann(100, 14), _ann(101, 14))])
    _run(3, tmp_path)
    # Now LS only returns annotation 100; 101 was deleted.
    _set_pages([_task(14, _ann(100, 14))])
    _run(3, tmp_path)
    log_lines = [
        json.loads(line) for line in
        (tmp_path / "project_3" / "events.jsonl").read_text().splitlines() if line
    ]
    delete_events = [e for e in log_lines
                     if e["payload"]["action"] == "ANNOTATIONS_DELETED"]
    assert len(delete_events) == 1
    assert delete_events[0]["payload"]["annotations"][0]["id"] == 101
    assert not (tmp_path / "project_3" / "annotations" / "101.json").exists()
    assert (tmp_path / "project_3" / "annotations" / "100.json").exists()


def test_volatile_fields_dont_trigger_updates(tmp_path: Path) -> None:
    """LS sometimes refreshes ``updated_at`` server-side without a real
    body change. The sync must not flag those as updates."""
    a = _ann(100, 14)
    a["updated_at"] = "2026-04-30T01:00:00"
    a["lead_time"] = 5.0
    _set_pages([_task(14, a)])
    _run(3, tmp_path)
    log_path = tmp_path / "project_3" / "events.jsonl"
    size_before = log_path.stat().st_size
    a2 = _ann(100, 14)
    a2["updated_at"] = "2026-04-30T02:00:00"   # different timestamp
    a2["lead_time"] = 99.0                       # different lead_time
    _set_pages([_task(14, a2)])
    _run(3, tmp_path)
    assert log_path.stat().st_size == size_before, \
        "volatile-only diffs must not produce an UPDATE event"


def test_multiple_tasks_isolated(tmp_path: Path) -> None:
    _set_pages([
        _task(14, _ann(100, 14)),
        _task(15, _ann(200, 15)),
    ])
    _run(3, tmp_path)
    agg14 = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    agg15 = json.loads((tmp_path / "project_3" / "tasks" / "15.json").read_text())
    assert agg14["annotation_ids"] == [100]
    assert agg15["annotation_ids"] == [200]
