"""Tests for the LS annotation push-backup handler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manual_reviewer.ml_backend.lswebhook import (
    _extract_project_id,
    process_event,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _annotation(
    *,
    annotation_id: int = 100,
    task_id: int = 14,
    project_id: int = 3,
    result: list[dict] | None = None,
    **extra,
) -> dict:
    base = {
        "id": annotation_id,
        "task": task_id,
        "project": project_id,
        "result": result if result is not None else [
            {
                "id": "region_1",
                "type": "rectanglelabels",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["car"]},
            }
        ],
    }
    base.update(extra)
    return base


def _payload(action: str, **extra) -> dict:
    return {"action": action, **extra}


# ---------------------------------------------------------------------------
# 1. Project id extraction (nested LS payload formats)
# ---------------------------------------------------------------------------


def test_extract_project_id_top_level_int() -> None:
    assert _extract_project_id({"project": 3}) == 3


def test_extract_project_id_top_level_object() -> None:
    assert _extract_project_id({"project": {"id": 3}}) == 3


def test_extract_project_id_nested_in_annotation() -> None:
    assert _extract_project_id({"annotation": {"project": 5}}) == 5


def test_extract_project_id_missing_returns_none() -> None:
    assert _extract_project_id({"action": "PROJECT_UPDATED"}) is None


# ---------------------------------------------------------------------------
# 2. ANNOTATION_CREATED — happy path
# ---------------------------------------------------------------------------


def test_created_writes_event_log(tmp_path: Path) -> None:
    payload = _payload("ANNOTATION_CREATED", annotation=_annotation())
    summary = process_event(payload, backup_dir=tmp_path)
    assert summary["ok"] is True
    assert summary["action"] == "ANNOTATION_CREATED"
    assert summary["project_id"] == 3
    assert summary["annotation_id"] == 100
    assert summary["task_id"] == 14

    log = (tmp_path / "project_3" / "events.jsonl").read_text()
    lines = [json.loads(line) for line in log.splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["payload"]["action"] == "ANNOTATION_CREATED"
    assert "received_at" in lines[0]


def test_created_writes_annotation_snapshot(tmp_path: Path) -> None:
    payload = _payload("ANNOTATION_CREATED", annotation=_annotation())
    process_event(payload, backup_dir=tmp_path)
    snap = tmp_path / "project_3" / "annotations" / "100.json"
    assert snap.exists()
    body = json.loads(snap.read_text())
    assert body["id"] == 100
    assert body["task"] == 14


def test_created_refreshes_task_aggregate(tmp_path: Path) -> None:
    payload = _payload("ANNOTATION_CREATED", annotation=_annotation())
    process_event(payload, backup_dir=tmp_path)
    agg = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    assert agg["task_id"] == 14
    assert agg["annotation_ids"] == [100]
    assert len(agg["annotations"]) == 1
    assert agg["annotations"][0]["id"] == 100


# ---------------------------------------------------------------------------
# 3. ANNOTATION_UPDATED — overwrite, audit appended
# ---------------------------------------------------------------------------


def test_updated_overwrites_snapshot(tmp_path: Path) -> None:
    process_event(
        _payload("ANNOTATION_CREATED", annotation=_annotation(result=[
            {"id": "r1", "type": "rectanglelabels",
             "value": {"x": 1, "y": 1, "width": 1, "height": 1,
                       "rectanglelabels": ["car"]}}
        ])),
        backup_dir=tmp_path,
    )
    process_event(
        _payload("ANNOTATION_UPDATED", annotation=_annotation(result=[
            {"id": "r1", "type": "rectanglelabels",
             "value": {"x": 50, "y": 50, "width": 5, "height": 5,
                       "rectanglelabels": ["truck"]}}
        ])),
        backup_dir=tmp_path,
    )
    snap = json.loads(
        (tmp_path / "project_3" / "annotations" / "100.json").read_text()
    )
    assert snap["result"][0]["value"]["rectanglelabels"] == ["truck"]
    # Audit log preserves both events.
    log_lines = [
        json.loads(l) for l in
        (tmp_path / "project_3" / "events.jsonl").read_text().splitlines()
        if l
    ]
    actions = [e["payload"]["action"] for e in log_lines]
    assert actions == ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]


# ---------------------------------------------------------------------------
# 4. ANNOTATIONS_DELETED — LS 1.23 payload (always plural, ID-only items)
# ---------------------------------------------------------------------------


def test_annotations_deleted_moves_snapshot_to_deleted_dir(tmp_path: Path) -> None:
    """LS 1.23 ANNOTATIONS_DELETED payload: ``annotations: [{"id": N}, ...]``
    — ID-only because the annotations are gone server-side."""
    process_event(
        _payload("ANNOTATION_CREATED", annotation=_annotation()),
        backup_dir=tmp_path,
    )
    summary = process_event(
        {"action": "ANNOTATIONS_DELETED",
         "project": {"id": 3},
         "annotations": [{"id": 100}]},
        backup_dir=tmp_path,
    )
    live = tmp_path / "project_3" / "annotations" / "100.json"
    deleted_dir = tmp_path / "project_3" / "annotations" / "deleted"
    assert not live.exists()
    assert deleted_dir.exists()
    deleted_files = list(deleted_dir.glob("100-*.json"))
    assert len(deleted_files) == 1
    body = json.loads(deleted_files[0].read_text())
    assert body["id"] == 100
    assert summary["moved_count"] == 1
    assert summary["affected_tasks"] == [14]


def test_annotations_deleted_refreshes_aggregate_to_empty(tmp_path: Path) -> None:
    process_event(
        _payload("ANNOTATION_CREATED", annotation=_annotation()),
        backup_dir=tmp_path,
    )
    process_event(
        {"action": "ANNOTATIONS_DELETED",
         "project": {"id": 3},
         "annotations": [{"id": 100}]},
        backup_dir=tmp_path,
    )
    agg = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    assert agg["annotation_ids"] == []
    assert agg["annotations"] == []


def test_annotations_deleted_bulk_clears_multiple(tmp_path: Path) -> None:
    """Bulk delete of multiple annotations across the same task."""
    process_event(
        _payload("ANNOTATION_CREATED",
                 annotation=_annotation(annotation_id=100, task_id=14)),
        backup_dir=tmp_path,
    )
    process_event(
        _payload("ANNOTATION_CREATED",
                 annotation=_annotation(annotation_id=101, task_id=14)),
        backup_dir=tmp_path,
    )
    summary = process_event(
        {"action": "ANNOTATIONS_DELETED",
         "project": {"id": 3},
         "annotations": [{"id": 100}, {"id": 101}]},
        backup_dir=tmp_path,
    )
    assert summary["moved_count"] == 2
    assert summary["affected_tasks"] == [14]
    agg = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    assert agg["annotation_ids"] == []


def test_annotations_deleted_when_snapshot_missing_is_idempotent(tmp_path: Path) -> None:
    """LS may re-deliver a DELETED for an annotation we never saw —
    or a delete that already ran. Don't crash; just record the event."""
    summary = process_event(
        {"action": "ANNOTATIONS_DELETED",
         "project": {"id": 3},
         "annotations": [{"id": 100}]},
        backup_dir=tmp_path,
    )
    assert summary["moved_count"] == 0
    log = (tmp_path / "project_3" / "events.jsonl").read_text()
    assert "ANNOTATIONS_DELETED" in log


# ---------------------------------------------------------------------------
# 5. Re-delivery (LS retries on 5xx) — idempotent
# ---------------------------------------------------------------------------


def test_redelivery_is_idempotent(tmp_path: Path) -> None:
    payload = _payload("ANNOTATION_CREATED", annotation=_annotation())
    process_event(payload, backup_dir=tmp_path)
    process_event(payload, backup_dir=tmp_path)
    # Both audit lines preserved.
    log_lines = (tmp_path / "project_3" / "events.jsonl").read_text().splitlines()
    assert len([l for l in log_lines if l]) == 2
    # Snapshot still single file, not duplicated.
    snap_dir = tmp_path / "project_3" / "annotations"
    assert len(list(snap_dir.glob("*.json"))) == 1
    # Aggregate still has one annotation, not two.
    agg = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    assert agg["annotation_ids"] == [100]


# ---------------------------------------------------------------------------
# 6. ANNOTATIONS_CREATED — bulk create variant (LS 1.23 ships this for batch ops)
# ---------------------------------------------------------------------------


def test_annotations_created_bulk_writes_all_snapshots(tmp_path: Path) -> None:
    """``ANNOTATIONS_CREATED`` payload: ``annotations: [<full>, <full>]``."""
    summary = process_event(
        {"action": "ANNOTATIONS_CREATED",
         "project": {"id": 3},
         "annotations": [
             _annotation(annotation_id=100, task_id=14),
             _annotation(annotation_id=101, task_id=14),
             _annotation(annotation_id=200, task_id=15),
         ]},
        backup_dir=tmp_path,
    )
    assert sorted(summary["annotation_ids"]) == [100, 101, 200]
    assert summary["affected_tasks"] == [14, 15]
    snap_dir = tmp_path / "project_3" / "annotations"
    assert (snap_dir / "100.json").exists()
    assert (snap_dir / "101.json").exists()
    assert (snap_dir / "200.json").exists()
    agg14 = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    agg15 = json.loads((tmp_path / "project_3" / "tasks" / "15.json").read_text())
    assert agg14["annotation_ids"] == [100, 101]
    assert agg15["annotation_ids"] == [200]


# ---------------------------------------------------------------------------
# 7. Cross-task isolation — refreshing task A doesn't surface task B's anns
# ---------------------------------------------------------------------------


def test_aggregate_filters_by_task_id(tmp_path: Path) -> None:
    process_event(
        _payload("ANNOTATION_CREATED",
                 annotation=_annotation(annotation_id=100, task_id=14)),
        backup_dir=tmp_path,
    )
    process_event(
        _payload("ANNOTATION_CREATED",
                 annotation=_annotation(annotation_id=200, task_id=15)),
        backup_dir=tmp_path,
    )
    agg_14 = json.loads((tmp_path / "project_3" / "tasks" / "14.json").read_text())
    agg_15 = json.loads((tmp_path / "project_3" / "tasks" / "15.json").read_text())
    assert agg_14["annotation_ids"] == [100]
    assert agg_15["annotation_ids"] == [200]


# ---------------------------------------------------------------------------
# 8. Unknown action — payload still captured to events.jsonl
# ---------------------------------------------------------------------------


def test_unknown_action_still_logged(tmp_path: Path) -> None:
    summary = process_event(
        {"action": "PROJECT_UPDATED", "project": {"id": 3}},
        backup_dir=tmp_path,
    )
    assert summary["ok"] is True
    assert summary["unhandled"] is True
    log = (tmp_path / "project_3" / "events.jsonl").read_text()
    assert "PROJECT_UPDATED" in log


def test_missing_project_id_still_records_to_unknown_bucket(tmp_path: Path) -> None:
    """Audit must never silently drop an event."""
    process_event(
        {"action": "ANNOTATION_CREATED", "annotation": {"id": 999}},
        backup_dir=tmp_path,
    )
    log = (tmp_path / "project_unknown" / "events.jsonl").read_text()
    assert "999" in log


# ---------------------------------------------------------------------------
# 9. Non-dict payload — graceful failure
# ---------------------------------------------------------------------------


def test_non_dict_payload_returns_error(tmp_path: Path) -> None:
    out = process_event("not a dict", backup_dir=tmp_path)  # type: ignore[arg-type]
    assert out == {"ok": False, "reason": "payload_not_dict"}


# ---------------------------------------------------------------------------
# 10. Atomic write robustness — partial files don't appear
# ---------------------------------------------------------------------------


def test_no_tmp_files_left_behind(tmp_path: Path) -> None:
    process_event(
        _payload("ANNOTATION_CREATED", annotation=_annotation()),
        backup_dir=tmp_path,
    )
    # Walk the project dir; no .tmp.* leftovers.
    leftovers = [p for p in (tmp_path / "project_3").rglob("*")
                 if ".tmp." in p.name]
    assert leftovers == []
