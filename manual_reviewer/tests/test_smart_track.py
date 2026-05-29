"""Tests for the SAM 3.1 tracker-driven cross-frame propagation.

Covers four layers, none requiring a live SAM 3.1 server or LS:

  1. ``pipeline_io.clip_id`` — the regex helper.
  2. ``ml_backend.smart_track_lib`` — JPEG folder builder, response filter,
     sibling discovery, and the orchestrator with mocked clients.
  3. ``ml_backend.routes.smart_track`` — route preconditions + delegation.
  4. ``ml_backend.ls_rest`` — REST client behavior (mocked httpx).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data_miner.auto_annotation_v4.configs.wire import (
    SAM3VideoTrackFrameOutput,
    SAM3VideoTrackObjectOutput,
    SAM3VideoTrackResponse,
    SAM3VideoTrackSeed,
)

from manual_reviewer.ml_backend.routes import smart_track
from manual_reviewer.ml_backend.smart_track_lib import (
    PropagateResult,
    Sibling,
    build_jpeg_folder,
    filter_track_response,
    find_siblings,
    propagate_via_tracker,
)
from manual_reviewer.pipeline_io.clip_id import clip_prefix


# ---------------------------------------------------------------------------
# clip_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_id, expected",
    [
        ("Caifu_Center_Fewer_2_f00516", "Caifu_Center_Fewer_2"),
        ("clip_a_f00000", "clip_a"),
        ("singleshot", "singleshot"),
        ("clip_with_underscore_f000", "clip_with_underscore"),
        # No frame index suffix: returns full id (one-element clip).
        ("Moto_Bicycle", "Moto_Bicycle"),
        # Pathological: id starts with _f<digits> — strip leaves empty,
        # falls back to full id.
        ("_f00000", "_f00000"),
    ],
)
def test_clip_prefix(image_id: str, expected: str) -> None:
    assert clip_prefix(image_id) == expected


# ---------------------------------------------------------------------------
# JPEG folder builder
# ---------------------------------------------------------------------------


def test_build_jpeg_folder_creates_ordered_symlinks(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jpg"
    sib_a = tmp_path / "sib_a.jpg"
    sib_b = tmp_path / "sib_b.jpg"
    for p in (seed, sib_a, sib_b):
        p.write_bytes(b"jpeg")

    folder = build_jpeg_folder(str(seed), [str(sib_a), str(sib_b)], parent=tmp_path)
    try:
        names = sorted(p.name for p in folder.iterdir())
        assert names == ["00000.jpg", "00001.jpg", "00002.jpg"]
        assert (folder / "00000.jpg").resolve() == seed.resolve()
        assert (folder / "00001.jpg").resolve() == sib_a.resolve()
        assert (folder / "00002.jpg").resolve() == sib_b.resolve()
    finally:
        import shutil
        shutil.rmtree(folder)


# ---------------------------------------------------------------------------
# filter_track_response
# ---------------------------------------------------------------------------


def _resp(frames: list[tuple[int, list[tuple[int, list[float] | None, float]]]]) -> SAM3VideoTrackResponse:
    """Helper: build a SAM3VideoTrackResponse from compact tuples."""
    out_frames = []
    for frame_index, objs in frames:
        out_objs = [
            SAM3VideoTrackObjectOutput(obj_id=oid, bbox=bbox, score=score)
            for oid, bbox, score in objs
        ]
        out_frames.append(SAM3VideoTrackFrameOutput(frame_index=frame_index, objects=out_objs))
    return SAM3VideoTrackResponse(frames=out_frames)


def test_filter_track_response_static_success() -> None:
    seed = [0.10, 0.10, 0.20, 0.20]
    resp = _resp([
        (0, [(1, seed, 0.95)]),                           # seed frame, ignored
        (1, [(1, [0.10, 0.10, 0.20, 0.20], 0.94)]),
        (2, [(1, [0.11, 0.11, 0.20, 0.21], 0.93)]),
    ])
    per_frame, stats = filter_track_response(resp, seed_bbox=seed)
    assert set(per_frame) == {1, 2}
    assert per_frame[1][1] == pytest.approx(0.94)
    assert stats == {"motion": 0, "score": 0, "missing": 0}


def test_filter_track_response_drift_rejected() -> None:
    seed = [0.10, 0.10, 0.20, 0.20]
    # Frame 1: object moved >0.05 from seed center — rejected.
    resp = _resp([
        (1, [(1, [0.40, 0.40, 0.50, 0.50], 0.95)]),
    ])
    per_frame, stats = filter_track_response(resp, seed_bbox=seed)
    assert per_frame == {}
    assert stats["motion"] == 1


def test_filter_track_response_low_score_rejected() -> None:
    seed = [0.10, 0.10, 0.20, 0.20]
    # Static-position but score below threshold.
    resp = _resp([
        (1, [(1, [0.10, 0.10, 0.20, 0.20], 0.30)]),
    ])
    per_frame, stats = filter_track_response(resp, seed_bbox=seed, score_thresh=0.5)
    assert per_frame == {}
    assert stats["score"] == 1


def test_filter_track_response_missing_object_rejected() -> None:
    seed = [0.10, 0.10, 0.20, 0.20]
    # Frame 1 has an object but no bbox — counted as missing.
    resp = _resp([
        (1, [(1, None, 0.0)]),
    ])
    per_frame, stats = filter_track_response(resp, seed_bbox=seed)
    assert per_frame == {}
    assert stats["missing"] == 1


def test_filter_track_response_picks_closest_when_multiple() -> None:
    """When temporal-disambiguation leaks extras, motion filter picks the seed match."""
    seed = [0.10, 0.10, 0.20, 0.20]
    resp = _resp([
        (1, [
            (0, [0.80, 0.80, 0.90, 0.90], 0.99),  # auto-detect, far from seed
            (1, [0.11, 0.11, 0.21, 0.21], 0.92),  # the actual seed propagation
            (2, [0.50, 0.10, 0.60, 0.20], 0.95),  # auto-detect, also far
        ]),
    ])
    per_frame, stats = filter_track_response(resp, seed_bbox=seed)
    # Only the seed-aligned object survives motion threshold.
    assert per_frame[1][0] == [0.11, 0.11, 0.21, 0.21]
    assert per_frame[1][1] == pytest.approx(0.92)


def test_filter_track_response_skips_seed_frame() -> None:
    """frame_index == 0 is the seed; never propagated even with high score."""
    seed = [0.10, 0.10, 0.20, 0.20]
    resp = _resp([
        (0, [(1, [0.10, 0.10, 0.20, 0.20], 0.99)]),
    ])
    per_frame, _ = filter_track_response(resp, seed_bbox=seed)
    assert per_frame == {}


# ---------------------------------------------------------------------------
# find_siblings
# ---------------------------------------------------------------------------


class _StubLSRest:
    """Mimic LSRestClient surface used by smart_track_lib.

    ``existing_predictions`` lets tests prime the per-task-id prediction
    list so list_predictions returns realistic state. Defaults to no
    existing predictions, in which case the new merge path falls back
    to post_prediction (preserving existing test semantics).
    """

    def __init__(
        self, tasks: list[dict[str, Any]],
        *,
        existing_predictions: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._tasks = tasks
        self._existing_predictions = existing_predictions or {}
        self.posted: list[dict[str, Any]] = []
        self.patched: list[dict[str, Any]] = []
        self.next_pid = 1000

    def iter_project_tasks(self, project_id: int, *, page_size: int = 200):
        for t in self._tasks:
            yield t

    def post_prediction(self, *, task_id, result, score, model_version):
        self.posted.append({
            "task_id": task_id,
            "result": result,
            "score": score,
            "model_version": model_version,
        })
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def list_predictions(self, task_id: int) -> list[dict[str, Any]]:
        return list(self._existing_predictions.get(task_id, []))

    def patch_prediction(self, prediction_id, *, result, score=None,
                         model_version=None) -> bool:
        self.patched.append({
            "prediction_id": prediction_id,
            "result": result,
            "score": score,
            "model_version": model_version,
        })
        return True


def test_find_siblings_filters_by_clip_prefix() -> None:
    tasks = [
        {"id": 1, "data": {"image_id": "vid_a_f00000", "image_path": "/a0.jpg"}},
        {"id": 2, "data": {"image_id": "vid_a_f00100", "image_path": "/a1.jpg"}},
        {"id": 3, "data": {"image_id": "vid_b_f00000", "image_path": "/b0.jpg"}},
        {"id": 4, "data": {"image_id": "vid_a_f00050", "image_path": "/a2.jpg"}},
    ]
    sibs = find_siblings(_StubLSRest(tasks), project_id=1, seed_image_id="vid_a_f00000")
    # Self excluded; vid_b excluded; remaining sorted by image_id.
    assert [s.image_id for s in sibs] == ["vid_a_f00050", "vid_a_f00100"]
    assert [s.task_id for s in sibs] == [4, 2]


def test_find_siblings_excludes_malformed_rows() -> None:
    tasks = [
        {"id": 1, "data": {"image_id": "vid_a_f00000", "image_path": "/a0.jpg"}},
        {"id": "not-int", "data": {"image_id": "vid_a_f00100", "image_path": "/a1.jpg"}},
        {"id": 3, "data": {"image_id": None, "image_path": "/x.jpg"}},
        {"id": 4, "data": {}},
    ]
    sibs = find_siblings(_StubLSRest(tasks), project_id=1, seed_image_id="vid_a_f00050")
    assert [s.image_id for s in sibs] == ["vid_a_f00000"]


def test_find_siblings_max_cap() -> None:
    tasks = [
        {"id": i, "data": {"image_id": f"v_f{i:05d}", "image_path": f"/{i}.jpg"}}
        for i in range(20)
    ]
    sibs = find_siblings(_StubLSRest(tasks), project_id=1, seed_image_id="v_f00000",
                        max_siblings=5)
    assert len(sibs) == 5


def test_find_siblings_filters_by_assigned_to() -> None:
    tasks = [
        {"id": 1, "data": {"image_id": "vid_a_f00000", "image_path": "/0.jpg",
                           "assigned_to": "pavan"}},
        {"id": 2, "data": {"image_id": "vid_a_f00100", "image_path": "/1.jpg",
                           "assigned_to": "pavan"}},
        {"id": 3, "data": {"image_id": "vid_a_f00200", "image_path": "/2.jpg",
                           "assigned_to": "sree"}},
        {"id": 4, "data": {"image_id": "vid_a_f00300", "image_path": "/3.jpg"}},
    ]
    sibs = find_siblings(
        _StubLSRest(tasks), project_id=1, seed_image_id="vid_a_f00000",
        assigned_to="pavan",
    )
    # sree's frame and the unassigned frame are both excluded.
    assert [s.image_id for s in sibs] == ["vid_a_f00100"]


def test_find_siblings_no_filter_returns_all_assignments() -> None:
    tasks = [
        {"id": 1, "data": {"image_id": "vid_a_f00000", "image_path": "/0.jpg",
                           "assigned_to": "pavan"}},
        {"id": 2, "data": {"image_id": "vid_a_f00100", "image_path": "/1.jpg",
                           "assigned_to": "sree"}},
    ]
    sibs = find_siblings(
        _StubLSRest(tasks), project_id=1, seed_image_id="vid_a_f00000",
    )
    # When assigned_to=None, the field is ignored — back-compat with
    # projects that haven't been rebuilt with --assignees yet.
    assert [s.image_id for s in sibs] == ["vid_a_f00100"]


# ---------------------------------------------------------------------------
# propagate_via_tracker (orchestrator)
# ---------------------------------------------------------------------------


class _StubSam3:
    def __init__(self, response: SAM3VideoTrackResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def track(self, *, resource_path, seeds, propagation_direction, return_masks=False, max_frames=None):
        self.calls.append({
            "resource_path": resource_path,
            "seeds": [s.model_dump() for s in seeds],
            "propagation_direction": propagation_direction,
        })
        return self._response


def test_propagate_via_tracker_happy_path(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jpg"
    sib_path_a = tmp_path / "sib_a.jpg"
    sib_path_b = tmp_path / "sib_b.jpg"
    for p in (seed_path, sib_path_a, sib_path_b):
        p.write_bytes(b"jpeg")

    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path_a)}},
        {"id": 102, "data": {"image_id": "clip_f00200", "image_path": str(sib_path_b)}},
    ])

    seed_bbox = [0.10, 0.10, 0.20, 0.20]
    resp = _resp([
        (0, [(1, seed_bbox, 0.95)]),
        (1, [(1, [0.10, 0.10, 0.20, 0.20], 0.94)]),
        (2, [(1, [0.40, 0.40, 0.50, 0.50], 0.95)]),  # drifted — rejected
    ])
    sam3 = _StubSam3(resp)

    result = propagate_via_tracker(
        sam3_client=sam3,
        ls_rest=ls,
        project_id=42,
        seed_image_path=str(seed_path),
        seed_image_id="clip_f00000",
        seed_bbox=seed_bbox,
        seed_label="forklift",
        current_task_id=100,
    )

    assert result.siblings_total == 2
    assert result.propagated == 1
    assert result.rejected_motion == 1
    assert result.written_task_ids == [101]
    # Posted prediction has correct structure.
    assert len(ls.posted) == 1
    posted = ls.posted[0]
    assert posted["task_id"] == 101
    assert posted["score"] == pytest.approx(0.94)
    assert posted["model_version"] == "sam3_1_track"
    region = posted["result"][0]
    assert region["type"] == "rectanglelabels"
    assert region["value"]["rectanglelabels"] == ["forklift"]
    assert region["meta"]["source"] == "smart_track"
    assert region["meta"]["from_image"] == "clip_f00000"
    assert region["meta"]["from_task"] == 100
    # SAM 3.1 was called with forward direction + correct seed.
    assert sam3.calls[0]["propagation_direction"] == "forward"
    assert sam3.calls[0]["seeds"][0]["bbox"] == seed_bbox


def test_propagate_via_tracker_no_siblings_short_circuits(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jpg"
    seed_path.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 1, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
    ])
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    result = propagate_via_tracker(
        sam3_client=sam3,
        ls_rest=ls,
        project_id=1,
        seed_image_path=str(seed_path),
        seed_image_id="clip_f00000",
        seed_bbox=[0.1, 0.1, 0.2, 0.2],
        seed_label="other",
    )
    assert result.siblings_total == 0
    assert result.propagated == 0
    assert sam3.calls == []  # Tracker not invoked when no siblings.
    assert ls.posted == []


def test_propagate_via_tracker_merges_into_existing_prediction(tmp_path: Path) -> None:
    """Sibling task has a YOLO prediction already → propagated region is
    PATCHed onto it (not POSTed as a separate prediction). Mirrors the
    project 9 task 222 fix from 2026-05-04: separate predictions caused
    LS to show only one of them on the canvas, hiding the YOLO labels.
    """
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")

    yolo_region = {
        "id": "yolo_0",
        "type": "rectanglelabels",
        "from_name": "bbox",
        "value": {"x": 5, "y": 5, "width": 10, "height": 10,
                  "rectanglelabels": ["car"]},
    }
    ls = _StubLSRest(
        [
            {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
            {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
        ],
        existing_predictions={
            101: [{
                "id": 9000,
                "model_version": "dataset_selection_yolo",
                "score": 0.5,
                "created_at": "2026-05-04T06:00:00Z",
                "result": [yolo_region],
            }],
        },
    )

    seed_bbox = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([
        (1, [(1, [0.10, 0.10, 0.20, 0.20], 0.93)]),
    ]))
    propagate_via_tracker(
        sam3_client=sam3, ls_rest=ls, project_id=42,
        seed_image_path=str(seed_path),
        seed_image_id="clip_f00000",
        seed_bbox=seed_bbox,
        seed_label="motorcycle",
        current_task_id=100,
    )
    # Did NOT POST a new prediction (would have broken LS one-prediction-on-canvas UX).
    assert ls.posted == []
    # Did PATCH the existing YOLO prediction with both regions.
    assert len(ls.patched) == 1
    patched = ls.patched[0]
    assert patched["prediction_id"] == 9000
    # 1 original YOLO + 1 propagated.
    assert len(patched["result"]) == 2
    assert patched["result"][0] == yolo_region
    propagated = patched["result"][1]
    assert propagated["meta"]["source"] == "smart_track"
    assert propagated["value"]["rectanglelabels"] == ["motorcycle"]
    # Score takes the max of existing + new.
    assert patched["score"] == pytest.approx(0.93)


def test_propagate_via_tracker_posts_when_no_existing_prediction(tmp_path: Path) -> None:
    """Sibling task has no predictions → fall back to POST, not PATCH."""
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")

    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])  # no existing_predictions
    sam3 = _StubSam3(_resp([
        (1, [(1, [0.10, 0.10, 0.20, 0.20], 0.9)]),
    ]))
    propagate_via_tracker(
        sam3_client=sam3, ls_rest=ls, project_id=1,
        seed_image_path=str(seed_path),
        seed_image_id="clip_f00000",
        seed_bbox=[0.10, 0.10, 0.20, 0.20],
        seed_label="motorcycle",
    )
    assert ls.patched == []
    assert len(ls.posted) == 1
    assert ls.posted[0]["task_id"] == 101


def test_propagate_via_tracker_sam3_failure_is_swallowed(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")

    class _FailingSam3:
        calls: list[Any] = []
        def track(self, **kwargs):
            raise RuntimeError("sam3 server down")

    ls = _StubLSRest([
        {"id": 1, "data": {"image_id": "v_f00000", "image_path": str(seed_path)}},
        {"id": 2, "data": {"image_id": "v_f00100", "image_path": str(sib_path)}},
    ])

    result = propagate_via_tracker(
        sam3_client=_FailingSam3(),
        ls_rest=ls,
        project_id=1,
        seed_image_path=str(seed_path),
        seed_image_id="v_f00000",
        seed_bbox=[0.1, 0.1, 0.2, 0.2],
        seed_label="other",
    )
    assert result.siblings_total == 1
    assert result.propagated == 0
    assert ls.posted == []


# ---------------------------------------------------------------------------
# routes.smart_track preconditions
# ---------------------------------------------------------------------------


def _track_context(bbox=(0.10, 0.10, 0.20, 0.20), label="forklift"):
    """Build an LS context with a smart_track rectangle draft."""
    x1, y1, x2, y2 = bbox
    return {
        "result": [
            {
                "from_name": "smart_track",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x1 * 100,
                    "y": y1 * 100,
                    "width": (x2 - x1) * 100,
                    "height": (y2 - y1) * 100,
                    "rectanglelabels": [label],
                    "rotation": 0,
                },
            }
        ]
    }


def _task(image_id="clip_f00000", image_path="/x.jpg", project=42):
    return {
        "id": 100,
        "project": project,
        "data": {"image_id": image_id, "image_path": image_path},
    }


def test_smart_track_returns_empty_when_no_ls_rest() -> None:
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    regions, result = smart_track(_task(), _track_context(), sam3, ls_rest=None)
    assert regions == []
    assert result is None


def test_smart_track_returns_empty_when_no_seed() -> None:
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    ls = _StubLSRest([])
    regions, result = smart_track(_task(), {"result": []}, sam3, ls_rest=ls)
    assert regions == []
    assert result is None


def test_smart_track_returns_empty_when_seed_degenerate() -> None:
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    ls = _StubLSRest([])
    regions, result = smart_track(
        _task(), _track_context(bbox=(0.10, 0.10, 0.10, 0.10)), sam3, ls_rest=ls
    )
    assert regions == []
    assert result is None


def test_smart_track_returns_empty_when_missing_image_id() -> None:
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    ls = _StubLSRest([])
    bad = {"id": 100, "project": 42, "data": {}}
    regions, result = smart_track(bad, _track_context(), sam3, ls_rest=ls)
    assert regions == []
    assert result is None


def test_smart_track_returns_empty_when_missing_project() -> None:
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    ls = _StubLSRest([])
    bad = {"id": 100, "data": {"image_id": "clip_f00000", "image_path": "/x.jpg"}}
    regions, result = smart_track(bad, _track_context(), sam3, ls_rest=ls)
    assert regions == []
    assert result is None


def test_smart_track_ignores_non_smart_track_rectangles() -> None:
    """A regular bbox draw shouldn't fire smart_track."""
    sam3 = _StubSam3(SAM3VideoTrackResponse(frames=[]))
    ls = _StubLSRest([])
    ctx = {
        "result": [
            {
                "from_name": "bbox",  # not smart_track
                "type": "rectanglelabels",
                "value": {"x": 10, "y": 10, "width": 10, "height": 10,
                          "rectanglelabels": ["forklift"], "rotation": 0},
            }
        ]
    }
    regions, result = smart_track(_task(), ctx, sam3, ls_rest=ls)
    assert regions == []
    assert result is None


def test_smart_track_invokes_propagate(tmp_path: Path) -> None:
    """Happy-path delegation: smart_track wires propagate_via_tracker."""
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])
    seed_bbox = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([
        (1, [(1, seed_bbox, 0.92)]),
    ]))
    task = _task(image_id="clip_f00000", image_path=str(seed_path), project=42)
    regions, result = smart_track(task, _track_context(bbox=tuple(seed_bbox)), sam3, ls_rest=ls)
    assert regions == []
    assert result is not None
    assert result.propagated == 1
    assert result.written_task_ids == [101]


def test_smart_track_contains_to_seed_owner(tmp_path: Path) -> None:
    """Per-user containment: pavan's smart_track must skip sree's frames."""
    seed_path = tmp_path / "seed.jpg"
    pavan_sib = tmp_path / "pavan_sib.jpg"
    sree_sib = tmp_path / "sree_sib.jpg"
    for p in (seed_path, pavan_sib, sree_sib):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path),
                             "assigned_to": "pavan"}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(pavan_sib),
                             "assigned_to": "pavan"}},
        {"id": 102, "data": {"image_id": "clip_f00200", "image_path": str(sree_sib),
                             "assigned_to": "sree"}},
    ])
    seed_bbox = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([
        (1, [(1, seed_bbox, 0.92)]),
        (2, [(1, seed_bbox, 0.93)]),
    ]))
    task = {
        "id": 100, "project": 42,
        "data": {"image_id": "clip_f00000", "image_path": str(seed_path),
                 "assigned_to": "pavan"},
    }
    _, result = smart_track(task, _track_context(bbox=tuple(seed_bbox)), sam3, ls_rest=ls)
    assert result is not None
    # Only pavan's sibling got the propagation; sree's never reached the tracker.
    assert result.siblings_total == 1
    assert result.written_task_ids == [101]


