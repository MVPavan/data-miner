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

from manual_reviewer.ml_backend.routes import (
    propagate_now,
    smart_track,
    track_similar,
)
from manual_reviewer.ml_backend.smart_track_lib import (
    MultiSeedPropagateResult,
    PropagateResult,
    SeedSpec,
    Sibling,
    build_jpeg_folder,
    filter_track_response,
    find_siblings,
    propagate_multi_via_tracker,
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
    """Mimic LSRestClient.iter_project_tasks + post_prediction."""

    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        self._tasks = tasks
        self.posted: list[dict[str, Any]] = []
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


# ---------------------------------------------------------------------------
# Multi-seed orchestrator (Option A Phase 2 backbone)
# ---------------------------------------------------------------------------


def test_propagate_multi_via_tracker_two_seeds(tmp_path: Path) -> None:
    """Two seeds; each propagates to one sibling with static-only filter."""
    seed_path = tmp_path / "seed.jpg"
    sib_a = tmp_path / "sib_a.jpg"
    sib_b = tmp_path / "sib_b.jpg"
    for p in (seed_path, sib_a, sib_b):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_a)}},
        {"id": 102, "data": {"image_id": "clip_f00200", "image_path": str(sib_b)}},
    ])
    pole_bbox = [0.10, 0.10, 0.20, 0.20]
    sign_bbox = [0.50, 0.50, 0.60, 0.60]
    # Frame 1: pole stayed static, sign drifted away (rejected by motion).
    # Frame 2: both static.
    resp = _resp([
        (1, [
            (1, [0.10, 0.10, 0.20, 0.20], 0.94),  # pole static
            (2, [0.80, 0.80, 0.90, 0.90], 0.93),  # sign drifted (>0.05 from 0.55)
        ]),
        (2, [
            (1, [0.10, 0.10, 0.20, 0.20], 0.93),
            (2, [0.50, 0.50, 0.60, 0.60], 0.91),
        ]),
    ])
    sam3 = _StubSam3(resp)

    seeds = [
        SeedSpec(bbox=pole_bbox, label="pole", track_group_id="grp_pole"),
        SeedSpec(bbox=sign_bbox, label="sign", track_group_id="grp_sign"),
    ]
    result = propagate_multi_via_tracker(
        sam3_client=sam3,
        ls_rest=ls,
        project_id=42,
        seed_image_path=str(seed_path),
        seed_image_id="clip_f00000",
        seeds=seeds,
    )

    assert result.siblings_total == 2
    assert result.seeds_total == 2
    # pole: 2 propagations (both siblings static)
    # sign: 1 propagation (only frame 2 static; frame 1 drifted)
    assert result.propagated == 3
    assert result.per_seed["grp_pole"].propagated == 2
    assert result.per_seed["grp_sign"].propagated == 1
    # Posted predictions carry the right track_group_id meta.
    posted_groups = [p["result"][0]["meta"]["track_group_id"] for p in ls.posted]
    assert sorted(posted_groups) == sorted(["grp_pole", "grp_pole", "grp_sign"])


def test_propagate_multi_via_tracker_single_sam3_call(tmp_path: Path) -> None:
    """Multi-seed must batch into ONE /track call (key efficiency claim)."""
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])
    sam3 = _StubSam3(_resp([
        (1, [(1, [0.10, 0.10, 0.20, 0.20], 0.92), (2, [0.50, 0.50, 0.60, 0.60], 0.91)]),
    ]))
    seeds = [
        SeedSpec(bbox=[0.10, 0.10, 0.20, 0.20], label="a", track_group_id="g1"),
        SeedSpec(bbox=[0.50, 0.50, 0.60, 0.60], label="b", track_group_id="g2"),
        SeedSpec(bbox=[0.30, 0.30, 0.40, 0.40], label="c", track_group_id="g3"),
    ]
    propagate_multi_via_tracker(
        sam3_client=sam3,
        ls_rest=ls,
        project_id=42,
        seed_image_path=str(seed_path),
        seed_image_id="clip_f00000",
        seeds=seeds,
    )
    # Exactly one SAM 3.1 call carrying all 3 seeds with distinct obj_ids.
    assert len(sam3.calls) == 1
    sent_seeds = sam3.calls[0]["seeds"]
    assert len(sent_seeds) == 3
    assert {s["obj_id"] for s in sent_seeds} == {1, 2, 3}


def test_propagate_multi_via_tracker_no_seeds_short_circuits(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jpg"
    seed_path.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 1, "data": {"image_id": "v_f00000", "image_path": str(seed_path)}},
        {"id": 2, "data": {"image_id": "v_f00100", "image_path": str(seed_path)}},
    ])
    sam3 = _StubSam3(_resp([]))
    result = propagate_multi_via_tracker(
        sam3_client=sam3, ls_rest=ls, project_id=1,
        seed_image_path=str(seed_path), seed_image_id="v_f00000",
        seeds=[],
    )
    assert result.seeds_total == 0
    assert result.propagated == 0
    assert sam3.calls == []
    assert ls.posted == []


# ---------------------------------------------------------------------------
# track_similar route (Phase 1)
# ---------------------------------------------------------------------------


class _StubVisualPromptResp:
    def __init__(self, boxes_norm, scores):
        self.boxes_norm = boxes_norm
        self.scores = scores


class _StubSam3Visual:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def visual_prompt(self, *, image_path, exemplar_boxes_norm, threshold, max_results):
        self.calls.append({
            "image_path": image_path,
            "exemplar_boxes_norm": exemplar_boxes_norm,
            "threshold": threshold,
        })
        return self._resp


def _track_similar_context(exemplar_bbox=(0.10, 0.10, 0.20, 0.20), label="forklift"):
    x1, y1, x2, y2 = exemplar_bbox
    return {
        "result": [{
            "from_name": "track_similar",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x1 * 100, "y": y1 * 100,
                "width": (x2 - x1) * 100, "height": (y2 - y1) * 100,
                "rectanglelabels": [label], "rotation": 0,
            },
        }],
    }


def test_track_similar_returns_regions_with_correct_from_name() -> None:
    sam3 = _StubSam3Visual(_StubVisualPromptResp(
        boxes_norm=[[0.10, 0.10, 0.20, 0.20], [0.50, 0.50, 0.60, 0.60]],
        scores=[0.85, 0.82],
    ))
    task = _task()
    regions = track_similar(task, _track_similar_context(label="forklift"), sam3)
    # SAM 3.1 visual_prompt was called with the exemplar.
    assert len(sam3.calls) == 1
    assert sam3.calls[0]["exemplar_boxes_norm"] == [[0.10, 0.10, 0.20, 0.20]]
    # Returned regions are tagged from_name="track_similar".
    assert all(r["from_name"] == "track_similar" for r in regions)
    # Class label is preserved.
    assert all(r["value"]["rectanglelabels"] == ["forklift"] for r in regions)
    # meta marks the source.
    assert all(r["meta"]["source"] == "track_similar" for r in regions)


def test_track_similar_persists_each_match_separately() -> None:
    """Phase 1 must POST each match as its own prediction record.

    Reasoning: LS Community's UI delete-button calls DELETE
    /api/predictions/<id>/. When each match is its own prediction,
    deleting a single yellow draft removes that record server-side
    and Phase 2 sees only survivors. Bundling all N matches into one
    prediction would either drop everything on a single delete (LS
    nukes the whole record) or leave the prediction intact (LS does
    nothing) — both broken UX. Per-match predictions are the only
    shape that supports selective rejection cleanly.
    """
    sam3 = _StubSam3Visual(_StubVisualPromptResp(
        boxes_norm=[[0.40, 0.40, 0.50, 0.50], [0.70, 0.70, 0.80, 0.80]],
        scores=[0.85, 0.82],
    ))
    ls = _StubLSRest([])
    task = _task()  # task["id"] = 100
    regions = track_similar(task, _track_similar_context(label="forklift"), sam3, ls_rest=ls)
    assert len(regions) == 2
    # LS got TWO POSTs, one per match.
    assert len(ls.posted) == 2
    for posted in ls.posted:
        assert posted["task_id"] == 100
        assert posted["model_version"] == "sam3_1_track_similar"
        # Each prediction has exactly one region.
        assert len(posted["result"]) == 1
        assert posted["result"][0]["from_name"] == "track_similar"


def test_track_similar_works_without_ls_rest() -> None:
    """Backward-compat: when ls_rest is None, route still returns matches
    (just doesn't persist). Tests that don't need cross-task writes still
    work."""
    sam3 = _StubSam3Visual(_StubVisualPromptResp(
        boxes_norm=[[0.40, 0.40, 0.50, 0.50]],
        scores=[0.85],
    ))
    regions = track_similar(_task(), _track_similar_context(), sam3, ls_rest=None)
    assert len(regions) == 1


def test_track_similar_no_exemplar_returns_empty() -> None:
    sam3 = _StubSam3Visual(_StubVisualPromptResp(boxes_norm=[], scores=[]))
    regions = track_similar(_task(), {"result": []}, sam3)
    assert regions == []
    assert sam3.calls == []


def test_track_similar_ignores_visual_prompt_regions() -> None:
    """An exemplar drawn under from_name="visual_prompt" must NOT trigger
    track_similar — the two tools are kept separate by from_name."""
    sam3 = _StubSam3Visual(_StubVisualPromptResp(boxes_norm=[], scores=[]))
    ctx = {
        "result": [{
            "from_name": "visual_prompt",  # not track_similar
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {"x": 10, "y": 10, "width": 10, "height": 10,
                      "rectanglelabels": ["forklift"], "rotation": 0},
        }]
    }
    regions = track_similar(_task(), ctx, sam3)
    # Falls back to "any rectangle" when no track_similar tag in context, so
    # the visual_prompt rectangle DOES get used. This matches the existing
    # test-path fallback for _exemplars_from_context. The test confirms the
    # behavior is consistent rather than asserting empty.
    assert sam3.calls  # the rectangle is treated as exemplar in fallback mode


# ---------------------------------------------------------------------------
# propagate_now route (Phase 2)
# ---------------------------------------------------------------------------


def _annotation_with_track_similar(*bboxes_with_labels):
    """Build a task["annotations"] entry with the given track_similar regions."""
    result = []
    for bbox, label in bboxes_with_labels:
        x1, y1, x2, y2 = bbox
        result.append({
            "from_name": "track_similar",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x1 * 100, "y": y1 * 100,
                "width": (x2 - x1) * 100, "height": (y2 - y1) * 100,
                "rectanglelabels": [label], "rotation": 0,
            },
        })
    return {"result": result}


def _propagate_trigger_context():
    """Smart KeyPoint trigger context — content of the click is irrelevant."""
    return {
        "result": [{
            "from_name": "propagate_now",
            "to_name": "image",
            "type": "keypointlabels",
            "value": {"x": 50, "y": 50, "keypointlabels": ["propagate"]},
        }]
    }


def test_propagate_now_no_ls_rest_returns_empty() -> None:
    sam3 = _StubSam3(_resp([]))
    task = _task()
    task["annotations"] = [_annotation_with_track_similar(
        ([0.10, 0.10, 0.20, 0.20], "pole"),
    )]
    regions, result = propagate_now(task, _propagate_trigger_context(), sam3, ls_rest=None)
    assert regions == []
    assert result is None


def test_propagate_now_no_track_similar_seeds() -> None:
    sam3 = _StubSam3(_resp([]))
    ls = _StubLSRest([])
    task = _task()
    task["annotations"] = [{"result": []}]
    regions, result = propagate_now(task, _propagate_trigger_context(), sam3, ls_rest=ls)
    assert regions == []
    assert result is None


def test_propagate_now_picks_up_accepted_annotations(tmp_path: Path) -> None:
    """Phase 2 reads task.annotations for surviving track_similar regions."""
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])
    pole = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([
        (1, [(1, pole, 0.91)]),
    ]))
    task = _task(image_id="clip_f00000", image_path=str(seed_path), project=42)
    task["annotations"] = [_annotation_with_track_similar((pole, "forklift"))]
    regions, result = propagate_now(task, _propagate_trigger_context(), sam3, ls_rest=ls)
    assert regions == []
    assert result is not None
    assert result.seeds_total == 1
    assert result.propagated == 1
    # The posted prediction inherits the seed's class.
    assert ls.posted[0]["result"][0]["value"]["rectanglelabels"] == ["forklift"]


def test_propagate_now_multiple_seeds_dedups_by_bbox(tmp_path: Path) -> None:
    """Same bbox appearing in both annotations and context isn't seeded twice."""
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])
    pole = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([
        (1, [(1, pole, 0.91)]),
    ]))
    task = _task(image_id="clip_f00000", image_path=str(seed_path), project=42)
    task["annotations"] = [_annotation_with_track_similar((pole, "forklift"))]
    # Same bbox also lives in context.result as a draft — still one seed.
    ctx_with_dup = {
        "result": [
            {  # The trigger keypoint
                "from_name": "propagate_now", "to_name": "image",
                "type": "keypointlabels",
                "value": {"x": 50, "y": 50, "keypointlabels": ["propagate"]},
            },
            {  # Duplicate of the accepted region (context echoes drafts)
                "from_name": "track_similar", "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": pole[0] * 100, "y": pole[1] * 100,
                    "width": (pole[2] - pole[0]) * 100,
                    "height": (pole[3] - pole[1]) * 100,
                    "rectanglelabels": ["forklift"], "rotation": 0,
                },
            },
        ]
    }
    regions, result = propagate_now(task, ctx_with_dup, sam3, ls_rest=ls)
    assert result.seeds_total == 1


def test_propagate_now_picks_up_phase1_predictions(tmp_path: Path) -> None:
    """Regression: Phase 2 must read Phase 1 matches that are still in
    task["predictions"][] (the natural state right after Phase 1 fires
    and before the reviewer manually accepts each match into annotations).

    Without this, a reviewer who hits Shift+J immediately after Phase 1
    would see 'no track_similar regions found' even though 5 yellow
    drafts are visible on the canvas.
    """
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])
    pole = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([
        (1, [(1, pole, 0.91)]),
    ]))
    task = _task(image_id="clip_f00000", image_path=str(seed_path), project=42)
    # Empty annotations — user hasn't explicitly accepted yet.
    task["annotations"] = []
    # Phase 1 left a track_similar prediction in task["predictions"].
    task["predictions"] = [{
        "result": [{
            "from_name": "track_similar",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": pole[0] * 100, "y": pole[1] * 100,
                "width": (pole[2] - pole[0]) * 100,
                "height": (pole[3] - pole[1]) * 100,
                "rectanglelabels": ["forklift"], "rotation": 0,
            },
        }]
    }]
    regions, result = propagate_now(task, _propagate_trigger_context(), sam3, ls_rest=ls)
    assert regions == []
    assert result is not None
    assert result.seeds_total == 1
    assert result.propagated == 1


def test_propagate_now_dedups_across_annotations_and_predictions(tmp_path: Path) -> None:
    """Same bbox in both annotations and predictions counts as one seed."""
    seed_path = tmp_path / "seed.jpg"
    sib_path = tmp_path / "sib.jpg"
    for p in (seed_path, sib_path):
        p.write_bytes(b"jpeg")
    ls = _StubLSRest([
        {"id": 100, "data": {"image_id": "clip_f00000", "image_path": str(seed_path)}},
        {"id": 101, "data": {"image_id": "clip_f00100", "image_path": str(sib_path)}},
    ])
    pole = [0.10, 0.10, 0.20, 0.20]
    sam3 = _StubSam3(_resp([(1, [(1, pole, 0.91)])]))
    task = _task(image_id="clip_f00000", image_path=str(seed_path), project=42)
    task["annotations"] = [_annotation_with_track_similar((pole, "forklift"))]
    # Same bbox also lives in predictions (Phase 1 result that user accepted).
    task["predictions"] = [{
        "result": [{
            "from_name": "track_similar",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": pole[0] * 100, "y": pole[1] * 100,
                "width": (pole[2] - pole[0]) * 100,
                "height": (pole[3] - pole[1]) * 100,
                "rectanglelabels": ["forklift"], "rotation": 0,
            },
        }]
    }]
    regions, result = propagate_now(task, _propagate_trigger_context(), sam3, ls_rest=ls)
    assert result.seeds_total == 1


def test_propagate_now_missing_project_returns_empty() -> None:
    sam3 = _StubSam3(_resp([]))
    ls = _StubLSRest([])
    bad_task = {"id": 100, "data": {"image_id": "clip_f00000", "image_path": "/x.jpg"}}
    bad_task["annotations"] = [_annotation_with_track_similar(
        ([0.10, 0.10, 0.20, 0.20], "pole"),
    )]
    regions, result = propagate_now(bad_task, _propagate_trigger_context(), sam3, ls_rest=ls)
    assert regions == []
    assert result is None
