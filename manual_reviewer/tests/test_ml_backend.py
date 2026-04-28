"""Tests for the manual_reviewer LS ML backend.

Three layers, no GPU required:

  1. ls_payload — pure LS↔aav4 translation helpers.
  2. routes — dispatch + each route with a stub Sam3 client.
  3. server — ManualReviewerMLBackend.predict end-to-end with the
     LabelStudioMLBase stub fall-back (we don't import label_studio_ml).
  4. SAM 3.1 wire/client/server — the new click_mask path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from manual_reviewer.ml_backend import dispatch
from manual_reviewer.ml_backend.aav4_client import (
    build_sam3_client,
    read_cached_proposals,
)
from manual_reviewer.ml_backend.ls_payload import (
    DEFAULT_LABEL,
    ls_box_to_norm,
    ls_keypoint_to_norm,
    ls_textarea_value_to_prompts,
    norm_box_to_ls_region,
    predictions_envelope,
)
from manual_reviewer.ml_backend.routes import (
    batch_proposals,
    smart_click,
    smart_text,
)
from manual_reviewer.ml_backend.server import ManualReviewerMLBackend


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubResp:
    """Duck-typed SAM 3.1 response — only the attrs the routes read."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubSam3Client:
    """Records every call and returns canned responses keyed by method."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._click_resp: Any = _StubResp(bbox=None, score=0.0, mask_rle=None)
        self._text_resp: Any = _StubResp(boxes=[], scores=[], labels=[])
        self._click_raises: BaseException | None = None
        self._text_raises: BaseException | None = None

    def click_mask(self, **kwargs: Any) -> Any:
        self.calls.append(("click_mask", kwargs))
        if self._click_raises is not None:
            raise self._click_raises
        return self._click_resp

    def text_detect(self, **kwargs: Any) -> Any:
        self.calls.append(("text_detect", kwargs))
        if self._text_raises is not None:
            raise self._text_raises
        return self._text_resp


# ---------------------------------------------------------------------------
# 1. ls_payload helpers
# ---------------------------------------------------------------------------


def test_ls_box_to_norm_clamps_and_rounds() -> None:
    norm = ls_box_to_norm({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
    assert norm is not None
    assert norm == pytest.approx([0.10, 0.20, 0.40, 0.60])


def test_ls_box_to_norm_returns_none_on_zero_area() -> None:
    assert ls_box_to_norm({"x": 10, "y": 10, "width": 0, "height": 0}) is None


def test_ls_box_to_norm_returns_none_on_garbage() -> None:
    assert ls_box_to_norm({"x": "abc"}) is None
    assert ls_box_to_norm({}) is None


def test_ls_keypoint_to_norm_clamps() -> None:
    assert ls_keypoint_to_norm({"x": 25.0, "y": 75.0}) == [0.25, 0.75]
    assert ls_keypoint_to_norm({"x": 150.0, "y": -10.0}) == [1.0, 0.0]


def test_ls_textarea_value_to_prompts_filters_empties() -> None:
    val = {"text": ["forklift", "  ", "person", ""]}
    assert ls_textarea_value_to_prompts(val) == ["forklift", "person"]


def test_ls_textarea_value_accepts_string_text() -> None:
    assert ls_textarea_value_to_prompts({"text": "person"}) == ["person"]


def test_norm_box_to_ls_region_scales_and_carries_label() -> None:
    region = norm_box_to_ls_region([0.10, 0.20, 0.40, 0.60], "forklift", score=0.9)
    val = region["value"]
    assert pytest.approx(val["x"]) == 10.0
    assert pytest.approx(val["y"]) == 20.0
    assert pytest.approx(val["width"]) == 30.0
    assert pytest.approx(val["height"]) == 40.0
    assert val["rectanglelabels"] == ["forklift"]
    assert region["type"] == "rectanglelabels"
    assert region["score"] == pytest.approx(0.9)


def test_norm_box_to_ls_region_falls_back_to_default_label() -> None:
    region = norm_box_to_ls_region([0.1, 0.1, 0.2, 0.2], "")
    assert region["value"]["rectanglelabels"] == [DEFAULT_LABEL]


def test_predictions_envelope_returns_empty_for_no_regions() -> None:
    assert predictions_envelope([], model_version="v") == []


def test_predictions_envelope_wraps_one_envelope_per_call() -> None:
    region = norm_box_to_ls_region([0, 0, 0.1, 0.1], "person")
    envelope = predictions_envelope([region], model_version="v_ml")
    assert len(envelope) == 1
    assert envelope[0]["model_version"] == "v_ml"
    assert envelope[0]["result"] == [region]


# ---------------------------------------------------------------------------
# 2. routes — smart_click
# ---------------------------------------------------------------------------


def _click_context(x: float, y: float, label: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"x": x, "y": y}
    if label is not None:
        value["keypointlabels"] = [label]
    return {
        "result": [
            {
                "type": "keypointlabels",
                "value": value,
                "from_name": "click",
                "to_name": "image",
            }
        ]
    }


def _text_context(prompts: list[str]) -> dict[str, Any]:
    return {
        "result": [
            {
                "type": "textarea",
                "value": {"text": prompts},
                "from_name": "text_query",
                "to_name": "image",
            }
        ]
    }


def _task(image_path: str = "/tmp/img.jpg", image_id: str = "img_a") -> dict[str, Any]:
    return {
        "id": 1,
        "data": {
            "image_path": image_path,
            "image_id": image_id,
            "image_size": [1920, 1080],
        },
    }


def test_smart_click_returns_box_when_sam_returns_bbox() -> None:
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    out = smart_click(_task(), _click_context(50, 60), client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == [DEFAULT_LABEL]
    assert ("click_mask", {
        "image_path": "/tmp/img.jpg",
        "point": [0.5, 0.6],
        "point_label": 1,
        "threshold": 0.5,
    }) in client.calls


def test_smart_click_uses_picked_label_from_context() -> None:
    """When the LS draft carries a class hint, the seeded box honors it."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "value": {"rectanglelabels": ["forklift"]},
            },
            {
                "type": "keypointlabels",
                "value": {"x": 50, "y": 60},
            },
        ]
    }
    out = smart_click(_task(), ctx, client)
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


def test_smart_click_negative_label_propagates() -> None:
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    smart_click(_task(), _click_context(50, 60, label="negative"), client)
    assert client.calls[0][1]["point_label"] == 0


def test_smart_click_returns_empty_when_no_bbox() -> None:
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=None, score=0.1)
    assert smart_click(_task(), _click_context(50, 60), client) == []


def test_smart_click_swallows_client_exceptions() -> None:
    client = _StubSam3Client()
    client._click_raises = RuntimeError("network down")
    assert smart_click(_task(), _click_context(50, 60), client) == []


def test_smart_click_no_image_path_returns_empty() -> None:
    client = _StubSam3Client()
    out = smart_click({"data": {}}, _click_context(50, 60), client)
    assert out == []
    assert client.calls == []


def test_smart_click_ignores_non_keypoint_context() -> None:
    client = _StubSam3Client()
    out = smart_click(_task(), _text_context(["forklift"]), client)
    assert out == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# 3. routes — smart_text
# ---------------------------------------------------------------------------


def test_smart_text_returns_one_region_per_box() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5], [0.5, 0.5, 0.9, 0.9]],
        scores=[0.9, 0.7],
        labels=["forklift", "forklift"],
    )
    out = smart_text(_task(), _text_context(["forklift"]), client)
    assert len(out) == 2
    assert all(r["value"]["rectanglelabels"] == ["forklift"] for r in out)
    assert client.calls[0][1] == {
        "image_path": "/tmp/img.jpg",
        "prompts": ["forklift"],
        "threshold": None,
    }


def test_smart_text_caps_max_regions() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0, 0, 0.1 * (i + 1), 0.1] for i in range(10)],
        scores=[0.5] * 10,
        labels=["x"] * 10,
    )
    out = smart_text(_task(), _text_context(["x"]), client, max_regions=3)
    assert len(out) == 3


def test_smart_text_skips_malformed_boxes() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5], "garbage", [0.6, 0.6, 0.9, 0.9]],
        scores=[0.9, 0.5, 0.7],
        labels=["a", "b", "c"],
    )
    out = smart_text(_task(), _text_context(["a"]), client)
    assert len(out) == 2


def test_smart_text_no_prompts_returns_empty_no_call() -> None:
    client = _StubSam3Client()
    out = smart_text(_task(), _text_context([]), client)
    assert out == []
    assert client.calls == []


def test_smart_text_swallows_client_exceptions() -> None:
    client = _StubSam3Client()
    client._text_raises = RuntimeError("boom")
    assert smart_text(_task(), _text_context(["forklift"]), client) == []


# ---------------------------------------------------------------------------
# 4. routes — batch_proposals (uses real sqlite via fixture)
# ---------------------------------------------------------------------------


def test_batch_proposals_returns_regions_for_seeded_image(
    seeded_pipeline_db: Path,
) -> None:
    out = batch_proposals(
        {"data": {"image_id": "img_a"}}, seeded_pipeline_db
    )
    assert len(out) == 1
    region = out[0]
    assert region["value"]["rectanglelabels"] == ["forklift"]
    assert region["meta"]["detector"] == "sam3_dart"


def test_batch_proposals_filters_by_min_confidence(
    seeded_pipeline_db: Path,
) -> None:
    out = batch_proposals(
        {"data": {"image_id": "img_a"}},
        seeded_pipeline_db,
        min_confidence=0.99,
    )
    assert out == []


def test_batch_proposals_returns_empty_for_unknown_image(
    seeded_pipeline_db: Path,
) -> None:
    assert (
        batch_proposals({"data": {"image_id": "missing"}}, seeded_pipeline_db) == []
    )


def test_batch_proposals_returns_empty_when_db_path_missing(tmp_path: Path) -> None:
    assert (
        batch_proposals({"data": {"image_id": "x"}}, tmp_path / "does_not_exist.db")
        == []
    )


def test_batch_proposals_returns_empty_when_db_path_none() -> None:
    assert batch_proposals({"data": {"image_id": "x"}}, None) == []


def test_batch_proposals_caps_max_regions(tmp_path: Path) -> None:
    """Build a synthetic DB with 5 candidates, cap at 2."""
    db_path = tmp_path / "tiny.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE proposals (image_id TEXT, model TEXT, data TEXT, PRIMARY KEY(image_id, model))"
    )
    candidates = [
        {
            "candidate_id": f"c{i}",
            "class_name": "forklift",
            "confidence": 0.5 + i * 0.1,
            "bbox": {"x1": 0.1 * i, "y1": 0.1, "x2": 0.1 * i + 0.05, "y2": 0.5},
        }
        for i in range(5)
    ]
    conn.execute(
        "INSERT INTO proposals VALUES (?, ?, ?)",
        ("img_x", "sam3_dart", json.dumps({"candidates": candidates})),
    )
    conn.commit()
    conn.close()

    out = batch_proposals(
        {"data": {"image_id": "img_x"}}, db_path, max_regions=2
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# 5. dispatch
# ---------------------------------------------------------------------------


def test_dispatch_routes_keypoint_to_smart_click() -> None:
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0, 0, 0.1, 0.1], score=0.6)
    out = dispatch(
        _task(), _click_context(50, 60), sam3_client=client, db_path=None
    )
    assert len(out) == 1
    assert client.calls[0][0] == "click_mask"


def test_dispatch_routes_textarea_to_smart_text() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.5, 0.5]], scores=[0.7], labels=["x"]
    )
    out = dispatch(
        _task(), _text_context(["x"]), sam3_client=client, db_path=None
    )
    assert len(out) == 1
    assert client.calls[0][0] == "text_detect"


def test_dispatch_no_context_routes_to_batch(seeded_pipeline_db: Path) -> None:
    client = _StubSam3Client()
    out = dispatch(
        _task(image_id="img_a"), None, sam3_client=client, db_path=seeded_pipeline_db
    )
    assert len(out) == 1
    assert client.calls == []


def test_dispatch_no_context_no_db_returns_empty() -> None:
    assert dispatch(_task(), None, sam3_client=None, db_path=None) == []


# ---------------------------------------------------------------------------
# 6. server (ManualReviewerMLBackend)
# ---------------------------------------------------------------------------


def test_server_predict_returns_one_envelope_per_task(
    seeded_pipeline_db: Path,
) -> None:
    backend = ManualReviewerMLBackend(
        sam3_client=_StubSam3Client(), db_path=seeded_pipeline_db
    )
    tasks = [
        {"id": 1, "data": {"image_path": "/tmp/x.jpg", "image_id": "img_a"}},
        {"id": 2, "data": {"image_path": "/tmp/y.jpg", "image_id": "missing"}},
    ]
    out = backend.predict(tasks, context=None)
    assert len(out) == 2
    assert out[0]["result"]  # img_a has cached proposals
    assert out[1]["result"] == []  # img_b has none
    assert out[0]["model_version"] == "manual_reviewer_v1"


def test_server_predict_smart_click_path() -> None:
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.1, 0.5, 0.5], score=0.8)
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    out = backend.predict([_task()], context=_click_context(50, 60))
    assert len(out) == 1
    assert out[0]["result"]
    assert out[0]["score"] == 1.0
    assert client.calls[0][0] == "click_mask"


def test_server_predict_swallows_per_task_exceptions() -> None:
    """One task throwing must not block the rest."""
    bad_client = _StubSam3Client()

    class _Boom:
        def click_mask(self, **kwargs: Any) -> Any:
            raise RuntimeError("nope")

        def text_detect(self, **kwargs: Any) -> Any:
            raise RuntimeError("nope")

    backend = ManualReviewerMLBackend(sam3_client=_Boom(), db_path=None)
    out = backend.predict(
        [_task(), {"id": 9, "data": {"image_path": "/tmp/z.jpg"}}],
        context=_click_context(10, 20),
    )
    assert len(out) == 2
    # Both tasks return empty (smart_click fails gracefully) but predict()
    # still produced two envelopes.
    assert all(r["result"] == [] for r in out)


def test_server_envelope_helper_wraps_regions() -> None:
    backend = ManualReviewerMLBackend(sam3_client=None, db_path=None)
    region = norm_box_to_ls_region([0, 0, 0.1, 0.1], "person")
    env = backend.envelope([region])
    assert len(env) == 1
    assert env[0]["result"] == [region]


# ---------------------------------------------------------------------------
# 7. aav4_client helpers
# ---------------------------------------------------------------------------


def test_read_cached_proposals_skips_malformed_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "tiny.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE proposals (image_id TEXT, model TEXT, data TEXT, PRIMARY KEY(image_id, model))"
    )
    conn.execute(
        "INSERT INTO proposals VALUES (?, ?, ?)",
        ("img_x", "sam3_dart", "not_json"),
    )
    conn.execute(
        "INSERT INTO proposals VALUES (?, ?, ?)",
        ("img_x", "sam3_1", json.dumps({"candidates": "not_a_list"})),
    )
    conn.commit()
    conn.close()
    assert read_cached_proposals(db_path, "img_x") == []


def test_build_sam3_client_honors_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM3_1_URL", "http://other:9999/predict")
    client = build_sam3_client()
    assert client._url == "http://other:9999/predict"


def test_build_sam3_client_default_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAM3_1_URL", raising=False)
    client = build_sam3_client()
    assert client._url == "http://localhost:3014/predict"


# ---------------------------------------------------------------------------
# 8. SAM 3.1 click_mask wire/client/server (no model load)
# ---------------------------------------------------------------------------


def test_sam3_1_click_mask_request_validates() -> None:
    from data_miner.auto_annotation_v4.configs.wire import SAM3ClickMaskRequest

    req = SAM3ClickMaskRequest(image_path="/x.jpg", point=[0.5, 0.5])
    assert req.point_label == 1  # default
    assert req.threshold == 0.5


def test_sam3_1_click_mask_response_round_trips() -> None:
    from data_miner.auto_annotation_v4.configs.wire import SAM3ClickMaskResponse

    resp = SAM3ClickMaskResponse(
        bbox=[0.1, 0.1, 0.4, 0.5], mask_rle={"counts": "abc"}, score=0.85
    )
    payload = resp.model_dump()
    assert payload["bbox"] == [0.1, 0.1, 0.4, 0.5]
    parsed = SAM3ClickMaskResponse.model_validate(payload)
    assert parsed.score == pytest.approx(0.85)


def test_sam3_one_api_decode_picks_click_for_point_request() -> None:
    from data_miner.auto_annotation_v4.model_servers.sam3_1 import (
        _CLICK_TAG,
        SAM3OneApi,
    )

    api = SAM3OneApi.__new__(SAM3OneApi)  # bypass setup
    decoded = api.decode_request(
        {"image_path": "/x.jpg", "point": [0.5, 0.5]}
    )
    assert decoded["__mode__"] == _CLICK_TAG


def test_sam3_one_api_predict_routes_click_mask() -> None:
    """Mode dispatch must reach SAM3OneModel.click_mask."""
    from data_miner.auto_annotation_v4.configs.wire import SAM3ClickMaskResponse
    from data_miner.auto_annotation_v4.model_servers.sam3_1 import SAM3OneApi

    class _StubModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def click_mask(self, **kwargs: Any) -> Any:
            self.calls.append(("click_mask", kwargs))
            return SAM3ClickMaskResponse(bbox=[0, 0, 0.1, 0.1], score=0.9)

    api = SAM3OneApi.__new__(SAM3OneApi)
    api.model = _StubModel()
    decoded = api.decode_request(
        {"image_path": "/x.jpg", "point": [0.4, 0.6]}
    )
    out = api.predict(decoded)
    assert isinstance(out, SAM3ClickMaskResponse)
    assert api.model.calls[0][1]["point_norm"] == [0.4, 0.6]


def test_sam3_one_http_client_click_mask_serializes_request() -> None:
    """Sam3OneHttpClient.click_mask sends a SAM3ClickMaskRequest body."""
    from manual_reviewer.reconcile import Sam3OneHttpClient

    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"bbox": [0.0, 0.0, 0.1, 0.1], "mask_rle": None, "score": 0.7}

    class _FakeSession:
        def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _FakeResp:
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    client = Sam3OneHttpClient(
        url="http://stub:1234/predict",
        click_url="http://stub:1234/predict",
        session=_FakeSession(),
    )
    resp = client.click_mask(image_path="/img.jpg", point=[0.5, 0.5])
    assert resp.bbox == [0.0, 0.0, 0.1, 0.1]
    assert captured["json"]["point"] == [0.5, 0.5]
    assert captured["json"]["point_label"] == 1


def test_sam3_one_http_client_text_detect_returns_detector_response() -> None:
    """Sam3OneHttpClient.text_detect parses DetectorResponse from server."""
    from manual_reviewer.reconcile import Sam3OneHttpClient

    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "boxes": [[0.1, 0.1, 0.4, 0.5]],
                "scores": [0.8],
                "labels": ["forklift"],
            }

    class _FakeSession:
        def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _FakeResp:
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    client = Sam3OneHttpClient(
        url="http://stub:1234/predict",
        text_url="http://stub:1234/predict",
        session=_FakeSession(),
    )
    resp = client.text_detect(image_path="/img.jpg", prompts=["forklift"])
    assert resp.labels == ["forklift"]
    assert captured["json"]["prompts"] == ["forklift"]
