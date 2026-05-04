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
    smart_search,
    smart_visual,
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
        self._visual_resp: Any = _StubResp(boxes_norm=[], scores=[])
        self._click_raises: BaseException | None = None
        self._text_raises: BaseException | None = None
        self._visual_raises: BaseException | None = None

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

    def visual_prompt(self, **kwargs: Any) -> Any:
        self.calls.append(("visual_prompt", kwargs))
        if self._visual_raises is not None:
            raise self._visual_raises
        return self._visual_resp


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


def test_norm_box_to_ls_region_drops_degenerate_box() -> None:
    """Zero-area input → ``None`` (no phantom 1e-6 region)."""
    assert norm_box_to_ls_region([0.5, 0.5, 0.5, 0.5], "forklift") is None
    assert norm_box_to_ls_region([0.6, 0.5, 0.5, 0.6], "forklift") is None


def test_norm_box_to_ls_region_emits_original_dims_when_provided() -> None:
    region = norm_box_to_ls_region(
        [0.1, 0.2, 0.4, 0.6],
        "forklift",
        original_width=1920,
        original_height=1080,
        original_rotation=0,
    )
    assert region is not None
    assert region["original_width"] == 1920
    assert region["original_height"] == 1080
    assert region["original_rotation"] == 0


def test_norm_box_to_ls_region_omits_original_dims_when_none() -> None:
    region = norm_box_to_ls_region([0.1, 0.2, 0.4, 0.6], "forklift")
    assert region is not None
    assert "original_width" not in region
    assert "original_height" not in region
    assert "original_rotation" not in region


def test_snap_label_passes_through_known_label() -> None:
    from manual_reviewer.ml_backend.ls_payload import snap_label

    assert snap_label("forklift") == "forklift"
    assert snap_label("person") == "person"


def test_snap_label_falls_back_for_unknown_label() -> None:
    from manual_reviewer.ml_backend.ls_payload import snap_label

    assert snap_label("a green forklift") == DEFAULT_LABEL
    assert snap_label("") == DEFAULT_LABEL
    assert snap_label(None) == DEFAULT_LABEL  # type: ignore[arg-type]


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


def _click_context(x: float, y: float) -> dict[str, Any]:
    """Phase A: keypoint draft no longer carries positive/negative labels.

    The shared ``<Labels>`` palette puts the active class on
    ``value.labels``; tests that need a class hint pass it explicitly via a
    paired labels region (see ``test_smart_click_uses_picked_label_from_context``).
    """
    return {
        "result": [
            {
                "type": "keypoint",
                "value": {"x": x, "y": y},
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
        "threshold": 0.0,
    }) in client.calls


def test_smart_click_uses_picked_label_from_context() -> None:
    """Phase A: shared ``<Labels>`` palette puts the active class on
    ``value.labels`` — smart_click must honor it on the seeded box."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    ctx = {
        "result": [
            {
                "type": "labels",
                "value": {"labels": ["forklift"]},
            },
            {
                "type": "keypoint",
                "value": {"x": 50, "y": 60},
            },
        ]
    }
    out = smart_click(_task(), ctx, client)
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


def test_smart_click_accepts_legacy_rectanglelabels_hint() -> None:
    """Backward compat: drafts that carry ``value.rectanglelabels`` still work."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "value": {"rectanglelabels": ["forklift"]},
            },
            {
                "type": "keypoint",
                "value": {"x": 50, "y": 60},
            },
        ]
    }
    out = smart_click(_task(), ctx, client)
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


def test_smart_click_accepts_keypointlabels_hint() -> None:
    """Phase A wire alignment: <KeyPointLabels> draft carries the class in
    ``value.keypointlabels``. The route must honor it so the reviewer's
    hotkey selection rides through to the seeded bbox."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    ctx = {
        "result": [
            {
                "type": "keypointlabels",
                "from_name": "click",
                "value": {"x": 50, "y": 60, "keypointlabels": ["bicycle"]},
            },
        ]
    }
    out = smart_click(_task(), ctx, client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["bicycle"]


def test_smart_click_prefers_keypoint_label_over_unrelated_rectangle() -> None:
    """When the triggering keypoint carries its own ``keypointlabels``, that
    class wins over an unrelated rectangle that happens to ride the same
    draft. Otherwise the smart_visual palette would override the click's class."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 50, "y": 50, "width": 10, "height": 10,
                          "rectanglelabels": ["bicycle"]},
            },
            {
                "type": "keypointlabels",
                "from_name": "click",
                "value": {"x": 50, "y": 60, "keypointlabels": ["forklift"]},
            },
        ]
    }
    out = smart_click(_task(), ctx, client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


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
# 3. routes — smart_search
# ---------------------------------------------------------------------------


def test_smart_search_returns_one_region_per_box() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5], [0.5, 0.5, 0.9, 0.9]],
        scores=[0.9, 0.7],
        labels=["forklift", "forklift"],
    )
    out = smart_search(_task(), _text_context(["forklift"]), client)
    assert len(out) == 2
    assert all(r["value"]["rectanglelabels"] == ["forklift"] for r in out)
    assert client.calls[0][1] == {
        "image_path": "/tmp/img.jpg",
        "prompts": ["forklift"],
        "threshold": None,
    }


def test_smart_search_caps_max_regions() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0, 0, 0.1 * (i + 1), 0.1] for i in range(10)],
        scores=[0.5] * 10,
        labels=["x"] * 10,
    )
    out = smart_search(_task(), _text_context(["x"]), client, max_regions=3)
    assert len(out) == 3


def test_smart_search_skips_malformed_boxes() -> None:
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5], "garbage", [0.6, 0.6, 0.9, 0.9]],
        scores=[0.9, 0.5, 0.7],
        labels=["a", "b", "c"],
    )
    out = smart_search(_task(), _text_context(["a"]), client)
    assert len(out) == 2


def test_smart_search_no_prompts_returns_empty_no_call() -> None:
    client = _StubSam3Client()
    out = smart_search(_task(), _text_context([]), client)
    assert out == []
    assert client.calls == []


def test_smart_search_swallows_client_exceptions() -> None:
    client = _StubSam3Client()
    client._text_raises = RuntimeError("boom")
    assert smart_search(_task(), _text_context(["forklift"]), client) == []


def test_smart_search_snaps_unknown_label_to_default() -> None:
    """Free-text prompt strings that don't match the LS palette get snapped
    to ``DEFAULT_LABEL``; the raw prompt is preserved on ``meta.prompt``."""
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5]],
        scores=[0.9],
        labels=["a green thing"],   # not in the 24-class palette
    )
    out = smart_search(_task(), _text_context(["a green thing"]), client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == [DEFAULT_LABEL]
    assert out[0]["meta"]["prompt"] == "a green thing"


def test_smart_search_keeps_known_palette_label() -> None:
    """Known palette label rides through unchanged."""
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5]],
        scores=[0.9],
        labels=["forklift"],
    )
    out = smart_search(_task(), _text_context(["forklift"]), client)
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


# ---------------------------------------------------------------------------
# 3b. routes — smart_visual
# ---------------------------------------------------------------------------


def _visual_context(
    *boxes: tuple[float, float, float, float],
    label: str | None = None,
    rtype: str = "rectanglelabels",
) -> dict[str, Any]:
    """Build an LS context with one or more rectangle exemplar regions."""
    result: list[dict[str, Any]] = []
    for x, y, w, h in boxes:
        value: dict[str, Any] = {"x": x, "y": y, "width": w, "height": h}
        if label is not None:
            # Phase A shape: label rides on value.labels.
            value["labels"] = [label]
        result.append(
            {
                "type": rtype,
                "value": value,
                "from_name": "bbox",
                "to_name": "image",
            }
        )
    return {"result": result}


def test_smart_visual_returns_one_region_per_match() -> None:
    client = _StubSam3Client()
    # Returned boxes are placed away from the exemplar so the IoU dedup
    # doesn't drop them. The exemplar-overlap case has its own test below.
    client._visual_resp = _StubResp(
        boxes_norm=[[0.5, 0.5, 0.7, 0.7], [0.8, 0.1, 0.95, 0.25]],
        scores=[0.9, 0.6],
    )
    out = smart_visual(
        _task(),
        _visual_context((10, 10, 20, 20), label="forklift"),
        client,
    )
    assert len(out) == 2
    assert all(r["value"]["rectanglelabels"] == ["forklift"] for r in out)
    assert client.calls[0][0] == "visual_prompt"
    kwargs = client.calls[0][1]
    assert kwargs["image_path"] == "/tmp/img.jpg"
    assert kwargs["exemplar_boxes_norm"][0] == pytest.approx([0.10, 0.10, 0.30, 0.30])
    assert kwargs["threshold"] == pytest.approx(0.4)


def test_smart_visual_falls_back_to_default_label() -> None:
    """Exemplar without a class hint → propagated boxes use DEFAULT_LABEL."""
    client = _StubSam3Client()
    # Match sits away from the exemplar so it isn't deduped.
    client._visual_resp = _StubResp(boxes_norm=[[0.5, 0.5, 0.7, 0.7]], scores=[0.8])
    out = smart_visual(
        _task(),
        _visual_context((10, 10, 20, 20), label=None),
        client,
    )
    assert out[0]["value"]["rectanglelabels"] == [DEFAULT_LABEL]


def test_smart_visual_accepts_multiple_exemplars() -> None:
    client = _StubSam3Client()
    client._visual_resp = _StubResp(boxes_norm=[[0.1, 0.1, 0.2, 0.2]], scores=[0.7])
    smart_visual(
        _task(),
        _visual_context(
            (10, 10, 20, 20),
            (40, 40, 10, 10),
            label="forklift",
        ),
        client,
    )
    kwargs = client.calls[0][1]
    assert len(kwargs["exemplar_boxes_norm"]) == 2


def test_smart_visual_caps_max_results() -> None:
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0, 0, 0.05 + 0.01 * i, 0.05] for i in range(10)],
        scores=[0.5] * 10,
    )
    out = smart_visual(
        _task(),
        _visual_context((10, 10, 20, 20), label="x"),
        client,
        max_results=3,
    )
    assert len(out) == 3


def test_smart_visual_no_exemplar_returns_empty_no_call() -> None:
    client = _StubSam3Client()
    out = smart_visual(_task(), {"result": []}, client)
    assert out == []
    assert client.calls == []


def test_smart_visual_swallows_client_exceptions() -> None:
    client = _StubSam3Client()
    client._visual_raises = RuntimeError("boom")
    out = smart_visual(
        _task(),
        _visual_context((10, 10, 20, 20), label="forklift"),
        client,
    )
    assert out == []


def test_smart_visual_no_image_path_returns_empty() -> None:
    client = _StubSam3Client()
    out = smart_visual(
        {"data": {}},
        _visual_context((10, 10, 20, 20), label="forklift"),
        client,
    )
    assert out == []
    assert client.calls == []


def test_smart_visual_meta_carries_source() -> None:
    client = _StubSam3Client()
    # Match away from the exemplar so it survives IoU dedup.
    client._visual_resp = _StubResp(boxes_norm=[[0.5, 0.5, 0.7, 0.7]], scores=[0.9])
    out = smart_visual(
        _task(),
        _visual_context((10, 10, 20, 20), label="forklift"),
        client,
    )
    assert out[0]["meta"]["source"] == "smart_visual"


def test_smart_visual_drops_match_overlapping_exemplar() -> None:
    """SAM almost always returns the exemplar position as a top match — that
    duplicate must be dropped via the IoU dedup."""
    client = _StubSam3Client()
    # First match overlaps the exemplar at IoU=1; second is a real new match.
    client._visual_resp = _StubResp(
        boxes_norm=[[0.10, 0.10, 0.30, 0.30], [0.50, 0.50, 0.70, 0.70]],
        scores=[0.95, 0.7],
    )
    out = smart_visual(
        _task(),
        _visual_context((10, 10, 20, 20), label="forklift"),
        client,
    )
    # Exemplar duplicate dropped, only the [0.5, 0.5, 0.7, 0.7] match kept.
    assert len(out) == 1
    val = out[0]["value"]
    assert val["x"] == pytest.approx(50.0)
    assert val["y"] == pytest.approx(50.0)


def test_smart_visual_drops_match_overlapping_existing_canvas_region() -> None:
    """Match that lands on a region already on the live canvas is dropped.

    Canvas state lives in ``context.result`` (live LS state), not on
    server-side ``task["annotations"]`` which is stale during a session.
    Live-test 2026-04-30: the dedup pool sources from context only.
    """
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0.50, 0.50, 0.70, 0.70], [0.80, 0.80, 0.95, 0.95]],
        scores=[0.8, 0.6],
    )
    # Context has the smart_visual draft AND an existing same-class rectangle
    # the reviewer drew earlier in the session.
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            },
            {
                "type": "rectanglelabels",
                "from_name": "bbox",
                "value": {"x": 50, "y": 50, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
        ]
    }
    task = {
        "id": 1,
        "data": {
            "image_path": "/tmp/img.jpg",
            "image_id": "img_a",
            "image_size": [1920, 1080],
        },
    }
    out = smart_visual(task, ctx, client)
    # First match overlaps the existing canvas region → dropped. Only [.8, .8, .95, .95] kept.
    assert len(out) == 1
    val = out[0]["value"]
    assert val["x"] == pytest.approx(80.0)


def test_smart_visual_skips_cancelled_annotations() -> None:
    """A reviewer-cancelled annotation is not part of the dedup pool."""
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0.50, 0.50, 0.70, 0.70]], scores=[0.8]
    )
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            },
        ]
    }
    task = {
        "id": 1,
        "data": {"image_path": "/tmp/img.jpg", "image_id": "img_a"},
        "annotations": [
            {
                "id": 99,
                "was_cancelled": True,
                "result": [
                    {
                        "type": "rectanglelabels",
                        "from_name": "bbox",
                        "value": {
                            "x": 50, "y": 50, "width": 20, "height": 20,
                            "rectanglelabels": ["forklift"],
                        },
                    }
                ],
            }
        ],
    }
    out = smart_visual(task, ctx, client)
    # Cancelled annotation is ignored → match at (50,50) survives.
    assert len(out) == 1


def test_smart_visual_drops_zero_area_exemplar() -> None:
    """Reviewer dragging exemplar to a zero-area drag → SAM gets nothing."""
    client = _StubSam3Client()
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 0, "height": 0,
                          "labels": ["forklift"]},
            },
        ]
    }
    out = smart_visual(_task(), ctx, client)
    assert out == []
    assert client.calls == []


def test_smart_visual_uses_only_smart_visual_region_as_exemplar() -> None:
    """Phase B-frontend dispatch: when a region with from_name=smart_visual
    is present, ONLY that region is sent to SAM. Other rectangles in the
    context (existing accepted boxes) must not be passed as exemplars."""
    client = _StubSam3Client()
    client._visual_resp = _StubResp(boxes_norm=[], scores=[])
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "bbox",
                "value": {"x": 50, "y": 50, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            },
        ]
    }
    smart_visual(_task(), ctx, client)
    kwargs = client.calls[0][1]
    # Only the smart_visual region is the exemplar, not the bbox region.
    assert len(kwargs["exemplar_boxes_norm"]) == 1
    assert kwargs["exemplar_boxes_norm"][0] == pytest.approx([0.10, 0.10, 0.30, 0.30])


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


def test_dispatch_routes_textarea_to_smart_search() -> None:
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


def test_dispatch_routes_v_tool_to_smart_visual() -> None:
    """A draft with from_name='smart_visual' must route to smart_visual,
    not fall through to batch_proposals. Mirrors server._predict_one
    behaviour so callers using the pure dispatch() get the same result.
    """
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0.4, 0.4, 0.6, 0.6]], scores=[0.9]
    )
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
        ]
    }
    out = dispatch(_task(), ctx, sam3_client=client, db_path=None)
    assert len(out) == 1
    assert client.calls and client.calls[0][0] == "visual_prompt"


def test_dispatch_v_tool_does_not_fall_to_batch(seeded_pipeline_db: Path) -> None:
    """Even with a viable db_path, a smart_visual draft must NEVER fall through
    to batch_proposals. The reviewer drew an exemplar, not asked for the
    cached audit layer.
    """
    client = _StubSam3Client()
    client._visual_resp = _StubResp(boxes_norm=[], scores=[])
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
        ]
    }
    out = dispatch(
        _task(image_id="img_a"), ctx, sam3_client=client, db_path=seeded_pipeline_db
    )
    assert out == []
    assert client.calls and client.calls[0][0] == "visual_prompt"


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
    # Envelope score now reflects the best per-region score, not 1.0.
    assert out[0]["score"] == pytest.approx(0.8)
    assert client.calls[0][0] == "click_mask"


def test_server_predict_smart_visual_path() -> None:
    """smart_visual draws fire smart_visual, not smart_click or smart_search."""
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0.5, 0.5, 0.7, 0.7]], scores=[0.9]
    )
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            }
        ]
    }
    out = backend.predict([_task()], context=ctx)
    assert len(out) == 1
    assert out[0]["result"]
    assert client.calls[0][0] == "visual_prompt"


def test_server_predict_dispatch_uses_latest_region() -> None:
    """Stale smart_track exemplar in context must NOT hijack a fresh
    smart_visual draft. Dispatch picks the LAST classifiable region.
    """
    client = _StubSam3Client()
    client._visual_resp = _StubResp(boxes_norm=[[0.5, 0.5, 0.7, 0.7]], scores=[0.9])
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    ctx = {
        "result": [
            # leftover smart_track exemplar from a previous tool switch
            {"type": "rectanglelabels", "from_name": "smart_track",
             "value": {"x": 5, "y": 5, "width": 5, "height": 5,
                       "rectanglelabels": ["forklift"]}},
            # fresh smart_visual draft the user just drew
            {"type": "rectanglelabels", "from_name": "smart_visual",
             "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                       "labels": ["forklift"]}},
        ]
    }
    out = backend.predict([_task()], context=ctx)
    assert out[0]["result"]  # smart_visual ran and produced regions
    assert client.calls[0][0] == "visual_prompt"
    # smart_track was NOT invoked even though one was in the context.
    assert all(call[0] != "track" for call in client.calls)


def test_server_pick_route_classifies_each_tool() -> None:
    """Direct table-test of the classifier — protects the dispatch contract."""
    pick = ManualReviewerMLBackend._pick_route
    assert pick([{"type": "rectanglelabels", "from_name": "smart_track"}]) == "smart_track"
    assert pick([{"type": "rectanglelabels", "from_name": "smart_visual"}]) == "smart_visual"
    assert pick([{"type": "keypointlabels", "from_name": "click"}]) == "smart_click"
    assert pick([{"type": "textarea", "from_name": "text_query"}]) == "smart_search"
    assert pick([{"type": "rectanglelabels", "from_name": "bbox"}]) is None
    assert pick([]) is None


def test_server_predict_regular_bbox_does_not_fire_ml() -> None:
    """A regular Rectangle (from_name=bbox) draw must NOT trigger ML —
    only smart_visual (from_name=smart_visual) drafts route to smart_visual."""
    client = _StubSam3Client()
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "bbox",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            }
        ]
    }
    out = backend.predict([_task()], context=ctx)
    # Falls through to batch_proposals (db_path=None → empty result).
    assert out[0]["result"] == []
    assert client.calls == []


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


def test_server_envelope_score_is_max_region_score() -> None:
    """Envelope score reflects the best per-region score, not a constant 1.0."""
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.5], [0.6, 0.6, 0.9, 0.9]],
        scores=[0.42, 0.81],
        labels=["forklift", "forklift"],
    )
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    out = backend.predict([_task()], context=_text_context(["forklift"]))
    assert out[0]["score"] == pytest.approx(0.81)


def test_server_envelope_score_zero_for_no_regions() -> None:
    backend = ManualReviewerMLBackend(sam3_client=None, db_path=None)
    out = backend.predict([_task()], context=None)
    assert out[0]["result"] == []
    assert out[0]["score"] == 0.0


def test_resolve_db_path_warns_when_env_path_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``AAV4_PIPELINE_DB`` env set to a non-existent path → logs WARNING."""
    import logging

    from manual_reviewer.ml_backend import server as server_mod

    monkeypatch.setenv("AAV4_PIPELINE_DB", str(tmp_path / "missing.db"))
    server_mod._DB_PATH_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="manual_reviewer.ml_backend.server"):
        result = server_mod._resolve_db_path(None)
    assert result is None
    assert any("AAV4_PIPELINE_DB" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 6b. Phase D-toggles: per-route env-var gates
# ---------------------------------------------------------------------------


def test_route_enabled_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from manual_reviewer.ml_backend.server import _route_enabled

    for var in (
        "ENABLE_BATCH_PROPOSALS",
        "ENABLE_SMART_CLICK",
        "ENABLE_SMART_SEARCH",
        "ENABLE_SMART_TRACK",
        "ENABLE_SMART_VISUAL",
    ):
        monkeypatch.delenv(var, raising=False)
    assert _route_enabled("smart_click") is True
    assert _route_enabled("smart_search") is True
    assert _route_enabled("smart_track") is True
    assert _route_enabled("smart_visual") is True
    assert _route_enabled("batch_proposals") is True


@pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "off", "disabled", ""])
def test_route_enabled_falsy_disables(
    monkeypatch: pytest.MonkeyPatch, falsy: str
) -> None:
    from manual_reviewer.ml_backend.server import _route_enabled

    monkeypatch.setenv("ENABLE_SMART_CLICK", falsy)
    assert _route_enabled("smart_click") is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "enabled"])
def test_route_enabled_truthy_keeps_on(
    monkeypatch: pytest.MonkeyPatch, truthy: str
) -> None:
    from manual_reviewer.ml_backend.server import _route_enabled

    monkeypatch.setenv("ENABLE_SMART_CLICK", truthy)
    assert _route_enabled("smart_click") is True


def test_route_enabled_unknown_name_is_open() -> None:
    """Unrecognized route names default to enabled — never silently drop."""
    from manual_reviewer.ml_backend.server import _route_enabled

    assert _route_enabled("not_a_real_route") is True


def test_server_smart_click_gate_off_skips_sam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_SMART_CLICK", "0")
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.1, 0.5, 0.5], score=0.8)
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    out = backend.predict([_task()], context=_click_context(50, 60))
    assert out[0]["result"] == []
    assert client.calls == []


def test_server_smart_search_gate_off_skips_sam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_SMART_SEARCH", "false")
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.1, 0.1, 0.4, 0.4]], scores=[0.9], labels=["forklift"]
    )
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    out = backend.predict([_task()], context=_text_context(["forklift"]))
    assert out[0]["result"] == []
    assert client.calls == []


def test_server_smart_visual_gate_off_skips_sam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_SMART_VISUAL", "off")
    client = _StubSam3Client()
    client._visual_resp = _StubResp(boxes_norm=[[0.5, 0.5, 0.7, 0.7]], scores=[0.9])
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            }
        ]
    }
    out = backend.predict([_task()], context=ctx)
    assert out[0]["result"] == []
    assert client.calls == []


def test_server_v_tool_gate_off_falls_through_to_keypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed-context: a smart_visual draft + a keypoint draft on the same predict
    call. With ENABLE_SMART_VISUAL off, the keypoint must still reach
    smart_click — gating one route off shouldn't suppress every other
    route on the same draft."""
    monkeypatch.setenv("ENABLE_SMART_VISUAL", "off")
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.2, 0.3, 0.4], score=0.7)
    client._visual_resp = _StubResp(boxes_norm=[[0.5, 0.5, 0.7, 0.7]], scores=[0.9])
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
            {
                "type": "keypoint",
                "from_name": "click",
                "value": {"x": 50, "y": 60},
            },
        ]
    }
    out = backend.predict([_task()], context=ctx)
    assert len(out[0]["result"]) == 1
    # smart_click ran; smart_visual did not.
    methods = [name for name, _ in client.calls]
    assert "click_mask" in methods
    assert "visual_prompt" not in methods


def test_server_batch_proposals_gate_off_skips_db(
    monkeypatch: pytest.MonkeyPatch, seeded_pipeline_db: Path
) -> None:
    monkeypatch.setenv("ENABLE_BATCH_PROPOSALS", "0")
    backend = ManualReviewerMLBackend(
        sam3_client=_StubSam3Client(), db_path=seeded_pipeline_db
    )
    out = backend.predict([_task(image_id="img_a")], context=None)
    assert out[0]["result"] == []


def test_server_disabled_smart_route_does_not_fall_through_to_batch(
    monkeypatch: pytest.MonkeyPatch, seeded_pipeline_db: Path
) -> None:
    """A click with smart_click disabled must NOT seed cached proposals.

    Previously a gated-off smart route fell through to batch_proposals,
    flooding the canvas with cached boxes after a click/textarea — not
    what the reviewer asked for. We now return empty for "had a draft
    but no enabled smart route handled it".
    """
    monkeypatch.setenv("ENABLE_SMART_CLICK", "0")
    monkeypatch.delenv("ENABLE_BATCH_PROPOSALS", raising=False)
    client = _StubSam3Client()
    backend = ManualReviewerMLBackend(
        sam3_client=client, db_path=seeded_pipeline_db
    )
    # img_a HAS cached proposals — confirm fallthrough would have produced them.
    out = backend.predict(
        [_task(image_id="img_a")], context=_click_context(50, 60)
    )
    assert out[0]["result"] == []
    assert client.calls == []


def test_server_mixed_draft_disabled_route_still_runs_enabled_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed context with disabled smart_search + enabled smart_click must
    run smart_click — the disabled draft is `continue`d past, not a
    short-circuit return."""
    monkeypatch.setenv("ENABLE_SMART_SEARCH", "0")
    monkeypatch.delenv("ENABLE_SMART_CLICK", raising=False)
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.1, 0.5, 0.5], score=0.8)
    backend = ManualReviewerMLBackend(sam3_client=client, db_path=None)
    # textarea draft listed FIRST (would short-circuit under the old bug),
    # keypoint draft listed second.
    ctx = {
        "result": [
            {
                "type": "textarea",
                "from_name": "text_query",
                "value": {"text": ["forklift"]},
            },
            {
                "type": "keypoint",
                "from_name": "click",
                "value": {"x": 50, "y": 60, "labels": ["forklift"]},
            },
        ]
    }
    out = backend.predict([_task()], context=ctx)
    assert out[0]["result"], "smart_click should have fired despite disabled smart_search"
    assert any(c[0] == "click_mask" for c in client.calls)
    assert not any(c[0] == "text_detect" for c in client.calls)


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
# 7b. Route-level dedup behaviour — uniform policy across smart routes.
# Per the dedup design (manual_reviewer/ml_backend/dedup.py): each route
# runs internal NMS (0.85, class-aware) then external dedup (0.7,
# class-aware) against task.annotations + task.predictions + context.result.
# batch_proposals is intentionally exempt — that route is the raw
# per-detector evidence layer.
# ---------------------------------------------------------------------------


def _accepted_box(
    bbox: list[float], label: str = "forklift"
) -> dict[str, Any]:
    """An LS task["annotations"][i] entry holding one accepted bbox."""
    return {
        "id": 1,
        "result": [
            norm_box_to_ls_region(bbox, label, score=1.0, region_id="acc")
        ],
        "result_count": 1,
    }


def _seeded_prediction(
    bbox: list[float], label: str = "forklift"
) -> dict[str, Any]:
    """An LS task["predictions"][i] entry holding one seeded bbox."""
    return {
        "id": 99,
        "model_version": "finalize",
        "result": [
            norm_box_to_ls_region(bbox, label, score=0.9, region_id="seed")
        ],
    }


def test_smart_click_keeps_overlap_with_seeded_prediction() -> None:
    """Click on a seeded yellow prediction → returned (not in canvas pool).

    Live test 2026-04-29 surfaced the bug: build_tasks seeds ~30
    finalize predictions per task, and an IoU>0.7 dedup that included
    task.predictions ate every click on an already-detected object.
    Predictions are not on the live canvas until accepted.
    """
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.10, 0.10, 0.30, 0.30], score=0.8)
    task = _task()
    task["predictions"] = [_seeded_prediction([0.10, 0.10, 0.30, 0.30], "forklift")]
    ctx = {
        "result": [
            {
                "type": "keypoint",
                "from_name": "click",
                "value": {"x": 50, "y": 60, "labels": ["forklift"]},
            }
        ]
    }
    out = smart_click(task, ctx, client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


def test_smart_click_drops_duplicate_of_canvas_region() -> None:
    """A second click whose mask matches a same-class rectangle already
    on the live canvas (in context.result) is deduped — that's the
    "double-click no-op" case. Stale task.annotations don't count;
    only context.result does."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.10, 0.10, 0.30, 0.30], score=0.8)
    task = _task()
    ctx = {
        "result": [
            {
                "type": "keypoint",
                "from_name": "click",
                "value": {"x": 50, "y": 60, "labels": ["forklift"]},
            },
            {
                "type": "rectanglelabels",
                "from_name": "bbox",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
        ]
    }
    out = smart_click(task, ctx, client)
    assert out == []


def test_smart_click_keeps_different_class_overlap() -> None:
    """Person clicked on top of a forklift bbox already on canvas →
    kept (class-aware dedup)."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.10, 0.10, 0.30, 0.30], score=0.8)
    task = _task()
    ctx = {
        "result": [
            {
                "type": "keypoint",
                "from_name": "click",
                "value": {"x": 50, "y": 60, "labels": ["person"]},
            },
            {
                "type": "rectanglelabels",
                "from_name": "bbox",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "rectanglelabels": ["forklift"]},
            },
        ]
    }
    out = smart_click(task, ctx, client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["person"]


def test_smart_search_internal_nms_collapses_near_duplicates() -> None:
    """Two SAM matches at near-identical coords collapse to one (NMS).

    Lower-scoring duplicate is suppressed.
    """
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[
            [0.10, 0.10, 0.30, 0.30],
            [0.105, 0.105, 0.305, 0.305],   # ~99% IoU
        ],
        scores=[0.9, 0.4],
        labels=["forklift", "forklift"],
    )
    out = smart_search(_task(), _text_context(["forklift"]), client)
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.9)


def test_smart_search_internal_nms_keeps_different_classes_at_same_spot() -> None:
    """Class-aware NMS — person and forklift at same coords both survive."""
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[
            [0.10, 0.10, 0.30, 0.30],
            [0.10, 0.10, 0.30, 0.30],
        ],
        scores=[0.9, 0.8],
        labels=["forklift", "person"],
    )
    out = smart_search(_task(), _text_context(["x"]), client)
    assert len(out) == 2
    assert {r["value"]["rectanglelabels"][0] for r in out} == {"forklift", "person"}


def test_smart_search_drops_match_overlapping_seeded_prediction() -> None:
    """Re-fire smart_search after task open with a seeded yellow
    prediction at the same coords → match is DROPPED.

    Live-test 2026-04-30 task 37: LS doesn't echo task.predictions in
    context.result for smart_search fires, so the canvas-dedup pool was
    empty and SAM=16 → kept=16. The fix re-includes
    ``task.predictions`` in the dedup pool for smart_search/smart_visual
    (but not smart_click — refine clicks on a seeded box still produce
    a region).
    """
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.10, 0.10, 0.30, 0.30]],
        scores=[0.9],
        labels=["forklift"],
    )
    task = _task()
    task["predictions"] = [_seeded_prediction([0.10, 0.10, 0.30, 0.30], "forklift")]
    out = smart_search(task, _text_context(["forklift"]), client)
    assert out == []


def test_smart_visual_internal_nms_collapses_near_duplicates() -> None:
    """SAM grounding head emitting two near-identical matches → one survives."""
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[
            [0.50, 0.50, 0.70, 0.70],
            [0.502, 0.502, 0.702, 0.702],   # near-duplicate
        ],
        scores=[0.95, 0.5],
    )
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            }
        ]
    }
    out = smart_visual(_task(), ctx, client)
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.95)


def test_smart_visual_drops_match_overlapping_seeded_prediction() -> None:
    """Seeded predictions ARE in the dedup pool for smart_visual —
    same reason as smart_search: LS doesn't echo task.predictions in
    context.result on smart-tool fires."""
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0.50, 0.50, 0.70, 0.70]],
        scores=[0.9],
    )
    task = _task()
    task["predictions"] = [_seeded_prediction([0.50, 0.50, 0.70, 0.70], "forklift")]
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            }
        ]
    }
    out = smart_visual(task, ctx, client)
    assert out == []


def test_smart_visual_dedup_is_class_aware_against_canvas() -> None:
    """Different-class overlap on the canvas shouldn't suppress the match."""
    client = _StubSam3Client()
    client._visual_resp = _StubResp(
        boxes_norm=[[0.50, 0.50, 0.70, 0.70]],
        scores=[0.9],
    )
    task = _task()
    # Existing accepted box at the same coords but different class.
    task["annotations"] = [_accepted_box([0.50, 0.50, 0.70, 0.70], "person")]
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            }
        ]
    }
    out = smart_visual(task, ctx, client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


def test_batch_proposals_is_not_deduped(seeded_pipeline_db: Path) -> None:
    """Raw per-detector evidence layer — must NOT be deduped by route.

    aav4's pipeline already produced the deduped consensus output as
    the green seeded predictions; this route is the audit/debug layer.
    Even when the canvas already has an accepted box at the same spot,
    every cached proposal must come through.

    Real signal: compare batch_proposals output WITH and WITHOUT a same-
    class accepted box on the canvas. If the route ever started deduping
    against ``task["annotations"]``, the count with annotations would
    drop and this test would fail.
    """
    from manual_reviewer.ml_backend.routes import batch_proposals

    task_clean = _task(image_id="img_a")
    raw_clean = batch_proposals(task_clean, seeded_pipeline_db)
    assert raw_clean, "fixture must produce at least one cached proposal"

    # Place an accepted box that overlaps every cached proposal.
    task_with_canvas = _task(image_id="img_a")
    task_with_canvas["annotations"] = [
        _accepted_box([0.0, 0.0, 1.0, 1.0], "forklift")
    ]
    raw_with_canvas = batch_proposals(task_with_canvas, seeded_pipeline_db)

    assert len(raw_with_canvas) == len(raw_clean), (
        "batch_proposals must be exempt from dedup; "
        f"clean={len(raw_clean)} with_canvas={len(raw_with_canvas)}"
    )


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


def test_sam3_1_visual_prompt_request_validates() -> None:
    from data_miner.auto_annotation_v4.configs.wire import SAM3VisualPromptRequest

    req = SAM3VisualPromptRequest(
        image_path="/x.jpg", exemplar_boxes_norm=[[0.1, 0.1, 0.3, 0.3]]
    )
    assert req.threshold == pytest.approx(0.4)
    assert req.max_results == 50
    assert req.exemplar_labels == []


def test_sam3_1_visual_prompt_response_round_trips() -> None:
    from data_miner.auto_annotation_v4.configs.wire import SAM3VisualPromptResponse

    resp = SAM3VisualPromptResponse(
        boxes_norm=[[0.1, 0.1, 0.4, 0.5]],
        scores=[0.85],
    )
    payload = resp.model_dump()
    assert payload["boxes_norm"] == [[0.1, 0.1, 0.4, 0.5]]
    parsed = SAM3VisualPromptResponse.model_validate(payload)
    assert parsed.scores == [pytest.approx(0.85)]


def test_sam3_one_api_decode_picks_visual_for_exemplar_request() -> None:
    from data_miner.auto_annotation_v4.model_servers.sam3_1 import (
        _VISUAL_TAG,
        SAM3OneApi,
    )

    api = SAM3OneApi.__new__(SAM3OneApi)  # bypass setup
    decoded = api.decode_request(
        {
            "image_path": "/x.jpg",
            "exemplar_boxes_norm": [[0.1, 0.1, 0.3, 0.3]],
        }
    )
    assert decoded["__mode__"] == _VISUAL_TAG


def test_sam3_one_api_predict_routes_visual_prompt() -> None:
    """Mode dispatch must reach SAM3OneModel.visual_prompt."""
    from data_miner.auto_annotation_v4.configs.wire import SAM3VisualPromptResponse
    from data_miner.auto_annotation_v4.model_servers.sam3_1 import SAM3OneApi

    class _StubModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def visual_prompt(self, **kwargs: Any) -> Any:
            self.calls.append(("visual_prompt", kwargs))
            return SAM3VisualPromptResponse(
                boxes_norm=[[0, 0, 0.2, 0.2]], scores=[0.6]
            )

    api = SAM3OneApi.__new__(SAM3OneApi)
    api.model = _StubModel()
    decoded = api.decode_request(
        {
            "image_path": "/x.jpg",
            "exemplar_boxes_norm": [[0.1, 0.1, 0.3, 0.3]],
        }
    )
    out = api.predict(decoded)
    assert isinstance(out, SAM3VisualPromptResponse)
    assert api.model.calls[0][1]["exemplar_boxes_norm"] == [[0.1, 0.1, 0.3, 0.3]]


def test_sam3_one_http_client_visual_prompt_serializes_request() -> None:
    """Sam3OneHttpClient.visual_prompt sends a SAM3VisualPromptRequest body."""
    from manual_reviewer.reconcile import Sam3OneHttpClient

    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "boxes_norm": [[0.1, 0.1, 0.4, 0.5]],
                "scores": [0.8],
                "mask_rles": None,
            }

    class _FakeSession:
        def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _FakeResp:
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    client = Sam3OneHttpClient(
        url="http://stub:1234/predict",
        visual_url="http://stub:1234/predict",
        session=_FakeSession(),
    )
    resp = client.visual_prompt(
        image_path="/img.jpg",
        exemplar_boxes_norm=[[0.1, 0.1, 0.3, 0.3]],
    )
    assert resp.boxes_norm == [[0.1, 0.1, 0.4, 0.5]]
    assert captured["json"]["exemplar_boxes_norm"] == [[0.1, 0.1, 0.3, 0.3]]
    assert captured["json"]["threshold"] == pytest.approx(0.4)


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


# ---------------------------------------------------------------------------
# 9. aav4_client retry-once shim + canvas contamination
# ---------------------------------------------------------------------------


def test_build_sam3_client_retries_once_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One transient ConnectionError is retried; the second call succeeds."""
    import requests

    from manual_reviewer.ml_backend import aav4_client as aav4

    calls: list[str] = []

    class _Inner:
        def click_mask(self, **kwargs: Any) -> Any:
            calls.append("click_mask")
            if len(calls) == 1:
                raise requests.ConnectionError("RST")
            return _StubResp(bbox=[0.1, 0.1, 0.2, 0.2], score=0.5)

        def text_detect(self, **kwargs: Any) -> Any:
            calls.append("text_detect")
            return _StubResp(boxes=[], scores=[], labels=[])

        def visual_prompt(self, **kwargs: Any) -> Any:
            calls.append("visual_prompt")
            return _StubResp(boxes_norm=[], scores=[])

    wrapped = aav4._RetryOnceClient(_Inner())  # type: ignore[arg-type]
    resp = wrapped.click_mask(image_path="/x.jpg", point=[0.5, 0.5])
    assert resp.score == pytest.approx(0.5)
    assert calls == ["click_mask", "click_mask"]


def test_build_sam3_client_does_not_retry_unrelated_errors() -> None:
    """A RuntimeError (not ConnectionError/Timeout) bubbles up unretried."""
    from manual_reviewer.ml_backend import aav4_client as aav4

    calls: list[str] = []

    class _Inner:
        def click_mask(self, **kwargs: Any) -> Any:
            calls.append("click_mask")
            raise RuntimeError("not retryable")

        def text_detect(self, **kwargs: Any) -> Any:
            return None

        def visual_prompt(self, **kwargs: Any) -> Any:
            return None

    wrapped = aav4._RetryOnceClient(_Inner())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        wrapped.click_mask(image_path="/x.jpg", point=[0.5, 0.5])
    assert calls == ["click_mask"]


def test_build_sam3_client_retries_once_on_timeout() -> None:
    import requests

    from manual_reviewer.ml_backend import aav4_client as aav4

    calls: list[str] = []

    class _Inner:
        def click_mask(self, **kwargs: Any) -> Any:
            calls.append("click_mask")
            if len(calls) == 1:
                raise requests.Timeout("slow")
            return _StubResp(bbox=[0, 0, 0.1, 0.1], score=0.4)

        def text_detect(self, **kwargs: Any) -> Any:
            return None

        def visual_prompt(self, **kwargs: Any) -> Any:
            return None

    wrapped = aav4._RetryOnceClient(_Inner())  # type: ignore[arg-type]
    resp = wrapped.click_mask(image_path="/x.jpg", point=[0.5, 0.5])
    assert resp.score == pytest.approx(0.4)
    assert calls == ["click_mask", "click_mask"]


def test_smart_search_does_not_dedup_against_v_tool_exemplar() -> None:
    """smart_visual exemplar on canvas must NOT suppress smart_search matches.

    Without filtering smart_visual regions out of the canvas pool, a
    same-class smart_search match overlapping a leftover exemplar would
    silently disappear.
    """
    client = _StubSam3Client()
    client._text_resp = _StubResp(
        boxes=[[0.10, 0.10, 0.30, 0.30]],
        scores=[0.9],
        labels=["forklift"],
    )
    ctx = {
        "result": [
            {
                "type": "rectanglelabels",
                "from_name": "smart_visual",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20,
                          "labels": ["forklift"]},
            },
            {
                "type": "textarea",
                "from_name": "text_query",
                "value": {"text": ["forklift"]},
            },
        ]
    }
    out = smart_search(_task(), ctx, client)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["forklift"]


def test_smart_click_emits_original_dims_when_image_size_known() -> None:
    """Image dims from ``task.data.image_size`` ride through onto the region."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.1, 0.5, 0.5], score=0.7)
    out = smart_click(_task(), _click_context(50, 60), client)
    assert len(out) == 1
    assert out[0]["original_width"] == 1920
    assert out[0]["original_height"] == 1080
    assert out[0]["original_rotation"] == 0


def test_smart_click_omits_original_dims_when_image_size_missing() -> None:
    """No ``image_size`` in task.data → keys absent rather than ``None``."""
    client = _StubSam3Client()
    client._click_resp = _StubResp(bbox=[0.1, 0.1, 0.5, 0.5], score=0.7)
    task = {"id": 1, "data": {"image_path": "/tmp/x.jpg"}}
    out = smart_click(task, _click_context(50, 60), client)
    assert len(out) == 1
    assert "original_width" not in out[0]
    assert "original_height" not in out[0]
