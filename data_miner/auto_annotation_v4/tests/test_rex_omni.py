"""Protocol-level tests for the Rex-Omni model + server.

GPU loading is not exercised here — we stub ``rex_omni.RexOmniWrapper`` with
an in-memory fake so the full prepare/infer/postprocess flow can be unit-
tested on CPU. These tests cover:

  * Enum + servers.yaml + serve.py registry plumbing.
  * Coordinate normalization (pixels → [0, 1]).
  * Empty / malformed predictions are skipped.
  * Multi-class output is flattened with the right labels.
  * RexOmniApi.setup uses the configured backend / max_new_tokens.
  * Wire format (DetectorRequest/Response) is unchanged.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from PIL import Image

from data_miner.auto_annotation_v4.configs.enums import DetectorName
from data_miner.auto_annotation_v4.configs.wire import (
    DetectorRequest,
    DetectorResponse,
    PreparedInput,
)
from data_miner.auto_annotation_v4.models.rex_omni import (
    RexOmniModel,
    _pixel_xyxy_to_norm,
)


# ---------------------------------------------------------------------------
# 1. Enum + registry plumbing
# ---------------------------------------------------------------------------


def test_detector_name_includes_rex_omni() -> None:
    assert DetectorName.REX_OMNI.value == "rex_omni"


def test_servers_yaml_has_rex_omni_block() -> None:
    """Sanity check that the servers.yaml file lists rex_omni."""
    from pathlib import Path

    yaml_path = (
        Path(__file__).resolve().parents[1] / "configs" / "servers.yaml"
    )
    text = yaml_path.read_text(encoding="utf-8")
    assert "rex_omni:" in text
    assert "IDEA-Research/Rex-Omni" in text
    assert "max_batch_size: 1" in text


def test_serve_registry_includes_rex_omni() -> None:
    """The registry entry must resolve to a LitAPI subclass."""
    from data_miner.auto_annotation_v4.model_servers.serve import _get_registry

    registry = _get_registry()
    assert DetectorName.REX_OMNI in registry
    api_cls = registry[DetectorName.REX_OMNI]
    assert api_cls.__name__ == "RexOmniApi"


# ---------------------------------------------------------------------------
# 2. Geometry helper
# ---------------------------------------------------------------------------


def test_pixel_xyxy_to_norm_basic() -> None:
    assert _pixel_xyxy_to_norm([0, 0, 100, 100], 200, 200) == pytest.approx(
        [0.0, 0.0, 0.5, 0.5]
    )


def test_pixel_xyxy_to_norm_clamps_overflow() -> None:
    out = _pixel_xyxy_to_norm([-10, -10, 1000, 1000], 200, 200)
    assert out == pytest.approx([0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "coords",
    [
        None,
        [0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, "x", 1],
        [50, 50, 10, 10],  # x2 < x1
    ],
)
def test_pixel_xyxy_to_norm_returns_none_on_garbage(coords: Any) -> None:
    assert _pixel_xyxy_to_norm(coords, 100, 100) is None


def test_pixel_xyxy_to_norm_zero_dim_returns_none() -> None:
    assert _pixel_xyxy_to_norm([0, 0, 1, 1], 0, 100) is None


# ---------------------------------------------------------------------------
# 3. Model with stubbed wrapper
# ---------------------------------------------------------------------------


class _FakeWrapper:
    """Simulate RexOmniWrapper.inference. Returns canned per-image predictions."""

    def __init__(self, *, model_path: str = "", backend: str = "transformers", **kw: Any) -> None:
        self.model_path = model_path
        self.backend = backend
        self.last_kwargs: dict[str, Any] | None = None
        self.canned_results: list[dict[str, Any]] = [
            {
                "extracted_predictions": {
                    "person": [
                        {"type": "box", "coords": [10, 20, 110, 220]},
                    ],
                    "car": [
                        {"type": "box", "coords": [50, 50, 150, 150]},
                    ],
                }
            }
        ]

    def inference(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_kwargs = kwargs
        return self.canned_results


@pytest.fixture
def stub_rex_module(monkeypatch: pytest.MonkeyPatch) -> _FakeWrapper:
    """Inject a fake ``rex_omni`` module that yields _FakeWrapper instances."""
    fake_mod = types.ModuleType("rex_omni")
    last: dict[str, _FakeWrapper] = {}

    def _factory(**kwargs: Any) -> _FakeWrapper:
        w = _FakeWrapper(**kwargs)
        last["wrapper"] = w
        return w

    fake_mod.RexOmniWrapper = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rex_omni", fake_mod)
    return last  # type: ignore[return-value]


def test_load_uses_provided_backend(stub_rex_module: dict[str, _FakeWrapper]) -> None:
    model = RexOmniModel()
    model.load("cuda:0", "IDEA-Research/Rex-Omni", backend="vllm")
    wrapper = stub_rex_module["wrapper"]
    assert wrapper.model_path == "IDEA-Research/Rex-Omni"
    assert wrapper.backend == "vllm"
    assert model.backend == "vllm"


def test_load_falls_back_when_wrapper_rejects_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older wrapper versions don't accept ``device=``; load() must retry."""
    calls: list[dict[str, Any]] = []

    def _factory(**kwargs: Any) -> _FakeWrapper:
        calls.append(dict(kwargs))
        if "device" in kwargs:
            raise TypeError("unexpected kwarg device")
        return _FakeWrapper(**kwargs)

    fake_mod = types.ModuleType("rex_omni")
    fake_mod.RexOmniWrapper = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rex_omni", fake_mod)

    model = RexOmniModel()
    model.load("cuda:0", "IDEA-Research/Rex-Omni")
    assert len(calls) == 2
    assert "device" in calls[0]
    assert "device" not in calls[1]


def test_prepare_returns_prepared_input(stub_rex_module: dict[str, _FakeWrapper]) -> None:
    model = RexOmniModel()
    model.load("cpu")
    img = Image.new("RGB", (300, 200), color="white")
    prepared = model.prepare(img, ["person", "car"])
    assert isinstance(prepared, PreparedInput)
    assert prepared.image_size == (300, 200)
    assert prepared.prompts == ["person", "car"]


def test_prepare_rejects_none_image() -> None:
    model = RexOmniModel()
    with pytest.raises(ValueError):
        model.prepare(None, ["person"])  # type: ignore[arg-type]


def test_full_inference_round_trip_normalizes_boxes(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    model = RexOmniModel()
    model.load("cpu")
    img = Image.new("RGB", (200, 400))  # easy normalization math
    prepared = model.prepare(img, ["person", "car"])
    raw = model.infer(prepared)
    resp = model.postprocess(raw)

    assert isinstance(resp, DetectorResponse)
    assert resp.labels == ["person", "car"]
    # person box [10, 20, 110, 220] in a 200x400 image
    assert resp.boxes[0] == pytest.approx([0.05, 0.05, 0.55, 0.55])
    # car box [50, 50, 150, 150] in a 200x400 image
    assert resp.boxes[1] == pytest.approx([0.25, 0.125, 0.75, 0.375])
    # No per-detection score → 1.0 each
    assert resp.scores == [1.0, 1.0]


def test_postprocess_skips_non_box_items(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    """Polygons / keypoints from other tasks must be dropped."""
    model = RexOmniModel()
    model.load("cpu")
    wrapper = stub_rex_module["wrapper"]
    wrapper.canned_results = [
        {
            "extracted_predictions": {
                "person": [
                    {"type": "polygon", "coords": [0, 0, 1, 1]},
                    {"type": "box", "coords": [10, 10, 20, 20]},
                ],
            }
        }
    ]
    img = Image.new("RGB", (100, 100))
    raw = model.infer(model.prepare(img, ["person"]))
    resp = model.postprocess(raw)
    assert len(resp.boxes) == 1
    assert resp.labels == ["person"]


def test_postprocess_skips_malformed_coords(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    model = RexOmniModel()
    model.load("cpu")
    wrapper = stub_rex_module["wrapper"]
    wrapper.canned_results = [
        {
            "extracted_predictions": {
                "x": [
                    {"type": "box", "coords": "garbage"},
                    {"type": "box", "coords": [0, 0]},
                    {"type": "box"},  # no coords
                    {"type": "box", "coords": [10, 10, 5, 5]},  # negative size
                    {"type": "box", "coords": [0, 0, 50, 50]},  # valid
                ]
            }
        }
    ]
    img = Image.new("RGB", (100, 100))
    raw = model.infer(model.prepare(img, ["x"]))
    resp = model.postprocess(raw)
    assert len(resp.boxes) == 1
    assert resp.labels == ["x"]


def test_infer_with_empty_prompts_returns_empty_response(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    model = RexOmniModel()
    model.load("cpu")
    img = Image.new("RGB", (100, 100))
    prepared = model.prepare(img, [])
    raw = model.infer(prepared)
    resp = model.postprocess(raw)
    assert resp.boxes == []
    assert resp.labels == []
    # Wrapper.inference must not be called when there are zero prompts.
    assert stub_rex_module["wrapper"].last_kwargs is None


def test_infer_passes_categories_and_task(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    model = RexOmniModel()
    model.load("cpu")
    img = Image.new("RGB", (100, 100))
    prepared = model.prepare(img, ["forklift", "person"])
    model.infer(prepared)
    kwargs = stub_rex_module["wrapper"].last_kwargs
    assert kwargs is not None
    assert kwargs["task"] == "detection"
    assert kwargs["categories"] == ["forklift", "person"]


def test_infer_max_new_tokens_passed_when_set(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    model = RexOmniModel()
    model.load("cpu", max_new_tokens=4096)
    img = Image.new("RGB", (50, 50))
    model.infer(model.prepare(img, ["x"]))
    kwargs = stub_rex_module["wrapper"].last_kwargs
    assert kwargs is not None
    assert kwargs["max_new_tokens"] == 4096


def test_infer_handles_dict_results(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    """Some wrapper builds return a single dict instead of a list."""
    model = RexOmniModel()
    model.load("cpu")
    wrapper = stub_rex_module["wrapper"]
    # Simulate the dict-shape return.
    wrapper.canned_results = {  # type: ignore[assignment]
        "extracted_predictions": {
            "person": [{"type": "box", "coords": [0, 0, 50, 50]}],
        }
    }
    img = Image.new("RGB", (100, 100))
    resp = model.postprocess(model.infer(model.prepare(img, ["person"])))
    assert resp.labels == ["person"]


# ---------------------------------------------------------------------------
# 4. RexOmniApi (LitAPI) — knob plumbing only
# ---------------------------------------------------------------------------


def test_rex_omni_api_setup_passes_backend_and_max_tokens(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    from data_miner.auto_annotation_v4.model_servers.rex_omni import RexOmniApi

    api = RexOmniApi.__new__(RexOmniApi)
    api.model_id = "IDEA-Research/Rex-Omni"
    api._backend = "vllm"
    api._max_new_tokens = 2048
    api.setup("cpu")

    assert api.model is not None  # type: ignore[attr-defined]
    assert isinstance(api.model, RexOmniModel)  # type: ignore[attr-defined]
    assert api.model.backend == "vllm"  # type: ignore[attr-defined]
    assert api.model.max_new_tokens == 2048  # type: ignore[attr-defined]


def test_rex_omni_api_uses_default_backend_when_unset(
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    from data_miner.auto_annotation_v4.model_servers.rex_omni import RexOmniApi

    api = RexOmniApi.__new__(RexOmniApi)
    api.model_id = "IDEA-Research/Rex-Omni"
    api._backend = "transformers"
    api._max_new_tokens = None
    api.setup("cpu")
    assert api.model.backend == "transformers"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 5. End-to-end wire round-trip via the LitAPI hooks
# ---------------------------------------------------------------------------


def test_rex_omni_api_decode_predict_encode_round_trip(
    tmp_path: Any,
    stub_rex_module: dict[str, _FakeWrapper],
) -> None:
    """decode_request → predict → encode_response, no LitServe needed."""
    from data_miner.auto_annotation_v4.model_servers.rex_omni import RexOmniApi

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (200, 400), color="red").save(img_path)

    api = RexOmniApi.__new__(RexOmniApi)
    api.model_id = "IDEA-Research/Rex-Omni"
    api._backend = "transformers"
    api._max_new_tokens = None
    api.setup("cpu")

    request = {
        "image_path": str(img_path),
        "prompts": ["person", "car"],
        "threshold": None,
    }
    # DetectorRequest validates this shape.
    DetectorRequest.model_validate(request)

    prepared = api.decode_request(request)
    raw = api.predict(prepared)
    encoded = api.encode_response(raw)
    parsed = DetectorResponse.model_validate(encoded)
    assert parsed.labels == ["person", "car"]
    assert all(0.0 <= x <= 1.0 for box in parsed.boxes for x in box)
