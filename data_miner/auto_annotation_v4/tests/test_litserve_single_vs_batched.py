"""Unit tests for LitServe single-loop vs batched-loop calling conventions.

LitServe switches between two predict() calling conventions based on
``max_batch_size``:

- ``max_batch_size >= 2`` → ``predict([item, item, ...])`` returns ``[resp, resp, ...]``
- ``max_batch_size == 1`` → ``predict(item)`` returns ``resp``

The ``DetectorServerBase.predict`` and ``SAM3DartApi.predict`` implementations
must tolerate both shapes. Locks in the fix for the
``'PreparedInput' object is not subscriptable`` regression triggered when
GDINO ran at ``max_batch_size=1``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from data_miner.auto_annotation_v4.model_servers.base import DetectorServerBase
from data_miner.auto_annotation_v4.model_servers.sam3_dart import (
    SAM3DartApi,
    _REFINE_TAG,
)


def _make_base_api(infer_batch_return: list) -> DetectorServerBase:
    api = DetectorServerBase.__new__(DetectorServerBase)  # skip __init__
    api.model = MagicMock()
    api.model.infer_batch = MagicMock(return_value=infer_batch_return)
    return api


def test_base_predict_batched_mode_passes_list_through():
    api = _make_base_api(infer_batch_return=["resp0", "resp1", "resp2"])
    out = api.predict(["a", "b", "c"])
    assert out == ["resp0", "resp1", "resp2"]
    api.model.infer_batch.assert_called_once_with(["a", "b", "c"])


def test_base_predict_single_mode_wraps_and_unwraps():
    api = _make_base_api(infer_batch_return=["only_response"])
    out = api.predict("solo_item")  # single-loop convention
    assert out == "only_response"
    # Must call infer_batch with a one-element list, never a bare item.
    api.model.infer_batch.assert_called_once_with(["solo_item"])


def test_base_predict_single_mode_preserves_pydantic_objects():
    # PreparedInput (or similar non-list) must not be subscripted as a list.
    class FakePrepared:
        prompts = ["truck"]

        def __getitem__(self, _):
            raise TypeError("'PreparedInput' object is not subscriptable")

    fake = FakePrepared()
    api = _make_base_api(infer_batch_return=["raw_pred"])
    out = api.predict(fake)
    assert out == "raw_pred"
    # Assert it was wrapped, not subscripted.
    (args, _kwargs) = api.model.infer_batch.call_args
    assert args[0] == [fake]


def _make_sam3_api() -> SAM3DartApi:
    api = SAM3DartApi.__new__(SAM3DartApi)  # skip __init__
    api.model = MagicMock()
    api.model.infer_batch = MagicMock(side_effect=lambda items: [f"raw:{i}" for i, _ in enumerate(items)])
    api.model.refine = MagicMock(side_effect=lambda item: f"refine:{item.get('pixel_box')}")
    return api


def test_sam3_predict_batched_mixed_proposal_and_refine():
    api = _make_sam3_api()
    batch = [
        "proposal_item_0",
        {"__mode__": _REFINE_TAG, "pixel_box": [1, 2, 3, 4]},
        "proposal_item_2",
    ]
    out = api.predict(batch)
    # Proposal indices 0, 2 → raw:0, raw:1 (infer_batch got 2 items)
    # Refine index 1 → refine:[1,2,3,4]
    assert out[0] == "raw:0"
    assert out[1] == "refine:[1, 2, 3, 4]"
    assert out[2] == "raw:1"


def test_sam3_predict_single_mode_proposal():
    api = _make_sam3_api()
    out = api.predict("solo_proposal")
    assert out == "raw:0"
    api.model.infer_batch.assert_called_once_with(["solo_proposal"])
    api.model.refine.assert_not_called()


def test_sam3_predict_single_mode_refine():
    api = _make_sam3_api()
    out = api.predict({"__mode__": _REFINE_TAG, "pixel_box": [0.1, 0.2, 0.3, 0.4]})
    assert out == "refine:[0.1, 0.2, 0.3, 0.4]"
    api.model.refine.assert_called_once()
    api.model.infer_batch.assert_not_called()
