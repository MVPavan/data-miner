"""HTTP wire contracts for model-server communication.

These Pydantic models define the request/response schemas exchanged over HTTP
between the auto_annotation_v4 pipeline and the detector / SAM3 LitServe
endpoints.  They are deliberately enum-independent: ``labels`` are raw strings
returned by model servers so that wire contracts never import from enums.py.

Internal hand-off models (``PreparedInput``, ``RawPrediction``) carry tensors
and PIL images between ``decode_request`` / ``predict`` / ``encode_response``
inside each LitServe server.  They refuse JSON serialization to prevent
accidental misuse.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer

__all__ = [
    "DetectorRequest",
    "DetectorResponse",
    "SAM3RefineRequest",
    "SAM3RefineResponse",
    "PreparedInput",
    "RawPrediction",
]

# ---------------------------------------------------------------------------
# Detector wire models
# ---------------------------------------------------------------------------


class DetectorRequest(BaseModel):
    """Uniform wire request accepted by every detector HTTP endpoint.

    Sent by the pipeline's detect stage to each configured detector server.
    ``prompts`` contains the text queries (class names / phrases) the server
    should detect; ``threshold`` optionally overrides the server default.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    prompts: list[str]
    threshold: float | None = None


class DetectorResponse(BaseModel):
    """Uniform wire response returned by every detector HTTP endpoint.

    ``labels`` is parallel to ``boxes`` / ``scores`` and contains the raw
    string label each detection was matched to.  The server is responsible for
    mapping its internal vocabulary back to one of the ``DetectorRequest.prompts``
    strings the caller sent.
    """

    model_config = ConfigDict(extra="forbid")

    boxes: list[list[float]]  # normalized [x1, y1, x2, y2]
    scores: list[float]
    labels: list[str]


# ---------------------------------------------------------------------------
# SAM3 refine wire models
# ---------------------------------------------------------------------------


class SAM3RefineRequest(BaseModel):
    """Wire request to the SAM3 ``/refine`` HTTP endpoint.

    Used by the refine stage to tighten or extend a bounding box via SAM3's
    mask-then-bbox flow.  ``points`` provides optional positive/negative point
    prompts as ``[[x, y, label], ...]``.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    bbox: list[float]  # normalized [x1, y1, x2, y2]
    points: list[list[float]] | None = None
    threshold: float = 0.5


class SAM3RefineResponse(BaseModel):
    """Wire response from the SAM3 ``/refine`` HTTP endpoint.

    Returns the refined bounding box and its confidence score, or ``None`` /
    ``0.0`` when SAM3 could not produce a valid mask.
    """

    model_config = ConfigDict(extra="forbid")

    box: list[float] | None = None  # normalized [x1, y1, x2, y2]
    score: float = 0.0


# ---------------------------------------------------------------------------
# Internal LitServe hand-off models (NOT JSON-serializable)
# ---------------------------------------------------------------------------


def _forbid_json(_self: object, _value: object) -> None:
    """Raise on any attempt to JSON-serialize a tensor-carrying model."""
    raise TypeError("PreparedInput is not JSON-serializable")


class PreparedInput(BaseModel):
    """Output of ``decode_request`` inside each LitServe detector server.

    Carries pre-processed tensors, PIL images, and processor outputs between
    ``decode_request`` and ``predict``.  Must never be serialized to JSON.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    image: Any
    processor_inputs: Any
    image_size: tuple[int, int]
    prompts: list[str]
    threshold: float | None = None
    extras: dict[str, Any] = {}

    @field_serializer("image", "processor_inputs", "extras", when_used="json")
    def _ser_nonjson(self, v: object) -> None:
        return _forbid_json(self, v)


class RawPrediction(BaseModel):
    """Output of ``predict`` inside each LitServe detector server.

    Carries raw model outputs and the corresponding inputs between ``predict``
    and ``encode_response``.  Must never be serialized to JSON.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    outputs: Any
    inputs: Any
    image_size: tuple[int, int]
    prompts: list[str]
    threshold: float | None = None
    extras: dict[str, Any] = {}

    @field_serializer("outputs", "inputs", "extras", when_used="json")
    def _ser_nonjson(self, v: object) -> None:
        return _forbid_json(self, v)
