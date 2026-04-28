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
    "SAM3ClickMaskRequest",
    "SAM3ClickMaskResponse",
    "SAM3RefineRequest",
    "SAM3RefineResponse",
    "SAM3VideoTrackRequest",
    "SAM3VideoTrackResponse",
    "SAM3VideoTrackSeed",
    "SAM3VideoTrackFrameOutput",
    "SAM3VideoTrackObjectOutput",
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
# SAM 3.1 click-mask wire models
# ---------------------------------------------------------------------------
#
# Used by the manual_reviewer Label Studio ML backend's smart_click route:
# reviewer drops a KeyPoint on the canvas, the backend turns it into a
# point prompt, SAM 3.1 returns a mask whose tight bbox is sent back as a
# RectangleLabels region. The full mask RLE is also returned for callers
# that want it (BrushLabels in v2).


class SAM3ClickMaskRequest(BaseModel):
    """Wire request for SAM 3.1 point→mask inference.

    Single positive (or negative) point prompt on a single image. ``point``
    is normalized [x, y] in [0, 1]. ``point_label`` follows SAM convention:
    1 = foreground / positive, 0 = background / negative.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    point: list[float]  # normalized [x, y]
    point_label: int = 1  # 1 = positive, 0 = negative
    threshold: float = 0.5


class SAM3ClickMaskResponse(BaseModel):
    """Wire response for SAM 3.1 point→mask inference."""

    model_config = ConfigDict(extra="forbid")

    bbox: list[float] | None = None  # normalized [x1, y1, x2, y2]
    mask_rle: dict[str, Any] | None = None
    score: float = 0.0


# ---------------------------------------------------------------------------
# SAM 3.1 video-track wire models
# ---------------------------------------------------------------------------
#
# Stateless RPC over the stateful SAM 3.1 video predictor: the server takes
# a resource (MP4, JPEG-folder, or single image), seeds the predictor with
# one or more prompts on specific frames, runs propagation, and returns
# per-frame outputs in a single response. Sessions are owned by the server
# and torn down on every call — clients never see session_ids.
#
# Mirrors ``Sam3VideoPredictor.handle_request`` / ``handle_stream_request``
# from facebookresearch/sam3.


class SAM3VideoTrackSeed(BaseModel):
    """One ``add_prompt`` call's worth of seeding info.

    At least one of ``bbox``, ``text``, or ``points`` must be set. ``obj_id``
    is required so the response can group masks by object across frames.
    Coordinates are normalized [0, 1] — the server converts to pixels.
    """

    model_config = ConfigDict(extra="forbid")

    obj_id: int
    frame_index: int = 0
    bbox: list[float] | None = None  # normalized [x1, y1, x2, y2]
    text: str | None = None
    points: list[list[float]] | None = None  # normalized [[x, y], ...]
    point_labels: list[int] | None = None  # 1 = positive, 0 = negative


class SAM3VideoTrackRequest(BaseModel):
    """Wire request to the SAM 3.1 ``/track`` HTTP endpoint."""

    model_config = ConfigDict(extra="forbid")

    resource_path: str  # MP4, JPEG-folder, or single image path
    seeds: list[SAM3VideoTrackSeed]
    propagation_direction: str = "both"  # "forward", "backward", "both"
    max_frames: int | None = None
    return_masks: bool = False  # mask_rle is large; off by default


class SAM3VideoTrackObjectOutput(BaseModel):
    """One object's per-frame mask + bbox + score."""

    model_config = ConfigDict(extra="forbid")

    obj_id: int
    bbox: list[float] | None = None  # normalized [x1, y1, x2, y2], None when absent
    mask_rle: dict[str, Any] | None = None  # only populated when return_masks=True
    score: float = 0.0


class SAM3VideoTrackFrameOutput(BaseModel):
    """All tracked objects on one frame."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int
    objects: list[SAM3VideoTrackObjectOutput] = []


class SAM3VideoTrackResponse(BaseModel):
    """Wire response from the SAM 3.1 ``/track`` HTTP endpoint."""

    model_config = ConfigDict(extra="forbid")

    frames: list[SAM3VideoTrackFrameOutput] = []


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
