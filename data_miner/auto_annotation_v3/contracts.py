"""All Pydantic data models for auto_annotation_v3. No loose dicts or strings."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DetectorName(StrEnum):
    """Canonical detector identifiers. YAML keys in ``servers.detectors`` and
    any Python reference must use these values — no free-form strings."""

    GROUNDING_DINO = "grounding_dino"
    FALCON         = "falcon"
    SAM3           = "sam3"
    SAM3_DART      = "sam3_dart"
    OWLVIT2        = "owlvit2"

    @property
    def is_sam3_family(self) -> bool:
        """Both SAM3 variants serve the same wire contract (proposal + refine).
        Used by refine.py / detect.py to find whichever is currently enabled."""
        return self in (DetectorName.SAM3, DetectorName.SAM3_DART)


class BboxQuality(str, Enum):
    GOOD = "good"
    NEEDS_EXPANSION = "needs_expansion"
    TOO_LOOSE = "too_loose"
    BAD = "bad"


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    FILTERED_OUT = "filtered_out"
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_EVALUATION = "needs_evaluation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REFINED = "refined"


class FinalAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Normalised bounding box (0.0–1.0 coordinates)."""

    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def cx(self) -> float:
        return self.x1 + self.width / 2

    @property
    def cy(self) -> float:
        return self.y1 + self.height / 2


# ---------------------------------------------------------------------------
# Candidate — core annotation unit tracked through all stages
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """A single detection candidate tracked from proposal through final decision."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    class_name: str
    label: str
    source_model: str
    expression: str
    bbox: BoundingBox
    score: float = 1.0
    agreement: int = 0
    agreeing_models: list[str] = Field(default_factory=list)
    status: CandidateStatus = CandidateStatus.PROPOSED
    mask_rle: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: DETECT
# ---------------------------------------------------------------------------


class ProposalResult(BaseModel):
    """Raw per-model output before any filtering or dedup."""

    model_config = ConfigDict(extra="forbid")

    model: str
    image_id: str
    image_size: list[int]  # [width, height]
    latency_ms: float
    candidates: list[Candidate] = Field(default_factory=list)


class DetectRouting(BaseModel):
    """Routing decisions produced by the detect stage."""

    model_config = ConfigDict(extra="forbid")

    auto_accepted: list[str] = Field(default_factory=list)
    needs_evaluation: list[str] = Field(default_factory=list)
    confusion_flags: list[dict[str, Any]] = Field(default_factory=list)


class DetectResult(BaseModel):
    """Stage 1 output: filtered, deduped, and routed candidates."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    image_path: str
    image_size: list[int]  # [width, height]
    models_used: list[str]
    candidates: list[Candidate]
    routing: DetectRouting = Field(default_factory=DetectRouting)
    filter_stats: dict[str, Any] = Field(default_factory=dict)
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 2: EVALUATE
# ---------------------------------------------------------------------------


class VLMVerdict(BaseModel):
    """Per-candidate verdict from the VLM classification + quality call."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    correct_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_quality: BboxQuality
    object_complete: bool
    reasoning: str


class RefinementInstruction(BaseModel):
    """Per-prompt VLM directive produced inside the refine stage.

    The class-driven refine flow asks the VLM, for each prompt configured
    against the candidate's class, whether to ``skip`` (no extension needed)
    or ``propose`` a ``target_region`` to extend the bbox to. Generated
    inside ``RefineWorker`` — evaluate no longer emits these.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    prompt_id: str
    action: Literal["skip", "propose"] = "skip"
    target_region: BoundingBox | None = None
    """VLM-proposed bbox over the area to extend into. Optional point fallback
    via ``point_x`` / ``point_y`` when VLM returns a single pixel instead.
    """
    load_vocab: list[str] = Field(default_factory=list)
    """Short noun phrases for the load (e.g. ``["pallet", "wooden crate"]``).
    Used as the SAM3 presence-head query on the proposed region.
    """
    point_x: int | None = None
    point_y: int | None = None
    vlm_reasoning: str = ""


class RefinementNeeded(BaseModel):
    """Record indicating a candidate requires refinement and why."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    reason: str
    class_rule: str


class PromptRef(BaseModel):
    """Reference to a versioned prompt used during a stage."""

    model_config = ConfigDict(extra="forbid")

    group: str | None = None
    prompt_id: str
    version: str
    hash: str


class EvaluateResult(BaseModel):
    """Stage 2 output: VLM evaluation verdicts and routing to refinement."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    vlm_calls: int = 0
    vlm_total_tokens: int = 0
    prompts_used: list[PromptRef] = Field(default_factory=list)
    verdicts: list[VLMVerdict] = Field(default_factory=list)
    refinement_needed: list[RefinementNeeded] = Field(default_factory=list)
    refinement_instructions: dict[str, RefinementInstruction] = Field(
        default_factory=dict
    )
    accepted: list[str] = Field(default_factory=list)
    review: list[str] = Field(default_factory=list)
    """Candidates with moderate confidence — pending human review or further
    refinement. Replaces the older ``needs_refine`` semantics now that refine
    is class-driven rather than VLM-signal driven (§10.1 vocab).
    """
    rejected: list[str] = Field(default_factory=list)
    relabels: dict[str, str] = Field(default_factory=dict)
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 3: REFINE
# ---------------------------------------------------------------------------


class PromptStepResult(BaseModel):
    """Trace of one prompt's pass through the refine inner loop."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    action: Literal["skip", "propose"]
    outcome: Literal[
        "skipped",
        "sam_no_mask",
        "presence_failed",
        "merge_failed",
        "merged",
        "vlm_error",
        "sam_error",
    ]
    presence_score: float | None = None
    proposed_bbox: BoundingBox | None = None
    merged_bbox: BoundingBox | None = None
    notes: str = ""


class RefinementResult(BaseModel):
    """Per-candidate result after the refine stage's per-prompt loop."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    original_bbox: BoundingBox
    refined_bbox: BoundingBox | None = None
    iou_with_original: float = 0.0
    accepted: bool = False
    method: str = "class_driven"
    prompt_steps: list[PromptStepResult] = Field(default_factory=list)
    adjudicate_verdict: Literal["accept", "review", "reject"] = "accept"
    """Final-VLM adjudication of the refined-vs-original bbox per the class's
    annotation rule. Combined with evaluate's verdict per the §10.4 table to
    yield the candidate's final routing.
    """
    final_verdict: Literal["accept", "review", "reject"] = "accept"
    """Combined verdict (evaluate × adjudicate per §10.4). Drives whether the
    final bbox is the refined or the original, and whether it lands in the
    review queue.
    """
    final_bbox_source: Literal["refined", "original"] = "original"


class RefineResult(BaseModel):
    """Stage 3 output: SAM refinement results."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    refinement_instructions: list[RefinementInstruction] = Field(default_factory=list)
    results: list[RefinementResult] = Field(default_factory=list)
    vlm_calls: int = 0
    sam_calls: int = 0
    prompt_used: PromptRef | None = None
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 4: FINALIZE — post-refine canonical list + dedup/geometry recheck
# ---------------------------------------------------------------------------


class FinalizeDrop(BaseModel):
    """One candidate dropped during the finalize stage."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    class_name: str
    reason: str  # geometric_filter | dedup | cross_class | per_class_cap | rejected_upstream
    bbox: BoundingBox | None = None


class FinalizeResult(BaseModel):
    """Stage 4 output: post-refine canonical annotation list + drop log."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    final_annotations: list[FinalAnnotation] = Field(default_factory=list)
    review_items: list[dict[str, Any]] = Field(default_factory=list)
    dropped: list[FinalizeDrop] = Field(default_factory=list)
    filter_stats: dict[str, int] = Field(default_factory=dict)
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------


class MetaCheckpoint(BaseModel):
    """Per-image pipeline metadata stored in meta.json."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    config_hash: str
    prompt_version: str
    status: str  # pending | running | complete | failed
    stages_completed: list[str] = Field(default_factory=list)
    total_timing_ms: float = 0.0
    final_counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Redis Streams message
# ---------------------------------------------------------------------------


class StageMessage(BaseModel):
    """Message envelope passed between pipeline stages via Redis Streams."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    image_path: str
    job_id: str
    stage: str
    attempt: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def forward(self, next_stage: str) -> "StageMessage":
        """Return a copy of this message addressed to the next stage."""
        return StageMessage(
            image_id=self.image_id,
            image_path=self.image_path,
            job_id=self.job_id,
            stage=next_stage,
            attempt=0,
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# Final annotation
# ---------------------------------------------------------------------------


class FinalAnnotation(BaseModel):
    """Final status of one annotation after the complete pipeline."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    class_name: str
    class_id: int
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    action: FinalAction
    source_model: str
    was_refined: bool = False
    trace: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector wire + internal models (Pydantic-everywhere; tensor-carrying
# intermediates use arbitrary_types_allowed + a JSON-refusing serializer).
# ---------------------------------------------------------------------------


class DetectorRequest(BaseModel):
    """Uniform wire request accepted by every detector server."""

    model_config = ConfigDict(extra="forbid")

    image_path: str
    prompts: list[str]
    threshold: float | None = None


class DetectorResponse(BaseModel):
    """Uniform wire response from every detector server.

    ``labels`` is parallel to ``boxes`` / ``scores`` and contains the canonical
    class name the box was matched to (server is responsible for mapping its
    internal label back to one of ``DetectorRequest.prompts`` and returning
    the same string the caller sent). One entry per detection.
    """

    model_config = ConfigDict(extra="forbid")

    boxes: list[list[float]]   # normalized [x1, y1, x2, y2]
    scores: list[float]
    labels: list[str]


# --- SAM3-specific refine wire (sibling shape, used only by the refine stage) ---


class SAM3RefineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str
    bbox: list[float]          # normalized [x1, y1, x2, y2]
    points: list[list[float]] | None = None   # [[x, y, label], ...]
    threshold: float = 0.5


class SAM3RefineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box: list[float] | None    # normalized [x1, y1, x2, y2] or None
    score: float = 0.0


# --- Internal stage handoff models (hold tensors / PIL / BatchFeature) ---
#
# These are NOT JSON-serializable. The @field_serializer on `obj`-typed
# fields below is registered ``when_used="json"`` so that an accidental
# ``model_dump_json()`` fails loud and fast rather than silently producing
# unusable output. ``model_dump()`` (Python-mode) still works and is zero-cost.


def _forbid_json(_self, _value):
    raise TypeError(
        "This model carries non-JSON payloads (tensors / PIL / BatchFeature). "
        "It is for in-process use only; do not model_dump_json() it."
    )


class PreparedInput(BaseModel):
    """Output of DetectorServerBase.decode_request → input to predict."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    image: Any
    processor_inputs: Any
    image_size: tuple[int, int]
    prompts: list[str]
    threshold: float | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("image", "processor_inputs", "extras", when_used="json")
    def _ser_nonjson(self, v):
        return _forbid_json(self, v)


class RawPrediction(BaseModel):
    """Output of DetectorServerBase.predict → input to encode_response."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    outputs: Any
    inputs: Any
    image_size: tuple[int, int]
    prompts: list[str]
    threshold: float | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("outputs", "inputs", "extras", when_used="json")
    def _ser_nonjson(self, v):
        return _forbid_json(self, v)
