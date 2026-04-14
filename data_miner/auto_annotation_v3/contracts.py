"""All Pydantic data models for auto_annotation_v3. No loose dicts or strings."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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


class RefinementStrategy(str, Enum):
    LOAD_EXTENSION = "load_extension"
    SAM_POINT = "sam_point"
    SAM_BOX = "sam_box"


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
    """Spatial refinement instruction produced by the VLM for one candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    strategy: RefinementStrategy
    direction: str | None = None  # left | right | up | down
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
    rejected: list[str] = Field(default_factory=list)
    relabels: dict[str, str] = Field(default_factory=dict)
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 3: REFINE
# ---------------------------------------------------------------------------


class RefinementResult(BaseModel):
    """Per-candidate result after SAM refinement."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    original_bbox: BoundingBox
    refined_bbox: BoundingBox | None = None
    iou_with_original: float = 0.0
    accepted: bool = False
    method: str  # sam_point | sam_box | load_extension


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
