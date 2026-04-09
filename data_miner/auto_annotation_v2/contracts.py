"""All Pydantic models for auto_annotation_v2. No loose dicts or strings."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums — typed stage & status identifiers
# ---------------------------------------------------------------------------

class StageName(str, Enum):
    PROPOSAL = "proposal"
    FILTERING = "filtering"
    VLM_REASONING = "vlm_reasoning"
    VLM_REFINEMENT = "vlm_refinement"
    VLM_VALIDATION = "vlm_validation"
    FINALIZE = "finalize"


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    FILTERED_OUT = "filtered_out"
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    REFINED = "refined"
    ESCALATED = "escalated"


class VLMDecision(str, Enum):
    ACCEPT = "accept"
    NEEDS_REVIEW = "needs_review"
    REJECT = "reject"


class RefinementStrategy(str, Enum):
    SAM_POINTS = "sam_points"
    SAM_BOX = "sam_box"
    REPROPOSE_TEXT = "repropose_text"


class FinalAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Normalised bounding box (0.0–1.0)."""
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


class PointPrompt(BaseModel):
    """A pixel-coordinate point prompt for SAM."""
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    label: int = 1  # 1 = foreground, 0 = background


# ---------------------------------------------------------------------------
# Candidate — the core annotation unit throughout the pipeline
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    """A single detection/annotation candidate tracked through the pipeline."""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    class_name: str
    label: str
    source_model: str
    expression: str
    bbox: BoundingBox
    score: float = 1.0
    status: CandidateStatus = CandidateStatus.PROPOSED
    mask_rle: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# VLM Reasoning outputs — structured output types for PydanticAI agents
# ---------------------------------------------------------------------------

class ScreeningVerdict(BaseModel):
    """Per-candidate verdict from Pass 1 (batch screening)."""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    decision: VLMDecision
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class ScreeningResult(BaseModel):
    """Output of Pass 1 screening for all candidates of one class."""
    model_config = ConfigDict(extra="forbid")

    verdicts: list[ScreeningVerdict]
    summary: str


class DetailedVerdict(BaseModel):
    """Per-candidate verdict from Pass 2 (detailed review)."""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    semantic_match: Literal["yes", "no", "uncertain"]
    bbox_quality: Literal["tight", "loose", "too_small", "uncertain"]
    decision: VLMDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    relabel_to: str | None = None
    refinement_hint: str | None = None


# ---------------------------------------------------------------------------
# VLM Refinement outputs
# ---------------------------------------------------------------------------

class RefinementAction(BaseModel):
    """What the refinement agent proposes for one candidate."""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    strategy: RefinementStrategy
    target_model: str
    points: list[PointPrompt] | None = None
    text_prompt: str | None = None
    reasoning: str


class RefinementProposal(BaseModel):
    """Batch output from the refinement agent."""
    model_config = ConfigDict(extra="forbid")

    actions: list[RefinementAction]
    summary: str


# ---------------------------------------------------------------------------
# Final annotation
# ---------------------------------------------------------------------------

class FinalAnnotation(BaseModel):
    """Final status of one annotation after the full pipeline."""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    class_name: str
    bbox: BoundingBox
    action: FinalAction
    confidence: float = Field(ge=0.0, le=1.0)
    source_model: str
    reasoning_trace: list[str] = Field(default_factory=list)
    was_refined: bool = False
    original_bbox: BoundingBox | None = None


# ---------------------------------------------------------------------------
# Per-image trace (full audit trail)
# ---------------------------------------------------------------------------

class StageRecord(BaseModel):
    """Record of one pipeline stage completing for one image."""
    model_config = ConfigDict(extra="forbid")

    stage: StageName
    started_at: str
    completed_at: str
    candidate_count_in: int
    candidate_count_out: int
    notes: list[str] = Field(default_factory=list)


class FailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: StageName
    error_type: str
    message: str
    candidate_id: str | None = None
    retriable: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ImageTrace(BaseModel):
    """Complete trace of everything that happened for one image."""
    model_config = ConfigDict(extra="forbid")

    image_path: str
    stages: list[StageRecord] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)
    proposal_candidates: list[Candidate] = Field(default_factory=list)
    filtered_candidates: list[Candidate] = Field(default_factory=list)
    screening_results: list[ScreeningVerdict] = Field(default_factory=list)
    detailed_verdicts: list[DetailedVerdict] = Field(default_factory=list)
    refinement_proposals: list[RefinementAction] = Field(default_factory=list)
    refined_candidates: list[Candidate] = Field(default_factory=list)
    validation_verdicts: list[DetailedVerdict] = Field(default_factory=list)
    final_annotations: list[FinalAnnotation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """Immutable output for one image."""
    model_config = ConfigDict(extra="forbid")

    image_path: str
    accepted: list[FinalAnnotation]
    rejected: list[FinalAnnotation]
    human_review: list[FinalAnnotation]
    yolo_lines: list[str]
    trace: ImageTrace
    partial: bool = False
