from __future__ import annotations

from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


Decision = Literal["accepted", "flagged", "rejected"]
Action = Literal["accept", "relabel", "refine", "reject", "escalate"]


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    class_name: str
    label: str
    source_model: str
    expression: str
    bbox: BoundingBox
    score: float = 1.0
    quality_score: float = 0.0
    uncertainty_score: float = 1.0
    status: str = "proposed"
    mask: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CandidateCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    class_name: str
    candidate_ids: list[str]
    source_models: list[str]
    fused_bbox: BoundingBox
    agreement_count: int
    quality_score: float
    uncertainty_score: float
    decision: Decision


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    semantic_match: Literal["yes", "no", "uncertain"]
    bbox_tight: Literal["tight", "loose", "too_small", "uncertain"]
    recommended_action: Action
    confidence_band: Literal["high", "medium", "low"]
    rationale_short: str
    relabel_to: str | None = None
    target_model: str | None = None
    retry_expression: str | None = None
    next_stage: Literal["proposal", "refinement", "escalation", "none"] = "none"


class PipelineState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    image_path: str
    image: Image.Image
    proposals: list[Candidate] = Field(default_factory=list)
    accepted: list[Candidate] = Field(default_factory=list)
    flagged: list[Candidate] = Field(default_factory=list)
    rejected: list[Candidate] = Field(default_factory=list)
    human_review: list[Candidate] = Field(default_factory=list)
    clusters: list[CandidateCluster] = Field(default_factory=list)
    reviews: dict[str, ReviewDecision] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)
    retry_round: int = 0


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str
    accepted: list[Candidate]
    rejected: list[Candidate]
    human_review: list[Candidate]
    reviews: dict[str, ReviewDecision]
    clusters: list[CandidateCluster]
    history: list[str]