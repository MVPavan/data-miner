"""Pipeline data contracts — all Pydantic models for stage results, candidates, and metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    BboxQuality,
    BboxSource,
    CandidateStatus,
    DropReason,
    FilterContext,
    FinalAction,
    ImageStatus,
    RefineAction,
    RefineOutcome,
    Stage,
    Verdict,
)

__all__ = [
    "BoundingBox",
    "Candidate",
    "DetectResult",
    "DetectRouting",
    "EvaluateResult",
    "FilterDrop",
    "FilterResult",
    "FinalAnnotation",
    "FinalizeDrop",
    "FinalizeResult",
    "MetaCheckpoint",
    "PromptRef",
    "PromptStepResult",
    "ProposalResult",
    "RefineResult",
    "RefinementInstruction",
    "RefinementNeeded",
    "RefinementResult",
    "StageMessage",
    "VLMVerdict",
]

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Normalised bounding box (0.0-1.0 coordinates)."""

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
# Pipeline message
# ---------------------------------------------------------------------------


class StageMessage(BaseModel):
    """Message envelope passed between pipeline stages via Redis Streams."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    image_path: str
    job_id: str
    stage: Stage


# ---------------------------------------------------------------------------
# Candidate -- core annotation unit tracked through all stages
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
# Stage 1b: FILTER (split out of DETECT)
# ---------------------------------------------------------------------------


class FilterDrop(BaseModel):
    """One candidate dropped by a filter, tagged with the invoking context."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    reason: DropReason
    context: FilterContext
    detail: str | None = None


class FilterResult(BaseModel):
    """Filter stage output: kept candidates, per-candidate drop log, and routing.

    Shape parallels :class:`DetectResult` — detect now saves raw merged
    proposals and the filter stage takes over the filter/route responsibility.
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str
    candidates: list[Candidate]
    drops: list[FilterDrop] = Field(default_factory=list)
    routing: DetectRouting = Field(default_factory=DetectRouting)
    filter_stats: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""
    created_at: float = 0.0
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 2: EVALUATE
# ---------------------------------------------------------------------------


class VLMVerdict(BaseModel):
    """Per-candidate verdict from the VLM classification and quality call.

    Two signals drive routing (class_match computed post-hoc from
    ``detected_class`` vs the original candidate's class):

      - ``class_confidence``: VLM's confidence in its ``detected_class``
        call. Anchored in the prompt: 0.9+ = certain, 0.7-0.9 = likely,
        0.5-0.7 = plausible, 0.3-0.5 = uncertain, <0.3 = wrong.
      - ``bbox_score``: continuous 0-1 score of how well the red TARGET
        bbox fits the object. 1.0 = perfect; 0.7 = loose/slightly tight;
        0.4 = major issue; <0.4 = bbox unusable.

    ``object_complete`` and ``bbox_quality`` (legacy) remain as telemetry
    only — not consumed by current routing.

    Backward compatibility: ``correct_class`` / ``confidence`` /
    ``bbox_quality`` kept as optional fields so old checkpoint rows still
    load. A pre-validator copies legacy field values into the v2 fields
    when the v2 fields are missing — so an old DB row with only
    ``correct_class`` / ``confidence`` becomes a valid v2 verdict on load.
    """

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    detected_class: str = ""
    class_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox_score: float = Field(default=1.0, ge=0.0, le=1.0)
    object_complete: bool = True
    reasoning: str = ""

    # ---- Legacy aliases (kept for old DB rows and viewer compatibility) ----
    correct_class: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox_quality: BboxQuality = BboxQuality.GOOD

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Accept old-format payloads: if the v2 fields are missing but the
        v1 fields are present, fill them in. Mirrors v2 fields back into
        the legacy aliases too so both schema views stay consistent.
        """
        if not isinstance(data, dict):
            return data
        # Normalize v1 → v2 when v2 is absent
        if data.get("detected_class") in (None, "") and data.get("correct_class"):
            data["detected_class"] = data["correct_class"]
        if data.get("class_confidence") is None and "confidence" in data:
            data["class_confidence"] = data["confidence"]
        # Mirror v2 → v1 so the viewer's legacy-field path keeps working
        if data.get("correct_class") in (None, "") and data.get("detected_class"):
            data["correct_class"] = data["detected_class"]
        if data.get("confidence") in (None, 0.0) and data.get("class_confidence") is not None:
            data["confidence"] = data["class_confidence"]
        # Clamp out-of-range floats (some VLMs emit 1.01 etc.) and coerce
        # None / malformed values to sensible defaults.
        defaults = {"class_confidence": 0.0, "confidence": 0.0, "bbox_score": 1.0}
        for k, default in defaults.items():
            if k in data:
                v = data[k]
                if v is None:
                    data[k] = default
                    continue
                try:
                    data[k] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    data[k] = default
        return data

    @classmethod
    def from_vlm_payload(
        cls, candidate_id: str, data: dict
    ) -> "VLMVerdict":
        """Build a VLMVerdict from parsed VLM JSON. The before-validator
        handles v1/v2 field migration + clamping.
        """
        payload = dict(data)
        payload["candidate_id"] = candidate_id
        return cls.model_validate(payload)


class RefinementInstruction(BaseModel):
    """Per-prompt VLM directive for the refine stage."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    prompt_id: str
    action: RefineAction = RefineAction.SKIP
    target_region: BoundingBox | None = None
    load_vocab: list[str] = Field(default_factory=list)
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
    rejected: list[str] = Field(default_factory=list)
    relabels: dict[str, str] = Field(default_factory=dict)
    drops: list[FilterDrop] = Field(default_factory=list)
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 3: REFINE
# ---------------------------------------------------------------------------


class PromptStepResult(BaseModel):
    """Trace of one prompt step through the refine inner loop."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    action: RefineAction
    outcome: RefineOutcome
    presence_score: float | None = None
    proposed_bbox: BoundingBox | None = None
    merged_bbox: BoundingBox | None = None
    notes: str = ""


class RefinementResult(BaseModel):
    """Per-candidate result after the refine stage per-prompt loop."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    original_bbox: BoundingBox
    refined_bbox: BoundingBox | None = None
    iou_with_original: float = 0.0
    accepted: bool = False
    method: str = "class_driven"
    prompt_steps: list[PromptStepResult] = Field(default_factory=list)
    adjudicate_verdict: Verdict = Verdict.ACCEPT
    final_verdict: Verdict = Verdict.ACCEPT
    final_bbox_source: BboxSource = BboxSource.ORIGINAL


class RefineResult(BaseModel):
    """Stage 3 output: SAM refinement results."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    refinement_instructions: list[RefinementInstruction] = Field(default_factory=list)
    results: list[RefinementResult] = Field(default_factory=list)
    vlm_calls: int = 0
    sam_calls: int = 0
    prompt_used: PromptRef | None = None
    drops: list[FilterDrop] = Field(default_factory=list)
    stage_timing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 4: FINALIZE
# ---------------------------------------------------------------------------


class FinalizeDrop(BaseModel):
    """One candidate dropped during the finalize stage."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    class_name: str
    reason: DropReason
    bbox: BoundingBox | None = None


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


class FinalizeResult(BaseModel):
    """Stage 4 output: post-refine canonical annotation list and drop log."""

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
    """Per-image pipeline metadata stored in checkpoint records."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    config_hash: str
    prompt_version: str
    status: ImageStatus
    stages_completed: list[Stage] = Field(default_factory=list)
    total_timing_ms: float = 0.0
    final_counts: dict[str, int] = Field(default_factory=dict)
