"""Frontend-neutral contracts for exchanging review annotations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    HumanCorrection,
    HumanReviewResult,
)

FrameState = Literal["clean", "needs_more_review", "ambiguous_skip"]


class FrontendName(StrEnum):
    """Supported human-review frontends."""

    LABEL_STUDIO = "label_studio"
    CVAT = "cvat"


class ReviewBoxSource(StrEnum):
    """How a reviewed box relates to the upstream pipeline output."""

    FINALIZE = "finalize"
    ADDED = "added"
    EDITED = "edited"
    RELABELED = "relabeled"
    KEPT_DROPPED = "kept_dropped"


class ReviewRegionOrigin(StrEnum):
    """Where a frontend exchange region originated before final review."""

    PREDICTION = "prediction"
    HUMAN = "human"
    MIGRATED_LABEL_STUDIO = "migrated_label_studio"
    MIGRATED_CVAT = "migrated_cvat"
    SMART_TOOL = "smart_tool"


class ReviewBox(BaseModel):
    """One box annotation in the frontend-neutral review exchange model."""

    model_config = ConfigDict(extra="forbid")

    region_id: str | None = None
    class_name: str
    bbox: BoundingBox
    source: ReviewBoxSource
    origin: ReviewRegionOrigin = ReviewRegionOrigin.HUMAN
    original_class: str | None = None
    original_bbox: BoundingBox | None = None
    track_id: str | None = None
    mask_rle: dict[str, Any] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def to_human_correction(self) -> HumanCorrection:
        """Convert this exchange box into the v4 human-review correction contract."""
        return HumanCorrection(
            candidate_id=self.region_id,
            class_name=self.class_name,
            bbox=self.bbox,
            mask_rle=self.mask_rle,
            track_id=self.track_id,
            source=self.source.value,
            original_class=self.original_class,
            original_bbox=self.original_bbox,
        )

    @classmethod
    def from_human_correction(cls, correction: HumanCorrection) -> ReviewBox:
        """Build an exchange box from a v4 human-review correction."""
        return cls(
            region_id=correction.candidate_id,
            class_name=correction.class_name,
            bbox=correction.bbox,
            source=ReviewBoxSource(correction.source),
            original_class=correction.original_class,
            original_bbox=correction.original_bbox,
            track_id=correction.track_id,
            mask_rle=correction.mask_rle,
        )


class ReviewExchangeTask(BaseModel):
    """A review task that can be materialized in Label Studio or CVAT."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    source_frontend: FrontendName | None = None
    source_task_id: str | None = None
    source_job_id: str | None = None
    media_uri: str | None = None
    clip_id: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    class_list_version: str | None = None
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    labels: list[str] = Field(default_factory=list)
    boxes: list[ReviewBox] = Field(default_factory=list)
    reviewer_id: str | None = None
    status: str = "new"
    notes: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class ReviewExchangeResult(BaseModel):
    """A completed review result from either Label Studio or CVAT."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    source_frontend: FrontendName
    source_task_id: str | None = None
    source_job_id: str | None = None
    reviewer_id: str
    reviewed_at: float
    duration_seconds: float = 0.0
    frame_state: FrameState = "clean"
    clip_id: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    class_list_version: str | None = None
    boxes: list[ReviewBox] = Field(default_factory=list)
    deletions: list[str] = Field(default_factory=list)
    notes: str = ""
    ml_modes_used: list[str] = Field(default_factory=list)
    source_completion_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def to_human_review_result(self) -> HumanReviewResult:
        """Convert this exchange result into the v4 human-review stage contract."""
        ls_completion_id = 0
        if self.source_frontend is FrontendName.LABEL_STUDIO:
            ls_completion_id = _safe_int(self.source_completion_id)

        return HumanReviewResult(
            image_id=self.image_id,
            reviewer_id=self.reviewer_id,
            reviewed_at=self.reviewed_at,
            duration_seconds=self.duration_seconds,
            frame_state=self.frame_state,
            corrections=[box.to_human_correction() for box in self.boxes],
            deletions=self.deletions,
            notes=self.notes,
            ml_modes_used=self.ml_modes_used,
            ls_completion_id=ls_completion_id,
        )

    @classmethod
    def from_human_review_result(
        cls,
        result: HumanReviewResult,
        *,
        source_frontend: FrontendName,
        source_completion_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        source_task_id: str | None = None,
        source_job_id: str | None = None,
    ) -> ReviewExchangeResult:
        """Build an exchange result from an existing v4 human-review result."""
        completion_id = source_completion_id
        if completion_id is None and source_frontend is FrontendName.LABEL_STUDIO:
            completion_id = str(result.ls_completion_id) if result.ls_completion_id else None

        return cls(
            image_id=result.image_id,
            source_frontend=source_frontend,
            source_task_id=source_task_id,
            source_job_id=source_job_id,
            reviewer_id=result.reviewer_id,
            reviewed_at=result.reviewed_at,
            duration_seconds=result.duration_seconds,
            frame_state=result.frame_state,
            boxes=[
                ReviewBox.from_human_correction(correction)
                for correction in result.corrections
            ],
            deletions=result.deletions,
            notes=result.notes,
            ml_modes_used=result.ml_modes_used,
            source_completion_id=completion_id,
            attributes=attributes or {},
        )


def _safe_int(value: str | None) -> int:
    """Convert an optional frontend completion id into v4's integer LS id field."""
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0