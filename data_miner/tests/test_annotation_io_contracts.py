"""Tests for frontend-neutral annotation exchange contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_miner.annotation_io import (
    FrontendName,
    ReviewBox,
    ReviewRegionOrigin,
    ReviewBoxSource,
    ReviewExchangeResult,
    ReviewExchangeTask,
)
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    HumanCorrection,
    HumanReviewResult,
)


def test_review_exchange_result_converts_to_human_review_result() -> None:
    """Exchange results should preserve fields needed by v4 human review."""
    original_bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.3, y2=0.4)
    edited_bbox = BoundingBox(x1=0.11, y1=0.2, x2=0.35, y2=0.42)
    result = ReviewExchangeResult(
        image_id="img_1",
        source_frontend=FrontendName.LABEL_STUDIO,
        reviewer_id="reviewer@example.com",
        reviewed_at=1714298400.0,
        duration_seconds=12.5,
        frame_state="needs_more_review",
        boxes=[
            ReviewBox(
                region_id="candidate_1",
                class_name="forklift",
                bbox=edited_bbox,
                source=ReviewBoxSource.EDITED,
                origin=ReviewRegionOrigin.MIGRATED_LABEL_STUDIO,
                original_bbox=original_bbox,
                track_id="track_7",
            )
        ],
        deletions=["candidate_2"],
        notes="needs second pass",
        ml_modes_used=["smart_click"],
        source_completion_id="42",
    )

    human_review = result.to_human_review_result()

    assert human_review == HumanReviewResult(
        image_id="img_1",
        reviewer_id="reviewer@example.com",
        reviewed_at=1714298400.0,
        duration_seconds=12.5,
        frame_state="needs_more_review",
        corrections=[
            HumanCorrection(
                candidate_id="candidate_1",
                class_name="forklift",
                bbox=edited_bbox,
                track_id="track_7",
                source="edited",
                original_bbox=original_bbox,
            )
        ],
        deletions=["candidate_2"],
        notes="needs second pass",
        ml_modes_used=["smart_click"],
        ls_completion_id=42,
    )


def test_human_review_result_converts_to_exchange_result() -> None:
    """Existing v4 human-review results should map into the neutral model."""
    bbox = BoundingBox(x1=0.2, y1=0.3, x2=0.5, y2=0.7)
    human_review = HumanReviewResult(
        image_id="img_2",
        reviewer_id="alice@example.com",
        reviewed_at=1714298500.0,
        corrections=[
            HumanCorrection(
                candidate_id="new_box",
                class_name="palletjack",
                bbox=bbox,
                source="added",
            )
        ],
        notes="added missing object",
        ls_completion_id=99,
    )

    exchange = ReviewExchangeResult.from_human_review_result(
        human_review,
        source_frontend=FrontendName.CVAT,
        source_completion_id="job-12-frame-3",
        source_task_id="task-44",
        source_job_id="job-12",
        attributes={"job_id": 12},
    )

    assert exchange.source_frontend is FrontendName.CVAT
    assert exchange.source_task_id == "task-44"
    assert exchange.source_job_id == "job-12"
    assert exchange.source_completion_id == "job-12-frame-3"
    assert exchange.attributes == {"job_id": 12}
    assert exchange.boxes == [
        ReviewBox(
            region_id="new_box",
            class_name="palletjack",
            bbox=bbox,
            source=ReviewBoxSource.ADDED,
        )
    ]


def test_exchange_task_rejects_invalid_image_dimensions() -> None:
    """Exchange tasks should reject non-positive image dimensions."""
    with pytest.raises(ValidationError):
        ReviewExchangeTask(
            image_id="img_3",
            image_width=0,
            image_height=720,
        )


def test_exchange_task_carries_cross_frontend_identity() -> None:
    """Exchange tasks should carry identifiers needed by both frontends."""
    task = ReviewExchangeTask(
        image_id="clip_1_frame_000003",
        source_frontend=FrontendName.LABEL_STUDIO,
        source_task_id="ls-task-8",
        source_job_id="ls-project-2",
        media_uri="file:///dataset/frame_000003.jpg",
        clip_id="clip_1",
        frame_index=3,
        class_list_version="forklift-v1",
        image_width=1280,
        image_height=720,
        labels=["forklift", "palletjack"],
    )

    assert task.clip_id == "clip_1"
    assert task.frame_index == 3
    assert task.class_list_version == "forklift-v1"