"""Tests for CVAT Datumaro export parsing."""

from __future__ import annotations

import pytest

from data_miner.annotation_io import (
    FrontendName,
    ReviewBoxSource,
    ReviewRegionOrigin,
)
from data_miner.auto_annotation_v4.configs.contracts import BoundingBox
from manual_reviewer_cvat.pipeline_io import parse_datumaro_review_results


def test_parse_datumaro_review_results_maps_bbox_to_exchange_result() -> None:
    """Datumaro bbox annotations should normalize into exchange boxes."""
    document = {
        "version": "1.0",
        "categories": {
            "label": [
                {"id": 0, "name": "forklift"},
                {"id": 1, "name": "palletjack"},
            ]
        },
        "items": [
            {
                "id": "frame_0001",
                "image": {"path": "images/frame_0001.jpg", "size": [640, 480]},
                "attributes": {
                    "image_id": "clip_a_frame_0001",
                    "task_id": 12,
                    "job_id": 34,
                    "clip_id": "clip_a",
                    "frame_index": 1,
                    "frame_state": "needs_more_review",
                    "class_list_version": "warehouse-v1",
                    "deletions": ["old_candidate"],
                    "notes": "check occlusion",
                    "ml_modes_used": ["sam_click"],
                },
                "annotations": [
                    {
                        "id": 99,
                        "type": "bbox",
                        "x": 64,
                        "y": 48,
                        "w": 128,
                        "h": 96,
                        "label": 0,
                        "attributes": {
                            "candidate_id": "cand_1",
                            "source": "edited",
                            "origin": "prediction",
                            "track_id": "trk_1",
                            "original_class": "palletjack",
                            "original_bbox": [0.1, 0.1, 0.25, 0.25],
                        },
                    }
                ],
            }
        ],
    }

    results = parse_datumaro_review_results(
        document,
        reviewer_id="reviewer@example.com",
        reviewed_at=1714298400.0,
    )

    assert len(results) == 1
    result = results[0]
    assert result.source_frontend is FrontendName.CVAT
    assert result.image_id == "clip_a_frame_0001"
    assert result.source_task_id == "12"
    assert result.source_job_id == "34"
    assert result.media_uri == "images/frame_0001.jpg"
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.clip_id == "clip_a"
    assert result.frame_index == 1
    assert result.frame_state == "needs_more_review"
    assert result.class_list_version == "warehouse-v1"
    assert result.deletions == ["old_candidate"]
    assert result.notes == "check occlusion"
    assert result.ml_modes_used == ["sam_click"]

    box = result.boxes[0]
    assert box.region_id == "cand_1"
    assert box.class_name == "forklift"
    assert box.bbox == BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    assert box.source is ReviewBoxSource.EDITED
    assert box.origin is ReviewRegionOrigin.PREDICTION
    assert box.original_class == "palletjack"
    assert box.original_bbox == BoundingBox(x1=0.1, y1=0.1, x2=0.25, y2=0.25)
    assert box.track_id == "trk_1"

    human_review = result.to_human_review_result()
    assert human_review.corrections[0].source == "edited"
    assert human_review.deletions == ["old_candidate"]


def test_parse_datumaro_review_results_rejects_unknown_label_id() -> None:
    """Unknown Datumaro label ids should fail instead of corrupting classes."""
    document = {
        "categories": {"label": [{"id": 0, "name": "forklift"}]},
        "items": [
            {
                "id": "frame_0001",
                "image": {"path": "frame_0001.jpg", "size": [640, 480]},
                "annotations": [
                    {
                        "id": 1,
                        "type": "bbox",
                        "x": 0,
                        "y": 0,
                        "w": 10,
                        "h": 10,
                        "label": 7,
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown Datumaro label id"):
        parse_datumaro_review_results(
            document,
            reviewer_id="reviewer@example.com",
            reviewed_at=1714298400.0,
        )