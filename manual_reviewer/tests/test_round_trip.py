"""End-to-end round-trip: pipeline.db → LS task → fake completion → pipeline.db."""

from __future__ import annotations

from pathlib import Path

import pytest

from manual_reviewer.pipeline_io import (
    build_task,
    iter_survivor_images,
    parse_ls_completion,
    read_image_payload,
    write_human_review,
)


def _make_completion(
    *,
    edited_id: str = "c1",
    new_class: str = "palletjack",
    extra_added: bool = True,
) -> dict:
    result = [
        {
            "id": edited_id,
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "value": {
                "x": 10.0,
                "y": 20.0,
                "width": 30.0,
                "height": 40.0,
                "rotation": 0,
                "rectanglelabels": [new_class],
            },
        },
        {
            "from_name": "frame_state",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": ["clean"]},
        },
        {
            "from_name": "notes",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": ["minor cleanup"]},
        },
    ]
    if extra_added:
        result.insert(
            1,
            {
                "id": "ls-new-1",
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "value": {
                    "x": 50.0,
                    "y": 50.0,
                    "width": 10.0,
                    "height": 10.0,
                    "rotation": 0,
                    "rectanglelabels": ["person"],
                },
            },
        )
    return {
        "id": 9001,
        "lead_time": 12.5,
        "completed_by": "alice@example.com",
        "created_at": "2026-04-27T10:00:00Z",
        "result": result,
    }


def test_iter_survivor_images_excludes_dropped(seeded_pipeline_db: Path) -> None:
    survivors = list(iter_survivor_images(seeded_pipeline_db))
    assert [s["image_id"] for s in survivors] == ["img_a"]


def test_build_task_seeds_finalize_predictions(seeded_pipeline_db: Path) -> None:
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload, job_id="testjob")

    assert task["meta"]["image_id"] == "img_a"
    assert task["data"]["job_id"] == "testjob"
    assert task["data"]["proposal_summary"]["sam3_dart"]["count"] == 1
    assert task["data"]["vlm_summary"][0]["detected_class"] == "forklift"
    assert task["data"]["ghost_drops"][0]["candidate_id"] == "c2"

    pred = task["predictions"][0]
    assert pred["model_version"] == "aa_v4_finalize"
    assert len(pred["result"]) == 1
    region = pred["result"][0]
    assert region["id"] == "c1"
    assert region["value"]["rectanglelabels"] == ["forklift"]
    # 0.10 * 100 = 10.0 etc. (use approx to absorb float noise from x2-x1)
    assert region["value"]["x"] == pytest.approx(10.0)
    assert region["value"]["width"] == pytest.approx(30.0)


def test_round_trip_relabel_and_add(seeded_pipeline_db: Path) -> None:
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload, job_id="testjob")
    seeded = task["predictions"][0]["result"]

    completion = _make_completion(new_class="palletjack")
    result = parse_ls_completion(
        completion,
        image_id="img_a",
        seeded_predictions=seeded,
    )

    sources = [(c.candidate_id, c.class_name, c.source) for c in result.corrections]
    assert ("c1", "palletjack", "relabeled") in sources
    assert ("ls-new-1", "person", "added") in sources

    # write + reread
    write_human_review(seeded_pipeline_db, result, config_hash="h1")
    payload2 = read_image_payload(seeded_pipeline_db, "img_a")
    assert "human_review" in payload2["stages"]
    assert payload2["stages"]["human_review"]["reviewer_id"] == "alice@example.com"
    assert "human_review" in payload2["meta"]["stages_completed"]


def test_round_trip_idempotent_on_reexport(seeded_pipeline_db: Path) -> None:
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload, job_id="testjob")
    seeded = task["predictions"][0]["result"]

    completion = _make_completion()
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    write_human_review(seeded_pipeline_db, result, config_hash="h1")
    write_human_review(seeded_pipeline_db, result, config_hash="h1")

    payload2 = read_image_payload(seeded_pipeline_db, "img_a")
    completed = payload2["meta"]["stages_completed"]
    assert completed.count("human_review") == 1


def test_unchanged_region_marked_finalize(seeded_pipeline_db: Path) -> None:
    """A region returned exactly as seeded → source='finalize'."""
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload, job_id="testjob")
    seeded = task["predictions"][0]["result"]

    # Build a completion that mirrors the seed value exactly
    completion = {
        "id": 1,
        "lead_time": 1.0,
        "completed_by": "bob",
        "created_at": "2026-04-27T11:00:00Z",
        "result": [
            {
                "id": "c1",
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "value": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 30.0,
                    "height": 40.0,
                    "rotation": 0,
                    "rectanglelabels": ["forklift"],
                },
            },
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    assert result.corrections[0].source == "finalize"
    assert result.corrections[0].original_class is None
    assert result.corrections[0].original_bbox is None


def test_deletion_recorded(seeded_pipeline_db: Path) -> None:
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload, job_id="testjob")
    seeded = task["predictions"][0]["result"]

    # Reviewer submitted no rectangle → c1 is implicitly deleted
    completion = {
        "id": 2,
        "lead_time": 1.0,
        "completed_by": "bob",
        "created_at": "2026-04-27T11:00:00Z",
        "result": [
            {
                "from_name": "frame_state",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": ["needs_more_review"]},
            }
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    assert result.corrections == []
    assert result.deletions == ["c1"]
    assert result.frame_state == "needs_more_review"
