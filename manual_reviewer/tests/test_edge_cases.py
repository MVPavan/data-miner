"""Edge-case tests for the manual_reviewer Label Studio adapter.

Covers ``task_builder.build_task`` and ``ls_export_parser.parse_ls_completion``
against malformed / missing / boundary inputs that the happy-path round-trip
suite (``test_round_trip.py``) does not exercise.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pytest

from manual_reviewer.pipeline_io import (
    build_task,
    parse_ls_completion,
    read_image_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_payload(**overrides: Any) -> dict[str, Any]:
    """Build a JSON-only image_payload mimicking ``read_image_payload`` output.

    Lets each test override only the keys it cares about.
    """
    payload: dict[str, Any] = {
        "image_id": "img_x",
        "meta": {
            "image_path": "/tmp/imgs/img_x.jpg",
            "dedup_cluster_id": "cluster-7",
        },
        "stages": {
            "detect": {"image_size": [800, 600]},
            "filter": {"drops": []},
            "evaluate": {"verdicts": []},
            "finalize": {
                "final_annotations": [],
                "review_items": [],
                "dropped": [],
            },
        },
        "proposals": {},
        "trace_excerpt": [],
    }
    for k, v in overrides.items():
        payload[k] = v
    return payload


def _ls_rect(
    region_id: str,
    *,
    x: float = 10.0,
    y: float = 20.0,
    width: float = 30.0,
    height: float = 40.0,
    label: str = "forklift",
) -> dict[str, Any]:
    return {
        "id": region_id,
        "type": "rectanglelabels",
        "from_name": "bbox",
        "to_name": "image",
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def _seed_pred(
    region_id: str,
    *,
    x: float = 10.0,
    y: float = 20.0,
    width: float = 30.0,
    height: float = 40.0,
    label: str = "forklift",
) -> dict[str, Any]:
    """A single seeded prediction region (the build_task output shape)."""
    return _ls_rect(region_id, x=x, y=y, width=width, height=height, label=label)


# ---------------------------------------------------------------------------
# build_task — edge cases
# ---------------------------------------------------------------------------


def test_build_task_empty_finalize_stage_no_final_annotations_key() -> None:
    """`finalize` stage missing `final_annotations` → empty predictions, no crash."""
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {}  # no final_annotations key at all
    task = build_task(payload)
    assert task["predictions"] == []
    assert task["data"]["pre_annotations_finalize"] == []


def test_build_task_image_size_missing_from_all_stages() -> None:
    """No `image_size` in detect or filter → predictions still build, w/h are None."""
    payload = _minimal_payload()
    payload["stages"]["detect"] = {}
    payload["stages"]["filter"] = {"drops": []}
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c1",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            }
        ],
    }
    task = build_task(payload)
    assert task["data"]["image_size"] is None
    region = task["predictions"][0]["result"][0]
    assert region["original_width"] is None
    assert region["original_height"] is None


def test_build_task_drops_malformed_bbox_values() -> None:
    """Annotation whose bbox holds a string-not-coercible-to-float is silently dropped."""
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c-bad",
                "class_name": "forklift",
                "bbox": {"x1": "not-a-number", "y1": 0.2, "x2": 0.4, "y2": 0.6},
            },
            {
                "candidate_id": "c-good",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            },
        ],
    }
    task = build_task(payload)
    region_ids = [r["id"] for r in task["predictions"][0]["result"]]
    assert region_ids == ["c-good"]


def test_build_task_drops_bbox_with_missing_keys_gracefully() -> None:
    """Bbox missing some keys: missing keys default to 0.0, doesn't crash."""
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c1",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2},  # missing x2/y2
            }
        ],
    }
    task = build_task(payload)
    # x2/y2 default to 0.0; produces a (negatively-sized) rect but does not crash.
    region = task["predictions"][0]["result"][0]
    assert region["id"] == "c1"
    assert region["value"]["x"] == pytest.approx(10.0)


def test_build_task_drops_annotation_with_empty_class_name() -> None:
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c1",
                "class_name": "",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            }
        ],
    }
    task = build_task(payload)
    assert task["predictions"] == []


def test_build_task_drops_annotation_with_empty_candidate_id() -> None:
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            }
        ],
    }
    task = build_task(payload)
    assert task["predictions"] == []


def test_build_task_review_items_mark_predictions_needs_review() -> None:
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c1",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            },
            {
                "candidate_id": "c2",
                "class_name": "person",
                "bbox": {"x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.7},
            },
        ],
        "review_items": [{"candidate_id": "c1", "reason": "low_conf"}],
    }
    task = build_task(payload)
    by_id = {r["id"]: r for r in task["predictions"][0]["result"]}
    assert by_id["c1"]["meta"]["needs_review"] is True
    assert by_id["c2"]["meta"]["needs_review"] is False


def test_build_task_omits_ghost_drops_when_disabled() -> None:
    payload = _minimal_payload()
    payload["stages"]["filter"] = {
        "drops": [{"candidate_id": "c-drop", "reason": "score_floor"}],
    }
    payload["stages"]["finalize"] = {
        "final_annotations": [],
        "dropped": [{"candidate_id": "c-fdrop", "class_name": "forklift"}],
    }
    task = build_task(payload, include_ghost_drops=False)
    assert "ghost_drops" not in task["data"]


def test_build_task_includes_ghost_drops_by_default() -> None:
    payload = _minimal_payload()
    payload["stages"]["filter"] = {
        "drops": [{"candidate_id": "c-drop", "reason": "score_floor"}],
    }
    payload["stages"]["finalize"] = {
        "final_annotations": [],
        "dropped": [
            {
                "candidate_id": "c-fdrop",
                "class_name": "forklift",
                "reason": "below_floor",
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            },
        ],
    }
    task = build_task(payload)
    assert "ghost_drops" in task["data"]
    stages = {g["stage"] for g in task["data"]["ghost_drops"]}
    assert stages == {"filter", "finalize"}


def test_build_task_vlm_summary_handles_none_verdicts() -> None:
    payload = _minimal_payload()
    payload["stages"]["evaluate"] = {"verdicts": None}
    task = build_task(payload)
    assert task["data"]["vlm_summary"] == []


def test_build_task_vlm_summary_skips_non_dict_entries() -> None:
    payload = _minimal_payload()
    payload["stages"]["evaluate"] = {
        "verdicts": [
            {
                "candidate_id": "c1",
                "detected_class": "forklift",
                "class_confidence": 0.9,
                "bbox_score": 0.8,
                "reasoning": "ok",
            },
            "not-a-dict",
            None,
            42,
            {
                "candidate_id": "c2",
                "detected_class": "person",
                "class_confidence": 0.5,
                "bbox_score": 0.6,
                "reasoning": "maybe",
            },
        ],
    }
    task = build_task(payload)
    classes = [v["detected_class"] for v in task["data"]["vlm_summary"]]
    assert classes == ["forklift", "person"]


def test_build_task_proposal_summary_multiple_models_class_counts() -> None:
    payload = _minimal_payload()
    payload["proposals"] = {
        "sam3_dart": {
            "candidates": [
                {"class_name": "forklift"},
                {"class_name": "forklift"},
                {"class_name": "person"},
            ],
            "latency_ms": 120.0,
        },
        "yolo": {
            "candidates": [
                {"class_name": "forklift"},
                {"class_name": "palletjack"},
                "garbage-non-dict",
            ],
            "latency_ms": 50.0,
        },
    }
    task = build_task(payload)
    summary = task["data"]["proposal_summary"]
    assert summary["sam3_dart"]["count"] == 3
    assert summary["sam3_dart"]["classes"] == {"forklift": 2, "person": 1}
    # non-dict candidate skipped, but the count reflects total candidates list length
    assert summary["yolo"]["count"] == 3
    assert summary["yolo"]["classes"] == {"forklift": 1, "palletjack": 1}
    assert summary["yolo"]["latency_ms"] == 50.0


def test_build_task_custom_image_url_template_substitutes_path() -> None:
    payload = _minimal_payload()
    task = build_task(
        payload,
        image_url_template="https://cdn.example.com/img?p={path}&v=1",
    )
    assert task["data"]["image"] == (
        "https://cdn.example.com/img?p=/tmp/imgs/img_x.jpg&v=1"
    )


def test_build_task_empty_image_path_returns_none() -> None:
    payload = _minimal_payload()
    payload["meta"]["image_path"] = ""
    assert build_task(payload) is None


def test_build_task_review_items_with_non_dict_entries_does_not_crash() -> None:
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c1",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            }
        ],
        "review_items": ["not-a-dict", None, 7, {"candidate_id": "c1"}],
    }
    task = build_task(payload)
    region = task["predictions"][0]["result"][0]
    assert region["meta"]["needs_review"] is True


def test_build_task_seeded_db_smoke(seeded_pipeline_db: Path) -> None:
    """Reuse the DB fixture to confirm new helpers work alongside the real reader."""
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(
        payload,
        job_id="testjob",
        image_url_template="custom://{path}",
        include_ghost_drops=False,
    )
    assert task["data"]["image"] == "custom:///tmp/imgs/img_a.jpg"
    assert "ghost_drops" not in task["data"]


# ---------------------------------------------------------------------------
# parse_ls_completion — edge cases
# ---------------------------------------------------------------------------


def test_parse_skips_region_with_no_value_key() -> None:
    completion = {
        "id": 1,
        "result": [
            {
                "id": "c1",
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                # no "value"
            }
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.corrections == []


def test_parse_skips_region_with_empty_rectanglelabels() -> None:
    completion = {
        "id": 1,
        "result": [
            {
                "id": "c1",
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "value": {
                    "x": 10.0,
                    "y": 10.0,
                    "width": 20.0,
                    "height": 20.0,
                    "rotation": 0,
                    "rectanglelabels": [],
                },
            }
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.corrections == []


def test_parse_clamps_out_of_bounds_bbox_to_unit_interval() -> None:
    completion = {
        "id": 1,
        "result": [
            _ls_rect("ls-1", x=110.0, y=-5.0, width=200.0, height=50.0, label="x"),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    bb = result.corrections[0].bbox
    # x=1.10 → clamp to 1.0; x+w=3.10 → clamp to 1.0 → both x1 and x2 are 1.0
    assert bb.x1 == pytest.approx(1.0)
    assert bb.x2 == pytest.approx(1.0)
    # y=-0.05 → clamp to 0.0; y+h=0.45 stays in range
    assert bb.y1 == pytest.approx(0.0)
    assert bb.y2 == pytest.approx(0.45)


def test_parse_accepts_zero_width_after_clamping() -> None:
    """LS box at the right edge with overflow → x1==x2==1.0 — still a valid BoundingBox."""
    completion = {
        "id": 1,
        "result": [_ls_rect("ls-1", x=100.0, y=10.0, width=10.0, height=20.0)],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    bb = result.corrections[0].bbox
    assert bb.x1 == pytest.approx(1.0)
    assert bb.x2 == pytest.approx(1.0)
    assert bb.width == pytest.approx(0.0)


def test_parse_same_class_moved_box_is_edited_not_relabeled() -> None:
    seeded = [_seed_pred("c1", x=10.0, y=20.0, width=30.0, height=40.0, label="forklift")]
    completion = {
        "id": 1,
        "result": [
            _ls_rect("c1", x=15.0, y=25.0, width=30.0, height=40.0, label="forklift"),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    c = result.corrections[0]
    assert c.source == "edited"
    assert c.original_class is None
    assert c.original_bbox is not None
    assert c.original_bbox.x1 == pytest.approx(0.10)


def test_parse_class_and_bbox_change_is_relabeled_with_both_originals() -> None:
    seeded = [_seed_pred("c1", x=10.0, y=20.0, width=30.0, height=40.0, label="forklift")]
    completion = {
        "id": 1,
        "result": [
            _ls_rect("c1", x=15.0, y=25.0, width=30.0, height=40.0, label="palletjack"),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    c = result.corrections[0]
    assert c.source == "relabeled"
    assert c.original_class == "forklift"
    assert c.original_bbox is not None
    assert c.original_bbox.x1 == pytest.approx(0.10)


def test_parse_ghost_drop_id_passed_but_not_promoted_yields_no_kept_dropped() -> None:
    seeded = [_seed_pred("c1", label="forklift")]
    completion = {
        "id": 1,
        "result": [_ls_rect("c1", label="forklift")],  # only the seeded id
    }
    result = parse_ls_completion(
        completion,
        image_id="img_a",
        seeded_predictions=seeded,
        ghost_drop_ids=["c-drop-1", "c-drop-2"],
    )
    sources = [c.source for c in result.corrections]
    assert "kept_dropped" not in sources


def test_parse_ghost_drop_promoted_yields_kept_dropped() -> None:
    seeded = [_seed_pred("c1", label="forklift")]
    completion = {
        "id": 1,
        "result": [
            _ls_rect("c1", label="forklift"),
            _ls_rect("c-drop-1", x=50.0, y=50.0, width=10.0, height=10.0, label="person"),
        ],
    }
    result = parse_ls_completion(
        completion,
        image_id="img_a",
        seeded_predictions=seeded,
        ghost_drop_ids=["c-drop-1"],
    )
    by_id = {c.candidate_id: c for c in result.corrections}
    assert by_id["c-drop-1"].source == "kept_dropped"
    assert by_id["c1"].source == "finalize"


def test_parse_picks_up_per_region_track_id_textarea() -> None:
    """The XML defines <TextArea name="track_id" perRegion="true">. LS emits
    one entry per rectangle the reviewer typed into, with parentID linking
    to the rectangle id. The parser walks results once collecting the
    parentID->text map and attaches it to each HumanCorrection.
    """
    completion = {
        "id": 1,
        "result": [
            _ls_rect("rect_1", label="forklift"),
            {
                "type": "textarea",
                "from_name": "track_id",
                "to_name": "image",
                "parentID": "rect_1",
                "value": {"text": ["TRK-007"]},
            },
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert len(result.corrections) == 1
    assert result.corrections[0].track_id == "TRK-007"


def test_parse_global_notes_textarea_does_not_become_track_id() -> None:
    """The global notes textarea has no parentID; per-region track_id has
    parentID. They live in the same result list — the parser must not mix
    them up."""
    completion = {
        "id": 1,
        "result": [
            _ls_rect("rect_1", label="forklift"),
            {
                "type": "textarea",
                "from_name": "notes",
                "to_name": "image",
                "value": {"text": ["seed frame for cluster A"]},
            },
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.notes == "seed frame for cluster A"
    assert result.corrections[0].track_id is None


def test_parse_with_no_seeded_predictions_marks_everything_added() -> None:
    completion = {
        "id": 1,
        "result": [_ls_rect("c1"), _ls_rect("c2", label="person")],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=None)
    assert {c.source for c in result.corrections} == {"added"}


def test_parse_duplicate_region_ids_first_classified_against_seed() -> None:
    """When LS emits two regions with the same id (defensive — LS shouldn't):
    both regions are classified independently against the seed since the parser
    does not de-dupe. Verifies the actual behavior so future changes don't
    silently regress.
    """
    seeded = [_seed_pred("c1", x=10.0, y=20.0, width=30.0, height=40.0, label="forklift")]
    completion = {
        "id": 1,
        "result": [
            _ls_rect("c1", x=10.0, y=20.0, width=30.0, height=40.0, label="forklift"),
            _ls_rect("c1", x=50.0, y=50.0, width=10.0, height=10.0, label="person"),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    sources = [c.source for c in result.corrections]
    # Both regions match against the same seed; the unchanged one is finalize,
    # the moved + relabeled one is relabeled. Neither becomes "added" because
    # both ids are in the seeded map.
    assert sources == ["finalize", "relabeled"]
    # And no deletion is emitted because the id was seen.
    assert result.deletions == []


def test_parse_completion_with_no_result_key() -> None:
    completion = {"id": 7, "completed_by": "alice", "lead_time": 1.0}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.corrections == []
    assert result.frame_state == "clean"
    assert result.notes == ""


def test_parse_completion_with_result_none() -> None:
    completion = {"id": 7, "result": None}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.corrections == []
    assert result.deletions == []
    assert result.frame_state == "clean"


def test_parse_cancelled_completion_does_not_crash() -> None:
    """LS-cancelled tasks may carry was_cancelled=True. The export script filters
    them, but the parser itself must not crash if handed one.
    """
    completion = {
        "id": 11,
        "was_cancelled": True,
        "completed_by": "alice@example.com",
        "result": [],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.corrections == []
    assert result.frame_state == "clean"
    assert result.reviewer_id == "alice@example.com"


def test_parse_completed_by_int_yields_stringified_reviewer_id() -> None:
    completion = {"id": 1, "completed_by": 42, "result": []}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.reviewer_id == "42"


def test_parse_completed_by_dict_email_preferred() -> None:
    completion = {
        "id": 1,
        "completed_by": {"id": 5, "email": "alice@example.com"},
        "result": [],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.reviewer_id == "alice@example.com"


def test_parse_completed_by_dict_falls_back_to_id() -> None:
    completion = {"id": 1, "completed_by": {"id": 5}, "result": []}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.reviewer_id == "5"


def test_parse_completed_by_missing_yields_unknown() -> None:
    completion = {"id": 1, "result": []}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.reviewer_id == "unknown"


def test_parse_explicit_reviewer_id_overrides_completed_by() -> None:
    completion = {"id": 1, "completed_by": "bob", "result": []}
    result = parse_ls_completion(
        completion, image_id="img_a", reviewer_id="override@x.com"
    )
    assert result.reviewer_id == "override@x.com"


def test_parse_created_at_iso_with_z_suffix() -> None:
    completion = {
        "id": 1,
        "result": [],
        "created_at": "2026-04-27T10:00:00Z",
    }
    result = parse_ls_completion(completion, image_id="img_a")
    # Match the exact UTC interpretation (Z → +00:00)
    from datetime import datetime
    expected = datetime.fromisoformat("2026-04-27T10:00:00+00:00").timestamp()
    assert result.reviewed_at == pytest.approx(expected)


def test_parse_created_at_unix_timestamp() -> None:
    completion = {"id": 1, "result": [], "created_at": 1714200000}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.reviewed_at == pytest.approx(1714200000.0)


def test_parse_created_at_missing_uses_now() -> None:
    before = time.time()
    result = parse_ls_completion({"id": 1, "result": []}, image_id="img_a")
    after = time.time()
    assert before <= result.reviewed_at <= after + 1.0


def test_parse_invalid_frame_state_choice_defaults_to_needs_more_review() -> None:
    completion = {
        "id": 1,
        "result": [
            {
                "from_name": "frame_state",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": ["totally_made_up_state"]},
            }
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.frame_state == "needs_more_review"


def test_parse_multiple_textarea_lines_joined_with_newlines() -> None:
    completion = {
        "id": 1,
        "result": [
            {
                "from_name": "notes",
                "to_name": "image",
                "type": "textarea",
                "value": {"text": ["first line", "second line", "third"]},
            }
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.notes == "first line\nsecond line\nthird"


def test_parse_non_rectangle_region_is_ignored() -> None:
    """A polygon (or other non-rectanglelabels) region must not crash the parser."""
    completion = {
        "id": 1,
        "result": [
            {
                "id": "poly-1",
                "type": "polygon",
                "from_name": "bbox",
                "to_name": "image",
                "value": {"points": [[0, 0], [1, 0], [1, 1]]},
            },
            _ls_rect("c1", label="forklift"),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a")
    assert len(result.corrections) == 1
    assert result.corrections[0].candidate_id == "c1"


def test_parse_float_precision_within_tolerance_is_finalize() -> None:
    """Reviewer's box differs from seed by < 1e-4 in normalized space → finalize.

    Seed: x=10.0, w=30.0 (LS percent) → x1=0.10, x2=0.40.
    Reviewer: x=10.000001, w=30.0 → x1=0.10000001, x2=0.40000001.
    Both within math.isclose abs_tol=1e-4 → source='finalize'.
    """
    seeded = [_seed_pred("c1", x=10.0, y=20.0, width=30.0, height=40.0, label="forklift")]
    completion = {
        "id": 1,
        "result": [
            _ls_rect(
                "c1",
                x=10.000001,
                y=20.000001,
                width=30.0,
                height=40.0,
                label="forklift",
            ),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    c = result.corrections[0]
    assert c.source == "finalize"
    assert c.original_class is None
    assert c.original_bbox is None


def test_parse_just_outside_tolerance_is_edited() -> None:
    """Confirms the tolerance boundary: a 1% LS-percent shift is well outside 1e-4 norm tol."""
    seeded = [_seed_pred("c1", x=10.0, y=20.0, width=30.0, height=40.0, label="forklift")]
    completion = {
        "id": 1,
        "result": [
            _ls_rect("c1", x=11.0, y=20.0, width=30.0, height=40.0, label="forklift"),
        ],
    }
    result = parse_ls_completion(completion, image_id="img_a", seeded_predictions=seeded)
    assert result.corrections[0].source == "edited"


def test_parse_lead_time_missing_defaults_to_zero() -> None:
    result = parse_ls_completion({"id": 1, "result": []}, image_id="img_a")
    assert result.duration_seconds == 0.0


def test_parse_id_missing_defaults_completion_id_zero() -> None:
    result = parse_ls_completion({"result": []}, image_id="img_a")
    assert result.ls_completion_id == 0


def test_parse_non_numeric_completion_id_falls_back_to_zero() -> None:
    completion = {"id": "abc-not-a-number", "result": []}
    result = parse_ls_completion(completion, image_id="img_a")
    assert result.ls_completion_id == 0


def test_build_task_path_traversal_image_path_returns_none() -> None:
    payload = _minimal_payload()
    payload["meta"]["image_path"] = "/tmp/imgs/../etc/passwd"
    assert build_task(payload) is None


def test_build_task_clamps_out_of_range_bbox_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "c1",
                "class_name": "forklift",
                "bbox": {"x1": -0.2, "y1": 0.2, "x2": 1.5, "y2": 0.6},
            }
        ],
    }
    with caplog.at_level("WARNING", logger="manual_reviewer.pipeline_io.task_builder"):
        task = build_task(payload)
    region = task["predictions"][0]["result"][0]
    assert region["value"]["x"] == pytest.approx(0.0)
    assert region["value"]["x"] + region["value"]["width"] == pytest.approx(100.0)
    assert any("out of [0,1]" in rec.message for rec in caplog.records)


def test_build_task_logs_drop_for_missing_candidate_id(caplog: pytest.LogCaptureFixture) -> None:
    payload = _minimal_payload()
    payload["stages"]["finalize"] = {
        "final_annotations": [
            {
                "candidate_id": "",
                "class_name": "forklift",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
            }
        ],
    }
    with caplog.at_level("WARNING", logger="manual_reviewer.pipeline_io.task_builder"):
        build_task(payload)
    assert any("dropping finalize annotation" in rec.message for rec in caplog.records)


def test_bbox_equal_with_image_size_is_pixel_aware() -> None:
    from manual_reviewer.pipeline_io.ls_export_parser import _bbox_equal
    from data_miner.auto_annotation_v4.configs.contracts import BoundingBox

    a = BoundingBox(x1=0.10, y1=0.10, x2=0.20, y2=0.20)
    # 0.4 px shift on a 1000-px-wide image — within 0.5/1000 = 5e-4
    b = BoundingBox(x1=0.1004, y1=0.10, x2=0.20, y2=0.20)
    assert _bbox_equal(a, b, image_size=(1000, 1000)) is True
    # 0.6 px shift — outside the pixel-aware tolerance
    c = BoundingBox(x1=0.1006, y1=0.10, x2=0.20, y2=0.20)
    assert _bbox_equal(a, c, image_size=(1000, 1000)) is False


def test_bbox_equal_default_tolerance_unchanged() -> None:
    from manual_reviewer.pipeline_io.ls_export_parser import _bbox_equal
    from data_miner.auto_annotation_v4.configs.contracts import BoundingBox

    a = BoundingBox(x1=0.10, y1=0.10, x2=0.20, y2=0.20)
    b = BoundingBox(x1=0.10005, y1=0.10, x2=0.20, y2=0.20)
    assert _bbox_equal(a, b) is True
