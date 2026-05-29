"""Unit tests for synonym routing in the detect stage.

Covers the two symmetric changes made in stages/detect_model.py:

1. ``_to_candidates`` collapses synonym labels to the canonical class name.
2. Unknown labels (not in prompts or synonyms) are dropped.

Prompt-list build (prompts + synonyms, deduped) is exercised by constructing
an ``active_classes`` dict and asserting the flat list the helper would emit.
"""

from __future__ import annotations

from data_miner.auto_annotation_v4.configs import (
    ClassConfig,
    DetectorName,
    DetectorResponse,
)
from data_miner.auto_annotation_v4.stages.detect_model import _to_candidates
from data_miner.auto_annotation_v4.utils import normalize_class_alias


def _make_class(
    name: str,
    prompts: list[str],
    synonyms: list[str] | None = None,
    *,
    class_id: int = 0,
    tier: int = 1,
) -> ClassConfig:
    return ClassConfig(
        id=class_id,
        tier=tier,
        prompts=prompts,
        synonyms=synonyms or [],
        tags=[],
        description="",
    )


def test_synonym_collapses_to_canonical():
    classes = {
        "truck": _make_class(
            "truck",
            prompts=["truck"],
            synonyms=["fire truck", "van", "ambulance", "tractor", "lorry"],
            class_id=7,
        ),
    }
    resp = DetectorResponse(
        boxes=[
            [0.0, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
        ],
        scores=[0.9, 0.8, 0.7, 0.6],
        labels=["ambulance", "van", "fire truck", "truck"],
    )

    candidates = _to_candidates(resp, DetectorName.GROUNDING_DINO, classes)

    assert len(candidates) == 4
    assert {c.class_name for c in candidates} == {"truck"}
    # Original detector label is preserved on the candidate for observability.
    assert [c.label for c in candidates] == ["ambulance", "van", "fire truck", "truck"]


def test_unknown_labels_are_dropped():
    classes = {
        "truck": _make_class("truck", prompts=["truck"], synonyms=["van"], class_id=7),
    }
    resp = DetectorResponse(
        boxes=[[0.0, 0.0, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]],
        scores=[0.9, 0.8, 0.7],
        labels=["van", "helicopter", "truck"],
    )

    candidates = _to_candidates(resp, DetectorName.GROUNDING_DINO, classes)

    # "helicopter" is not in truck's aliases and no helicopter class is registered
    # in this test — should be dropped.
    assert len(candidates) == 2
    assert [c.class_name for c in candidates] == ["truck", "truck"]
    assert [c.label for c in candidates] == ["van", "truck"]


def test_alias_matching_is_case_and_whitespace_insensitive():
    classes = {
        "motorcycle": _make_class(
            "motorcycle", prompts=["motorcycle"], synonyms=["motorbike", "scooter"],
            class_id=3,
        ),
    }
    resp = DetectorResponse(
        boxes=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
        scores=[0.9, 0.8, 0.7],
        labels=["Motorbike", "  SCOOTER  ", "motorcycle"],
    )

    candidates = _to_candidates(resp, DetectorName.GROUNDING_DINO, classes)

    assert len(candidates) == 3
    assert {c.class_name for c in candidates} == {"motorcycle"}


def test_multi_class_disambiguation():
    # Two classes, each with synonyms. Verify candidates route to the right one.
    classes = {
        "truck": _make_class("truck", prompts=["truck"], synonyms=["van"], class_id=7),
        "car": _make_class("car", prompts=["car"], synonyms=["suv"], class_id=2),
    }
    resp = DetectorResponse(
        boxes=[[0, 0, 1, 1]] * 4,
        scores=[0.9, 0.8, 0.7, 0.6],
        labels=["van", "suv", "truck", "car"],
    )

    candidates = _to_candidates(resp, DetectorName.GROUNDING_DINO, classes)

    by_label = {c.label: c.class_name for c in candidates}
    assert by_label == {
        "van": "truck",
        "suv": "car",
        "truck": "truck",
        "car": "car",
    }


def test_prompts_and_synonyms_dedup_by_normalized_form():
    # Simulate the flat-prompt build logic from DetectModelWorker._call_server.
    classes = {
        "truck": _make_class(
            "truck",
            prompts=["truck"],
            synonyms=["Truck", "fire truck", "van"],  # "Truck" dupes "truck"
            class_id=7,
        ),
        "car": _make_class(
            "car",
            prompts=["car"],
            synonyms=["suv", "CAR"],  # "CAR" dupes "car"
            class_id=2,
        ),
    }

    seen: set[str] = set()
    flat: list[str] = []
    for cls_cfg in classes.values():
        for prompt in (*cls_cfg.prompts, *cls_cfg.synonyms):
            key = normalize_class_alias(prompt)
            if key in seen:
                continue
            seen.add(key)
            flat.append(prompt)

    # Duplicates (case/whitespace variants) must be removed; canonical comes first.
    assert flat == ["truck", "fire truck", "van", "car", "suv"]
