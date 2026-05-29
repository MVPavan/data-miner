"""Tier 2 test: detect-only run saves raw merged candidates (no filter drops).

Proves that :class:`DetectMergeWorker.process` merges per-model proposals into
a :class:`DetectResult` whose ``candidates`` is the full union of inputs — the
worker performs ZERO filter drops post-split (filtering now lives in
:class:`FilterWorker`).

Inspection-level: confirms ``FilterPipeline`` is not imported/called from
``stages/detect.py``.

Behavioural: instantiates a real :class:`CheckpointDB` on ``tmp_path``, seeds
two per-model :class:`ProposalResult` rows via ``save_proposal``, runs the
worker's ``process()`` directly, and asserts the returned + persisted
``DetectResult`` contains every input candidate with no drop metadata.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    Candidate,
    DetectResult,
    ProposalResult,
    StageMessage,
)
from data_miner.auto_annotation_v4.configs.enums import DetectorName, Stage
from data_miner.auto_annotation_v4.configs.settings import (
    AutoAnnotationV4Config,
    ClassConfig,
)
from data_miner.auto_annotation_v4.stages import detect as detect_module
from data_miner.auto_annotation_v4.stages.detect import DetectMergeWorker


def _run(coro):
    return asyncio.run(coro)


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _make_synthetic_image(path: Path, size: tuple[int, int] = (640, 480)) -> None:
    Image.new("RGB", size, color=(127, 127, 127)).save(path, format="PNG")


def _build_config() -> AutoAnnotationV4Config:
    return AutoAnnotationV4Config(
        class_registry={
            "person": ClassConfig(id=0, tier=1, prompts=["person"], tags=[]),
            "car": ClassConfig(id=2, tier=1, prompts=["car"], tags=[]),
        },
        prompts_dir="prompts_nonexistent_for_test",
    )


def _gdino_candidates() -> list[Candidate]:
    return [
        Candidate(
            candidate_id="g1",
            class_name="person",
            label="person",
            source_model="grounding_dino",
            expression="person",
            bbox=_bbox(0.10, 0.10, 0.30, 0.50),
            score=0.80,
        ),
        Candidate(
            candidate_id="g2",
            class_name="person",
            label="person",
            source_model="grounding_dino",
            expression="person",
            bbox=_bbox(0.60, 0.10, 0.80, 0.50),
            score=0.70,
        ),
        Candidate(
            candidate_id="g3",
            class_name="car",
            label="car",
            source_model="grounding_dino",
            expression="car",
            bbox=_bbox(0.00, 0.00, 0.005, 0.005),  # tiny area — would be filtered
            score=0.05,                             # and below any sane floor
        ),
    ]


def _sam3_candidates() -> list[Candidate]:
    return [
        Candidate(
            candidate_id="s1",
            class_name="person",
            label="person",
            source_model="sam3",
            expression="person",
            bbox=_bbox(0.11, 0.11, 0.30, 0.50),  # near-duplicate of g1 (would dedup)
            score=0.90,
        ),
        Candidate(
            candidate_id="s2",
            class_name="car",
            label="car",
            source_model="sam3",
            expression="car",
            bbox=_bbox(0.40, 0.40, 0.70, 0.60),
            score=0.85,
        ),
    ]


def test_filter_pipeline_not_imported_in_detect_module():
    """Inspection-level proof: detect.py has no FilterPipeline dependency."""
    src = Path(detect_module.__file__).read_text()
    assert "FilterPipeline" not in src, (
        "stages/detect.py must NOT reference FilterPipeline — "
        "filtering lives in stages/filter.py post-split."
    )


def test_detect_merge_drop_field_absent_from_contract():
    """DetectResult has no ``drops`` field — raw merged candidates only."""
    assert "drops" not in DetectResult.model_fields
    assert "candidates" in DetectResult.model_fields


def test_detect_merge_saves_raw_candidates(tmp_path):
    """DetectMergeWorker.process returns union of per-model candidates, no drops."""

    async def scenario():
        image_path = tmp_path / "img1.png"
        _make_synthetic_image(image_path)

        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            await db.register_image("img1", str(image_path))

            gdino_cands = _gdino_candidates()
            sam3_cands = _sam3_candidates()

            await db.save_proposal(
                "img1",
                DetectorName.GROUNDING_DINO,
                ProposalResult(
                    model=DetectorName.GROUNDING_DINO.value,
                    image_id="img1",
                    image_size=[640, 480],
                    latency_ms=1.0,
                    candidates=gdino_cands,
                ),
            )
            await db.save_proposal(
                "img1",
                DetectorName.SAM3,
                ProposalResult(
                    model=DetectorName.SAM3.value,
                    image_id="img1",
                    image_size=[640, 480],
                    latency_ms=1.0,
                    candidates=sam3_cands,
                ),
            )

            config = _build_config()
            worker = DetectMergeWorker(config, db, worker_id="test-merge")

            msg = StageMessage(
                image_id="img1",
                image_path=str(image_path),
                job_id="test-job",
                stage=Stage.DETECT,
            )
            result = await worker.process(msg)

            assert isinstance(result, DetectResult)
            expected_ids = {c.candidate_id for c in gdino_cands + sam3_cands}
            actual_ids = {c.candidate_id for c in result.candidates}
            assert actual_ids == expected_ids, (
                f"DetectMergeWorker dropped candidates: "
                f"missing={expected_ids - actual_ids}, extra={actual_ids - expected_ids}"
            )
            assert len(result.candidates) == len(gdino_cands) + len(sam3_cands)

            assert set(result.models_used) == {
                DetectorName.GROUNDING_DINO.value,
                DetectorName.SAM3.value,
            }

            assert result.routing.auto_accepted == []
            assert result.routing.needs_evaluation == []
            assert result.routing.confusion_flags == []
            assert result.filter_stats == {}

            next_stage = worker._resolve_next_stage(result)
            assert next_stage == Stage.FILTER

            await db.save_stage("img1", Stage.DETECT, result, config_hash="test")
            reloaded = await db.load_stage("img1", Stage.DETECT, DetectResult)
            assert reloaded is not None
            reloaded_ids = {c.candidate_id for c in reloaded.candidates}
            assert reloaded_ids == expected_ids
            assert reloaded.routing.auto_accepted == []
            assert reloaded.routing.needs_evaluation == []
            assert reloaded.filter_stats == {}
        finally:
            await db.close()

    _run(scenario())
