"""Shared fixtures for manual_reviewer round-trip tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    Candidate,
    EvaluateResult,
    FilterDrop,
    FilterResult,
    FinalAnnotation,
    FinalizeResult,
    ProposalResult,
    VLMVerdict,
)
from data_miner.auto_annotation_v4.configs.enums import (
    DetectorName,
    DropReason,
    FilterContext,
    FinalAction,
    Stage,
)


@pytest.fixture
def seeded_pipeline_db(tmp_path: Path) -> Path:
    """Build a tiny pipeline.db with one survivor + one drop, full pipeline output."""
    db_path = tmp_path / "pipeline.db"
    asyncio.run(_seed(db_path))
    return db_path


async def _seed(db_path: Path) -> None:
    async with CheckpointDB(db_path) as db:
        await db.save_job_info(
            job_id="testjob",
            image_dir="/tmp/imgs",
            config_hash="h1",
            prompt_version="v1",
        )
        await db.register_image_batch(
            [("img_a", "/tmp/imgs/img_a.jpg"), ("img_b", "/tmp/imgs/img_b.jpg")]
        )
        async with db._transaction() as tx:
            await tx.execute(
                "UPDATE image_meta SET dedup_status='dropped' WHERE image_id=?",
                ("img_b",),
            )

        bbox = BoundingBox(x1=0.10, y1=0.20, x2=0.40, y2=0.60)
        annotation = FinalAnnotation(
            candidate_id="c1",
            class_name="forklift",
            class_id=0,
            bbox=bbox,
            confidence=0.95,
            action=FinalAction.ACCEPT,
            source_model="sam3_dart",
        )
        await db.save_stage(
            "img_a",
            Stage.FINALIZE,
            FinalizeResult(image_id="img_a", final_annotations=[annotation]),
            "h1",
        )

        drop = FilterDrop(
            candidate_id="c2",
            reason=DropReason.SCORE_FLOOR,
            context=FilterContext.POST_DETECT,
            detail="below floor",
        )
        await db.save_stage(
            "img_a",
            Stage.FILTER,
            FilterResult(image_id="img_a", candidates=[], drops=[drop]),
            "h1",
        )

        verdict = VLMVerdict(
            candidate_id="c1",
            detected_class="forklift",
            class_confidence=0.88,
            bbox_score=0.9,
            reasoning="loaders visible",
        )
        await db.save_stage(
            "img_a",
            Stage.EVALUATE,
            EvaluateResult(image_id="img_a", verdicts=[verdict]),
            "h1",
        )

        cand = Candidate(
            candidate_id="c1",
            class_name="forklift",
            label="forklift",
            source_model="sam3_dart",
            expression="forklift",
            bbox=bbox,
            score=0.9,
        )
        await db.save_proposal(
            "img_a",
            DetectorName.SAM3_DART,
            ProposalResult(
                model="sam3_dart",
                image_id="img_a",
                image_size=[1920, 1080],
                latency_ms=120.0,
                candidates=[cand],
            ),
        )

        async with db._transaction() as tx:
            await tx.execute(
                "UPDATE image_meta SET stages_completed=? WHERE image_id=?",
                (
                    json.dumps(["detect", "filter", "evaluate", "refine", "finalize"]),
                    "img_a",
                ),
            )
