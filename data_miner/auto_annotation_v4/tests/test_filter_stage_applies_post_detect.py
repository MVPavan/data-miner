"""Tier 2 test: FilterWorker applies the POST_DETECT chain after detect.

Proves that running :class:`FilterWorker.process` on a persisted
:class:`DetectResult` with obvious filter violations produces a
:class:`FilterResult` whose ``drops`` covers at least three
:class:`DropReason` values, and that surviving auto-accepted candidates
produce a YOLO label file on disk via the fast-path in
:meth:`FilterWorker._write_auto_accepted_output`.

Uses the real loaded config (via ``load_config`` with no overrides) and then
tweaks a handful of thresholds in place so auto-accept fires for single-model
survivors and ``per_class_cap`` trips with a small set of candidates.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    Candidate,
    DetectResult,
    DetectRouting,
    FilterResult,
    StageMessage,
)
from data_miner.auto_annotation_v4.configs.enums import (
    DropReason,
    Stage,
    WorkStatus,
)
from data_miner.auto_annotation_v4.configs.loader import load_config
from data_miner.auto_annotation_v4.output import OutputWriter
from data_miner.auto_annotation_v4.stages.filter import FilterWorker


def _run(coro):
    return asyncio.run(coro)


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _build_violation_candidates() -> list[Candidate]:
    """Synthetic candidates engineered to exercise multiple DropReasons."""
    return [
        # 1) GEOMETRIC_FILTER — area ~2.5e-5 << min_area 0.0005.
        Candidate(
            candidate_id="geom_fail",
            class_name="car",
            label="car",
            source_model="sam3",
            expression="car",
            bbox=_bbox(0.01, 0.01, 0.015, 0.015),
            score=0.90,
        ),
        # 2) SCORE_FLOOR — grounding_dino floor is 0.35; this is 0.10.
        Candidate(
            candidate_id="score_fail",
            class_name="car",
            label="car",
            source_model="grounding_dino",
            expression="car",
            bbox=_bbox(0.05, 0.05, 0.10, 0.10),
            score=0.10,
        ),
        # 3) DEDUP — two heavily overlapping same-class candidates, different
        #    models — one is dropped with reason DEDUP. Survivor gets
        #    agreement=2, easily auto-accepting later.
        Candidate(
            candidate_id="dedup_A",
            class_name="dog",
            label="dog",
            source_model="sam3",
            expression="dog",
            bbox=_bbox(0.20, 0.20, 0.32, 0.35),
            score=0.90,
        ),
        Candidate(
            candidate_id="dedup_B",
            class_name="dog",
            label="dog",
            source_model="grounding_dino",
            expression="dog",
            bbox=_bbox(0.205, 0.205, 0.32, 0.35),
            score=0.60,
        ),
        # 4) PER_CLASS_CAP — three non-overlapping birds with max_per_class=2
        #    => the lowest-score one is capped out.
        Candidate(
            candidate_id="bird_1",
            class_name="bird",
            label="bird",
            source_model="sam3",
            expression="bird",
            bbox=_bbox(0.40, 0.05, 0.48, 0.15),
            score=0.95,
        ),
        Candidate(
            candidate_id="bird_2",
            class_name="bird",
            label="bird",
            source_model="sam3",
            expression="bird",
            bbox=_bbox(0.50, 0.05, 0.58, 0.15),
            score=0.90,
        ),
        Candidate(
            candidate_id="bird_3",
            class_name="bird",
            label="bird",
            source_model="sam3",
            expression="bird",
            bbox=_bbox(0.40, 0.18, 0.48, 0.28),
            score=0.80,
        ),
        # 5) CROSS_CLASS — bus vs boat, both tier 1, no overlap-exempt tags,
        #    no shared confusion tag — heavy overlap triggers cross-class
        #    suppression of the loser (tiebreak: agreement -> model_priority
        #    -> score).
        Candidate(
            candidate_id="cross_bus",
            class_name="bus",
            label="bus",
            source_model="sam3",
            expression="bus",
            bbox=_bbox(0.60, 0.60, 0.80, 0.80),
            score=0.90,
        ),
        Candidate(
            candidate_id="cross_boat",
            class_name="boat",
            label="boat",
            source_model="grounding_dino",
            expression="boat",
            bbox=_bbox(0.61, 0.61, 0.80, 0.80),
            score=0.70,
        ),
    ]


def test_filter_worker_applies_post_detect_chain(tmp_path):
    """FilterWorker.process drops violations across reasons and writes YOLO."""

    async def scenario():
        config = load_config()
        # Tweak to make the test deterministic:
        #   - auto-accept on single-model agreement so survivors auto-accept,
        #     producing a YOLO file via the fast path.
        #   - max_per_class=2 so three non-overlapping birds trigger the cap.
        config.auto_accept.min_model_agreement = 1
        config.filtering.max_per_class = 2

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        output_writer = OutputWriter(job_dir)

        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_id = "img-filter-test"
            image_path = str(tmp_path / "img-filter-test.png")
            await db.register_image(image_id, image_path)

            candidates = _build_violation_candidates()
            detect_result = DetectResult(
                image_id=image_id,
                image_path=image_path,
                image_size=[640, 480],
                models_used=["grounding_dino", "sam3"],
                candidates=candidates,
                routing=DetectRouting(),
                filter_stats={},
                stage_timing_ms=0.0,
            )
            await db.save_stage(
                image_id, Stage.DETECT, detect_result, config_hash="test-hash"
            )

            # Seed a work_queue row so save_and_forward semantics hold, even
            # though we invoke process() directly (not the run() loop).
            conn = db._require_db()
            await conn.execute(
                "INSERT INTO work_queue (image_id, stage, status, score)"
                " VALUES (?, ?, ?, ?)",
                (image_id, Stage.FILTER.value, WorkStatus.PROCESSING, time.time()),
            )
            await conn.commit()

            worker = FilterWorker(
                config,
                db,
                output_writer=output_writer,
                worker_id="test-filter",
            )

            msg = StageMessage(
                image_id=image_id,
                image_path=image_path,
                job_id="test-job",
                stage=Stage.FILTER,
            )
            result = await worker.process(msg)
            assert isinstance(result, FilterResult)

            # ---- drops: must span at least 3 DropReason values. ------------
            drop_reasons = {d.reason for d in result.drops}
            assert len(drop_reasons) >= 3, (
                f"Expected drops across >=3 DropReasons, got {drop_reasons}. "
                f"drops={[(d.candidate_id, d.reason) for d in result.drops]}"
            )
            # The violations we engineered should all fire:
            for expected in (
                DropReason.GEOMETRIC_FILTER,
                DropReason.SCORE_FLOOR,
                DropReason.DEDUP,
                DropReason.PER_CLASS_CAP,
                DropReason.CROSS_CLASS,
            ):
                assert expected in drop_reasons, (
                    f"Missing expected drop reason {expected}. "
                    f"All: {drop_reasons}"
                )

            # ---- routing: survivors are all auto-accepted (fast path). -----
            surviving_ids = {c.candidate_id for c in result.candidates}
            routed_ids = (
                set(result.routing.auto_accepted)
                | set(result.routing.needs_evaluation)
            )
            # Every surviving candidate is assigned to one of the buckets.
            assert routed_ids == surviving_ids, (
                f"Routing buckets do not cover survivors: routed={routed_ids}, "
                f"surviving={surviving_ids}"
            )
            assert result.routing.auto_accepted, (
                "Expected at least one auto-accepted candidate."
            )
            assert not result.routing.needs_evaluation, (
                "min_model_agreement=1 + tier-1 classes should yield "
                "zero needs_evaluation; got "
                f"{result.routing.needs_evaluation}"
            )

            # ---- save + reload round-trip to exercise the checkpoint path. -
            await db.save_stage(
                image_id, Stage.FILTER, result, config_hash="test-hash"
            )
            reloaded = await db.load_stage(
                image_id, Stage.FILTER, FilterResult
            )
            assert reloaded is not None
            reloaded_reasons = {d.reason for d in reloaded.drops}
            assert reloaded_reasons == drop_reasons

            # ---- YOLO labels: fast path wrote labels/{image_id}.txt --------
            labels_path = job_dir / "labels" / f"{image_id}.txt"
            assert labels_path.exists(), (
                f"Expected YOLO labels at {labels_path} but the file is missing. "
                "Fast-path write only fires when auto_accepted is non-empty "
                "and needs_evaluation is empty."
            )
            contents = labels_path.read_text().strip()
            assert contents, "Labels file is empty."
            lines = [ln for ln in contents.splitlines() if ln.strip()]
            assert len(lines) == len(result.routing.auto_accepted), (
                f"YOLO line count {len(lines)} != auto_accepted count "
                f"{len(result.routing.auto_accepted)}"
            )
            # Each line is "class_id cx cy w h" — 5 space-separated tokens.
            for ln in lines:
                toks = ln.split()
                assert len(toks) == 5, f"Bad YOLO line: {ln!r}"
                int(toks[0])  # class_id must be int-parseable
                for t in toks[1:]:
                    float(t)  # floats must be parseable

            trace_path = job_dir / "traces" / f"{image_id}.json"
            assert trace_path.exists(), (
                "Auto-accept fast path should also write a trace file."
            )

            # ---- filter_stats bookkeeping is sane. -------------------------
            stats = result.filter_stats
            assert stats["total_proposed"] == len(candidates)
            assert stats["after_filters"] == len(result.candidates)
            assert stats["dropped"] == len(result.drops)
            assert stats["auto_accepted"] == len(result.routing.auto_accepted)
            assert stats["sent_to_vlm"] == len(result.routing.needs_evaluation)
        finally:
            await db.close()

    _run(scenario())
