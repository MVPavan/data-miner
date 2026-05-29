"""Tier 3 lifecycle scenarios — simulation-mode (CPU, DB-level).

These tests exercise the pipeline's state transitions (restart, force,
hash-change) by seeding synthetic rows directly into a CheckpointDB and
invoking the relevant primitives — no GPU, no model servers, no live
inference.
"""

from __future__ import annotations

import asyncio
import json
import time

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    Candidate,
    DetectResult,
    EvaluateResult,
    FilterResult,
    FinalizeResult,
    ProposalResult,
    RefineResult,
)
from data_miner.auto_annotation_v4.configs.enums import (
    STAGE_ORDER,
    DetectorName,
    ImageStatus,
    Stage,
    WorkStatus,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Builders for minimal-but-valid Pydantic payloads
# ---------------------------------------------------------------------------


def _detect_result(image_id: str) -> DetectResult:
    return DetectResult(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.jpg",
        image_size=[100, 100],
        models_used=["grounding_dino"],
        candidates=[],
    )


def _filter_result(image_id: str) -> FilterResult:
    return FilterResult(image_id=image_id, candidates=[])


def _evaluate_result(image_id: str) -> EvaluateResult:
    return EvaluateResult(image_id=image_id)


def _refine_result(image_id: str) -> RefineResult:
    return RefineResult(image_id=image_id)


def _finalize_result(image_id: str) -> FinalizeResult:
    return FinalizeResult(image_id=image_id)


def _proposal(image_id: str, model: DetectorName) -> ProposalResult:
    return ProposalResult(
        model=model.value,
        image_id=image_id,
        image_size=[100, 100],
        latency_ms=1.0,
        candidates=[],
    )


async def _save_all_stages(db: CheckpointDB, image_id: str, config_hash: str = "v1") -> None:
    """Directly write every canonical stage checkpoint via save_stage.

    Also marks image_meta.stages_completed = STAGE_ORDER and status = complete
    so all_stages_complete() reports True.
    """
    await db.save_stage(image_id, Stage.DETECT, _detect_result(image_id), config_hash)
    await db.save_stage(image_id, Stage.FILTER, _filter_result(image_id), config_hash)
    await db.save_stage(image_id, Stage.EVALUATE, _evaluate_result(image_id), config_hash)
    await db.save_stage(image_id, Stage.REFINE, _refine_result(image_id), config_hash)
    await db.save_stage(image_id, Stage.FINALIZE, _finalize_result(image_id), config_hash)
    conn = db._require_db()
    all_stages = [s.value for s in STAGE_ORDER]
    await conn.execute(
        "UPDATE image_meta SET stages_completed = ?, status = ? WHERE image_id = ?",
        (json.dumps(all_stages), ImageStatus.COMPLETE, image_id),
    )
    # Seed DONE rows in work_queue for every stage as a completed run would.
    now = time.time()
    for s in STAGE_ORDER:
        await conn.execute(
            "INSERT OR REPLACE INTO work_queue (image_id, stage, status, score)"
            " VALUES (?, ?, ?, ?)",
            (image_id, s.value, WorkStatus.DONE, now),
        )
    await conn.commit()


# ---------------------------------------------------------------------------
# S2: SIGKILL simulated restart
# ---------------------------------------------------------------------------


def test_s2_sigkill_simulated_restart(tmp_path):
    async def scenario():
        db_path = tmp_path / "pipeline.db"
        db = CheckpointDB(db_path, lock_ttl=60, max_retries=3)
        await db.connect()
        try:
            image_ids = [f"img{i:02d}" for i in range(20)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            await db.add_work_batch(Stage.DETECT.value, image_ids)

            # Claim 5 — these are the "in-flight when SIGKILL hits" images.
            claimed = []
            for i in range(5):
                c = await db.claim_work(Stage.DETECT.value, f"worker-{i}")
                assert c is not None
                claimed.append(c)
            assert len(set(claimed)) == 5

            # Complete 3 others through DETECT -> FILTER -> ... -> FINALIZE.
            completed = []
            for i in range(5, 8):
                c = await db.claim_work(Stage.DETECT.value, f"worker-{i}")
                assert c is not None
                completed.append(c)
                await db.save_and_forward(
                    image_id=c,
                    stage=Stage.DETECT,
                    data=_detect_result(c),
                    config_hash="v1",
                    next_stage=Stage.FILTER.value,
                    timing_ms=1.0,
                    work_stage=Stage.DETECT.value,
                )
                # Claim+forward through the rest of the stages.
                chain = [
                    (Stage.FILTER, _filter_result(c), Stage.EVALUATE.value),
                    (Stage.EVALUATE, _evaluate_result(c), Stage.REFINE.value),
                    (Stage.REFINE, _refine_result(c), Stage.FINALIZE.value),
                    (Stage.FINALIZE, _finalize_result(c), "done"),
                ]
                for stg, payload, nxt in chain:
                    got = await db.claim_work(stg.value, f"worker-{i}")
                    assert got == c
                    await db.save_and_forward(
                        image_id=c,
                        stage=stg,
                        data=payload,
                        config_hash="v1",
                        next_stage=nxt,
                        timing_ms=1.0,
                        work_stage=stg.value,
                    )

            # Confirm the 3 completed images report all_stages_complete.
            for c in completed:
                assert await db.all_stages_complete(c) is True

            # Force the 5 claimed rows stale via raw UPDATE (mirrors
            # test_checkpoint_db.py::test_stale_recovery pattern).
            conn = db._require_db()
            stale_time = time.time() - 9999
            placeholders = ",".join("?" for _ in claimed)
            await conn.execute(
                f"UPDATE work_queue SET claimed_at = ? WHERE image_id IN ({placeholders})"
                f" AND stage = ?",
                [stale_time, *claimed, Stage.DETECT.value],
            )
            await conn.commit()

            # Recover stale: 5 rows reset to pending, attempts=1.
            recovered = await db.recover_stale(lock_ttl=60, max_retries=3)
            assert recovered == 5
            cur = await conn.execute(
                f"SELECT image_id, status, attempts FROM work_queue"
                f" WHERE image_id IN ({placeholders}) AND stage = ?",
                [*claimed, Stage.DETECT.value],
            )
            rows = await cur.fetchall()
            assert len(rows) == 5
            for r in rows:
                assert r["status"] == WorkStatus.PENDING
                assert r["attempts"] == 1

            # Simulate submitter restart: iterate every image, skip complete
            # ones, re-queue incomplete ones via add_work (INSERT OR IGNORE).
            requeued = 0
            skipped = 0
            for img in image_ids:
                if await db.all_stages_complete(img):
                    skipped += 1
                    continue
                await db.add_work(Stage.DETECT.value, img)
                requeued += 1
            assert skipped == 3
            assert requeued == 17

            # Final checks: 3 completed images have DETECT row marked DONE,
            # the other 17 are PENDING, zero duplicates.
            cur = await conn.execute(
                "SELECT image_id, status, COUNT(*) as cnt FROM work_queue"
                " WHERE stage = ? GROUP BY image_id",
                (Stage.DETECT.value,),
            )
            rows = await cur.fetchall()
            assert len(rows) == 20
            for r in rows:
                assert r["cnt"] == 1, f"duplicate for {r['image_id']}"
            cur = await conn.execute(
                "SELECT status, COUNT(*) as cnt FROM work_queue WHERE stage = ?"
                " GROUP BY status",
                (Stage.DETECT.value,),
            )
            counts = {r["status"]: r["cnt"] for r in await cur.fetchall()}
            assert counts.get(WorkStatus.DONE, 0) == 3
            assert counts.get(WorkStatus.PENDING, 0) == 17
            assert counts.get(WorkStatus.PROCESSING, 0) == 0
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S3: SIGINT graceful release via release_work
# ---------------------------------------------------------------------------


def test_s3_sigint_graceful_release(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(5)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            await db.add_work_batch(Stage.DETECT.value, image_ids)

            claimed = []
            for i in range(3):
                c = await db.claim_work(Stage.DETECT.value, f"worker-{i}")
                assert c is not None
                claimed.append(c)

            # Graceful release — mimics StageWorker on asyncio.CancelledError.
            for c in claimed:
                await db.release_work(c, Stage.DETECT.value)

            conn = db._require_db()
            placeholders = ",".join("?" for _ in claimed)
            cur = await conn.execute(
                f"SELECT image_id, status, claimed_at, worker_id, attempts"
                f" FROM work_queue WHERE image_id IN ({placeholders}) AND stage = ?",
                [*claimed, Stage.DETECT.value],
            )
            rows = await cur.fetchall()
            assert len(rows) == 3
            for r in rows:
                assert r["status"] == WorkStatus.PENDING
                assert r["claimed_at"] is None
                assert r["worker_id"] is None
                assert r["attempts"] == 0

            # All 3 must be reclaimable immediately (no wait for stale recovery).
            reclaimed = set()
            for i in range(3):
                c = await db.claim_work(Stage.DETECT.value, f"worker-new-{i}")
                assert c is not None
                reclaimed.add(c)
            assert reclaimed == set(claimed)
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S5: force_detect_models=[grounding_dino] simulated
# ---------------------------------------------------------------------------


def test_s5_force_detect_models_simulated(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(5)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )

            # Save both per-model proposals + DETECT + FILTER checkpoints.
            for img in image_ids:
                await db.save_proposal(
                    img, DetectorName.GROUNDING_DINO, _proposal(img, DetectorName.GROUNDING_DINO)
                )
                await db.save_proposal(
                    img, DetectorName.SAM3_DART, _proposal(img, DetectorName.SAM3_DART)
                )
                await db.save_stage(img, Stage.DETECT, _detect_result(img), "v1")
                await db.save_stage(img, Stage.FILTER, _filter_result(img), "v1")

            for img in image_ids:
                assert await db.proposal_exists(img, DetectorName.GROUNDING_DINO)
                assert await db.proposal_exists(img, DetectorName.SAM3_DART)

            # Simulate force_detect_models=[grounding_dino].
            compound_stage = f"{Stage.DETECT.value}:{DetectorName.GROUNDING_DINO.value}"
            for img in image_ids:
                await db.delete_proposal(img, DetectorName.GROUNDING_DINO.value)
                # clear_downstream(DETECT) wipes detect + filter + ... plus
                # compound stage:* queue rows per its implementation.
                await db.clear_downstream(img, Stage.DETECT)
                await db.add_work(compound_stage, img)

            conn = db._require_db()
            for img in image_ids:
                assert await db.proposal_exists(img, DetectorName.SAM3_DART) is True
                assert await db.proposal_exists(img, DetectorName.GROUNDING_DINO) is False
                assert await db.stage_exists(img, Stage.DETECT) is False
                assert await db.stage_exists(img, Stage.FILTER) is False

            cur = await conn.execute(
                "SELECT image_id, status FROM work_queue WHERE stage = ?",
                (compound_stage,),
            )
            rows = await cur.fetchall()
            assert len(rows) == 5
            for r in rows:
                assert r["status"] == WorkStatus.PENDING
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S6: force_stages=[filter] on completed job
# ---------------------------------------------------------------------------


def test_s6_force_filter_simulated(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(3)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            for img in image_ids:
                await _save_all_stages(db, img, config_hash="v1")
                assert await db.all_stages_complete(img) is True

            for img in image_ids:
                await db.clear_downstream(img, Stage.FILTER)

            conn = db._require_db()
            for img in image_ids:
                assert await db.stage_exists(img, Stage.DETECT) is True
                for s in [Stage.FILTER, Stage.EVALUATE, Stage.REFINE, Stage.FINALIZE]:
                    assert await db.stage_exists(img, s) is False

                cur = await conn.execute(
                    "SELECT stages_completed FROM image_meta WHERE image_id = ?",
                    (img,),
                )
                row = await cur.fetchone()
                completed = json.loads(row["stages_completed"])
                assert completed == [Stage.DETECT.value]

                cur = await conn.execute(
                    "SELECT stage FROM work_queue WHERE image_id = ?", (img,)
                )
                remaining = {r["stage"] for r in await cur.fetchall()}
                assert remaining == {Stage.DETECT.value}

            # Simulate re-queue for filter stage.
            for img in image_ids:
                await db.add_work(Stage.FILTER.value, img)
            cur = await conn.execute(
                "SELECT image_id, status FROM work_queue WHERE stage = ?",
                (Stage.FILTER.value,),
            )
            rows = await cur.fetchall()
            assert len(rows) == 3
            for r in rows:
                assert r["status"] == WorkStatus.PENDING
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S7: downstream-only run after proposal-only completion
# ---------------------------------------------------------------------------


def test_s7_downstream_only_after_proposal(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(3)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            for img in image_ids:
                await db.save_proposal(
                    img, DetectorName.GROUNDING_DINO, _proposal(img, DetectorName.GROUNDING_DINO)
                )
                await db.save_stage(img, Stage.DETECT, _detect_result(img), "v1")
                await db.save_stage(img, Stage.FILTER, _filter_result(img), "v1")

            # Simulate submitter: if FILTER exists, enqueue EVALUATE.
            for img in image_ids:
                if await db.stage_exists(img, Stage.FILTER):
                    await db.add_work(Stage.EVALUATE.value, img)

            conn = db._require_db()
            cur = await conn.execute(
                "SELECT image_id, status FROM work_queue WHERE stage = ?",
                (Stage.EVALUATE.value,),
            )
            rows = await cur.fetchall()
            assert len(rows) == 3
            for r in rows:
                assert r["status"] == WorkStatus.PENDING

            cur = await conn.execute(
                "SELECT stage, COUNT(*) as cnt FROM work_queue GROUP BY stage"
            )
            by_stage = {r["stage"]: r["cnt"] for r in await cur.fetchall()}
            assert Stage.DETECT.value not in by_stage
            assert Stage.FILTER.value not in by_stage
            assert by_stage.get(Stage.EVALUATE.value) == 3
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S10: mid-run stage-addition via stop+restart
#   Scenario: detect-only pipeline was running, stopped, restarted with
#   stages=[filter,...]. Submitter's _plan_first_stage checks prereq
#   (Stage.DETECT) via stage_exists — queues filter where detect exists,
#   skips where it doesn't. Proves the workaround for "can I add a stage
#   to a running pipeline" (answer: no while running — pidfile blocks —
#   but yes via restart on cached checkpoints).
# ---------------------------------------------------------------------------


def test_s10_add_filter_stage_after_partial_detect(tmp_path):
    async def scenario():
        from data_miner.auto_annotation_v4.configs.enums import STAGE_ORDER

        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            all_imgs = [f"img{i}" for i in range(5)]
            with_detect = all_imgs[:3]
            without_detect = all_imgs[3:]

            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in all_imgs]
            )
            for img in with_detect:
                await db.save_stage(img, Stage.DETECT, _detect_result(img), "v1")

            first_stage = Stage.FILTER
            idx = STAGE_ORDER.index(first_stage)
            prereq = STAGE_ORDER[idx - 1]
            assert prereq == Stage.DETECT

            planned: list[str] = []
            skipped: list[str] = []
            for img in all_imgs:
                if await db.stage_exists(img, prereq):
                    await db.add_work(first_stage.value, img)
                    planned.append(img)
                else:
                    skipped.append(img)

            assert planned == with_detect
            assert skipped == without_detect

            conn = db._require_db()
            cur = await conn.execute(
                "SELECT image_id, status FROM work_queue WHERE stage = ?",
                (first_stage.value,),
            )
            rows = await cur.fetchall()
            assert {r["image_id"] for r in rows} == set(with_detect)
            for r in rows:
                assert r["status"] == WorkStatus.PENDING

            cur = await conn.execute(
                "SELECT COUNT(*) as cnt FROM work_queue WHERE stage = ?",
                (Stage.DETECT.value,),
            )
            row = await cur.fetchone()
            assert row["cnt"] == 0
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S9: config_hash change invalidates downstream via should_run_stage
# ---------------------------------------------------------------------------


def test_s9_config_hash_change(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(3)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            for img in image_ids:
                await _save_all_stages(db, img, config_hash="v1")

            for img in image_ids:
                should = await db.should_run_stage(img, Stage.FILTER, config_hash="v2")
                assert should is True

            conn = db._require_db()
            for img in image_ids:
                assert await db.stage_exists(img, Stage.DETECT) is True
                for s in [Stage.FILTER, Stage.EVALUATE, Stage.REFINE, Stage.FINALIZE]:
                    assert await db.stage_exists(img, s) is False

                cur = await conn.execute(
                    "SELECT stages_completed FROM image_meta WHERE image_id = ?",
                    (img,),
                )
                row = await cur.fetchone()
                completed = json.loads(row["stages_completed"])
                assert completed == [Stage.DETECT.value]
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# S6 bug 3: submitter routing when force_stages=[filter] with all caches hot.
#   Ensures the submitter enqueues filter work (not detect) so the pipeline
#   doesn't hang at 0/N after clear_downstream wipes filter+downstream.
# ---------------------------------------------------------------------------


def test_s6_force_filter_submitter_routing(tmp_path):
    async def scenario():
        from data_miner.auto_annotation_v4.configs.loader import (
            compute_config_hash,
            load_config,
        )
        from data_miner.auto_annotation_v4.workers.submitter import JobSubmitter

        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(5)]
            image_paths = [f"/tmp/{img}.jpg" for img in image_ids]
            await db.register_image_batch(list(zip(image_ids, image_paths)))

            # Config with force_stages=[filter] — the live S6 shape.
            config = load_config(
                overrides=[
                    "runtime.stages=[detect,filter,evaluate,refine,finalize]",
                    "runtime.force_stages=[filter]",
                ]
            )
            # Seed with the same hash the submitter will compute so the
            # hash-mismatch gate (Bug 6) doesn't invalidate detect too.
            seed_hash = compute_config_hash(config, config.prompts_dir)

            # Seed all stages + both detector proposals so the cache is hot
            # before submitter runs (mirrors the live S6 setup).
            for img in image_ids:
                await db.save_proposal(
                    img, DetectorName.GROUNDING_DINO,
                    _proposal(img, DetectorName.GROUNDING_DINO),
                )
                await db.save_proposal(
                    img, DetectorName.SAM3_DART,
                    _proposal(img, DetectorName.SAM3_DART),
                )
                await _save_all_stages(db, img, config_hash=seed_hash)
                assert await db.all_stages_complete(img) is True

            conn = db._require_db()
            # Capture proposal created_at so we can later assert they weren't
            # touched by the submitter path.
            cur = await conn.execute(
                "SELECT image_id, model, created_at FROM proposals"
            )
            proposal_timestamps = {
                (r["image_id"], r["model"]): r["created_at"]
                for r in await cur.fetchall()
            }
            assert len(proposal_timestamps) == 10  # 5 imgs × 2 models

            submitter = JobSubmitter(config, db)

            submitted, total = await submitter.submit_images(
                image_paths, job_id="s6_test"
            )
            assert total == 5
            assert submitted == 5, (
                "All 5 images must be re-submitted because force_stages "
                "cleared filter+downstream for each one."
            )

            # Filter work queued for all 5 (status=pending).
            cur = await conn.execute(
                "SELECT image_id, status FROM work_queue WHERE stage = ?",
                (Stage.FILTER.value,),
            )
            rows = await cur.fetchall()
            assert {r["image_id"] for r in rows} == set(image_ids)
            for r in rows:
                assert r["status"] == WorkStatus.PENDING

            # NO pending detect work queued — neither the per-model keys
            # nor the merge barrier queue should have pending items. (The
            # original DONE detect row from seeding is fine; we're asserting
            # the submitter didn't re-enqueue detect.)
            cur = await conn.execute(
                "SELECT stage, status FROM work_queue"
                " WHERE (stage = ? OR stage LIKE 'detect:%') AND status = ?",
                (Stage.DETECT.value, WorkStatus.PENDING),
            )
            detect_pending = await cur.fetchall()
            assert detect_pending == [], (
                f"Detect path must not be re-enqueued when force_stages=[filter]. "
                f"Found pending: {[(r['stage'], r['status']) for r in detect_pending]}"
            )

            # Proposals untouched — same created_at, same count.
            cur = await conn.execute(
                "SELECT image_id, model, created_at FROM proposals"
            )
            after = {
                (r["image_id"], r["model"]): r["created_at"]
                for r in await cur.fetchall()
            }
            assert after == proposal_timestamps, (
                "Proposals must not have been rewritten by the submitter."
            )
        finally:
            await db.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# Bug 5 regression: _check_config_continuity must run BEFORE save_job_info
# so get_job_info() returns the PREVIOUS run's hash (not this run's).
# Source-level gate: assert ordering in pipeline.py::_run_locked.
# ---------------------------------------------------------------------------


def test_bug5_continuity_check_runs_before_save_job_info():
    from pathlib import Path

    pipeline_src = Path(
        "data_miner/auto_annotation_v4/pipeline.py"
    ).read_text()
    locked = pipeline_src.split("async def _run_locked")[1].split(
        "async def run_single_image"
    )[0]
    check_pos = locked.index("_check_config_continuity()")
    save_pos = locked.index("save_job_info(")
    assert check_pos < save_pos, (
        "_check_config_continuity() must be called BEFORE save_job_info() "
        "in _run_locked, so the continuity check sees the previous run's "
        "stored config_hash instead of the one we're about to write."
    )


# ---------------------------------------------------------------------------
# Bug 6 regression: submitter skip gate must honour config_hash drift.
# If a previous run completed all stages with hash=v1 and the current run
# is hash=v2, the submitter must NOT skip based on all_stages_complete
# alone — should_run_stage() detects the mismatch and clear_downstream
# invalidates the stale rows so work can be re-queued.
# ---------------------------------------------------------------------------


def test_s9_submitter_hash_gate_invalidates(tmp_path):
    async def scenario():
        db = CheckpointDB(tmp_path / "pipeline.db")
        await db.connect()
        try:
            image_ids = [f"img{i}" for i in range(3)]
            await db.register_image_batch(
                [(img, f"/tmp/{img}.jpg") for img in image_ids]
            )
            for img in image_ids:
                await _save_all_stages(db, img, config_hash="v1")
                assert await db.all_stages_complete(img) is True

            enabled_stages = list(STAGE_ORDER)

            # Replay the fixed submitter skip-gate logic with current_hash=v2.
            invalidated_map: dict[str, bool] = {}
            for img in image_ids:
                invalidated = False
                for stg in enabled_stages:
                    if await db.should_run_stage(img, stg, config_hash="v2"):
                        invalidated = True
                        break
                invalidated_map[img] = invalidated

            assert all(invalidated_map.values()), (
                f"Hash mismatch must flag every image as invalidated. "
                f"Got: {invalidated_map}"
            )

            # clear_downstream fired on the first stage (DETECT) for each
            # image, wiping every stage and flipping status to RUNNING.
            for img in image_ids:
                assert await db.all_stages_complete(img) is False, (
                    f"{img} still reports complete after invalidation"
                )
                for s in STAGE_ORDER:
                    assert await db.stage_exists(img, s) is False, (
                        f"{img}/{s.value} checkpoint should have been cleared"
                    )

            # The skip gate: "all_stages_complete AND not invalidated" must
            # be False for every image, so they proceed into the work planner.
            for img in image_ids:
                skip = (
                    await db.all_stages_complete(img)
                    and not invalidated_map[img]
                )
                assert skip is False
        finally:
            await db.close()

    _run(scenario())
