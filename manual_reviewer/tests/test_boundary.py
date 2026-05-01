"""Integration boundary tests across the manual_reviewer ↔ aa_v4 contract.

These pin behaviour at the seams that previous tests skipped:

* Pydantic round-trip from manual_reviewer's sync writer through aa_v4's async
  reader (catches contract drift — field rename, enum value rename — at test
  time rather than production).
* Stage enum string values pinned: viewer regex / DB row consumers depend on
  the exact strings.
* LS task ``data`` blob shape strictly asserted (regression for the
  ``original_width`` / ``rectanglelabels`` keys recently fixed).
* Concurrent write-then-read sanity: WAL + ``closing()`` doesn't break the
  async aa_v4 writer pattern.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    HumanCorrection,
    HumanReviewResult,
    PropagationVote,
    ReconcileResult,
    ReconciledDetection,
)
from data_miner.auto_annotation_v4.configs.enums import Stage
from manual_reviewer.pipeline_io import (
    build_task,
    iter_survivor_images,
    read_image_payload,
    write_human_review,
)
from manual_reviewer.pipeline_io.db_writer import write_reconcile_results


# ---------------------------------------------------------------------------
# 1. Round-trip via the aa_v4 async reader
# ---------------------------------------------------------------------------


def test_human_review_round_trip_via_aa_v4_reader(seeded_pipeline_db: Path) -> None:
    """``write_human_review`` (sync, manual_reviewer) → ``CheckpointDB.load_stage``
    (async, aa_v4) must round-trip every field byte-for-byte.

    This is the contract canary: if aa_v4 ever renames a field on
    ``HumanReviewResult`` or changes the JSON serialization, this fails at
    test-time instead of at production import-time.
    """
    bbox = BoundingBox(x1=0.10, y1=0.20, x2=0.40, y2=0.60)
    correction = HumanCorrection(
        candidate_id="c1",
        class_name="palletjack",
        bbox=bbox,
        source="relabeled",
        original_class="forklift",
        original_bbox=bbox,
    )
    original = HumanReviewResult(
        image_id="img_a",
        reviewer_id="alice@example.com",
        reviewed_at=1714298400.0,
        duration_seconds=42.5,
        frame_state="needs_more_review",
        corrections=[correction],
        deletions=["c2"],
        notes="some notes",
        ml_modes_used=["smart_search", "click_mask"],
        ls_completion_id=9001,
        stage_timing_ms=12.0,
    )

    write_human_review(seeded_pipeline_db, original, config_hash="h1")

    async def _load() -> HumanReviewResult | None:
        async with CheckpointDB(seeded_pipeline_db) as db:
            return await db.load_stage(
                "img_a", Stage.HUMAN_REVIEW, HumanReviewResult
            )

    reloaded = asyncio.run(_load())
    assert reloaded is not None
    # Deep equality on model_dump() catches every field, including nested
    # bbox + corrections list.
    assert reloaded.model_dump() == original.model_dump()


def test_reconcile_round_trip_via_aa_v4_reader(seeded_pipeline_db: Path) -> None:
    """Same canary for ``ReconcileResult`` — propagated detections must
    deserialize identically through ``CheckpointDB.load_stage``."""
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4)
    propagated = ReconciledDetection(
        candidate_id="prop_1",
        class_name="forklift",
        class_id=0,
        bbox=bbox,
        seed_bbox=bbox,
        mask_score=0.85,
        seed_iou=0.7,
        cluster_id="cluster_42",
        votes=[
            PropagationVote(
                image_id="img_a",
                candidate_id="c1",
                score=0.9,
                source_model="sam3_dart",
            )
        ],
    )
    original = ReconcileResult(
        image_id="img_a",
        group_id="clip_1",
        propagated=[propagated],
        rejected=[],
        propagation_strategy="image_mode_sam3_dart",
        stage_timing_ms=44.0,
    )

    written = write_reconcile_results(
        seeded_pipeline_db, [original], config_hash="h1"
    )
    assert written == 1

    async def _load() -> ReconcileResult | None:
        async with CheckpointDB(seeded_pipeline_db) as db:
            return await db.load_stage(
                "img_a", Stage.RECONCILE, ReconcileResult
            )

    reloaded = asyncio.run(_load())
    assert reloaded is not None
    assert reloaded.model_dump() == original.model_dump()


# ---------------------------------------------------------------------------
# 2. Stage enum value pinning
# ---------------------------------------------------------------------------


def test_stage_enum_values_pinned() -> None:
    """The viewer regex (``human_review|reconcile``) and SQL row writes use the
    string values directly. If aa_v4 renames these to camelCase or anything
    else, half the consumers break silently — pin the literals.
    """
    assert Stage.HUMAN_REVIEW.value == "human_review"
    assert Stage.RECONCILE.value == "reconcile"


# ---------------------------------------------------------------------------
# 3. LS task data-blob schema asserter
# ---------------------------------------------------------------------------


def test_ls_task_data_blob_strict_schema(seeded_pipeline_db: Path) -> None:
    """Pin the LS task contract: ``data`` keys, ``predictions[0].result[0]``
    shape. This is the regression test for the ``original_width`` /
    ``rectanglelabels`` LS schema.
    """
    # Patch a detect stage onto the seeded DB so ``_resolve_image_size``
    # returns the canvas dimensions (the fixture doesn't include detect).
    from data_miner.auto_annotation_v4.configs.contracts import DetectResult
    from data_miner.auto_annotation_v4.configs.enums import Stage as _Stage

    async def _patch() -> None:
        async with CheckpointDB(seeded_pipeline_db) as db:
            await db.save_stage(
                "img_a",
                _Stage.DETECT,
                DetectResult(
                    image_id="img_a",
                    image_path="/tmp/imgs/img_a.jpg",
                    image_size=[1920, 1080],
                    models_used=["sam3_dart"],
                    candidates=[],
                ),
                "h1",
            )

    asyncio.run(_patch())

    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload, job_id="testjob")
    assert task is not None

    # data blob keys
    data = task["data"]
    for key in (
        "image",
        "image_id",
        "image_size",
        "cluster_id",
        "pre_annotations_finalize",
        "review_items",
        "ghost_drops",
        "vlm_summary",
        "proposal_summary",
        "trace_excerpt",
    ):
        assert key in data, f"missing data.{key}"
    assert data["image"]
    # cluster_id may be None for an unclustered survivor — the key still must exist
    assert "cluster_id" in data

    # predictions[0].result[0] shape
    pred = task["predictions"][0]
    region = pred["result"][0]
    assert region["type"] == "rectanglelabels"
    assert region["from_name"] == "bbox"
    assert region["to_name"] == "image"
    assert region["original_width"] is not None
    assert region["original_height"] is not None

    # value keys: exactly the LS-required + rotation; nothing extra leaked
    expected_keys = {"x", "y", "width", "height", "rectanglelabels", "rotation"}
    assert set(region["value"].keys()) == expected_keys
    # x/y/width/height percentage clamp
    for k in ("x", "y", "width", "height"):
        v = region["value"][k]
        assert isinstance(v, float)
        assert 0.0 <= v <= 100.0, f"value.{k}={v} outside LS percent range"
    # rectanglelabels must be a non-empty list of strings
    rl = region["value"]["rectanglelabels"]
    assert isinstance(rl, list) and rl and all(isinstance(s, str) for s in rl)


# ---------------------------------------------------------------------------
# 4. Concurrent write-then-read sanity
# ---------------------------------------------------------------------------


def test_concurrent_write_does_not_corrupt_async_reads(
    seeded_pipeline_db: Path,
) -> None:
    """``write_human_review`` uses WAL + ``closing()``. A concurrent thread
    doing writes must not break a read happening on the main thread (no
    "database is locked" or empty result).
    """
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2)
    payload = HumanReviewResult(
        image_id="img_a",
        reviewer_id="bot",
        reviewed_at=1.0,
        corrections=[
            HumanCorrection(
                candidate_id="c1",
                class_name="forklift",
                bbox=bbox,
                source="finalize",
            )
        ],
    )

    errors: list[BaseException] = []

    def _writer() -> None:
        try:
            for _ in range(5):
                write_human_review(seeded_pipeline_db, payload, config_hash="h1")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_writer)
    t.start()

    # Read via the same path the live builder uses.
    for _ in range(5):
        rows = list(iter_survivor_images(seeded_pipeline_db))
        assert any(r["image_id"] == "img_a" for r in rows)

    t.join(timeout=10.0)
    assert not t.is_alive(), "writer thread hung — DB likely locked"
    assert not errors, f"writer errors: {errors!r}"

    # Final sanity: the row is readable and intact.
    final = read_image_payload(seeded_pipeline_db, "img_a")
    assert "human_review" in final["stages"]
    assert final["stages"]["human_review"]["reviewer_id"] == "bot"


# ---------------------------------------------------------------------------
# 5. _rewrite_yolo_label — broader coverage (happy, precondition, unknown
#    class, ambiguous_skip, dry_run). Empty/BOM cases are in test_export_to_aa_v4.
# ---------------------------------------------------------------------------


def _yolo_result(*, frame_state: str = "clean", class_name: str = "cat"):
    """Tiny duck-typed stand-in for HumanReviewResult that ``_rewrite_yolo_label``
    walks. We don't want to validate the full Pydantic shape here; the function
    only reads ``corrections[*].class_name``, ``corrections[*].bbox.{x1..y2}``,
    and ``frame_state``."""

    class _Bbox:
        x1, y1, x2, y2 = 0.10, 0.20, 0.40, 0.60

    class _C:
        pass

    _C.class_name = class_name
    _C.bbox = _Bbox()

    class _R:
        pass

    _R.corrections = [_C()]
    _R.frame_state = frame_state
    return _R()


def test_rewrite_yolo_happy_path_writes_normalized_text(tmp_path: Path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _rewrite_yolo_label

    classes = tmp_path / "classes.txt"
    classes.write_text("cat\ndog\nbird\n", encoding="utf-8")
    labels_dir = tmp_path / "labels"

    _rewrite_yolo_label(labels_dir, "img1", _yolo_result(class_name="dog"), classes)

    out = (labels_dir / "img1.txt").read_text(encoding="utf-8")
    parts = out.strip().split()
    assert parts[0] == "1"  # dog → id 1
    # YOLO is (cls, cx, cy, w, h) normalized in [0,1]
    cx, cy, w, h = (float(p) for p in parts[1:])
    assert cx == pytest.approx(0.25)  # (0.10+0.40)/2
    assert cy == pytest.approx(0.40)  # (0.20+0.60)/2
    assert w == pytest.approx(0.30)
    assert h == pytest.approx(0.40)


def test_rewrite_yolo_precondition_missing_classes_file_returns_2(tmp_path: Path) -> None:
    """``main`` must hard-fail when ``--rewrite-yolo`` and ``--labels-dir`` are
    given without ``--classes-file`` — silently writing every class as id 0
    would corrupt the dataset."""
    from manual_reviewer.scripts import export_to_aa_v4 as mod

    db = tmp_path / "pipeline.db"
    db.write_text("")  # any non-empty file — main returns before opening it
    rc = mod.main(
        [
            "--db",
            str(db),
            "--in-file",
            str(tmp_path / "missing.json"),
            "--rewrite-yolo",
            "--labels-dir",
            str(tmp_path / "labels"),
        ]
    )
    assert rc == 2


def test_rewrite_yolo_unknown_class_raises(tmp_path: Path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _rewrite_yolo_label

    classes = tmp_path / "classes.txt"
    classes.write_text("cat\ndog\n", encoding="utf-8")
    labels_dir = tmp_path / "labels"

    with pytest.raises(ValueError, match="not in"):
        _rewrite_yolo_label(
            labels_dir, "img1", _yolo_result(class_name="elephant"), classes
        )


def test_rewrite_yolo_dry_run_skips_disk(tmp_path: Path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _rewrite_yolo_label

    classes = tmp_path / "classes.txt"
    classes.write_text("cat\n", encoding="utf-8")
    labels_dir = tmp_path / "labels"

    _rewrite_yolo_label(
        labels_dir, "img1", _yolo_result(), classes, dry_run=True
    )
    # No file written, no labels dir created
    assert not (labels_dir / "img1.txt").exists()


def test_main_skips_yolo_when_frame_state_ambiguous_skip(tmp_path: Path) -> None:
    """``frame_state == 'ambiguous_skip'`` must not rewrite the YOLO label —
    that frame's labels are intentionally undecided. Drives through ``main``
    so the gate (which lives in main, not _rewrite_yolo_label) is exercised.
    """
    from manual_reviewer.scripts import export_to_aa_v4 as mod

    # Build a real pipeline.db so main() can open it.
    db_path = tmp_path / "pipeline.db"

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.save_job_info(
                job_id="j", image_dir="/tmp", config_hash="h", prompt_version="v"
            )
            await db.register_image_batch([("img1", "/tmp/img1.jpg")])

    asyncio.run(_seed())

    # Pre-existing YOLO label that must NOT be overwritten.
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    pre = labels_dir / "img1.txt"
    pre.write_text("preserve me\n", encoding="utf-8")

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")

    # Minimal LS export with frame_state=ambiguous_skip.
    in_file = tmp_path / "in.json"
    in_file.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "data": {"image_id": "img1"},
                    "predictions": [],
                    "annotations": [
                        {
                            "id": 7,
                            "completed_by": 1,
                            "lead_time": 1.0,
                            "result": [
                                {
                                    "from_name": "frame_state",
                                    "to_name": "image",
                                    "type": "choices",
                                    "value": {"choices": ["ambiguous_skip"]},
                                }
                            ],
                        }
                    ],
                }
            ]
        )
    )

    rc = mod.main(
        [
            "--db",
            str(db_path),
            "--in-file",
            str(in_file),
            "--rewrite-yolo",
            "--labels-dir",
            str(labels_dir),
            "--classes-file",
            str(classes_file),
        ]
    )
    assert rc == 0
    # Pre-existing label intact: no clobber on ambiguous_skip
    assert pre.read_text(encoding="utf-8") == "preserve me\n"


# ---------------------------------------------------------------------------
# 6. _append_trace — first-append, idempotent on same id, atomicity
# ---------------------------------------------------------------------------


def test_append_trace_first_call_creates_file(tmp_path: Path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _append_trace

    traces = tmp_path / "traces"
    result = HumanReviewResult(
        image_id="img1",
        reviewer_id="r",
        reviewed_at=10.0,
        ls_completion_id=42,
    )
    _append_trace(traces, "img1", result)

    out = traces / "img1.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["stage"] == "human_review"
    assert payload[0]["data"]["ls_completion_id"] == 42


def test_append_trace_dedups_on_same_completion_id(tmp_path: Path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _append_trace

    traces = tmp_path / "traces"
    result = HumanReviewResult(
        image_id="img1",
        reviewer_id="r",
        reviewed_at=10.0,
        ls_completion_id=42,
    )
    _append_trace(traces, "img1", result)
    _append_trace(traces, "img1", result)

    payload = json.loads((traces / "img1.json").read_text(encoding="utf-8"))
    assert len(payload) == 1


def test_append_trace_writes_full_json_list_atomically(tmp_path: Path) -> None:
    """After two distinct appends the file must contain a well-formed JSON
    list — the os.replace should not leave a half-written tmp file.
    """
    from manual_reviewer.scripts.export_to_aa_v4 import _append_trace

    traces = tmp_path / "traces"
    r1 = HumanReviewResult(
        image_id="img1", reviewer_id="r", reviewed_at=10.0, ls_completion_id=1
    )
    r2 = HumanReviewResult(
        image_id="img1", reviewer_id="r", reviewed_at=11.0, ls_completion_id=2
    )
    _append_trace(traces, "img1", r1)
    _append_trace(traces, "img1", r2)

    out = traces / "img1.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 2
    # No leftover tmp file
    assert not (traces / "img1.json.tmp").exists()


# ---------------------------------------------------------------------------
# 7. build_tasks.main — happy path
# ---------------------------------------------------------------------------


def test_build_tasks_main_writes_valid_task_json(
    seeded_pipeline_db: Path, tmp_path: Path
) -> None:
    """End-to-end: ``python -m manual_reviewer.scripts.build_tasks`` against the
    seeded pipeline.db must exit 0 and produce a JSON task array LS can
    consume.
    """
    from manual_reviewer.scripts import build_tasks as mod

    out_path = tmp_path / "tasks.json"
    rc = mod.main(["--db", str(seeded_pipeline_db), "--out-file", str(out_path)])
    assert rc == 0
    assert out_path.exists()

    tasks = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(tasks, list) and len(tasks) == 1
    task = tasks[0]
    # The task must carry data, predictions, meta
    assert "data" in task and "predictions" in task and "meta" in task
    assert task["meta"]["image_id"] == "img_a"
    # Sanity: at least one rectanglelabels region, with the LS-required keys.
    region = task["predictions"][0]["result"][0]
    assert region["type"] == "rectanglelabels"
    assert region["from_name"] == "bbox"
    assert region["to_name"] == "image"


# ---------------------------------------------------------------------------
# 8. Viewer regex — accepts human_review and reconcile (no 422 / 400)
# ---------------------------------------------------------------------------


def test_viewer_search_accepts_human_review_and_reconcile(
    seeded_pipeline_db: Path,
) -> None:
    """``/api/search?stage=human_review`` and ``stage=reconcile`` must not be
    rejected by the FastAPI Query regex. Status 422 here would mean the
    pattern was reverted to the pre-fix regex.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_miner.auto_annotation_v4.viewer.app import create_app

    app = create_app(seeded_pipeline_db.parent)
    client = TestClient(app)
    for stage in ("human_review", "reconcile"):
        resp = client.get("/api/search", params={"stage": stage})
        assert resp.status_code == 200, (
            f"stage={stage} returned {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        assert "items" in body
