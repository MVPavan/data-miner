"""Phase 3 — cross-frame static-object propagation tests.

Layered:

1. Pure-logic units: ``iou``, ``group_images``, ``build_clusters``.
2. Orchestrator: ``reconcile_group`` with a stub SAM3-DART client; covers
   the accept/reject thresholds, missing-frame iteration, and per-image
   result shape.
3. DB writer: ``write_reconcile_results`` round-trips through ``stages``
   and updates ``image_meta.stages_completed``.
4. CLI smoke: ``run_reconcile.main`` with ``Sam3HttpClient`` monkeypatched
   to a stub on the ``seeded_pipeline_db`` fixture.

No GPU, no network. The Sam3Client protocol is the single mock seam.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    FinalAnnotation,
    FinalizeResult,
    ReconciledDetection,
    ReconcileResult,
)
from data_miner.auto_annotation_v4.configs.enums import FinalAction, Stage
from manual_reviewer.pipeline_io import (
    build_task,
    read_image_payload,
    write_reconcile_results,
)
from manual_reviewer.reconcile import (
    AnnotationRef,
    DEFAULT_CLIP_REGEX,
    ImageContext,
    PropagationConfig,
    RefineResponse,
    build_clusters,
    group_images,
    iou,
    reconcile_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _ann(
    cid: str,
    klass: str,
    score: float,
    bbox: BoundingBox,
    *,
    source_model: str = "sam3_dart",
) -> FinalAnnotation:
    return FinalAnnotation(
        candidate_id=cid,
        class_name=klass,
        class_id=0,
        bbox=bbox,
        confidence=score,
        action=FinalAction.ACCEPT,
        source_model=source_model,
    )


class _StubClient:
    """Records refine() calls; returns scripted responses keyed on image_path.

    Default behaviour: echo the seed bbox at the configured score.
    """

    def __init__(
        self,
        *,
        default_score: float = 0.85,
        per_image: dict[str, RefineResponse] | None = None,
    ) -> None:
        self.default_score = default_score
        self.per_image = per_image or {}
        self.calls: list[tuple[str, BoundingBox, float]] = []

    def refine(
        self, *, image_path: str, bbox: BoundingBox, threshold: float = 0.5
    ) -> RefineResponse:
        self.calls.append((image_path, bbox, threshold))
        if image_path in self.per_image:
            return self.per_image[image_path]
        return RefineResponse(box=bbox, score=self.default_score)


# ---------------------------------------------------------------------------
# 1. Pure logic
# ---------------------------------------------------------------------------


def test_iou_identical_box() -> None:
    b = _bbox(0.1, 0.2, 0.5, 0.6)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_box() -> None:
    a = _bbox(0.0, 0.0, 0.2, 0.2)
    b = _bbox(0.8, 0.8, 1.0, 1.0)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap() -> None:
    a = _bbox(0.0, 0.0, 0.5, 0.5)
    b = _bbox(0.25, 0.25, 0.75, 0.75)
    # intersect = 0.25*0.25; union = 0.5*0.5 + 0.5*0.5 - 0.0625 = 0.4375
    assert iou(a, b) == pytest.approx(0.0625 / 0.4375)


def test_iou_zero_area_box() -> None:
    a = _bbox(0.1, 0.2, 0.1, 0.2)
    b = _bbox(0.3, 0.4, 0.5, 0.6)
    assert iou(a, b) == 0.0


def test_group_clip_id_default_regex() -> None:
    g = group_images(
        [
            ("a", "/d/clip_x_0001.jpg"),
            ("b", "/d/clip_x_0002.jpg"),
            ("c", "/d/clip_y_0001.jpg"),
            ("d", "/d/poster.jpg"),
        ],
        strategy="clip_id",
    )
    assert sorted(g["clip_x"]) == ["a", "b"]
    assert g["clip_y"] == ["c"]
    # poster.jpg falls through to image_id-keyed singleton
    assert g["d"] == ["d"]


def test_group_all_strategy_lumps_everything() -> None:
    g = group_images(
        [("a", "/x.jpg"), ("b", "/y.jpg")], strategy="all"
    )
    assert g == {"all": ["a", "b"]}


def test_group_per_image_strategy_disables_propagation() -> None:
    g = group_images(
        [("a", "/x.jpg"), ("b", "/y.jpg")], strategy="per_image"
    )
    assert g == {"a": ["a"], "b": ["b"]}


def test_group_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown grouping strategy"):
        group_images([("a", "/x.jpg")], strategy="bogus")


def test_default_clip_regex_handles_dash_prefix() -> None:
    m = DEFAULT_CLIP_REGEX.match("warehouse-cam2-frame-0123.png")
    assert m is not None
    assert m.group(1) == "warehouse-cam2-frame"


def test_build_clusters_groups_same_class_high_iou() -> None:
    b1 = _bbox(0.10, 0.10, 0.50, 0.50)
    b2 = _bbox(0.12, 0.11, 0.51, 0.49)
    refs = [
        AnnotationRef("img1", _ann("c1", "forklift", 0.9, b1)),
        AnnotationRef("img2", _ann("c2", "forklift", 0.85, b2)),
    ]
    clusters = build_clusters(refs, iou_threshold=0.5, group_id="g")
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.positive_image_ids == {"img1", "img2"}
    assert cluster.class_name == "forklift"
    canon = cluster.canonical_bbox()
    # Score-weighted average should sit between the two member boxes.
    assert b1.x1 <= canon.x1 <= b2.x1 + 1e-9
    assert b1.x2 - 1e-9 <= canon.x2 <= b2.x2


def test_build_clusters_keeps_different_class_separate() -> None:
    b = _bbox(0.10, 0.10, 0.50, 0.50)
    refs = [
        AnnotationRef("img1", _ann("c1", "forklift", 0.9, b)),
        AnnotationRef("img2", _ann("c2", "person", 0.85, b)),
    ]
    clusters = build_clusters(refs)
    assert len(clusters) == 2
    classes = {c.class_name for c in clusters}
    assert classes == {"forklift", "person"}


def test_build_clusters_one_per_image_per_cluster() -> None:
    """Two annotations from same image at high IoU never collapse into one
    cluster — that's aav4's job, not the reconciler's."""
    b1 = _bbox(0.10, 0.10, 0.50, 0.50)
    b2 = _bbox(0.11, 0.11, 0.51, 0.51)
    refs = [
        AnnotationRef("img1", _ann("c1", "forklift", 0.9, b1)),
        AnnotationRef("img1", _ann("c2", "forklift", 0.85, b2)),
    ]
    clusters = build_clusters(refs)
    assert len(clusters) == 2  # one per annotation, not merged


def test_build_clusters_low_iou_stays_separate() -> None:
    refs = [
        AnnotationRef("img1", _ann("c1", "forklift", 0.9, _bbox(0.0, 0.0, 0.2, 0.2))),
        AnnotationRef("img2", _ann("c2", "forklift", 0.85, _bbox(0.7, 0.7, 0.9, 0.9))),
    ]
    clusters = build_clusters(refs, iou_threshold=0.5)
    assert len(clusters) == 2


def test_canonical_bbox_handles_zero_confidence() -> None:
    b = _bbox(0.10, 0.10, 0.50, 0.50)
    refs = [
        AnnotationRef("img1", _ann("c1", "forklift", 0.0, b)),
        AnnotationRef("img2", _ann("c2", "forklift", 0.0, b)),
    ]
    [cluster] = build_clusters(refs)
    canon = cluster.canonical_bbox()
    # Equal members + equal weights → centroid equals the shared bbox.
    assert (canon.x1, canon.y1, canon.x2, canon.y2) == (0.10, 0.10, 0.50, 0.50)


# ---------------------------------------------------------------------------
# 2. Orchestrator: reconcile_group
# ---------------------------------------------------------------------------


def _imgs_three_frames(missing_idx: int = 2) -> list[ImageContext]:
    bbox = _bbox(0.10, 0.10, 0.50, 0.50)
    contexts = []
    for i in range(3):
        anns = [] if i == missing_idx else [_ann(f"c{i}", "forklift", 0.9, bbox)]
        contexts.append(
            ImageContext(
                image_id=f"img{i}",
                image_path=f"/data/clip_a_{i:04d}.jpg",
                final_annotations=anns,
            )
        )
    return contexts


def test_reconcile_propagates_to_missing_frame_when_sam3_confirms() -> None:
    imgs = _imgs_three_frames(missing_idx=2)
    client = _StubClient(default_score=0.9)
    results = reconcile_group("clip_a", imgs, client=client)

    assert len(results) == 3
    assert not results["img0"].propagated  # already had it
    assert not results["img1"].propagated
    propagated = results["img2"].propagated
    assert len(propagated) == 1
    det = propagated[0]
    assert det.class_name == "forklift"
    assert det.cluster_id.startswith("clip_a::")
    assert det.mask_score == pytest.approx(0.9)
    assert det.seed_iou == pytest.approx(1.0)
    assert {v.image_id for v in det.votes} == {"img0", "img1"}
    # Only one SAM3 call — for the missing frame.
    assert len(client.calls) == 1
    assert client.calls[0][0] == "/data/clip_a_0002.jpg"


def test_reconcile_rejects_when_mask_score_too_low() -> None:
    imgs = _imgs_three_frames(missing_idx=2)
    client = _StubClient(default_score=0.3)  # below default accept_score 0.5
    results = reconcile_group("clip_a", imgs, client=client)
    assert not results["img2"].propagated
    assert len(results["img2"].rejected) == 1
    assert results["img2"].rejected[0].mask_score == pytest.approx(0.3)


def test_reconcile_rejects_when_refined_box_drifts_far_from_seed() -> None:
    imgs = _imgs_three_frames(missing_idx=2)
    drifted = _bbox(0.80, 0.80, 0.95, 0.95)  # IoU vs seed ≈ 0
    client = _StubClient(
        per_image={
            "/data/clip_a_0002.jpg": RefineResponse(box=drifted, score=0.95),
        },
    )
    results = reconcile_group("clip_a", imgs, client=client)
    assert not results["img2"].propagated
    assert len(results["img2"].rejected) == 1
    assert results["img2"].rejected[0].seed_iou < 0.1


def test_reconcile_skips_when_sam3_returns_no_box() -> None:
    imgs = _imgs_three_frames(missing_idx=2)
    client = _StubClient(
        per_image={
            "/data/clip_a_0002.jpg": RefineResponse(box=None, score=0.0),
        },
    )
    results = reconcile_group("clip_a", imgs, client=client)
    # No box → not in propagated, not in rejected (silently skipped).
    assert not results["img2"].propagated
    assert not results["img2"].rejected


def test_reconcile_respects_min_positive_frames() -> None:
    bbox = _bbox(0.10, 0.10, 0.50, 0.50)
    imgs = [
        ImageContext(
            image_id="img0",
            image_path="/data/clip_a_0000.jpg",
            final_annotations=[_ann("c0", "forklift", 0.9, bbox)],
        ),
        ImageContext(
            image_id="img1",
            image_path="/data/clip_a_0001.jpg",
            final_annotations=[],
        ),
    ]
    # min_positive_frames=2 — only one frame has the box, so cluster ineligible.
    client = _StubClient(default_score=0.95)
    results = reconcile_group("clip_a", imgs, client=client, config=PropagationConfig(min_positive_frames=2))
    assert not results["img1"].propagated
    assert not client.calls

    # min_positive_frames=1 lets singletons propagate.
    client2 = _StubClient(default_score=0.95)
    results2 = reconcile_group(
        "clip_a", imgs, client=client2, config=PropagationConfig(min_positive_frames=1)
    )
    assert len(results2["img1"].propagated) == 1
    assert len(client2.calls) == 1


def test_reconcile_records_transport_error_audit_row() -> None:
    """Network/server failures must materialise a synthetic rejected row so
    operators can tell "infra failed" from "object not present" — silent
    skips bias the system toward false negatives without operator signal.
    """
    imgs = _imgs_three_frames(missing_idx=2)

    class ExplodingClient:
        def refine(self, *, image_path, bbox, threshold=0.5):
            raise RuntimeError("server died")

    results = reconcile_group("clip_a", imgs, client=ExplodingClient())
    assert not results["img2"].propagated
    assert len(results["img2"].rejected) == 1
    failed = results["img2"].rejected[0]
    assert failed.candidate_id.endswith("#transport_error")
    assert failed.mask_score == 0.0
    assert failed.seed_iou == 0.0
    # bbox falls back to the seed bbox; cluster_id and votes stay attached
    # so the audit row points at the right cluster.
    assert failed.bbox == failed.seed_bbox
    assert failed.cluster_id == "clip_a::0"
    assert failed.votes


def test_reconcile_all_frames_have_detection_no_calls() -> None:
    bbox = _bbox(0.10, 0.10, 0.50, 0.50)
    imgs = [
        ImageContext(
            image_id=f"img{i}",
            image_path=f"/data/clip_a_{i:04d}.jpg",
            final_annotations=[_ann(f"c{i}", "forklift", 0.9, bbox)],
        )
        for i in range(3)
    ]
    client = _StubClient()
    results = reconcile_group("clip_a", imgs, client=client)
    assert sum(len(r.propagated) for r in results.values()) == 0
    assert not client.calls


# ---------------------------------------------------------------------------
# 3. DB writer
# ---------------------------------------------------------------------------


def _make_reconcile_result(image_id: str, *, propagated: int = 1) -> ReconcileResult:
    bbox = _bbox(0.10, 0.10, 0.50, 0.50)
    detections = [
        ReconciledDetection(
            candidate_id=f"prop_{image_id}_{i}",
            class_name="forklift",
            class_id=0,
            bbox=bbox,
            seed_bbox=bbox,
            mask_score=0.9,
            seed_iou=0.95,
            cluster_id="clip_a::0",
        )
        for i in range(propagated)
    ]
    return ReconcileResult(
        image_id=image_id,
        group_id="clip_a",
        propagated=detections,
    )


def test_write_reconcile_results_persists_and_updates_stages_completed(
    seeded_pipeline_db: Path,
) -> None:
    result = _make_reconcile_result("img_a")
    written = write_reconcile_results(seeded_pipeline_db, [result])
    assert written == 1

    with sqlite3.connect(str(seeded_pipeline_db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT data FROM stages WHERE image_id = ? AND stage = ?",
            ("img_a", Stage.RECONCILE.value),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["data"])
        assert payload["group_id"] == "clip_a"
        assert len(payload["propagated"]) == 1

        meta = conn.execute(
            "SELECT stages_completed FROM image_meta WHERE image_id = ?",
            ("img_a",),
        ).fetchone()
        completed = json.loads(meta["stages_completed"])
        assert "reconcile" in completed


def test_write_reconcile_results_skip_empty_default(seeded_pipeline_db: Path) -> None:
    empty = ReconcileResult(image_id="img_a", group_id="clip_a")
    written = write_reconcile_results(seeded_pipeline_db, [empty])
    assert written == 0  # skipped


def test_write_reconcile_results_keep_empty_when_requested(
    seeded_pipeline_db: Path,
) -> None:
    empty = ReconcileResult(image_id="img_a", group_id="clip_a")
    written = write_reconcile_results(seeded_pipeline_db, [empty], skip_empty=False)
    assert written == 1


def test_write_reconcile_results_idempotent_on_rerun(
    seeded_pipeline_db: Path,
) -> None:
    r1 = _make_reconcile_result("img_a", propagated=1)
    r2 = _make_reconcile_result("img_a", propagated=2)
    write_reconcile_results(seeded_pipeline_db, [r1])
    write_reconcile_results(seeded_pipeline_db, [r2])

    with sqlite3.connect(str(seeded_pipeline_db)) as conn:
        rows = conn.execute(
            "SELECT data FROM stages WHERE image_id=? AND stage=?",
            ("img_a", Stage.RECONCILE.value),
        ).fetchall()
        assert len(rows) == 1  # INSERT OR REPLACE — single row
        payload = json.loads(rows[0][0])
        assert len(payload["propagated"]) == 2  # second write won

        completed = json.loads(
            conn.execute(
                "SELECT stages_completed FROM image_meta WHERE image_id=?",
                ("img_a",),
            ).fetchone()[0]
        )
        # 'reconcile' added exactly once even after two writes.
        assert completed.count("reconcile") == 1


# ---------------------------------------------------------------------------
# 4. task_builder integration
# ---------------------------------------------------------------------------


def test_task_builder_surfaces_cross_frame_suggestions(
    seeded_pipeline_db: Path,
) -> None:
    write_reconcile_results(
        seeded_pipeline_db, [_make_reconcile_result("img_a", propagated=1)]
    )
    payload = read_image_payload(seeded_pipeline_db, "img_a")
    task = build_task(payload)
    assert "cross_frame_suggestions" in task["data"]
    assert len(task["data"]["cross_frame_suggestions"]) == 1
    # Predictions list must include the cross-frame box tagged with source.
    pred_results = task["predictions"][0]["result"]
    cross_frame_preds = [r for r in pred_results if r.get("meta", {}).get("source") == "cross_frame"]
    assert len(cross_frame_preds) == 1
    assert cross_frame_preds[0]["meta"]["cluster_id"] == "clip_a::0"


# ---------------------------------------------------------------------------
# 5. CLI smoke (no real network)
# ---------------------------------------------------------------------------


def _seed_two_clip_frames(db_path: Path) -> None:
    """Build a 3-frame clip pipeline.db: img1 + img2 detected, img3 missing."""
    bbox = BoundingBox(x1=0.10, y1=0.10, x2=0.50, y2=0.50)

    async def _seed() -> None:
        async with CheckpointDB(db_path) as db:
            await db.save_job_info(
                job_id="cli-test",
                image_dir="/tmp",
                config_hash="h1",
                prompt_version="v1",
            )
            await db.register_image_batch(
                [
                    ("clip_a_001", "/data/clip_a_0001.jpg"),
                    ("clip_a_002", "/data/clip_a_0002.jpg"),
                    ("clip_a_003", "/data/clip_a_0003.jpg"),
                ]
            )
            for i, image_id in enumerate(("clip_a_001", "clip_a_002")):
                ann = FinalAnnotation(
                    candidate_id=f"c{i}",
                    class_name="forklift",
                    class_id=0,
                    bbox=bbox,
                    confidence=0.9,
                    action=FinalAction.ACCEPT,
                    source_model="sam3_dart",
                )
                await db.save_stage(
                    image_id,
                    Stage.FINALIZE,
                    FinalizeResult(image_id=image_id, final_annotations=[ann]),
                    "h1",
                )
            # img3 has no finalize annotations but did "complete" the pipeline.
            await db.save_stage(
                "clip_a_003",
                Stage.FINALIZE,
                FinalizeResult(image_id="clip_a_003", final_annotations=[]),
                "h1",
            )
            async with db._transaction() as tx:
                await tx.execute(
                    "UPDATE image_meta SET stages_completed=? "
                    "WHERE image_id IN ('clip_a_001','clip_a_002','clip_a_003')",
                    (json.dumps(["detect", "filter", "evaluate", "refine", "finalize"]),),
                )

    asyncio.run(_seed())


def test_run_reconcile_cli_writes_propagation_for_missing_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "pipeline.db"
    _seed_two_clip_frames(db_path)

    # Default backend is sam3_1 — patch Sam3OneHttpClient.
    from manual_reviewer.scripts import run_reconcile

    captured: dict[str, object] = {}

    class _CLIStub:
        def __init__(self, *args, **kwargs):
            captured["url"] = kwargs.get("url")

        def refine(self, *, image_path: str, bbox: BoundingBox, threshold: float = 0.5):
            return RefineResponse(box=bbox, score=0.95)

    monkeypatch.setattr(run_reconcile, "Sam3OneHttpClient", _CLIStub)

    rc = run_reconcile.main(
        [
            "--db",
            str(db_path),
            "--sam3-url",
            "http://stub:9999/predict",
            "--grouping",
            "clip_id",
        ]
    )
    assert rc == 0
    assert captured["url"] == "http://stub:9999/predict"

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT image_id FROM stages WHERE stage=?",
            (Stage.RECONCILE.value,),
        ).fetchall()
    image_ids = {r[0] for r in rows}
    # Only the missing-frame row gets persisted (skip_empty=True default).
    assert image_ids == {"clip_a_003"}


def test_run_reconcile_cli_backend_sam3_dart_picks_legacy_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "pipeline.db"
    _seed_two_clip_frames(db_path)

    from manual_reviewer.scripts import run_reconcile

    sam3_1_calls = {"used": False}
    sam3_dart_calls = {"used": False}

    class _OneStub:
        def __init__(self, *args, **kwargs):
            sam3_1_calls["used"] = True

        def refine(self, *, image_path, bbox, threshold=0.5):
            return RefineResponse(box=bbox, score=0.95)

    class _DartStub:
        def __init__(self, *args, **kwargs):
            sam3_dart_calls["used"] = True

        def refine(self, *, image_path, bbox, threshold=0.5):
            return RefineResponse(box=bbox, score=0.95)

    monkeypatch.setattr(run_reconcile, "Sam3OneHttpClient", _OneStub)
    monkeypatch.setattr(run_reconcile, "Sam3HttpClient", _DartStub)

    rc = run_reconcile.main(
        ["--db", str(db_path), "--backend", "sam3_dart", "--grouping", "clip_id"]
    )
    assert rc == 0
    assert sam3_dart_calls["used"] is True
    assert sam3_1_calls["used"] is False


# ---------------------------------------------------------------------------
# 6. SAM 3.1 client + wire (no GPU)
# ---------------------------------------------------------------------------


def test_sam3_one_http_client_refine_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sam3OneHttpClient.refine builds the right SAM3RefineRequest and parses
    the SAM3RefineResponse back into a BoundingBox."""
    from manual_reviewer.reconcile import Sam3OneHttpClient

    captured = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"box": [0.12, 0.13, 0.51, 0.52], "score": 0.91}

        def raise_for_status(self):
            return None

    class _FakeSession:
        def post(self, url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    client = Sam3OneHttpClient(
        url="http://stub:1234/predict", session=_FakeSession()
    )
    bbox = BoundingBox(x1=0.10, y1=0.10, x2=0.50, y2=0.50)
    result = client.refine(image_path="/img.jpg", bbox=bbox, threshold=0.5)

    assert captured["url"] == "http://stub:1234/predict"
    assert captured["json"]["image_path"] == "/img.jpg"
    assert captured["json"]["bbox"] == [0.10, 0.10, 0.50, 0.50]
    assert captured["json"]["threshold"] == 0.5

    assert result.box is not None
    assert result.box.x1 == pytest.approx(0.12)
    assert result.box.x2 == pytest.approx(0.51)
    assert result.score == pytest.approx(0.91)


def test_sam3_one_http_client_refine_handles_no_box() -> None:
    from manual_reviewer.reconcile import Sam3OneHttpClient

    class _FakeResp:
        def json(self):
            return {"box": None, "score": 0.0}

        def raise_for_status(self):
            return None

    class _FakeSession:
        def post(self, *_args, **_kwargs):
            return _FakeResp()

    client = Sam3OneHttpClient(session=_FakeSession())
    result = client.refine(
        image_path="/img.jpg",
        bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5),
    )
    assert result.box is None
    assert result.score == 0.0


def test_sam3_one_http_client_track_serializes_seeds() -> None:
    from data_miner.auto_annotation_v4.configs.wire import SAM3VideoTrackSeed
    from manual_reviewer.reconcile import Sam3OneHttpClient

    captured = {}

    class _FakeResp:
        def json(self):
            return {
                "frames": [
                    {
                        "frame_index": 0,
                        "objects": [
                            {"obj_id": 1, "bbox": [0.1, 0.1, 0.5, 0.5], "score": 0.9}
                        ],
                    },
                    {
                        "frame_index": 5,
                        "objects": [
                            {"obj_id": 1, "bbox": [0.11, 0.10, 0.51, 0.50], "score": 0.85}
                        ],
                    },
                ]
            }

        def raise_for_status(self):
            return None

    class _FakeSession:
        def post(self, url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    client = Sam3OneHttpClient(
        url="http://stub:1234/predict",
        track_url="http://stub:1234/predict",
        session=_FakeSession(),
    )
    seeds = [
        SAM3VideoTrackSeed(obj_id=1, frame_index=0, bbox=[0.1, 0.1, 0.5, 0.5]),
    ]
    resp = client.track(
        resource_path="/data/clip", seeds=seeds, propagation_direction="both"
    )

    assert captured["json"]["resource_path"] == "/data/clip"
    assert len(captured["json"]["seeds"]) == 1
    assert captured["json"]["seeds"][0]["obj_id"] == 1
    assert captured["json"]["propagation_direction"] == "both"

    assert len(resp.frames) == 2
    assert resp.frames[0].frame_index == 0
    assert resp.frames[1].objects[0].score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# 7. SAM 3.1 LitAPI server dispatch (no model load)
# ---------------------------------------------------------------------------


def test_sam3_one_api_decode_dispatches_three_modes() -> None:
    """The mode tag picked by decode_request determines which model method
    runs in predict — verify all three paths route correctly without
    instantiating SAM 3.1."""
    from data_miner.auto_annotation_v4.model_servers.sam3_1 import (
        SAM3OneApi,
        _REFINE_TAG,
        _TRACK_TAG,
        _TEXT_TAG,
    )

    api = SAM3OneApi.__new__(SAM3OneApi)  # bypass setup() / model load

    refine = api.decode_request(
        {"image_path": "/img.jpg", "bbox": [0.1, 0.2, 0.5, 0.6]}
    )
    assert refine["__mode__"] == _REFINE_TAG

    track = api.decode_request(
        {
            "resource_path": "/clip",
            "seeds": [{"obj_id": 1, "bbox": [0.1, 0.1, 0.5, 0.5]}],
        }
    )
    assert track["__mode__"] == _TRACK_TAG

    text = api.decode_request(
        {"image_path": "/img.jpg", "prompts": ["forklift", "person"]}
    )
    assert text["__mode__"] == _TEXT_TAG


def test_sam3_one_api_predict_routes_to_correct_model_method() -> None:
    """Predict should call refine/track/text_detect on the wrapped model
    based on the decoded mode tag — no actual inference needed."""
    from data_miner.auto_annotation_v4.configs.wire import SAM3VideoTrackResponse
    from data_miner.auto_annotation_v4.model_servers.sam3_1 import (
        SAM3OneApi,
        _REFINE_TAG,
        _TRACK_TAG,
        _TEXT_TAG,
    )

    class _StubModel:
        def __init__(self):
            self.calls = []

        def refine(self, **kwargs):
            self.calls.append(("refine", kwargs))
            from data_miner.auto_annotation_v4.configs.wire import SAM3RefineResponse
            return SAM3RefineResponse(box=[0, 0, 1, 1], score=0.9)

        def track(self, **kwargs):
            self.calls.append(("track", kwargs))
            return SAM3VideoTrackResponse(frames=[])

        def text_detect(self, **kwargs):
            self.calls.append(("text_detect", kwargs))
            from data_miner.auto_annotation_v4.configs.wire import DetectorResponse
            return DetectorResponse(boxes=[], scores=[], labels=[])

    api = SAM3OneApi.__new__(SAM3OneApi)
    api.model = _StubModel()

    refine_decoded = api.decode_request(
        {"image_path": "/img.jpg", "bbox": [0.1, 0.2, 0.5, 0.6]}
    )
    track_decoded = api.decode_request(
        {"resource_path": "/clip", "seeds": [{"obj_id": 1, "text": "person"}]}
    )
    text_decoded = api.decode_request(
        {"image_path": "/img.jpg", "prompts": ["forklift"]}
    )

    api.predict(refine_decoded)
    api.predict(track_decoded)
    api.predict(text_decoded)

    # All three model methods got called exactly once in order.
    methods = [c[0] for c in api.model.calls]
    assert methods == ["refine", "track", "text_detect"]
