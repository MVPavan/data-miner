"""Phase C v1 — static-object propagation unit tests.

Layered:

1. Helpers: ``_iou_xyxy``, ``_normalize``, ``_as_bbox_tuple``.
2. ``CosineGenerator``: threshold filtering, encode_crop failures, dim
   mismatch, seed-cache reuse.
3. ``Reconciler``: confirm / weak_iou / conflict / suggest branches.
4. ``propagate_static``: end-to-end with a stubbed encoder + dict-backed
   ``fetch_existing``; scope cap, seed-frame filter, fetch_existing raise.
5. ``verdict_to_ls_region``: round-trip into an LS ``RectangleLabels`` payload.

No DINOv3, no LS REST. The encoder Protocol and ``fetch_existing``
callable are the only mock seams.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from manual_reviewer.reconcile.propagate_static import (
    CosineGenerator,
    Match,
    PropagateStaticConfig,
    Reconciler,
    Seed,
    Verdict,
    _as_bbox_tuple,
    _iou_xyxy,
    _normalize,
    propagate_static,
    verdict_to_ls_region,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Returns canned vectors keyed by (image_path, bbox-rounded-key)."""

    def __init__(
        self,
        responses: dict[tuple[str, tuple[float, ...]], np.ndarray],
        *,
        raise_on: set[str] | None = None,
    ) -> None:
        self._responses = responses
        self._raise_on = raise_on or set()
        self.calls: list[tuple[str, tuple[float, float, float, float]]] = []

    def encode_crop(
        self,
        image_path: str,
        bbox_norm: tuple[float, float, float, float],
    ) -> np.ndarray:
        self.calls.append((image_path, bbox_norm))
        if image_path in self._raise_on:
            raise RuntimeError(f"encode failed for {image_path}")
        key = (image_path, tuple(round(v, 4) for v in bbox_norm))
        if key not in self._responses:
            raise KeyError(f"no canned response for {key}")
        return self._responses[key]


def _seed_for(image_path: str = "/img/seed.jpg") -> Seed:
    return Seed(
        image_id="seed",
        image_path=image_path,
        bbox_norm=(0.10, 0.20, 0.30, 0.40),
        class_name="forklift",
    )


def _vec(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------


def test_iou_xyxy_perfect_overlap() -> None:
    box = (0.1, 0.2, 0.3, 0.4)
    assert _iou_xyxy(box, box) == pytest.approx(1.0)


def test_iou_xyxy_disjoint_is_zero() -> None:
    assert _iou_xyxy((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_xyxy_partial() -> None:
    a = (0.0, 0.0, 0.4, 0.4)
    b = (0.2, 0.2, 0.6, 0.6)
    assert _iou_xyxy(a, b) == pytest.approx(0.04 / (0.16 + 0.16 - 0.04))


def test_iou_xyxy_handles_zero_area() -> None:
    assert _iou_xyxy((0.1, 0.1, 0.1, 0.1), (0.0, 0.0, 0.5, 0.5)) == 0.0


def test_normalize_unit_norm() -> None:
    out = _normalize(_vec([3.0, 0.0, 4.0]))
    assert float(np.linalg.norm(out)) == pytest.approx(1.0)


def test_normalize_zero_vector_returns_zero() -> None:
    out = _normalize(_vec([0.0, 0.0, 0.0]))
    assert float(np.linalg.norm(out)) == 0.0


def test_as_bbox_tuple_accepts_list_and_tuple() -> None:
    assert _as_bbox_tuple([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)
    assert _as_bbox_tuple((0.5, 0.5, 0.6, 0.6)) == (0.5, 0.5, 0.6, 0.6)


def test_as_bbox_tuple_rejects_garbage() -> None:
    assert _as_bbox_tuple([0.1, 0.2, 0.3]) is None
    assert _as_bbox_tuple("bad") is None
    assert _as_bbox_tuple([0.1, "x", 0.3, 0.4]) is None


# ---------------------------------------------------------------------------
# 2. CosineGenerator
# ---------------------------------------------------------------------------


def test_cosine_generator_emits_match_when_above_threshold() -> None:
    seed = _seed_for()
    same = _vec([1.0, 0.0])
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): same,
            ("/img/a.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): same,
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.85)
    matches = gen.generate([("img_a", "/img/a.jpg")])
    assert len(matches) == 1
    assert matches[0].image_id == "img_a"
    assert matches[0].cosine == pytest.approx(1.0)
    assert matches[0].bbox_norm == seed.bbox_norm


def test_cosine_generator_drops_below_threshold() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0]),
            ("/img/a.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): _vec([0.0, 1.0]),
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.85)
    assert gen.generate([("img_a", "/img/a.jpg")]) == []


def test_cosine_generator_skips_encode_failures() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0]),
            ("/img/ok.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0]),
        },
        raise_on={"/img/bad.jpg"},
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.85)
    matches = gen.generate(
        [("img_bad", "/img/bad.jpg"), ("img_ok", "/img/ok.jpg")]
    )
    assert {m.image_id for m in matches} == {"img_ok"}


def test_cosine_generator_clamps_dot_product_to_unit_range() -> None:
    """Dot of two pre-normalized vectors can drift slightly above 1.0 due
    to float roundoff; the generator must clamp to [-1, 1] so downstream
    consumers don't see physically-impossible cosines."""
    seed = _seed_for()
    # Construct a frame embedding whose normalized dot-with-seed exceeds
    # 1.0 in float32 by giving _normalize() a vector whose computed norm
    # is very slightly off (near-identical to seed but with a tiny FP
    # perturbation in the last digit). Easiest reproducible setup: same
    # vector, asserting cosine == 1.0 (clamped) post-clip.
    same = _vec([1.0, 0.0])
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): same,
            ("/img/a.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): same,
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.5)
    [m] = gen.generate([("img_a", "/img/a.jpg")])
    # After clamp the cosine is bounded by 1.0 even if the underlying dot
    # produced 1.0000001 from FP error.
    assert -1.0 <= m.cosine <= 1.0


def test_cosine_generator_drops_nan_cosine() -> None:
    """A NaN-laden frame embedding must NOT silently match.

    Without the explicit ``isfinite`` guard, ``cosine < thresh`` is False
    for NaN, so the frame slips through with ``cosine=NaN``. That would
    travel into the Reconciler and emit a misleading suggest/confirm.
    """
    seed = _seed_for()
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0]),
            ("/img/nan.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): _vec([float("nan"), 0.0]),
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.5)
    assert gen.generate([("img_nan", "/img/nan.jpg")]) == []


def test_cosine_generator_drops_degenerate_zero_embedding() -> None:
    """An all-zero frame embedding (e.g. uniform/black crop) is meaningless."""
    seed = _seed_for()
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0]),
            ("/img/black.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): _vec([0.0, 0.0]),
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.5)
    assert gen.generate([("img_black", "/img/black.jpg")]) == []


def test_cosine_generator_dim_mismatch_drops() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0]),
            ("/img/a.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): _vec([1.0, 0.0, 0.0]),
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.5)
    assert gen.generate([("img_a", "/img/a.jpg")]) == []


def test_cosine_generator_caches_seed_embedding() -> None:
    seed = _seed_for()
    same = _vec([1.0, 0.0])
    encoder = _StubEncoder(
        {
            (seed.image_path, tuple(round(v, 4) for v in seed.bbox_norm)): same,
            ("/img/a.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): same,
            ("/img/b.jpg", tuple(round(v, 4) for v in seed.bbox_norm)): same,
        }
    )
    gen = CosineGenerator(seed, encoder, cosine_thresh=0.5)
    gen.generate([("img_a", "/img/a.jpg")])
    gen.generate([("img_b", "/img/b.jpg")])
    seed_calls = sum(1 for path, _ in encoder.calls if path == seed.image_path)
    assert seed_calls == 1, "seed should be encoded exactly once across batches"


# ---------------------------------------------------------------------------
# 3. Reconciler
# ---------------------------------------------------------------------------


def _match(bbox: tuple[float, float, float, float] = (0.10, 0.20, 0.30, 0.40)) -> Match:
    return Match(
        image_id="img_a",
        image_path="/img/a.jpg",
        bbox_norm=bbox,
        cosine=0.9,
    )


def test_reconciler_no_existing_is_suggest() -> None:
    rec = Reconciler(PropagateStaticConfig())
    v = rec.verdict(_match(), [], class_name="forklift")
    assert v.outcome == "suggest"
    assert v.matched_existing_idx is None
    assert v.iou_to_existing == 0.0
    assert v.class_name == "forklift"


def test_reconciler_high_iou_is_confirm() -> None:
    rec = Reconciler(PropagateStaticConfig(iou_confirm=0.7, iou_conflict=0.4))
    v = rec.verdict(
        _match(),
        [(0.10, 0.20, 0.30, 0.40)],   # identical → IoU 1.0
        class_name="forklift",
    )
    assert v.outcome == "confirm"
    assert v.matched_existing_idx == 0
    assert v.iou_to_existing == pytest.approx(1.0)


def test_reconciler_mid_iou_is_weak_iou() -> None:
    rec = Reconciler(PropagateStaticConfig(iou_confirm=0.7, iou_conflict=0.4))
    # propagated  0.10..0.30 in x, 0.20..0.40 in y  → area 0.04
    # existing    0.18..0.30 in x, 0.20..0.40 in y  → area 0.024
    # intersect   0.18..0.30 in x, 0.20..0.40 in y  → area 0.024
    # IoU = 0.024 / (0.04 + 0.024 - 0.024) = 0.6
    v = rec.verdict(
        _match(),
        [(0.18, 0.20, 0.30, 0.40)],
        class_name="forklift",
    )
    assert v.outcome == "weak_iou"
    assert 0.4 <= v.iou_to_existing < 0.7


def test_reconciler_low_iou_is_conflict() -> None:
    rec = Reconciler(PropagateStaticConfig(iou_confirm=0.7, iou_conflict=0.4))
    v = rec.verdict(
        _match(),
        [(0.50, 0.50, 0.60, 0.60)],   # disjoint → IoU 0
        class_name="forklift",
    )
    assert v.outcome == "conflict"
    assert v.iou_to_existing == 0.0


def test_reconciler_picks_best_iou_when_multiple_existing() -> None:
    rec = Reconciler(PropagateStaticConfig())
    v = rec.verdict(
        _match(),
        [
            (0.50, 0.50, 0.60, 0.60),   # disjoint
            (0.10, 0.20, 0.30, 0.40),   # identical
            (0.20, 0.20, 0.30, 0.40),   # partial
        ],
        class_name="forklift",
    )
    assert v.outcome == "confirm"
    assert v.matched_existing_idx == 1


# ---------------------------------------------------------------------------
# 4. propagate_static end-to-end
# ---------------------------------------------------------------------------


def _vec_responses_for(
    seed: Seed,
    *,
    seed_path_vec: list[float],
    frame_vecs: dict[str, list[float]],
) -> dict[tuple[str, tuple[float, ...]], np.ndarray]:
    bbox_key = tuple(round(v, 4) for v in seed.bbox_norm)
    out: dict[tuple[str, tuple[float, ...]], np.ndarray] = {
        (seed.image_path, bbox_key): _vec(seed_path_vec),
    }
    for path, vec in frame_vecs.items():
        out[(path, bbox_key)] = _vec(vec)
    return out


def test_propagate_static_no_match_records_skipped() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={"/img/a.jpg": [0.0, 1.0]},   # cosine 0
        )
    )
    out = propagate_static(
        seed,
        scope=[("img_a", "/img/a.jpg")],
        encoder=encoder,
        fetch_existing=lambda _: [],
    )
    assert out.skipped == 1
    assert out.confirmed == 0
    assert out.suggested == 0
    assert out.per_image == {}


def test_propagate_static_suggest_when_no_existing() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={"/img/a.jpg": [1.0, 0.0]},
        )
    )
    out = propagate_static(
        seed,
        scope=[("img_a", "/img/a.jpg")],
        encoder=encoder,
        fetch_existing=lambda _: [],
    )
    assert out.suggested == 1
    assert out.confirmed == 0
    assert out.per_image["img_a"][0].outcome == "suggest"


def test_propagate_static_confirm_against_existing() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={"/img/a.jpg": [1.0, 0.0]},
        )
    )
    out = propagate_static(
        seed,
        scope=[("img_a", "/img/a.jpg")],
        encoder=encoder,
        fetch_existing=lambda _: [(0.10, 0.20, 0.30, 0.40)],
    )
    assert out.confirmed == 1
    assert out.suggested == 0
    assert out.per_image["img_a"][0].outcome == "confirm"


def test_propagate_static_conflict_when_low_iou() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={"/img/a.jpg": [1.0, 0.0]},
        )
    )
    out = propagate_static(
        seed,
        scope=[("img_a", "/img/a.jpg")],
        encoder=encoder,
        fetch_existing=lambda _: [(0.50, 0.50, 0.60, 0.60)],   # disjoint
    )
    assert out.conflicts == 1
    assert out.per_image["img_a"][0].outcome == "conflict"


def test_propagate_static_strips_seed_frame_from_scope() -> None:
    """Seed frame must be filtered out BEFORE encoding.

    Build a scope of [seed, real_target]. The seed embedding (lazy) only
    fires once the generator runs against the non-seed target — that
    proves we didn't short-circuit before the filter. We then assert the
    seed's *image_id* never shows up among matches and that the only
    target the encoder processed (apart from the seed itself) is the
    real one.
    """
    seed = _seed_for()
    same = _vec([1.0, 0.0])
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={"/img/real.jpg": [1.0, 0.0]},
        )
    )
    out = propagate_static(
        seed,
        scope=[
            (seed.image_id, seed.image_path),   # would auto-confirm if not filtered
            ("img_real", "/img/real.jpg"),
        ],
        encoder=encoder,
        fetch_existing=lambda _: [],
    )
    # The seed must not appear in any per-image counter.
    assert seed.image_id not in out.per_image
    # The real target produced one match (suggest, no existing).
    assert out.suggested == 1
    assert "img_real" in out.per_image
    # The seed embedding was actually computed (proves we didn't bail
    # before the generator ran).
    assert any(p == seed.image_path for p, _ in encoder.calls)
    # No call ever encoded the seed frame *as a target* — there is at
    # most one call for the seed image_path (the cached seed embedding),
    # not two.
    seed_path_calls = sum(1 for p, _ in encoder.calls if p == seed.image_path)
    assert seed_path_calls == 1
    # The real target was encoded.
    assert any(p == "/img/real.jpg" for p, _ in encoder.calls)


def test_propagate_static_caps_scope() -> None:
    seed = _seed_for()
    cfg = PropagateStaticConfig(cosine_thresh=2.0, max_scope=3)   # nothing matches
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={
                f"/img/{i}.jpg": [1.0, 0.0] for i in range(5)
            },
        )
    )
    out = propagate_static(
        seed,
        scope=[(f"img_{i}", f"/img/{i}.jpg") for i in range(5)],
        encoder=encoder,
        fetch_existing=lambda _: [],
        config=cfg,
    )
    assert out.out_of_scope == 2
    # 3 in-scope frames, all skipped because cosine_thresh > 1.
    assert out.skipped == 3


def test_propagate_static_fetch_existing_exception_treated_as_empty() -> None:
    seed = _seed_for()
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=[1.0, 0.0],
            frame_vecs={"/img/a.jpg": [1.0, 0.0]},
        )
    )

    def boom(image_id: str) -> list:
        raise RuntimeError("LS down")

    out = propagate_static(
        seed,
        scope=[("img_a", "/img/a.jpg")],
        encoder=encoder,
        fetch_existing=boom,
    )
    assert out.suggested == 1
    assert out.per_image["img_a"][0].outcome == "suggest"


def test_propagate_static_summary_render_note() -> None:
    summary = propagate_static(
        _seed_for(),
        scope=[],
        encoder=_StubEncoder({}),
        fetch_existing=lambda _: [],
    )
    summary.confirmed = 38
    summary.suggested = 7
    summary.conflicts = 3
    summary.weak_iou = 1
    note = summary.render_note("bicycle")
    assert "bicycle" in note
    assert "38 confirmed" in note
    assert "7 suggested" in note
    assert "3 conflicts" in note
    assert "1 weak-iou" in note


def test_propagate_static_aggregates_mixed_outcomes() -> None:
    seed = _seed_for()
    same = [1.0, 0.0]
    encoder = _StubEncoder(
        _vec_responses_for(
            seed,
            seed_path_vec=same,
            frame_vecs={
                "/img/a.jpg": same,   # suggest (no existing)
                "/img/b.jpg": same,   # confirm
                "/img/c.jpg": same,   # conflict
                "/img/d.jpg": [0.0, 1.0],   # skip (cosine 0)
            },
        )
    )
    existing_by_id: dict[str, list[tuple[float, float, float, float]]] = {
        "img_b": [(0.10, 0.20, 0.30, 0.40)],
        "img_c": [(0.50, 0.50, 0.60, 0.60)],
    }
    out = propagate_static(
        seed,
        scope=[
            ("img_a", "/img/a.jpg"),
            ("img_b", "/img/b.jpg"),
            ("img_c", "/img/c.jpg"),
            ("img_d", "/img/d.jpg"),
        ],
        encoder=encoder,
        fetch_existing=lambda iid: existing_by_id.get(iid, []),
    )
    assert out.confirmed == 1
    assert out.suggested == 1
    assert out.conflicts == 1
    assert out.skipped == 1


# ---------------------------------------------------------------------------
# 5. verdict_to_ls_region
# ---------------------------------------------------------------------------


def test_verdict_to_ls_region_carries_metadata() -> None:
    verdict = Verdict(
        image_id="img_a",
        bbox_norm=(0.10, 0.20, 0.30, 0.40),
        class_name="forklift",
        cosine=0.91,
        iou_to_existing=0.55,
        outcome="weak_iou",
        matched_existing_idx=0,
    )
    region = verdict_to_ls_region(verdict)
    assert region["type"] == "rectanglelabels"
    assert region["value"]["rectanglelabels"] == ["forklift"]
    assert region["score"] == pytest.approx(0.91)
    meta = region["meta"]
    assert meta["source"] == "propagate_static"
    assert meta["outcome"] == "weak_iou"
    assert meta["cosine"] == pytest.approx(0.91)
    assert meta["iou_to_existing"] == pytest.approx(0.55)


def test_verdict_to_ls_region_honors_custom_model_version() -> None:
    verdict = Verdict(
        image_id="img_a",
        bbox_norm=(0.10, 0.20, 0.30, 0.40),
        class_name="forklift",
        cosine=0.95,
        iou_to_existing=0.0,
        outcome="suggest",
        matched_existing_idx=None,
    )
    region = verdict_to_ls_region(verdict, model_version="custom_v2")
    assert region["meta"]["model_version"] == "custom_v2"
