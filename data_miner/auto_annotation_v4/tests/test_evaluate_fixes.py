"""Tests for the evaluate-stage fixes landed with VLM-review pass:

  C5  — VLMVerdict mirror-back bug (legacy 0.0 confidence was clobbered).
  C3  — draw_focus_on_image edge-touching bbox regression (inward fallback).
  C4  — pil_to_data_url JPEG + configurable max_size / format / quality.
  C1  — _resolve_verdicts split bbox thresholds from class thresholds.
  H2  — VLM DropReason categories attached to rejects.
  H5  — VLMVerdict parse failure → VLM_MALFORMED drop.
  MED — BboxQuality removed; old rows still load (extra='ignore' discards).
"""

from __future__ import annotations

from PIL import Image

from data_miner.auto_annotation_v4.configs import (
    BoundingBox,
    Candidate,
    DropReason,
    VLMVerdict,
)
from data_miner.auto_annotation_v4.utils import (
    draw_focus_on_image,
    pil_to_data_url,
)


def _mk_cand(x1: float, y1: float, x2: float, y2: float) -> Candidate:
    return Candidate(
        candidate_id="cx",
        class_name="head",
        label="head",
        source_model="sam3_dart",
        expression="head",
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        score=0.8,
    )


# ---------------------------------------------------------------------------
# C5 — mirror-back: legitimate 0.0 confidence must not be overwritten
# ---------------------------------------------------------------------------


def test_vlm_verdict_legacy_zero_confidence_preserved():
    """Pre-fix bug: ``data.get('confidence') in (None, 0.0)`` treated a real
    legacy ``confidence=0.0`` (hard reject) as unset and silently mirrored
    a v2 non-zero ``class_confidence`` over it. Key-presence check fixes it.
    """
    v = VLMVerdict.model_validate({
        "candidate_id": "a",
        "correct_class": "forklift",
        "confidence": 0.0,          # real VLM reject, must be preserved
        # also provide v2 field — mirror-back must NOT overwrite the 0.0
        "class_confidence": 0.75,
    })
    # v1 field stays at its explicit value
    assert v.confidence == 0.0
    # v2 field carries its own payload value
    assert v.class_confidence == 0.75


def test_vlm_verdict_legacy_empty_correct_class_preserved():
    v = VLMVerdict.model_validate({
        "candidate_id": "b",
        "correct_class": "",
        "detected_class": "truck",
    })
    assert v.correct_class == ""
    assert v.detected_class == "truck"


def test_vlm_verdict_v1_only_row_migrates_to_v2():
    """Old DB row with only v1 fields must still load and populate v2."""
    v = VLMVerdict.model_validate({
        "candidate_id": "c",
        "correct_class": "head",
        "confidence": 0.7,
        "bbox_quality": "good",      # stripped (extra='ignore')
    })
    assert v.detected_class == "head"
    assert v.class_confidence == 0.7
    assert not hasattr(v, "bbox_quality")


def test_vlm_verdict_clamps_out_of_range():
    v = VLMVerdict.model_validate({
        "candidate_id": "d",
        "detected_class": "car",
        "class_confidence": 1.5,
        "bbox_score": -0.3,
    })
    assert v.class_confidence == 1.0
    assert v.bbox_score == 0.0


def test_vlm_verdict_none_coerced_to_defaults():
    """None → conservative defaults. bbox_score goes to 0.0 (not 1.0) so
    a VLM that omits / nulls the field does NOT silently pass the bbox
    gate — a truncated/refused response reads as "unusable" and rejects."""
    v = VLMVerdict.model_validate({
        "candidate_id": "e",
        "detected_class": "other",
        "class_confidence": None,
        "bbox_score": None,
    })
    assert v.class_confidence == 0.0
    assert v.bbox_score == 0.0


def test_vlm_verdict_missing_bbox_score_defaults_to_reject():
    """Truncated VLM response omits bbox_score entirely — field default
    should read as 'unusable' so the stricter bbox gate rejects rather
    than accepts."""
    v = VLMVerdict.model_validate({
        "candidate_id": "f",
        "detected_class": "forklift",
        "class_confidence": 0.9,
        # bbox_score intentionally absent
    })
    assert v.bbox_score == 0.0


# ---------------------------------------------------------------------------
# C3 — draw_focus_on_image: edge-touching bbox gets full outward border
# ---------------------------------------------------------------------------


def test_edge_touching_bbox_has_full_outward_border():
    """Pre-fix bug: bbox at (0,0) had left/top border silently falling back
    to inward draw (PIL ``width=N`` at col=0 grows inward), re-occluding
    object pixels. Post-fix: canvas is pre-padded so outward draw has room.
    """
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    cand = _mk_cand(0.0, 0.0, 0.2, 0.2)
    out = draw_focus_on_image(img, cand)
    # Canvas padded: size grew from 100x100 to ~104x104 (stroke=1 + pad=1+1)
    assert out.size[0] > 100 and out.size[1] > 100
    px = out.load()
    # Outward border lands at row=1 and col=1 (pad=2, stroke=1 → rectangle
    # origin at (1, 1)) so red must be visible on both.
    red = (255, 0, 0)
    assert any(px[x, 1] == red for x in range(out.size[0])), "top border missing"
    assert any(px[1, y] == red for y in range(out.size[1])), "left border missing"
    # Bbox interior (shifted by pad=2) must be unoccluded grey — the
    # previous inward-fallback bug would have made this red.
    assert px[12, 12] == (128, 128, 128), "bbox interior occluded by border"


def test_bottom_right_corner_bbox_has_full_border():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    cand = _mk_cand(0.8, 0.8, 1.0, 1.0)
    out = draw_focus_on_image(img, cand)
    w, h = out.size
    px = out.load()
    red = (255, 0, 0)
    assert any(px[x, h - 2] == red for x in range(w)), "bottom border missing"
    assert any(px[w - 2, y] == red for y in range(h)), "right border missing"


def test_degenerate_bbox_does_not_raise():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    out = draw_focus_on_image(img, _mk_cand(0.5, 0.5, 0.5, 0.5))
    assert out.size[0] >= 100 and out.size[1] >= 100


# ---------------------------------------------------------------------------
# C4 — pil_to_data_url JPEG default + configurable
# ---------------------------------------------------------------------------


def test_pil_to_data_url_defaults_to_jpeg():
    img = Image.new("RGB", (200, 200), (50, 80, 120))
    url = pil_to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")


def test_pil_to_data_url_png_still_available():
    img = Image.new("RGB", (200, 200), (50, 80, 120))
    url = pil_to_data_url(img, fmt="PNG")
    assert url.startswith("data:image/png;base64,")


def test_pil_to_data_url_respects_max_size():
    img = Image.new("RGB", (2000, 2000), (100, 100, 100))
    small = pil_to_data_url(img, max_size=640)
    large = pil_to_data_url(img, max_size=1600)
    # Smaller cap produces smaller payload (same image, both JPEG).
    assert len(small) < len(large)


# ---------------------------------------------------------------------------
# C1 + H2 — _resolve_verdicts split bbox thresholds + typed drops
# ---------------------------------------------------------------------------


def _run_resolve(verdicts: list[VLMVerdict], cands: list[Candidate],
                 *, class_accept: float, class_reject: float,
                 bbox_accept: float, bbox_reject: float):
    """Drive EvaluateWorker._resolve_verdicts without constructing a worker.

    We call the unbound method with a tiny Object-stub that carries the
    three attributes it uses: ``config.evaluate`` (thresholds) and
    ``alias_map`` (canonical-class lookup).
    """
    from data_miner.auto_annotation_v4.stages.evaluate import EvaluateWorker

    class _EvalCfg:
        accept_above = class_accept
        reject_below = class_reject
        bbox_accept_above = bbox_accept
        bbox_reject_below = bbox_reject

    class _Cfg:
        evaluate = _EvalCfg()

    class _Self:
        config = _Cfg()
        # identity alias map: {'head': 'head', 'forklift': 'forklift', ...}
        alias_map = {c.class_name: c.class_name for c in cands}

    return EvaluateWorker._resolve_verdicts(_Self(), verdicts, cands)


def _mk_verdict(cid: str, detected: str, class_conf: float, bbox: float) -> VLMVerdict:
    return VLMVerdict(
        candidate_id=cid,
        detected_class=detected,
        class_confidence=class_conf,
        bbox_score=bbox,
    )


def test_bbox_unusable_attaches_typed_drop_reason():
    """Class-correct, HIGH conf, BAD bbox → reject with VLM_BBOX_UNUSABLE."""
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4)
    cand = cand.model_copy(update={"candidate_id": "v1"})
    v = _mk_verdict("v1", "head", class_conf=0.9, bbox=0.2)
    accepted, review, rejected, relabels, drops = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert rejected == ["v1"]
    assert len(drops) == 1 and drops[0].candidate_id == "v1"
    assert drops[0].reason == DropReason.VLM_BBOX_UNUSABLE


def test_low_confidence_attaches_typed_drop():
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(update={"candidate_id": "v2"})
    v = _mk_verdict("v2", "head", class_conf=0.1, bbox=0.9)
    _, _, rejected, _, drops = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert rejected == ["v2"]
    assert drops[0].reason == DropReason.VLM_LOW_CONFIDENCE


def test_other_class_attaches_typed_drop():
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(update={"candidate_id": "v3"})
    v = _mk_verdict("v3", "other", class_conf=0.2, bbox=0.9)
    _, _, rejected, _, drops = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert rejected == ["v3"]
    assert drops[0].reason == DropReason.VLM_OTHER_CLASS


def test_unknown_class_attaches_typed_drop():
    """VLM returns a string that can't resolve via alias_map."""
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(update={"candidate_id": "v4"})
    v = _mk_verdict("v4", "spaceship", class_conf=0.9, bbox=0.9)
    _, _, rejected, _, drops = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert rejected == ["v4"]
    assert drops[0].reason == DropReason.VLM_UNKNOWN_CLASS


def test_split_bbox_thresholds_allow_independent_tuning():
    """Class ≥ class_accept, bbox in [bbox_reject, bbox_accept) → review,
    NOT accept (would have auto-accepted when thresholds were mirrored)."""
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(update={"candidate_id": "v5"})
    v = _mk_verdict("v5", "head", class_conf=0.9, bbox=0.55)
    accepted, review, rejected, _, drops = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert review == ["v5"]
    assert accepted == []
    assert rejected == []
    assert drops == []  # review carries no drop


def test_split_thresholds_independent_of_class_floor():
    """Old code aliased bbox_accept = class_accept. The split lets bbox be
    stricter: class 0.9 trusted, but bbox 0.72 is below bbox_accept=0.8."""
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(update={"candidate_id": "v6"})
    v = _mk_verdict("v6", "head", class_conf=0.9, bbox=0.72)
    accepted, review, rejected, _, _ = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.8, bbox_reject=0.5,
    )
    assert review == ["v6"]
    assert accepted == []


def test_all_gates_accept_path():
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(update={"candidate_id": "v7"})
    v = _mk_verdict("v7", "head", class_conf=0.95, bbox=0.95)
    accepted, review, rejected, _, _ = _run_resolve(
        [v], [cand], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert accepted == ["v7"]


def test_relabel_on_class_mismatch_with_high_conf():
    cand = _mk_cand(0.2, 0.2, 0.4, 0.4).model_copy(
        update={"candidate_id": "v8", "class_name": "palletjack"}
    )
    # Add 'forklift' to the alias map via a second cand so the identity
    # map picks it up.
    other = _mk_cand(0.6, 0.6, 0.7, 0.7).model_copy(
        update={"candidate_id": "vOther", "class_name": "forklift"}
    )
    v = _mk_verdict("v8", "forklift", class_conf=0.95, bbox=0.9)
    accepted, review, rejected, relabels, _ = _run_resolve(
        [v], [cand, other], class_accept=0.7, class_reject=0.4,
        bbox_accept=0.7, bbox_reject=0.4,
    )
    assert accepted == ["v8"]
    assert relabels == {"v8": "forklift"}
