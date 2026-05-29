"""Shared utility functions for auto_annotation_v4.

Covers: bbox math, geometric filtering, dedup, cross-class routing,
image manipulation, YOLO export, class alias resolution, logging, and
robust VLM JSON parsing.

All bbox operations accept both BoundingBox Pydantic objects (with .x1 etc.
attributes) and plain dicts (with "x1" etc. keys).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Union

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

BboxLike = Union["BoundingBox", dict]  # noqa: F821  (contract imported at runtime)


def _x1(b: BboxLike) -> float:
    return b.x1 if hasattr(b, "x1") else b["x1"]


def _y1(b: BboxLike) -> float:
    return b.y1 if hasattr(b, "y1") else b["y1"]


def _x2(b: BboxLike) -> float:
    return b.x2 if hasattr(b, "x2") else b["x2"]


def _y2(b: BboxLike) -> float:
    return b.y2 if hasattr(b, "y2") else b["y2"]


# ---------------------------------------------------------------------------
# Bbox math
# ---------------------------------------------------------------------------


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


def bbox_iou(a: BboxLike, b: BboxLike) -> float:
    """Compute IoU between two BoundingBox objects or dicts with x1,y1,x2,y2."""
    ix1 = max(_x1(a), _x1(b))
    iy1 = max(_y1(a), _y1(b))
    ix2 = min(_x2(a), _x2(b))
    iy2 = min(_y2(a), _y2(b))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def bbox_to_pixels(
    bbox: BboxLike, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    """Convert normalized bbox to pixel coords (x1, y1, x2, y2)."""
    return (
        int(clamp(_x1(bbox)) * image_w),
        int(clamp(_y1(bbox)) * image_h),
        int(clamp(_x2(bbox)) * image_w),
        int(clamp(_y2(bbox)) * image_h),
    )


def pixels_to_bbox(
    x1: int, y1: int, x2: int, y2: int, image_w: int, image_h: int
) -> dict:
    """Convert pixel coords to normalized bbox dict."""
    return {
        "x1": clamp(x1 / image_w),
        "y1": clamp(y1 / image_h),
        "x2": clamp(x2 / image_w),
        "y2": clamp(y2 / image_h),
    }


def bbox_area(bbox: BboxLike) -> float:
    """Return the normalized area of a bbox."""
    w = max(0.0, _x2(bbox) - _x1(bbox))
    h = max(0.0, _y2(bbox) - _y1(bbox))
    return w * h


def bbox_aspect_ratio(bbox: BboxLike) -> float:
    """Return width/height. Returns 0.0 if height is zero."""
    h = max(0.0, _y2(bbox) - _y1(bbox))
    w = max(0.0, _x2(bbox) - _x1(bbox))
    return w / h if h > 0 else 0.0


# ---------------------------------------------------------------------------
# Geometric filtering (ported from v2 filtering.py)
# ---------------------------------------------------------------------------


def passes_area_filter(bbox: BboxLike, min_area: float, max_area: float) -> bool:
    """Return True if bbox area is within [min_area, max_area]."""
    area = bbox_area(bbox)
    return min_area <= area <= max_area


def passes_aspect_ratio_filter(
    bbox: BboxLike, min_ratio: float, max_ratio: float
) -> bool:
    """Return True if bbox aspect ratio is within [min_ratio, max_ratio]."""
    ar = bbox_aspect_ratio(bbox)
    if ar <= 0:
        return False
    return min_ratio <= ar <= max_ratio


def passes_edge_distance_filter(bbox: BboxLike, min_dist: float) -> bool:
    """Check bbox is at least min_dist from image edges (normalized coords)."""
    if min_dist <= 0:
        return True
    d = min_dist
    return (
        _x1(bbox) >= d
        and _y1(bbox) >= d
        and _x2(bbox) <= (1.0 - d)
        and _y2(bbox) <= (1.0 - d)
    )


def geometric_filter(candidates: list, config: Any) -> list:
    """Apply area, aspect ratio, and edge distance filters.

    config is expected to have a .filtering attribute with min_area, max_area,
    min_aspect_ratio, max_aspect_ratio, and min_edge_distance fields
    (matches AutoAnnotationV4Config.filtering / FilterConfig). An optional
    ``per_class_min_area`` dict overrides ``min_area`` for listed classes;
    max_area and aspect ratio stay uniform.

    Returns the list of candidates that pass all filters.
    """
    logger = get_logger("utils.geometric_filter")
    cfg = config.filtering
    per_class_min = getattr(cfg, "per_class_min_area", {}) or {}
    passed = []
    for cand in candidates:
        bbox = cand.bbox
        min_area = per_class_min.get(cand.class_name, cfg.min_area)
        if not passes_area_filter(bbox, min_area, cfg.max_area):
            logger.debug(
                "Filtered %s: area=%.6f not in [%.6f, %.6f]",
                cand.candidate_id,
                bbox_area(bbox),
                min_area,
                cfg.max_area,
            )
            continue
        if not passes_aspect_ratio_filter(
            bbox, cfg.min_aspect_ratio, cfg.max_aspect_ratio
        ):
            logger.debug(
                "Filtered %s: aspect_ratio=%.3f not in [%.3f, %.3f]",
                cand.candidate_id,
                bbox_aspect_ratio(bbox),
                cfg.min_aspect_ratio,
                cfg.max_aspect_ratio,
            )
            continue
        if not passes_edge_distance_filter(bbox, cfg.min_edge_distance):
            logger.debug("Filtered %s: too close to image edge", cand.candidate_id)
            continue
        passed.append(cand)

    logger.info(
        "Geometric filtering: %d → %d candidates", len(candidates), len(passed)
    )
    return passed


# ---------------------------------------------------------------------------
# Source-model allowlist
# ---------------------------------------------------------------------------


def filter_by_source_model(candidates: list, allowed: list[str]) -> list:
    """Drop candidates whose ``source_model`` is not in ``allowed``.

    Empty/None ``allowed`` is treated as "allow all" so this is a no-op
    in default configs. Called both at the merge boundary and as the first
    step of every ``FilterPipeline`` plan, so flipping the allowlist in
    config only requires a filter-stage re-run to take effect.
    """
    if not allowed:
        return list(candidates)
    allow = set(allowed)
    return [c for c in candidates if c.source_model in allow]


# ---------------------------------------------------------------------------
# Per-model score filtering
# ---------------------------------------------------------------------------


def filter_by_model_score(candidates: list, per_model_score: dict[str, float]) -> list:
    """Drop candidates below their model's score floor.

    Args:
        candidates: list of objects with ``.source_model`` and ``.score``.
        per_model_score: mapping ``{model_name: min_score}``. Models absent
            from the map are not filtered (all their candidates pass).

    Returns:
        Filtered list of candidates that meet their model's floor.
    """
    if not per_model_score:
        return list(candidates)

    logger = get_logger("utils.filter_by_model_score")
    passed = []
    for cand in candidates:
        floor = per_model_score.get(cand.source_model)
        if floor is not None and cand.score < floor:
            logger.debug(
                "Filtered %s: score=%.4f < %s floor=%.4f",
                cand.candidate_id,
                cand.score,
                cand.source_model,
                floor,
            )
            continue
        passed.append(cand)

    logger.info(
        "Per-model score filtering: %d → %d candidates",
        len(candidates),
        len(passed),
    )
    return passed


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def cluster_and_collapse(candidates: list, iou_dedup_cfg: Any) -> list:
    """Per-class IoU clustering + tiebreak cascade — replaces dedup + agreement.

    For each class, builds connected components over the IoU>=threshold graph
    across ALL detector models (no per-model grouping). Each component is one
    physical object; its agreement metadata (count + sorted list of distinct
    source models) is computed during clustering and attached to the survivor.

    The survivor is chosen by walking ``iou_dedup_cfg.tiebreak_by`` in order
    and applying the first discriminator that distinguishes a single winner:

    - ``agreement``     — most distinct source_models wins.
    - ``model_priority`` — earliest in ``iou_dedup_cfg.model_priority`` wins
      (lower index = higher trust).
    - ``score``         — highest raw score wins (last-resort fallback).

    Inputs:
      candidates: list of objects with .class_name, .source_model, .score, .bbox
      iou_dedup_cfg: an IouDedupConfig (or duck-typed equivalent) with
        ``threshold``, ``tiebreak_by``, ``model_priority``.

    Returns:
      list of survivors; each has ``agreement`` and ``agreeing_models`` set.
    """
    threshold = iou_dedup_cfg.threshold
    contain_min = float(
        getattr(iou_dedup_cfg, "same_class_containment_min", 0.0) or 0.0
    )
    tiebreak_by = list(iou_dedup_cfg.tiebreak_by)
    priority_index = {m: i for i, m in enumerate(iou_dedup_cfg.model_priority)}
    fallback_priority = len(iou_dedup_cfg.model_priority)  # unknown models last

    if (threshold <= 0 and contain_min <= 0) or not candidates:
        # Still attach singleton agreement metadata so downstream is consistent.
        for c in candidates:
            c.agreement = 1
            c.agreeing_models = [c.source_model]
        return list(candidates)

    logger = get_logger("utils.cluster_and_collapse")

    # Group by class.
    by_class: dict[str, list] = {}
    for c in candidates:
        by_class.setdefault(c.class_name, []).append(c)

    survivors: list = []

    for class_name, group in by_class.items():
        # Union-find over IoU >= threshold within this class.
        n = len(group)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = group[i].bbox, group[j].bbox
                if threshold > 0 and bbox_iou(bi, bj) >= threshold:
                    _union(i, j)
                    continue
                # Containment fallback: catches nested pairs that IoU misses
                # (small-in-large can have IoU ~0.4 while containment ~1.0).
                if contain_min > 0:
                    ai = max(0.0, bi.x2 - bi.x1) * max(0.0, bi.y2 - bi.y1)
                    aj = max(0.0, bj.x2 - bj.x1) * max(0.0, bj.y2 - bj.y1)
                    if ai <= 0 or aj <= 0:
                        continue
                    ix1, iy1 = max(bi.x1, bj.x1), max(bi.y1, bj.y1)
                    ix2, iy2 = min(bi.x2, bj.x2), min(bi.y2, bj.y2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    if inter > 0 and inter / min(ai, aj) >= contain_min:
                        _union(i, j)

        # Bucket members by cluster root.
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(_find(i), []).append(i)

        for member_idxs in clusters.values():
            members = [group[i] for i in member_idxs]
            agreeing_models = sorted({m.source_model for m in members})
            agreement = len(agreeing_models)

            # Cascade tiebreak: filter winners by each discriminator until 1 remains.
            pool = list(members)
            for key in tiebreak_by:
                if len(pool) == 1:
                    break
                if key == "agreement":
                    # Within a cluster, every member shares the same agreement
                    # value (cluster-level metadata) — this discriminator only
                    # distinguishes across clusters, so it's a no-op here. Kept
                    # for symmetry / future cross-cluster ranking use.
                    continue
                elif key == "model_priority":
                    best = min(
                        priority_index.get(m.source_model, fallback_priority)
                        for m in pool
                    )
                    pool = [
                        m
                        for m in pool
                        if priority_index.get(m.source_model, fallback_priority)
                        == best
                    ]
                elif key == "score":
                    best_score = max(m.score for m in pool)
                    pool = [m for m in pool if m.score == best_score]
                else:
                    raise ValueError(f"Unknown tiebreak discriminator: {key!r}")

            # Stable final fallback: take the first remaining (lexical by id).
            if len(pool) > 1:
                pool.sort(key=lambda m: m.candidate_id)
            survivor = pool[0]
            survivor.agreement = agreement
            survivor.agreeing_models = agreeing_models
            survivors.append(survivor)

        if len(clusters) < n:
            logger.debug(
                "Cluster-and-collapse class '%s': %d → %d (%d clusters)",
                class_name,
                n,
                len(clusters),
                len(clusters),
            )

    return survivors


def limit_per_class(candidates: list, max_per_class: int = 30) -> list:
    """Cap candidates per class, keeping the highest-scoring ones."""
    if max_per_class <= 0:
        return list(candidates)

    logger = get_logger("utils.limit_per_class")
    by_class: dict[str, list] = {}
    for c in candidates:
        by_class.setdefault(c.class_name, []).append(c)

    result: list = []
    for class_name, group in by_class.items():
        sorted_group = sorted(group, key=lambda c: c.score, reverse=True)
        if len(sorted_group) > max_per_class:
            logger.info(
                "Class '%s': limited from %d to %d candidates",
                class_name,
                len(sorted_group),
                max_per_class,
            )
        result.extend(sorted_group[:max_per_class])
    return result


# ---------------------------------------------------------------------------
# Tag-based co-existence resolution
# ---------------------------------------------------------------------------


def _resolve_co_existence(config: Any) -> tuple[set[str], list[frozenset]]:
    """Resolve tag-based co-existence rules into exempt set + confusion pairs.

    Uses ``config.co_existence`` (overlap_exempt_tags, confusion_tags,
    extra_confusion_pairs) and ``config.class_registry`` (dict[str, ClassConfig]
    with ``.tags`` lists) to compute:

    Returns:
        (exempt_set, confusion_pairs) where exempt_set is class names with any
        overlap_exempt tag and confusion_pairs is list of frozenset({a, b}).
    """
    co = config.co_existence
    registry = config.class_registry  # dict[str, ClassConfig]

    # Resolve overlap-exempt classes: any class whose tags intersect overlap_exempt_tags
    exempt_tags = set(co.overlap_exempt_tags)
    exempt_names: set[str] = set()
    for name, cls_cfg in registry.items():
        if exempt_tags & set(getattr(cls_cfg, "tags", [])):
            exempt_names.add(name)

    # Resolve confusion pairs: classes sharing a confusion_tag are confusion pairs
    confusion_pairs: list[frozenset] = []
    for tag in co.confusion_tags:
        # Collect all classes with this tag
        tagged = [name for name, cls_cfg in registry.items()
                  if tag in getattr(cls_cfg, "tags", [])]
        # All pairs within the tagged group
        for i, a in enumerate(tagged):
            for b in tagged[i + 1:]:
                pair = frozenset((a, b))
                if pair not in confusion_pairs:
                    confusion_pairs.append(pair)

    # Add explicit extra pairs
    for pair in co.extra_confusion_pairs:
        fp = frozenset(pair)
        if fp not in confusion_pairs:
            confusion_pairs.append(fp)

    return exempt_names, confusion_pairs


# ---------------------------------------------------------------------------
# Cross-class routing
# ---------------------------------------------------------------------------


def route_candidates(candidates: list, config: Any) -> dict:
    """Apply routing logic to produce auto_accepted / needs_evaluation splits.

    Rules (evaluated in order):
    - Tier 1 class AND agreement >= min_model_agreement AND score >= min_score
      → auto_accept
    - Otherwise → needs_evaluation
    - Confusion pairs: overlapping boxes of confused classes are flagged.

    config must have .auto_accept (AutoAcceptConfig), .classes (dict[str, ClassConfig]),
    and .co_existence (CoExistenceConfig) attributes.

    Returns dict with keys: auto_accepted, needs_evaluation, confusion_flags.
    """
    aa_cfg = config.auto_accept
    eligible_tiers = set(aa_cfg.tiers)
    eligible_names = {
        name for name, cls_cfg in config.classes.items()
        if cls_cfg.tier in eligible_tiers
    }
    _, confusion_pairs = _resolve_co_existence(config)

    # Head+person co-existence shortcut: if a head is contained in a person
    # bbox above filtering.head_person_containment_min AND auto_accept
    # has opted in, both candidates auto-accept. The spatial reinforcement
    # is strong enough that VLM disambiguation rarely changes the verdict,
    # EXCEPT when either leg of the pair is itself a weak detection — a
    # marginal head riding a spurious person box must not auto-accept.
    # Both head and person therefore must clear
    # ``head_person_coexistence_min_score`` (default 0.0 = no gate).
    coexist_auto_ids: set[str] = set()
    coexist_enabled = getattr(
        aa_cfg, "head_person_coexistence", False
    ) and getattr(
        config.filtering, "head_person_containment_min", 0.0
    ) > 0
    if coexist_enabled:
        thr = config.filtering.head_person_containment_min
        coexist_min_score = getattr(
            aa_cfg, "head_person_coexistence_min_score", 0.0
        ) or 0.0
        cand_by_id = {c.candidate_id: c for c in candidates}
        _coex_log = get_logger("utils.route_candidates.coexist")
        for hid, pid in head_person_coexist_pairs(candidates, thr):
            head = cand_by_id.get(hid)
            person = cand_by_id.get(pid)
            if head is None or person is None:
                continue
            if head.score < coexist_min_score or person.score < coexist_min_score:
                # One leg too weak — pair doesn't qualify for the
                # shortcut; each candidate still routes via baseline
                # tier/score/agreement logic below. Log so an operator
                # can see *why* a plausible-looking pair didn't auto-accept.
                _coex_log.debug(
                    "coexist shortcut declined: head=%s(%.2f) person=%s(%.2f) "
                    "gate=%.2f",
                    hid, head.score, pid, person.score, coexist_min_score,
                )
                continue
            coexist_auto_ids.add(hid)
            coexist_auto_ids.add(pid)

    auto_accepted: list[str] = []
    needs_evaluation: list[str] = []

    per_model_score = getattr(config.filtering, "per_model_score", {}) or {}
    fallback_score = aa_cfg.min_score
    # Per-model high-confidence shortcut — a strong score bypasses the
    # agreement requirement for single-detector runs (e.g. sam3_dart-only
    # where agreement can never reach 2).
    hi_conf_scores = getattr(aa_cfg, "high_confidence_scores", {}) or {}

    for cand in candidates:
        if cand.candidate_id in coexist_auto_ids:
            auto_accepted.append(cand.candidate_id)
            continue
        is_eligible = cand.class_name in eligible_names
        score_floor = per_model_score.get(cand.source_model, fallback_score)
        hi_conf_floor = hi_conf_scores.get(cand.source_model)
        passes_agreement = cand.agreement >= aa_cfg.min_model_agreement
        passes_hi_conf = (
            hi_conf_floor is not None and cand.score >= hi_conf_floor
        )
        qualifies = (
            is_eligible
            and cand.score >= score_floor
            and (passes_agreement or passes_hi_conf)
        )
        if qualifies:
            auto_accepted.append(cand.candidate_id)
        else:
            needs_evaluation.append(cand.candidate_id)

    # Detect confusion-pair overlaps
    confusion_flags: list[dict] = []
    checked: set[frozenset] = set()
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if j <= i:
                continue
            pair_key = frozenset((a.candidate_id, b.candidate_id))
            if pair_key in checked:
                continue
            checked.add(pair_key)
            class_pair = frozenset((a.class_name, b.class_name))
            if class_pair in confusion_pairs and a.class_name != b.class_name:
                iou = bbox_iou(a.bbox, b.bbox)
                if iou > 0.3:
                    confusion_flags.append(
                        {
                            "candidate_ids": [a.candidate_id, b.candidate_id],
                            "classes": [a.class_name, b.class_name],
                            "iou": iou,
                        }
                    )

    return {
        "auto_accepted": auto_accepted,
        "needs_evaluation": needs_evaluation,
        "confusion_flags": confusion_flags,
    }


def _cross_class_winner(a: Any, b: Any, iou_dedup_cfg: Any) -> Any:
    """Pick the winner of a cross-class overlap using the same tiebreak cascade
    as cluster_and_collapse: agreement → model_priority → score.

    Returns the *loser* (the candidate to suppress).
    """
    tiebreak_by = list(iou_dedup_cfg.tiebreak_by)
    priority_index = {m: i for i, m in enumerate(iou_dedup_cfg.model_priority)}
    fallback_priority = len(iou_dedup_cfg.model_priority)

    pool = [a, b]
    for key in tiebreak_by:
        if len(pool) == 1:
            break
        if key == "agreement":
            best = max(m.agreement for m in pool)
            winners = [m for m in pool if m.agreement == best]
            if len(winners) < len(pool):
                pool = winners
        elif key == "model_priority":
            best = min(
                priority_index.get(m.source_model, fallback_priority)
                for m in pool
            )
            winners = [
                m for m in pool
                if priority_index.get(m.source_model, fallback_priority) == best
            ]
            if len(winners) < len(pool):
                pool = winners
        elif key == "score":
            best_score = max(m.score for m in pool)
            winners = [m for m in pool if m.score == best_score]
            if len(winners) < len(pool):
                pool = winners

    # pool[0] is the winner; return the loser
    winner = pool[0]
    return b if winner is a else a


def apply_cross_class_rules(candidates: list, config: Any) -> list:
    """Apply co-existence rules across class boundaries.

    - overlap_exempt classes (resolved via tags) are never suppressed.
    - confusion pairs (resolved via tags) are flagged but not suppressed.
    - For all other cross-class overlaps with IoU > 0.5: suppress the loser
      chosen by the tiebreak cascade (agreement → model_priority → score),
      the same cascade used by cluster_and_collapse.

    Returns a filtered list of candidates.
    """
    exempt, confusion_pairs = _resolve_co_existence(config)
    iou_dedup_cfg = config.filtering.iou_dedup

    suppressed: set[str] = set()

    for i, a in enumerate(candidates):
        if a.candidate_id in suppressed:
            continue
        for j, b in enumerate(candidates):
            if j <= i:
                continue
            if b.candidate_id in suppressed:
                continue
            if a.class_name == b.class_name:
                continue  # same class handled by cluster_and_collapse

            # Neither may be suppressed if exempt
            a_exempt = a.class_name in exempt
            b_exempt = b.class_name in exempt
            if a_exempt or b_exempt:
                continue

            class_pair = frozenset((a.class_name, b.class_name))
            if class_pair in confusion_pairs:
                continue  # confusion pairs: flag only, do not suppress here

            iou = bbox_iou(a.bbox, b.bbox)
            if iou > 0.5:
                loser = _cross_class_winner(a, b, iou_dedup_cfg)
                suppressed.add(loser.candidate_id)

    return [c for c in candidates if c.candidate_id not in suppressed]


def _containment_in(small_bbox, big_bbox) -> float:
    """Fraction of ``small_bbox`` area that falls inside ``big_bbox``.

    Intersection / area(small). 0.0 means no overlap; 1.0 means fully
    contained. More surgical than IoU when ``small`` is much smaller than
    ``big`` (e.g. a head vs a full-body person bbox).
    """
    x1 = max(_x1(small_bbox), _x1(big_bbox))
    y1 = max(_y1(small_bbox), _y1(big_bbox))
    x2 = min(_x2(small_bbox), _x2(big_bbox))
    y2 = min(_y2(small_bbox), _y2(big_bbox))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = bbox_area(small_bbox)
    return inter / a if a > 0 else 0.0


def head_person_coexist_pairs(
    candidates: list, containment_min: float
) -> set[tuple[str, str]]:
    """Return (head_id, person_id) pairs whose head is contained in person
    above ``containment_min``. A head may appear in multiple pairs if it
    overlaps several persons.
    """
    if containment_min <= 0:
        return set()
    heads = [c for c in candidates if c.class_name == "head"]
    persons = [c for c in candidates if c.class_name == "person"]
    pairs: set[tuple[str, str]] = set()
    for h in heads:
        for p in persons:
            if _containment_in(h.bbox, p.bbox) >= containment_min:
                pairs.add((h.candidate_id, p.candidate_id))
    return pairs


def drop_head_without_person(candidates: list, config: Any = None) -> list:
    """Drop stray ``head`` candidates that are not part of a detected person.

    Two modes, chosen by config:

    - **Per-head containment** (when
      ``config.filtering.head_person_containment_min > 0``): each head is
      kept only if its maximum containment-in-person across all persons in
      the image is >= threshold. Heads in images with no persons, and heads
      floating away from any person, are dropped. More surgical.

    - **Image-level presence** (when
      ``config.filtering.reject_head_without_person = True`` and containment
      threshold is 0): legacy coarse check — drop all heads if the image has
      no person candidates.

    If ``config`` is None, falls back to the legacy image-level check
    (previously the only behavior).
    """
    thr = 0.0
    legacy_presence = False
    if config is not None:
        thr = getattr(config.filtering, "head_person_containment_min", 0.0) or 0.0
        legacy_presence = getattr(
            config.filtering, "reject_head_without_person", False
        )

    heads = [c for c in candidates if c.class_name == "head"]
    if not heads:
        return list(candidates)

    logger = get_logger("utils.drop_head_without_person")

    if thr > 0:
        persons = [c for c in candidates if c.class_name == "person"]
        keep_head_ids: set[str] = set()
        if persons:
            for h in heads:
                for p in persons:
                    if _containment_in(h.bbox, p.bbox) >= thr:
                        keep_head_ids.add(h.candidate_id)
                        break
        kept = [c for c in candidates
                if c.class_name != "head" or c.candidate_id in keep_head_ids]
        dropped = len(candidates) - len(kept)
        if dropped:
            logger.info(
                "head_person_containment (thr=%.2f): %d head(s) lacked "
                "containment, dropped",
                thr, dropped,
            )
        return kept

    if legacy_presence:
        if any(c.class_name == "person" for c in candidates):
            return list(candidates)
        kept = [c for c in candidates if c.class_name != "head"]
        dropped = len(candidates) - len(kept)
        if dropped:
            logger.info(
                "reject_head_without_person (presence): no person, "
                "dropped %d head(s)", dropped,
            )
        return kept

    return list(candidates)


def apply_class_agnostic_nms(candidates: list, config: Any) -> list:
    """Class-agnostic IoU NMS across all surviving candidates.

    Runs AFTER within-class dedup (cluster_and_collapse) and the legacy
    cross_class step. Uses ``config.filtering.class_agnostic_nms`` for
    configuration:
      - ``enabled``: if False, returns candidates unchanged.
      - ``threshold``: IoU threshold for suppression.
      - ``respect_overlap_exempt``: if True, any pair where either side has
        an ``overlap_exempt`` tag is skipped (preserves head-inside-person,
        bag-on-person).
      - ``suppress_confusion_pairs``: if False, pairs belonging to the same
        confusion tag are left intact (kept for VLM disambiguation).

    Tiebreak cascade matches cluster_and_collapse: agreement → model_priority
    → score.
    """
    cfg = getattr(config.filtering, "class_agnostic_nms", None)
    if cfg is None or not cfg.enabled:
        return list(candidates)

    iou_dedup_cfg = config.filtering.iou_dedup
    threshold = cfg.threshold
    exempt, confusion_pairs = _resolve_co_existence(config)
    logger = get_logger("utils.apply_class_agnostic_nms")

    suppressed: set[str] = set()
    for i, a in enumerate(candidates):
        if a.candidate_id in suppressed:
            continue
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if b.candidate_id in suppressed:
                continue
            # Same-class already deduped by cluster_and_collapse.
            if a.class_name == b.class_name:
                continue
            if cfg.respect_overlap_exempt and (
                a.class_name in exempt or b.class_name in exempt
            ):
                continue
            if not cfg.suppress_confusion_pairs:
                class_pair = frozenset((a.class_name, b.class_name))
                if class_pair in confusion_pairs:
                    continue
            if bbox_iou(a.bbox, b.bbox) >= threshold:
                loser = _cross_class_winner(a, b, iou_dedup_cfg)
                suppressed.add(loser.candidate_id)

    kept = [c for c in candidates if c.candidate_id not in suppressed]
    if suppressed:
        logger.info(
            "class_agnostic_nms @ IoU≥%.2f: %d → %d (%d suppressed)",
            threshold, len(candidates), len(kept), len(suppressed),
        )
    return kept


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 56, 56),
    (56, 200, 56),
    (56, 56, 255),
    (255, 220, 0),
    (255, 56, 220),
    (0, 220, 220),
    (180, 0, 0),
    (0, 140, 0),
    (0, 0, 180),
    (140, 140, 0),
    (140, 0, 140),
    (0, 140, 140),
]


def get_image_size(image_path: str) -> tuple[int, int]:
    """Return (width, height) without loading full image data into memory."""
    with Image.open(image_path) as img:
        return img.size  # PIL reads header only when .size is accessed before load()


def draw_focus_on_image(
    image: Image.Image,
    candidate: Any,
    color: tuple[int, int, int] = (255, 0, 0),
    label: str = "TARGET",
) -> Image.Image:
    """Draw ONLY this candidate's bbox on a copy of *image* — for per-candidate
    VLM overview input. Keeps the rest of the scene visible (spatial context)
    without any other bboxes that could confuse which one is being asked about.

    Border is drawn OUTWARD from the bbox so the object pixels inside the
    bbox are never obscured. For bboxes that touch the image edge we
    **pad the canvas** by ``stroke+1`` px on all sides with a neutral
    dark-grey gutter before drawing, so the border has room to extend
    outward on every side. Without this pad the previous implementation
    clamped the outward offset at the image boundary, which caused PIL's
    ``width=N`` stroke to fall back to drawing *inward* on the edge side —
    re-occluding the very pixels the outward-draw change was meant to
    protect.

    Border width scales with bbox size (1-2 px for tiny boxes up to 4 px
    for large). Label has a filled background for contrast and renders
    above the outward border when there's room.
    """
    base = image if image.mode == "RGB" else image.convert("RGB")
    w, h = base.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    x1, y1, x2, y2 = bbox_to_pixels(candidate.bbox, w, h)
    # Degenerate bbox guard: collapse to a 1-px seed so PIL doesn't
    # silently draw nothing on a zero-area rectangle. We still emit the
    # overview (with a tiny marker) so the caller isn't left wondering
    # why a candidate vanished.
    side_px = max(1, min(x2 - x1, y2 - y1))
    stroke = max(1, min(4, side_px // 15))

    # Pad the canvas with a neutral dark gutter so the outward stroke
    # ALWAYS has room, even when the bbox touches an image edge. +1 so
    # the outline never shares a row/col with the padded boundary.
    pad = stroke + 1
    padded = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (32, 32, 32))
    padded.paste(base, (pad, pad))
    draw = ImageDraw.Draw(padded)

    # Shift bbox into padded coords, then offset outward by stroke.
    sx1, sy1 = x1 + pad, y1 + pad
    sx2, sy2 = x2 + pad, y2 + pad
    ox1, oy1 = sx1 - stroke, sy1 - stroke
    ox2, oy2 = sx2 + stroke, sy2 + stroke
    draw.rectangle((ox1, oy1, ox2, oy2), outline=color, width=stroke)

    # Label: filled background + white text for readability on any
    # scene colour. Try above the outward border first, then below; if
    # neither fits, skip the label (the red border alone is enough to
    # identify the target, and drawing inside the bbox would occlude
    # the very object we want the VLM to see).
    try:
        tw = int(draw.textlength(label, font=font))
    except AttributeError:  # old PIL
        tw = 8 * len(label)
    label_h = 20
    ph = padded.size[1]
    label_xy: tuple[int, int] | None = None
    if oy1 >= label_h + 2:
        label_xy = (ox1, oy1 - (label_h + 2))
    elif oy2 + label_h + 2 <= ph:
        label_xy = (ox1, oy2 + 2)
    if label_xy is not None:
        lx, ly = label_xy
        draw.rectangle((lx, ly, lx + tw + 6, ly + label_h), fill=color)
        draw.text((lx + 3, ly + 1), label, fill=(255, 255, 255), font=font)
    return padded


def draw_candidates_on_image(
    image: Image.Image,
    candidates: list,
    class_colors: dict | None = None,
) -> Image.Image:
    """Draw numbered bounding boxes on image for VLM input.

    Each candidate gets a number label and a colored rectangle.
    class_colors maps class_name -> (R, G, B); if None, cycles through defaults.
    """
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    w, h = rendered.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, cand in enumerate(candidates):
        if class_colors and cand.class_name in class_colors:
            color = class_colors[cand.class_name]
        else:
            color = _COLORS[idx % len(_COLORS)]

        px = bbox_to_pixels(cand.bbox, w, h)
        draw.rectangle(px, outline=color, width=3)
        label = f"[{idx}] {cand.class_name} ({cand.score:.2f})"
        draw.text((px[0], max(0, px[1] - 16)), label, fill=color, font=font)

    return rendered


def crop_candidate(
    image: Image.Image, bbox: BboxLike, padding: float = 0.08
) -> Image.Image:
    """Crop image around a candidate bbox with relative padding.

    bbox can be a BoundingBox object or a plain dict with x1/y1/x2/y2 keys.
    padding is a fraction of the bbox's own width/height.
    """
    w, h = image.size
    bw = max(0.0, _x2(bbox) - _x1(bbox))
    bh = max(0.0, _y2(bbox) - _y1(bbox))
    px = int(bw * w * padding)
    py = int(bh * h * padding)
    x1, y1, x2, y2 = bbox_to_pixels(bbox, w, h)
    return image.crop(
        (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))
    )


def pil_to_data_url(
    image: Image.Image,
    max_size: int = 1280,
    fmt: str = "JPEG",
    quality: int = 90,
) -> str:
    """Encode PIL image as a base64 data URL for the VLM payload.

    Defaults: JPEG q=90 at 1280 px longest side. JPEG is 5-10x smaller
    than PNG at negligible quality cost for natural-image VLM inputs —
    the bench harness (tests/bench_vllm.py) already uses JPEG, so this
    brings production in line. 1280 px keeps patch density for small
    objects (a 40x40 head bbox on a 4K frame lands at ~20 px here
    instead of ~10 px under the old 1024 cap).

    When ``fmt == "PNG"`` ``quality`` is ignored.
    """
    img = image.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    fmt_up = fmt.upper()
    if fmt_up == "JPEG":
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def pil_to_png_bytes(image: Image.Image) -> bytes:
    """Return PNG bytes for a PIL image without resizing."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------


def annotation_to_yolo_line(class_id: int, bbox: BboxLike) -> str:
    """Format as YOLO: 'class_id cx cy w h' with normalized coords."""
    x1, y1, x2, y2 = _x1(bbox), _y1(bbox), _x2(bbox), _y2(bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_yolo_labels(annotations: list, output_path: str, class_map: dict) -> None:
    """Write YOLO label file.

    annotations must have .class_name and .bbox attributes.
    class_map maps class_name -> class_id (int).
    Lines for unknown classes are skipped.
    """
    logger = get_logger("utils.write_yolo_labels")
    lines: list[str] = []
    for ann in annotations:
        class_id = class_map.get(ann.class_name)
        if class_id is None:
            logger.warning("Unknown class '%s'; skipping annotation.", ann.class_name)
            continue
        lines.append(annotation_to_yolo_line(class_id, ann.bbox))

    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""))


def write_classes_file(class_map: dict, output_path: str) -> None:
    """Write classes.txt sorted by class id.

    class_map maps class_name -> class_id.
    """
    sorted_names = sorted(class_map, key=lambda n: class_map[n])
    Path(output_path).write_text("\n".join(sorted_names) + "\n")


# ---------------------------------------------------------------------------
# Class alias utilities
# ---------------------------------------------------------------------------


def normalize_class_alias(name: str) -> str:
    """Lowercase, strip, replace underscores/hyphens with spaces, collapse whitespace."""
    return " ".join(
        name.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def build_class_alias_map(classes_config: list) -> dict:
    """Build map from normalized aliases to canonical class names.

    classes_config is a list of objects with at minimum a .name attribute and
    optionally an .aliases attribute (list[str]) or an .all_names() method.
    """
    alias_map: dict[str, str] = {}
    for cls in classes_config:
        canonical = cls.name
        # Support objects with all_names() (v2 ClassPackConfig style)
        if hasattr(cls, "all_names") and callable(cls.all_names):
            names = cls.all_names()
        else:
            names = [canonical]
            if hasattr(cls, "aliases"):
                names += list(cls.aliases)

        for alias in names:
            alias_map[normalize_class_alias(alias)] = canonical

    return alias_map


def resolve_canonical_class(name: str, alias_map: dict) -> str | None:
    """Resolve a string to canonical class name, or None if unknown."""
    return alias_map.get(normalize_class_alias(name))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the auto_annotation_v4 namespace."""
    return logging.getLogger(f"data_miner.auto_annotation_v4.{name}")


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure console + optional file logging for the pipeline."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("data_miner.auto_annotation_v4")
    root_logger.setLevel(numeric_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# JSON parsing (robust, for VLM output)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def parse_vlm_json(text: str) -> dict | list:
    """Parse JSON from VLM output, handling common formatting issues.

    Strips <think>...</think> blocks, markdown code fences, trailing commas,
    and single-quoted string keys/values (converted to double-quoted).
    Raises ValueError if no valid JSON can be extracted.
    """
    # 1. Strip <think> blocks
    text = _THINK_RE.sub("", text)

    # 2. Extract from code fence if present
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)

    text = text.strip()

    # 3. Try direct parse first (common case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Remove trailing commas before ] or }
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 5. Attempt to locate the first {...} or [...] block
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    fragment = cleaned[start : i + 1]
                    try:
                        return json.loads(fragment)
                    except json.JSONDecodeError:
                        break

    raise ValueError(
        f"Could not extract valid JSON from VLM output. "
        f"First 200 chars: {text[:200]!r}"
    )
