"""Shared utility functions for auto_annotation_v3.

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
    (matches AutoAnnotationV3Config.filtering / FilterConfig).

    Returns the list of candidates that pass all filters.
    """
    logger = get_logger("utils.geometric_filter")
    cfg = config.filtering
    passed = []
    for cand in candidates:
        bbox = cand.bbox
        if not passes_area_filter(bbox, cfg.min_area, cfg.max_area):
            logger.debug(
                "Filtered %s: area=%.6f not in [%.6f, %.6f]",
                cand.candidate_id,
                bbox_area(bbox),
                cfg.min_area,
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
    tiebreak_by = list(iou_dedup_cfg.tiebreak_by)
    priority_index = {m: i for i, m in enumerate(iou_dedup_cfg.model_priority)}
    fallback_priority = len(iou_dedup_cfg.model_priority)  # unknown models last

    if threshold <= 0 or not candidates:
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
                if bbox_iou(group[i].bbox, group[j].bbox) >= threshold:
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
# Cross-class routing (new for v3)
# ---------------------------------------------------------------------------


def route_candidates(candidates: list, config: Any) -> dict:
    """Apply routing logic to produce auto_accepted / needs_evaluation splits.

    Rules (evaluated in order):
    - Tier 1 class AND agreement >= min_model_agreement AND score >= min_score
      → auto_accept
    - Otherwise → needs_evaluation
    - Confusion pairs: overlapping boxes of confused classes are flagged.

    config must have .auto_accept (AutoAcceptConfig) and .classes (list[ClassConfig])
    and .co_existence (CoExistenceConfig) attributes.

    Returns dict:
        {
            "auto_accepted":    list[str],   # candidate_ids
            "needs_evaluation": list[str],   # candidate_ids
            "confusion_flags":  list[dict],  # {candidate_ids, classes, iou}
        }
    """
    aa_cfg = config.auto_accept
    eligible_tiers = set(aa_cfg.tiers)
    eligible_names = {
        c.name for c in config.classes if c.tier in eligible_tiers
    }
    confusion_pairs: list[frozenset] = [
        frozenset(pair) for pair in config.co_existence.confusion_pairs
    ]

    auto_accepted: list[str] = []
    needs_evaluation: list[str] = []

    per_model_score = getattr(aa_cfg, "per_model_score", {}) or {}
    fallback_score = aa_cfg.min_score

    for cand in candidates:
        is_eligible = cand.class_name in eligible_names
        score_floor = per_model_score.get(cand.source_model, fallback_score)
        qualifies = (
            is_eligible
            and cand.agreement >= aa_cfg.min_model_agreement
            and cand.score >= score_floor
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


def apply_cross_class_rules(candidates: list, config: Any) -> list:
    """Apply co-existence rules across class boundaries.

    - globally_exempt classes (e.g. person, head) are never suppressed.
    - confusion_pairs with high IoU are flagged but not suppressed (handled in evaluate).
    - For all other cross-class overlaps with IoU > 0.5: suppress the lower-score box.

    Returns a filtered list of candidates.
    """
    co = config.co_existence
    exempt: set[str] = set(co.globally_exempt)
    confusion_pairs: list[frozenset] = [
        frozenset(pair) for pair in co.confusion_pairs
    ]

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
                continue  # same class handled by dedup_by_iou

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
                # Suppress the lower-score candidate
                loser = b if a.score >= b.score else a
                suppressed.add(loser.candidate_id)

    return [c for c in candidates if c.candidate_id not in suppressed]


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


def pil_to_data_url(image: Image.Image, max_size: int = 1024) -> str:
    """Encode PIL image as base64 data URL. Resize if larger than max_size."""
    img = image.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


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
    """Get a logger under the auto_annotation_v3 namespace."""
    return logging.getLogger(f"data_miner.auto_annotation_v3.{name}")


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure console + optional file logging for the pipeline."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("data_miner.auto_annotation_v3")
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
