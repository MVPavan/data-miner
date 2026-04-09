"""Filtering stage: programmatic bbox filtering based on config thresholds."""

from __future__ import annotations

from ..config import AutoAnnotationV2Config, FilterConfig
from ..contracts import Candidate, CandidateStatus
from ..log_utils import get_logger
from ..utils import bbox_iou

logger = get_logger(__name__)


def _passes_area(candidate: Candidate, cfg: FilterConfig) -> bool:
    area = candidate.bbox.area
    return cfg.min_area <= area <= cfg.max_area


def _passes_aspect_ratio(candidate: Candidate, cfg: FilterConfig) -> bool:
    ar = candidate.bbox.aspect_ratio
    if ar <= 0:
        return False
    return cfg.min_aspect_ratio <= ar <= cfg.max_aspect_ratio


def _passes_edge_distance(candidate: Candidate, cfg: FilterConfig) -> bool:
    if cfg.min_edge_distance <= 0:
        return True
    box = candidate.bbox
    d = cfg.min_edge_distance
    return (
        box.x1 >= d
        and box.y1 >= d
        and box.x2 <= (1.0 - d)
        and box.y2 <= (1.0 - d)
    )


def _dedup_by_iou(candidates: list[Candidate], threshold: float) -> list[Candidate]:
    """Remove overlapping candidates, keeping higher-scored ones."""
    if threshold <= 0 or len(candidates) <= 1:
        return candidates

    # Sort by score descending so we keep the best
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []

    for cand in sorted_cands:
        is_dup = False
        for existing in kept:
            if bbox_iou(cand.bbox, existing.bbox) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)

    return kept


def _limit_per_class(
    candidates: list[Candidate], max_per_class: int
) -> list[Candidate]:
    """Keep at most N candidates per class, preferring higher scores."""
    if max_per_class <= 0:
        return candidates

    by_class: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_class.setdefault(c.class_name, []).append(c)

    result: list[Candidate] = []
    for class_name, class_cands in by_class.items():
        sorted_cands = sorted(class_cands, key=lambda c: c.score, reverse=True)
        kept = sorted_cands[:max_per_class]
        if len(sorted_cands) > max_per_class:
            logger.info(
                "Class '%s': limited from %d to %d candidates",
                class_name, len(sorted_cands), max_per_class,
            )
        result.extend(kept)
    return result


def run_filtering(
    candidates: list[Candidate],
    config: AutoAnnotationV2Config,
) -> list[Candidate]:
    """Apply all configured filters and return surviving candidates."""
    cfg = config.filtering
    initial_count = len(candidates)

    passed: list[Candidate] = []
    for cand in candidates:
        if not _passes_area(cand, cfg):
            logger.debug(
                "Filtered %s: area=%.6f not in [%.6f, %.6f]",
                cand.candidate_id, cand.bbox.area, cfg.min_area, cfg.max_area,
            )
            continue
        if not _passes_aspect_ratio(cand, cfg):
            logger.debug(
                "Filtered %s: aspect_ratio=%.3f not in [%.3f, %.3f]",
                cand.candidate_id, cand.bbox.aspect_ratio,
                cfg.min_aspect_ratio, cfg.max_aspect_ratio,
            )
            continue
        if not _passes_edge_distance(cand, cfg):
            logger.debug("Filtered %s: too close to image edge", cand.candidate_id)
            continue
        passed.append(cand)

    after_filter = len(passed)
    logger.info(
        "Geometric filtering: %d → %d candidates", initial_count, after_filter
    )

    # IoU dedup per class
    by_class: dict[str, list[Candidate]] = {}
    for c in passed:
        by_class.setdefault(c.class_name, []).append(c)

    deduped: list[Candidate] = []
    for class_name, class_cands in by_class.items():
        before = len(class_cands)
        kept = _dedup_by_iou(class_cands, cfg.iou_dedup_threshold)
        if len(kept) < before:
            logger.info(
                "IoU dedup class '%s': %d → %d", class_name, before, len(kept)
            )
        deduped.extend(kept)

    # Per-class limit
    limited = _limit_per_class(deduped, cfg.max_candidates_per_class)

    logger.info(
        "Filtering complete: %d → %d candidates", initial_count, len(limited)
    )
    return limited
