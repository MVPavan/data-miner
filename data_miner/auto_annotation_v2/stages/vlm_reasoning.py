"""VLM reasoning stage: two-pass evaluation (screening + detailed review)."""

from __future__ import annotations

import asyncio
from typing import Literal

from PIL import Image
from pydantic_ai.messages import BinaryContent

from ..agents.reasoning import (
    DetailedReviewDeps,
    ScreeningDeps,
    build_detailed_agent,
    build_screening_agent,
    parse_detailed_verdict,
    parse_screening_result,
)
from ..config import AutoAnnotationV2Config, ClassPackConfig
from ..contracts import (
    Candidate,
    DetailedVerdict,
    ScreeningResult,
    ScreeningVerdict,
    VLMDecision,
)
from ..log_utils import get_logger
from ..utils import crop_candidate, draw_candidates_numbered, pil_to_png_bytes

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_image_content(
    image: Image.Image,
    candidates: list[Candidate],
    candidate: Candidate | None,
    inputs: list[Literal["original", "annotated", "crop"]],
) -> list[BinaryContent]:
    """Build image content parts for VLM based on configured inputs."""
    parts: list[BinaryContent] = []
    for kind in inputs:
        if kind == "original":
            parts.append(
                BinaryContent(data=pil_to_png_bytes(image), media_type="image/png")
            )
        elif kind == "annotated":
            annotated = draw_candidates_numbered(image, candidates)
            parts.append(
                BinaryContent(data=pil_to_png_bytes(annotated), media_type="image/png")
            )
        elif kind == "crop" and candidate is not None:
            cropped = crop_candidate(image, candidate)
            parts.append(
                BinaryContent(data=pil_to_png_bytes(cropped), media_type="image/png")
            )
    return parts


def _format_candidate_list(candidates: list[Candidate]) -> str:
    """Format candidates as a numbered text list for the screening prompt."""
    lines: list[str] = []
    for idx, cand in enumerate(candidates):
        box = cand.bbox
        lines.append(
            f"[{idx}] class={cand.class_name} label={cand.label} "
            f"source={cand.source_model} score={cand.score:.3f} "
            f"bbox=({box.x1:.3f},{box.y1:.3f},{box.x2:.3f},{box.y2:.3f})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass 1: Screening (one call per class with all candidates)
# ---------------------------------------------------------------------------


async def _run_screening_for_class(
    image: Image.Image,
    class_candidates: list[Candidate],
    class_pack: ClassPackConfig,
    config: AutoAnnotationV2Config,
) -> ScreeningResult:
    """Run Pass 1 screening for all candidates of one class."""
    agent = build_screening_agent(config)
    image_inputs = config.reasoning.screening.image_inputs

    deps = ScreeningDeps(
        class_pack=class_pack,
        candidate_descriptions=_format_candidate_list(class_candidates),
        num_candidates=len(class_candidates),
    )

    # Build multimodal user prompt
    image_parts = _build_image_content(image, class_candidates, None, image_inputs)

    user_prompt_parts: list[str | BinaryContent] = []
    for part in image_parts:
        user_prompt_parts.append(part)
    user_prompt_parts.append(
        f"Here are {len(class_candidates)} candidate annotations to evaluate:\n\n"
        f"{deps.candidate_descriptions}\n\n"
        f"Evaluate each candidate and return your verdict."
    )

    result = await agent.run(user_prompt_parts, deps=deps)
    return parse_screening_result(result.output)


# ---------------------------------------------------------------------------
# Pass 2: Detailed review (one call per uncertain candidate)
# ---------------------------------------------------------------------------


async def _run_detailed_for_candidate(
    image: Image.Image,
    candidate: Candidate,
    all_candidates: list[Candidate],
    class_pack: ClassPackConfig,
    config: AutoAnnotationV2Config,
) -> DetailedVerdict:
    """Run Pass 2 detailed review for one candidate."""
    agent = build_detailed_agent(config)
    image_inputs = config.reasoning.detailed.image_inputs

    deps = DetailedReviewDeps(
        class_pack=class_pack,
        candidate_id=candidate.candidate_id,
        candidate_label=candidate.label,
        candidate_source=candidate.source_model,
        candidate_score=candidate.score,
    )

    image_parts = _build_image_content(image, all_candidates, candidate, image_inputs)
    user_prompt_parts: list[str | BinaryContent] = []
    for part in image_parts:
        user_prompt_parts.append(part)
    user_prompt_parts.append(
        f"Examine candidate [{candidate.candidate_id}] in detail.\n"
        f"Label: {candidate.label}, Score: {candidate.score:.3f}\n"
        f"BBox: ({candidate.bbox.x1:.3f}, {candidate.bbox.y1:.3f}, "
        f"{candidate.bbox.x2:.3f}, {candidate.bbox.y2:.3f})\n\n"
        f"Provide your detailed assessment."
    )

    result = await agent.run(user_prompt_parts, deps=deps)
    return parse_detailed_verdict(result.output)


# ---------------------------------------------------------------------------
# Public API: two-pass VLM reasoning
# ---------------------------------------------------------------------------


async def run_vlm_reasoning(
    image: Image.Image,
    candidates: list[Candidate],
    config: AutoAnnotationV2Config,
) -> tuple[list[ScreeningVerdict], list[DetailedVerdict]]:
    """Run two-pass VLM reasoning on candidates.

    Returns:
        (screening_verdicts, detailed_verdicts)
    """
    if not candidates:
        return [], []

    # Group candidates by class
    by_class: dict[str, list[Candidate]] = {}
    class_pack_map: dict[str, ClassPackConfig] = {}
    for cp in config.classes:
        class_pack_map[cp.name] = cp
    for cand in candidates:
        by_class.setdefault(cand.class_name, []).append(cand)

    # --- Pass 1: Screening (one call per class) ---
    logger.info("Pass 1: Screening %d classes", len(by_class))
    all_screening: list[ScreeningVerdict] = []
    screening_tasks = []
    for class_name, class_cands in by_class.items():
        cp = class_pack_map.get(class_name)
        if cp is None:
            logger.warning("No class pack for '%s', skipping screening", class_name)
            continue
        screening_tasks.append(_run_screening_for_class(image, class_cands, cp, config))

    screening_results: list[ScreeningResult] = await asyncio.gather(
        *screening_tasks, return_exceptions=True
    )
    for result in screening_results:
        if isinstance(result, Exception):
            logger.exception("Screening failed: %s", result)
            continue
        all_screening.extend(result.verdicts)

    logger.info(
        "Pass 1 complete: %d verdicts (%d accept, %d needs_review, %d reject)",
        len(all_screening),
        sum(1 for v in all_screening if v.decision == VLMDecision.ACCEPT),
        sum(1 for v in all_screening if v.decision == VLMDecision.NEEDS_REVIEW),
        sum(1 for v in all_screening if v.decision == VLMDecision.REJECT),
    )

    # --- Pass 2: Detailed review for needs_review candidates ---
    needs_review_ids = {
        v.candidate_id for v in all_screening if v.decision == VLMDecision.NEEDS_REVIEW
    }
    review_candidates = [c for c in candidates if c.candidate_id in needs_review_ids]

    if not review_candidates:
        logger.info("Pass 2: No candidates need detailed review")
        return all_screening, []

    logger.info("Pass 2: Detailed review for %d candidates", len(review_candidates))
    max_concurrent = config.reasoning.max_concurrent_calls
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_review(cand: Candidate) -> DetailedVerdict | Exception:
        async with semaphore:
            try:
                cp = class_pack_map[cand.class_name]
                class_cands = by_class[cand.class_name]
                return await _run_detailed_for_candidate(
                    image, cand, class_cands, cp, config
                )
            except Exception as exc:
                logger.exception("Detailed review failed for %s", cand.candidate_id)
                return exc

    detail_results = await asyncio.gather(
        *[_limited_review(c) for c in review_candidates]
    )

    all_detailed: list[DetailedVerdict] = []
    for result in detail_results:
        if isinstance(result, Exception):
            continue
        all_detailed.append(result)

    logger.info(
        "Pass 2 complete: %d detailed verdicts (%d accept, %d needs_review, %d reject)",
        len(all_detailed),
        sum(1 for v in all_detailed if v.decision == VLMDecision.ACCEPT),
        sum(1 for v in all_detailed if v.decision == VLMDecision.NEEDS_REVIEW),
        sum(1 for v in all_detailed if v.decision == VLMDecision.REJECT),
    )

    return all_screening, all_detailed
