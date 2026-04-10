"""VLM refinement stage: propose refinements via VLM, execute via detection models."""

from __future__ import annotations

from PIL import Image
from pydantic_ai.messages import BinaryContent

from ..agents.refinement import RefinementDeps, build_refinement_agent, parse_refinement_proposal
from ..config import AutoAnnotationV2Config
from ..contracts import (
    Candidate,
    CandidateStatus,
    DetailedVerdict,
    RefinementAction,
    RefinementProposal,
    RefinementStrategy,
    ScreeningVerdict,
    VLMDecision,
)
from ..log_utils import get_logger
from ..stages.proposal import _refine_with_sam, get_loaded_model
from ..utils import draw_candidates_numbered, pil_to_png_bytes

logger = get_logger(__name__)


def _candidates_needing_refinement(
    candidates: list[Candidate],
    screening: list[ScreeningVerdict],
    detailed: list[DetailedVerdict],
) -> list[Candidate]:
    """Identify candidates that should be refined."""
    # From screening: needs_review that weren't resolved in detailed
    detailed_ids = {v.candidate_id for v in detailed}
    screening_review_ids = {
        v.candidate_id for v in screening if v.decision == VLMDecision.NEEDS_REVIEW
    }

    # From detailed: still needs_review
    still_review_ids = {
        v.candidate_id for v in detailed if v.decision == VLMDecision.NEEDS_REVIEW
    }

    # Candidates that need refinement: unresolved needs_review
    refine_ids = (screening_review_ids - detailed_ids) | still_review_ids
    return [c for c in candidates if c.candidate_id in refine_ids]


def _format_refinement_candidates(
    candidates: list[Candidate],
    detailed: list[DetailedVerdict],
) -> str:
    """Format candidates with their review feedback for the refinement prompt."""
    verdict_map = {v.candidate_id: v for v in detailed}
    lines: list[str] = []
    for cand in candidates:
        verdict = verdict_map.get(cand.candidate_id)
        feedback = ""
        if verdict:
            feedback = f" feedback=[{verdict.reasoning}]"
            if verdict.refinement_hint:
                feedback += f" hint=[{verdict.refinement_hint}]"
        box = cand.bbox
        lines.append(
            f"- id={cand.candidate_id} class={cand.class_name} "
            f"source={cand.source_model} score={cand.score:.3f} "
            f"bbox=({box.x1:.3f},{box.y1:.3f},{box.x2:.3f},{box.y2:.3f})"
            f"{feedback}"
        )
    return "\n".join(lines)


async def _get_refinement_proposals(
    image: Image.Image,
    to_refine: list[Candidate],
    all_candidates: list[Candidate],
    detailed: list[DetailedVerdict],
    config: AutoAnnotationV2Config,
) -> RefinementProposal:
    """Ask VLM what refinements to make."""
    agent = build_refinement_agent(config)

    # Find class pack for these candidates (assume same class for simplicity)
    class_name = to_refine[0].class_name
    class_pack = next(
        (cp for cp in config.classes if cp.name == class_name), config.classes[0]
    )

    deps = RefinementDeps(
        class_pack=class_pack,
        candidate_descriptions=_format_refinement_candidates(to_refine, detailed),
        available_models=config.refinement.refinement_models,
    )

    annotated = draw_candidates_numbered(image, to_refine)
    image_parts: list[str | BinaryContent] = [
        BinaryContent(data=pil_to_png_bytes(annotated), media_type="image/png"),
        f"These candidates need refinement:\n\n{deps.candidate_descriptions}\n\n"
        f"Image dimensions: {image.size[0]}x{image.size[1]} pixels.\n"
        f"Propose refinements for each candidate.",
    ]

    result = await agent.run(image_parts, deps=deps)
    return parse_refinement_proposal(result.output)


def _execute_sam_refinement(
    image: Image.Image,
    candidate: Candidate,
    action: RefinementAction,
    config: AutoAnnotationV2Config,
) -> Candidate | None:
    """Execute a SAM-based refinement."""
    loaded = get_loaded_model("sam", config)

    if action.strategy == RefinementStrategy.SAM_POINTS and action.points:
        points = [(p.x, p.y, p.label) for p in action.points]
        return _refine_with_sam(loaded, image, candidate, points=points)
    else:
        return _refine_with_sam(loaded, image, candidate)


def _execute_repropose(
    image: Image.Image,
    candidate: Candidate,
    action: RefinementAction,
    config: AutoAnnotationV2Config,
) -> list[Candidate]:
    """Execute a re-proposal with a new text prompt."""
    from ..stages.proposal import _RUNNERS
    from ..stages.proposal import get_loaded_model as _get_model

    model_name = action.target_model
    model_cfg = config.detection_models.get(model_name)
    if model_cfg is None:
        logger.warning("Refinement target model '%s' not found", model_name)
        return []

    loaded = _get_model(model_name, config)
    runner = _RUNNERS.get(model_cfg.kind)
    if runner is None:
        return []

    class_pack = next(
        (cp for cp in config.classes if cp.name == candidate.class_name),
        config.classes[0],
    )
    expression = action.text_prompt or candidate.expression

    try:
        return runner(
            loaded, image, class_pack, expression, model_cfg.params, model_name
        )
    except Exception:
        logger.exception("Re-proposal failed for %s", candidate.candidate_id)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_vlm_refinement(
    image: Image.Image,
    candidates: list[Candidate],
    screening: list[ScreeningVerdict],
    detailed: list[DetailedVerdict],
    config: AutoAnnotationV2Config,
) -> tuple[list[Candidate], list[RefinementAction]]:
    """Run VLM-guided refinement. Returns (refined_candidates, actions_taken)."""
    if not config.refinement.enabled:
        logger.info("Refinement disabled, skipping")
        return candidates, []

    to_refine = _candidates_needing_refinement(candidates, screening, detailed)
    if not to_refine:
        logger.info("No candidates need refinement")
        return candidates, []

    logger.info("Refinement: %d candidates need improvement", len(to_refine))

    # Group by class for VLM call efficiency
    by_class: dict[str, list[Candidate]] = {}
    for c in to_refine:
        by_class.setdefault(c.class_name, []).append(c)

    all_actions: list[RefinementAction] = []
    for class_name, class_cands in by_class.items():
        try:
            proposal = await _get_refinement_proposals(
                image, class_cands, candidates, detailed, config
            )
            all_actions.extend(proposal.actions)
        except Exception:
            logger.exception("Refinement proposal failed for class '%s'", class_name)

    # Execute refinement actions
    cand_map = {c.candidate_id: c for c in candidates}
    refined_ids: set[str] = set()

    for action in all_actions:
        cand = cand_map.get(action.candidate_id)
        if cand is None:
            continue

        if action.strategy in (
            RefinementStrategy.SAM_POINTS,
            RefinementStrategy.SAM_BOX,
        ):
            refined = _execute_sam_refinement(image, cand, action, config)
            if refined:
                cand_map[action.candidate_id] = refined
                refined_ids.add(action.candidate_id)
                logger.info("SAM refined %s", action.candidate_id)

        elif action.strategy == RefinementStrategy.REPROPOSE_TEXT:
            new_candidates = _execute_repropose(image, cand, action, config)
            if new_candidates:
                best = max(new_candidates, key=lambda c: c.score)
                refined = cand.model_copy(
                    update={
                        "bbox": best.bbox,
                        "score": best.score,
                        "status": CandidateStatus.REFINED,
                        "source_model": f"{cand.source_model}+{action.target_model}",
                        "notes": [*cand.notes, f"reproposed:{action.text_prompt}"],
                    }
                )
                cand_map[action.candidate_id] = refined
                refined_ids.add(action.candidate_id)
                logger.info(
                    "Re-proposed %s via %s", action.candidate_id, action.target_model
                )

    logger.info(
        "Refinement complete: %d/%d candidates refined",
        len(refined_ids),
        len(to_refine),
    )
    return list(cand_map.values()), all_actions
