"""PydanticAI agent for VLM-guided refinement proposals."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from ..config import AutoAnnotationV2Config, ClassPackConfig, VLMConfig
from ..contracts import RefinementProposal


@dataclass
class RefinementDeps:
    """Dependencies for the refinement agent."""

    class_pack: ClassPackConfig
    candidate_descriptions: str
    available_models: list[str]


def _build_vlm_model(vlm_cfg: VLMConfig) -> OpenAIModel:
    return OpenAIModel(
        vlm_cfg.model_name,
        provider="openai",
        base_url=vlm_cfg.base_url,
        api_key=vlm_cfg.api_key,
    )


_REFINEMENT_INSTRUCTIONS = """\
You are an annotation refinement specialist. You receive candidates that need \
improvement based on prior review feedback.

Target class: {class_name}
Available refinement models: {available_models}

For each candidate, propose a refinement strategy:
- "sam_points": provide specific pixel coordinate points (foreground/background) \
for SAM to re-segment. Use this when the box boundary is wrong.
- "sam_box": provide a corrected box prompt for SAM. Use when the box needs \
tightening or expansion.
- "repropose_text": provide a new text prompt for re-detection with Falcon or \
GroundingDINO. Use when the original expression was too vague.

For sam_points, provide point coordinates as [x, y] pixel values relative to \
the image dimensions shown.
For repropose_text, provide a clear, specific expression.

Be specific and actionable in your proposals.
"""


def build_refinement_agent(
    config: AutoAnnotationV2Config,
) -> Agent[RefinementDeps, RefinementProposal]:
    """Create the refinement proposal agent."""
    vlm_model = _build_vlm_model(config.vlm)

    agent: Agent[RefinementDeps, RefinementProposal] = Agent(
        vlm_model,
        deps_type=RefinementDeps,
        output_type=RefinementProposal,
        retries=config.vlm.max_retries,
        model_settings=ModelSettings(
            temperature=config.vlm.temperature,
            max_tokens=config.vlm.max_tokens,
            timeout=config.vlm.timeout,
        ),
    )

    @agent.instructions
    def refinement_instructions(ctx) -> str:
        deps: RefinementDeps = ctx.deps
        return _REFINEMENT_INSTRUCTIONS.format(
            class_name=deps.class_pack.name,
            available_models=", ".join(deps.available_models),
        )

    return agent
