"""PydanticAI agents for VLM reasoning (screening + detailed review).

Uses output_type=str with JSON schema in system prompt because vLLM
doesn't support tool_choice (requires --tool-call-parser).
The caller parses the JSON manually via Pydantic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..config import AutoAnnotationV2Config, ClassPackConfig, VLMConfig
from ..contracts import DetailedVerdict, ScreeningResult

# ---------------------------------------------------------------------------
# Dependencies injected into agents at runtime
# ---------------------------------------------------------------------------


@dataclass
class ScreeningDeps:
    """Dependencies for the screening pass."""

    class_pack: ClassPackConfig
    candidate_descriptions: str  # Numbered list of candidates for the prompt
    num_candidates: int


@dataclass
class DetailedReviewDeps:
    """Dependencies for the detailed per-candidate review."""

    class_pack: ClassPackConfig
    candidate_id: str
    candidate_label: str
    candidate_source: str
    candidate_score: float


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_vlm_model(vlm_cfg: VLMConfig) -> OpenAIChatModel:
    provider = OpenAIProvider(base_url=vlm_cfg.base_url, api_key=vlm_cfg.api_key)
    return OpenAIChatModel(vlm_cfg.model_name, provider=provider)


def _build_model_settings(vlm_cfg: VLMConfig) -> ModelSettings:
    return ModelSettings(
        temperature=vlm_cfg.temperature,
        max_tokens=vlm_cfg.max_tokens,
        timeout=vlm_cfg.timeout,
    )


def _extract_json(text: str) -> dict:
    """Extract JSON from model response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` or ``` ... ```
        lines = text.split("\n")
        lines = lines[1:]  # remove opening ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Screening Agent (Pass 1): batch evaluation of all candidates for one class
# ---------------------------------------------------------------------------

_SCREENING_SCHEMA = ScreeningResult.model_json_schema()

_SCREENING_INSTRUCTIONS = """\
You are a strict annotation quality reviewer for object detection.
You are evaluating candidate bounding box annotations on an image.

Target class: {class_name}
Known synonyms: {synonyms}
Hard negatives (things that look similar but are NOT this class): {negatives}
Positive prompt variants (any one matching is valid): {positive_prompts}

You will see an annotated image with numbered bounding boxes.
For EACH numbered candidate, determine:
- "accept": the box correctly contains the target object with a tight fit
- "needs_review": uncertain — the box might be correct but needs closer inspection
- "reject": the box does NOT contain the target object, or is very poorly fitted

Be strict. When in doubt, mark as "needs_review" rather than "accept".
Return your assessment for ALL {num_candidates} candidates.

You MUST respond with ONLY valid JSON (no extra text) matching this schema:
{schema}

decision must be one of: "accept", "needs_review", "reject".
confidence must be between 0.0 and 1.0.
"""


def build_screening_agent(
    config: AutoAnnotationV2Config,
) -> Agent[ScreeningDeps, str]:
    """Create the screening agent (Pass 1). Returns raw JSON string."""
    vlm_model = _build_vlm_model(config.vlm)

    agent: Agent[ScreeningDeps, str] = Agent(
        vlm_model,
        deps_type=ScreeningDeps,
        output_type=str,
        retries=config.vlm.max_retries,
        model_settings=_build_model_settings(config.vlm),
    )

    @agent.instructions
    def screening_instructions(ctx) -> str:
        deps: ScreeningDeps = ctx.deps
        cp = deps.class_pack
        return _SCREENING_INSTRUCTIONS.format(
            class_name=cp.name,
            synonyms=", ".join(cp.synonyms) or "(none)",
            negatives=", ".join(cp.negatives) or "(none)",
            positive_prompts=", ".join(cp.prompt_variants) or cp.name,
            num_candidates=deps.num_candidates,
            schema=json.dumps(_SCREENING_SCHEMA, indent=2),
        )

    return agent


def parse_screening_result(raw: str) -> ScreeningResult:
    """Parse raw JSON string from screening agent into ScreeningResult."""
    return ScreeningResult.model_validate(_extract_json(raw))


# ---------------------------------------------------------------------------
# Detailed Review Agent (Pass 2): per-candidate deep analysis
# ---------------------------------------------------------------------------

_DETAILED_SCHEMA = DetailedVerdict.model_json_schema()

_DETAILED_INSTRUCTIONS = """\
You are a meticulous annotation quality reviewer for object detection.
You are examining ONE specific candidate annotation in detail.

Target class: {class_name}
Known synonyms: {synonyms}
Hard negatives (NOT this class): {negatives}
Positive prompt variants (any one matching is valid): {positive_prompts}

Candidate info:
- ID: {candidate_id}
- Label: {candidate_label}
- Source model: {candidate_source}
- Detection score: {candidate_score:.3f}

Evaluate carefully:
1. Semantic match: does the box contain the target class? (yes/no/uncertain)
2. Bbox quality: is the box tight/loose/too_small/uncertain?
3. If the label is wrong, suggest what it should be relabeled to.
4. If the box needs adjustment, describe how in refinement_hint.

Be precise and strict in your assessment.

You MUST respond with ONLY valid JSON (no extra text) matching this schema:
{schema}

decision must be one of: "accept", "needs_review", "reject".
semantic_match must be one of: "yes", "no", "uncertain".
bbox_quality must be one of: "tight", "loose", "too_small", "uncertain".
confidence must be between 0.0 and 1.0.
"""


def build_detailed_agent(
    config: AutoAnnotationV2Config,
) -> Agent[DetailedReviewDeps, str]:
    """Create the detailed review agent (Pass 2). Returns raw JSON string."""
    vlm_model = _build_vlm_model(config.vlm)

    agent: Agent[DetailedReviewDeps, str] = Agent(
        vlm_model,
        deps_type=DetailedReviewDeps,
        output_type=str,
        retries=config.vlm.max_retries,
        model_settings=_build_model_settings(config.vlm),
    )

    @agent.instructions
    def detailed_instructions(ctx) -> str:
        deps: DetailedReviewDeps = ctx.deps
        cp = deps.class_pack
        return _DETAILED_INSTRUCTIONS.format(
            class_name=cp.name,
            synonyms=", ".join(cp.synonyms) or "(none)",
            negatives=", ".join(cp.negatives) or "(none)",
            positive_prompts=", ".join(cp.prompt_variants) or cp.name,
            candidate_id=deps.candidate_id,
            candidate_label=deps.candidate_label,
            candidate_source=deps.candidate_source,
            candidate_score=deps.candidate_score,
            schema=json.dumps(_DETAILED_SCHEMA, indent=2),
        )

    return agent


def parse_detailed_verdict(raw: str) -> DetailedVerdict:
    """Parse raw JSON string from detailed agent into DetailedVerdict."""
    return DetailedVerdict.model_validate(_extract_json(raw))
