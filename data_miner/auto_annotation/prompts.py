from __future__ import annotations

from .config import ClassPackConfig
from .contracts import Candidate


def build_verification_prompt(class_pack: ClassPackConfig, candidate: Candidate) -> str:
    synonyms = ", ".join(class_pack.synonyms) or "none"
    negatives = ", ".join(class_pack.negatives) or "none"
    return (
        "You are validating one candidate annotation. "
        f"Expected class: {class_pack.name}. "
        f"Known synonyms: {synonyms}. "
        f"Hard negatives: {negatives}. "
        f"Current label: {candidate.label}. "
        f"Current source model: {candidate.source_model}. "
        "Return JSON only with: "
        '{"semantic_match":"yes|no|uncertain",'
        '"bbox_tight":"tight|loose|too_small|uncertain",'
        '"recommended_action":"accept|relabel|refine|reject|escalate",'
        '"confidence_band":"high|medium|low",'
        '"rationale_short":"...",'
        '"relabel_to":"optional",'
        '"target_model":"optional",'
        '"retry_expression":"optional",'
        '"next_stage":"proposal|refinement|escalation|none"}'
    )