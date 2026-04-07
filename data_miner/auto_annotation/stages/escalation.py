from __future__ import annotations

from ..contracts import PipelineState
from ..log_utils import get_logger
from ..registry import register_stage
from .base import Stage


logger = get_logger(__name__)


@register_stage("escalation")
class EscalationStage(Stage):
    kind = "escalation"

    def run(self, state: PipelineState) -> PipelineState:
        escalated = list(state.human_review)
        for candidate in state.flagged:
            review = state.reviews.get(candidate.candidate_id)
            if review is None or review.recommended_action in {"refine", "escalate"} or review.confidence_band == "low":
                escalated.append(candidate.model_copy(update={"status": "human_review"}))
        state.human_review = escalated
        state.history.append(f"escalation:human_review={len(state.human_review)}")
        logger.info("escalation.summary human_review=%s", len(state.human_review))
        return state