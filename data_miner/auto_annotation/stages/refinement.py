from __future__ import annotations

from ..contracts import Candidate, PipelineState
from ..registry import register_stage
from ..utils import bbox_area
from .base import Stage


@register_stage("refinement")
class RefinementStage(Stage):
    kind = "refinement"

    def run(self, state: PipelineState) -> PipelineState:
        targets = set(self.config.params.get("targets", ["flagged"]))
        if "accepted" in targets:
            state.accepted = [self._refine_candidate(state, candidate) for candidate in state.accepted]
        if "flagged" in targets or state.reviews:
            state.flagged = [self._refine_candidate(state, candidate) for candidate in state.flagged]
        state.history.append(f"refinement:round={state.retry_round}")
        return state

    def _refine_candidate(self, state: PipelineState, candidate: Candidate) -> Candidate:
        class_pack = next(pack for pack in self.pipeline_config.classes if pack.name == candidate.class_name)
        review = state.reviews.get(candidate.candidate_id)
        model_names = self._models_for_candidate(review)
        best = candidate
        for model_name in model_names:
            adapter = self.adapters.get(model_name)
            if adapter is None:
                continue
            refined = None
            if review and review.next_stage == "proposal" and adapter.supports("proposal"):
                proposals = adapter.propose(state.image, class_pack, review.retry_expression or candidate.expression, self.config.params)
                if proposals:
                    refined = max(proposals, key=lambda item: item.score)
            elif adapter.supports("refinement"):
                refined = adapter.refine(state.image, best, class_pack, self.config.params, review)
            if refined and self._is_better(best, refined):
                best = refined.model_copy(update={"candidate_id": candidate.candidate_id})
        return best

    def _models_for_candidate(self, review) -> list[str]:
        if review and review.target_model:
            return [review.target_model]
        return self.config.models

    def _is_better(self, current: Candidate, refined: Candidate) -> bool:
        old_area = bbox_area(current.bbox)
        new_area = bbox_area(refined.bbox)
        area_penalty = abs(new_area - old_area)
        new_signal = refined.score + refined.quality_score - area_penalty
        old_signal = current.score + current.quality_score
        return new_signal >= old_signal