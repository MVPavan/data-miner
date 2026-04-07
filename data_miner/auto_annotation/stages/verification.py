from __future__ import annotations

from ..contracts import PipelineState
from ..registry import register_stage
from .base import Stage


@register_stage("verification")
class VerificationStage(Stage):
    kind = "verification"

    def run(self, state: PipelineState) -> PipelineState:
        accepted = list(state.accepted)
        rejected = list(state.rejected)
        flagged = []
        for candidate in state.flagged:
            class_pack = next(pack for pack in self.pipeline_config.classes if pack.name == candidate.class_name)
            decision = None
            for model_name in self.config.models:
                adapter = self.adapters[model_name]
                if adapter.supports("verification"):
                    decision = adapter.verify(state.image, candidate, class_pack, self.config.params)
                    break
            if decision is None:
                flagged.append(candidate)
                continue
            state.reviews[candidate.candidate_id] = decision
            if decision.recommended_action == "accept":
                accepted.append(candidate.model_copy(update={"status": "accepted"}))
            elif decision.recommended_action == "relabel":
                label = decision.relabel_to or candidate.label
                class_name = decision.relabel_to or candidate.class_name
                accepted.append(candidate.model_copy(update={"status": "accepted", "label": label, "class_name": class_name}))
            elif decision.recommended_action == "reject":
                rejected.append(candidate.model_copy(update={"status": "rejected"}))
            else:
                flagged.append(candidate.model_copy(update={"status": "flagged"}))
        state.accepted = accepted
        state.rejected = rejected
        state.flagged = flagged
        state.history.append(f"verification:flagged={len(flagged)}")
        return state