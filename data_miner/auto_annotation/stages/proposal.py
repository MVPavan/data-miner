from __future__ import annotations

from ..contracts import PipelineState
from ..registry import register_stage
from .base import Stage


@register_stage("proposal")
class ProposalStage(Stage):
    kind = "proposal"

    def run(self, state: PipelineState) -> PipelineState:
        proposals = list(state.proposals)
        per_class_limit = self.pipeline_config.limits.max_candidates_per_class
        for class_pack in self.pipeline_config.classes:
            class_candidates = []
            expressions = class_pack.prompt_variants or [class_pack.name]
            for model_name in self.config.models:
                adapter = self.adapters[model_name]
                if not adapter.supports("proposal"):
                    continue
                for expression in expressions:
                    class_candidates.extend(adapter.propose(state.image, class_pack, expression, self.config.params))
            proposals.extend(class_candidates[:per_class_limit])
            state.history.append(f"proposal:{class_pack.name}:{len(class_candidates[:per_class_limit])}")
        state.proposals = proposals
        return state