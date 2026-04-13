from __future__ import annotations

from collections import OrderedDict

from ..contracts import PipelineState
from ..log_utils import get_logger
from ..registry import register_stage
from .base import Stage


logger = get_logger(__name__)


@register_stage("proposal")
class ProposalStage(Stage):
    kind = "proposal"

    def run(self, state: PipelineState) -> PipelineState:
        proposals = list(state.proposals)
        per_class_limit = self.pipeline_config.limits.max_candidates_per_class
        for class_pack in self.pipeline_config.classes:
            class_candidates: list = []
            buckets: OrderedDict[str, list] = OrderedDict()
            expressions = class_pack.prompt_variants or [class_pack.name]
            for model_name in self.config.models:
                adapter = self.adapters[model_name]
                if not adapter.supports("proposal"):
                    continue
                for expression in expressions:
                    bucket_key = f"{model_name}:{expression}"
                    bucket_candidates = adapter.propose(state.image, class_pack, expression, self.config.params)
                    buckets[bucket_key] = bucket_candidates
                    class_candidates.extend(bucket_candidates)
                    logger.info(
                        "proposal.bucket class=%s source=%s expression=%s count=%s",
                        class_pack.name,
                        model_name,
                        expression,
                        len(bucket_candidates),
                    )
            interleaved = self._interleave_buckets(buckets, per_class_limit)
            proposals.extend(interleaved)
            state.history.append(f"proposal:{class_pack.name}:{len(interleaved)}")
            logger.info(
                "proposal.class class=%s total=%s selected=%s buckets=%s",
                class_pack.name,
                len(class_candidates),
                len(interleaved),
                len(buckets),
            )
        state.proposals = proposals
        return state

    def _interleave_buckets(self, buckets: OrderedDict[str, list], limit: int) -> list:
        queues = OrderedDict((key, list(values)) for key, values in buckets.items() if values)
        selected: list = []
        while queues and len(selected) < limit:
            empty_keys = []
            for key, values in queues.items():
                if len(selected) >= limit:
                    break
                if values:
                    selected.append(values.pop(0))
                if not values:
                    empty_keys.append(key)
            for key in empty_keys:
                queues.pop(key, None)
        return selected