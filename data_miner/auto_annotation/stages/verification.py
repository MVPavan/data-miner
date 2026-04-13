from __future__ import annotations

from ..contracts import FailureRecord, PipelineState, ReviewDecision
from ..log_utils import get_logger
from ..registry import register_stage
from ..utils import resolve_canonical_class_name
from .base import Stage


logger = get_logger(__name__)


@register_stage("verification")
class VerificationStage(Stage):
    kind = "verification"

    def requests_retry(self, state: PipelineState) -> bool:
        return any(review.recommended_action == "refine" for review in state.reviews.values())

    def run(self, state: PipelineState) -> PipelineState:
        accepted = list(state.accepted)
        rejected = list(state.rejected)
        flagged = []
        class_alias_map = {pack.name: pack for pack in self.pipeline_config.classes}
        for candidate in state.flagged:
            class_pack = class_alias_map[candidate.class_name]
            decision = None
            for model_name in self.config.models:
                adapter = self.adapters[model_name]
                if adapter.supports("verification"):
                    try:
                        decision = adapter.verify(state.image, candidate, class_pack, self.config.params)
                    except Exception as exc:
                        decision = ReviewDecision(
                            candidate_id=candidate.candidate_id,
                            semantic_match="uncertain",
                            bbox_tight="uncertain",
                            recommended_action="escalate",
                            confidence_band="low",
                            rationale_short="Verification failed.",
                            next_stage="escalation",
                            failure_type="verification_exception",
                            failure_reason=str(exc),
                        )
                        logger.exception(
                            "verification.candidate_failed candidate_id=%s model=%s",
                            candidate.candidate_id,
                            model_name,
                        )
                    logger.info(
                        "verification.candidate candidate_id=%s model=%s action=%s confidence=%s",
                        candidate.candidate_id,
                        model_name,
                        decision.recommended_action,
                        decision.confidence_band,
                    )
                    break
            if decision is None:
                flagged.append(candidate)
                continue
            if decision.failure_type:
                state.failures.append(
                    FailureRecord(
                        scope="transport" if decision.failure_type in {"transport_error", "http_error", "timeout_error"} else "candidate",
                        stage=self.kind,
                        candidate_id=candidate.candidate_id,
                        error_type=decision.failure_type,
                        message=decision.failure_reason or decision.rationale_short,
                        retriable=decision.failure_type in {"transport_error", "http_error", "timeout_error"},
                    )
                )
            if decision.recommended_action == "relabel":
                canonical_name = resolve_canonical_class_name(decision.relabel_to, self.pipeline_config.classes)
                if canonical_name is None:
                    raw_label = decision.relabel_to or "<missing>"
                    decision = decision.model_copy(
                        update={
                            "recommended_action": "escalate",
                            "confidence_band": "low",
                            "rationale_short": f"{decision.rationale_short} Unresolved relabel target: {raw_label}.",
                            "next_stage": "escalation",
                        }
                    )
                    logger.warning(
                        "verification.relabel_unresolved candidate_id=%s relabel_to=%s",
                        candidate.candidate_id,
                        raw_label,
                    )
            state.reviews[candidate.candidate_id] = decision
            if decision.recommended_action == "accept":
                accepted.append(candidate.model_copy(update={"status": "accepted"}))
            elif decision.recommended_action == "relabel":
                canonical_name = resolve_canonical_class_name(decision.relabel_to, self.pipeline_config.classes) or candidate.class_name
                label = decision.relabel_to or canonical_name
                accepted.append(
                    candidate.model_copy(
                        update={
                            "status": "accepted",
                            "label": label,
                            "class_name": canonical_name,
                            "notes": [*candidate.notes, f"relabel:{label}->{canonical_name}"],
                        }
                    )
                )
            elif decision.recommended_action == "reject":
                rejected.append(candidate.model_copy(update={"status": "rejected"}))
            else:
                flagged.append(
                    candidate.model_copy(
                        update={
                            "status": "flagged",
                            "notes": [*candidate.notes, *( [f"verification_failed:{decision.failure_type}"] if decision.failure_type else [] )],
                        }
                    )
                )
        state.accepted = accepted
        state.rejected = rejected
        state.flagged = flagged
        state.history.append(f"verification:flagged={len(flagged)}")
        logger.info(
            "verification.summary accepted=%s rejected=%s flagged=%s",
            len(accepted),
            len(rejected),
            len(flagged),
        )
        return state