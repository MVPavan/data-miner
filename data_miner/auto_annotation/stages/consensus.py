from __future__ import annotations

from ..contracts import Candidate, CandidateCluster, PipelineState
from ..registry import register_stage
from ..utils import bbox_area, bbox_iou, weighted_box_fusion
from .base import Stage


@register_stage("consensus")
class ConsensusStage(Stage):
    kind = "consensus"

    def run(self, state: PipelineState) -> PipelineState:
        threshold = float(self.config.params.get("iou_threshold", 0.5))
        groups: dict[str, list[Candidate]] = {}
        for candidate in state.proposals:
            groups.setdefault(candidate.class_name, []).append(candidate)

        state.clusters = []
        state.accepted = []
        state.flagged = []
        state.rejected = []

        for class_name, candidates in groups.items():
            pending = list(candidates)
            index = 0
            while pending:
                seed = pending.pop(0)
                cluster_members = [seed]
                remainder = []
                for candidate in pending:
                    if bbox_iou(seed.bbox, candidate.bbox) >= threshold:
                        cluster_members.append(candidate)
                    else:
                        remainder.append(candidate)
                pending = remainder

                fused = weighted_box_fusion(cluster_members)
                agreement = len({candidate.source_model for candidate in cluster_members})
                quality = self._quality(fused)
                uncertainty = self._uncertainty(quality, agreement)
                decision = self._decision(quality, uncertainty, agreement)
                best = max(cluster_members, key=lambda item: item.score)
                final_candidate = best.model_copy(
                    update={
                        "bbox": fused,
                        "quality_score": quality,
                        "uncertainty_score": uncertainty,
                        "status": decision,
                    }
                )
                state.clusters.append(
                    CandidateCluster(
                        cluster_id=f"{class_name}:{index}",
                        class_name=class_name,
                        candidate_ids=[candidate.candidate_id for candidate in cluster_members],
                        source_models=sorted({candidate.source_model for candidate in cluster_members}),
                        fused_bbox=fused,
                        agreement_count=agreement,
                        quality_score=quality,
                        uncertainty_score=uncertainty,
                        decision=decision,
                    )
                )
                getattr(state, decision).append(final_candidate)
                index += 1

        state.history.append(f"consensus:accepted={len(state.accepted)}:flagged={len(state.flagged)}:rejected={len(state.rejected)}")
        return state

    def _quality(self, candidate_box) -> float:
        area = bbox_area(candidate_box)
        width = candidate_box.x2 - candidate_box.x1
        height = candidate_box.y2 - candidate_box.y1
        aspect = width / max(height, 1e-6)
        checks = [
            area > 0.001,
            area < 0.9,
            width > 0.01,
            height > 0.01,
            0.05 < aspect < 20.0,
            not (candidate_box.x1 <= 0.0 and candidate_box.x2 >= 0.99),
            not (candidate_box.y1 <= 0.0 and candidate_box.y2 >= 0.99),
        ]
        return sum(checks) / len(checks)

    def _uncertainty(self, quality: float, agreement: int) -> float:
        agreement_penalty = 1.0 - min(agreement / max(int(self.config.params.get("min_agreement", 2)), 1), 1.0)
        return min(1.0, (1.0 - quality) * 0.6 + agreement_penalty * 0.4)

    def _decision(self, quality: float, uncertainty: float, agreement: int) -> str:
        limits = self.pipeline_config.limits
        min_agreement = int(self.config.params.get("min_agreement", 2))
        if agreement >= min_agreement and quality >= limits.auto_accept_quality and uncertainty <= limits.auto_accept_uncertainty:
            return "accepted"
        if quality < limits.reject_quality or uncertainty > limits.reject_uncertainty:
            return "rejected"
        return "flagged"