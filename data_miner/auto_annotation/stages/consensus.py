from __future__ import annotations

from collections import defaultdict

from ..contracts import Candidate, CandidateCluster, PipelineState
from ..log_utils import get_logger
from ..registry import register_stage
from ..utils import bbox_area, bbox_iou, weighted_box_fusion
from .base import Stage


logger = get_logger(__name__)


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
            index = 0
            for cluster_members in self._build_clusters(candidates, threshold):
                fused = weighted_box_fusion(cluster_members)
                agreement = len({candidate.source_model for candidate in cluster_members})
                geometry = self._geometry_score(fused)
                max_confidence = self._max_scored_confidence(cluster_members)
                quality = self._quality(cluster_members, agreement, geometry, max_confidence)
                uncertainty = self._uncertainty(cluster_members, agreement, geometry, max_confidence, quality)
                decision = self._decision(cluster_members, agreement, geometry, max_confidence, quality, uncertainty)
                best = max(cluster_members, key=lambda item: item.score)
                final_candidate = best.model_copy(
                    update={
                        "bbox": fused,
                        "quality_score": quality,
                        "uncertainty_score": uncertainty,
                        "status": decision,
                        "metadata": {
                            **best.metadata,
                            "support_models": sorted({candidate.source_model for candidate in cluster_members}),
                            "agreement_count": agreement,
                            "geometry_score": geometry,
                            "max_scored_confidence": max_confidence,
                        },
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
                logger.info(
                    "consensus.cluster class=%s cluster=%s members=%s agreement=%s quality=%.3f uncertainty=%.3f decision=%s",
                    class_name,
                    index,
                    len(cluster_members),
                    agreement,
                    quality,
                    uncertainty,
                    decision,
                )
                index += 1

        state.history.append(f"consensus:accepted={len(state.accepted)}:flagged={len(state.flagged)}:rejected={len(state.rejected)}")
        logger.info(
            "consensus.summary accepted=%s flagged=%s rejected=%s",
            len(state.accepted),
            len(state.flagged),
            len(state.rejected),
        )
        return state

    def _build_clusters(self, candidates: list[Candidate], threshold: float) -> list[list[Candidate]]:
        size = len(candidates)
        if size <= 1:
            return [candidates] if candidates else []

        parent = list(range(size))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(size):
            for right in range(left + 1, size):
                if bbox_iou(candidates[left].bbox, candidates[right].bbox) >= threshold:
                    union(left, right)

        grouped: dict[int, list[Candidate]] = defaultdict(list)
        for index, candidate in enumerate(candidates):
            grouped[find(index)].append(candidate)
        return [grouped[key] for key in sorted(grouped)]

    def _geometry_score(self, candidate_box) -> float:
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

    def _max_scored_confidence(self, candidates: list[Candidate]) -> float:
        scored = [candidate.score for candidate in candidates if not candidate.metadata.get("binary_support_only", False)]
        return max(scored) if scored else 0.0

    def _quality(self, candidates: list[Candidate], agreement: int, geometry: float, max_confidence: float) -> float:
        min_agreement = max(int(self.config.params.get("min_agreement", 2)), 1)
        support_strength = min(agreement / min_agreement, 1.0)
        evidence = support_strength if max_confidence <= 0 else (0.55 * support_strength + 0.45 * max_confidence)
        penalty = max(0.0, 0.7 - geometry) * 0.15
        return max(0.0, min(1.0, evidence - penalty))

    def _uncertainty(self, candidates: list[Candidate], agreement: int, geometry: float, max_confidence: float, quality: float) -> float:
        min_agreement = max(int(self.config.params.get("min_agreement", 2)), 1)
        support_gap = 1.0 - min(agreement / min_agreement, 1.0)
        confidence_gap = 0.0 if max_confidence <= 0 else max(0.0, 0.65 - max_confidence)
        geometry_gap = max(0.0, 0.6 - geometry)
        uncertainty = 0.45 * support_gap + 0.4 * confidence_gap + 0.15 * geometry_gap
        return max(0.0, min(1.0, max(uncertainty, 1.0 - quality)))

    def _decision(self, candidates: list[Candidate], agreement: int, geometry: float, max_confidence: float, quality: float, uncertainty: float) -> str:
        limits = self.pipeline_config.limits
        min_agreement = int(self.config.params.get("min_agreement", 2))
        if geometry < 0.35:
            return "rejected"
        if agreement >= min_agreement and quality >= limits.auto_accept_quality and uncertainty <= limits.auto_accept_uncertainty:
            return "accepted"
        if quality < limits.reject_quality and uncertainty > limits.reject_uncertainty and max_confidence < 0.5 and agreement < min_agreement:
            return "rejected"
        return "flagged"