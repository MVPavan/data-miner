"""Cross-frame cluster detections that refer to the same physical object.

Within one frame group, every finalize annotation across every image is
flattened into a single pool. Annotations of the *same class* whose bboxes
overlap at IoU ≥ threshold are clustered — the assumption is that within a
group (similar camera pose), high-IoU same-class detections in different
frames are looking at the same physical (static) object.

Clustering is greedy single-link: start with the highest-confidence
annotation, pull in any compatible annotation, repeat. Output is a list of
``DetectionCluster`` entries. Singletons (only seen in 1 image) are emitted
too — the caller decides whether to skip them via ``min_positive_frames``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from data_miner.auto_annotation_v4.configs.contracts import (
    BoundingBox,
    FinalAnnotation,
)

__all__ = ["DetectionCluster", "AnnotationRef", "iou", "build_clusters"]


@dataclass(frozen=True)
class AnnotationRef:
    """A pointer to one ``FinalAnnotation`` in one image."""

    image_id: str
    annotation: FinalAnnotation


@dataclass
class DetectionCluster:
    """A set of cross-frame finalize annotations that refer to the same object.

    ``positive_image_ids`` is the set of images that contributed an annotation;
    every other image in the group is a *propagation candidate* for this
    cluster.
    """

    cluster_id: str
    class_name: str
    members: list[AnnotationRef] = field(default_factory=list)

    @property
    def positive_image_ids(self) -> set[str]:
        return {m.image_id for m in self.members}

    @property
    def class_id(self) -> int:
        # All members share class_name; class_id may differ per finalize row
        # in pathological data — use the most common or first.
        if not self.members:
            return 0
        return self.members[0].annotation.class_id

    def canonical_bbox(self) -> BoundingBox:
        """Score-weighted average of member bboxes.

        Falls back to a simple mean if all confidences are zero.
        """
        weights = [max(m.annotation.confidence, 0.0) for m in self.members]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(self.members)
            total = float(len(self.members))
        x1 = sum(w * m.annotation.bbox.x1 for w, m in zip(weights, self.members)) / total
        y1 = sum(w * m.annotation.bbox.y1 for w, m in zip(weights, self.members)) / total
        x2 = sum(w * m.annotation.bbox.x2 for w, m in zip(weights, self.members)) / total
        y2 = sum(w * m.annotation.bbox.y2 for w, m in zip(weights, self.members)) / total
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union for two normalized bboxes. 0.0 on empty union."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_clusters(
    refs: Sequence[AnnotationRef],
    *,
    iou_threshold: float = 0.5,
    group_id: str = "",
) -> list[DetectionCluster]:
    """Greedy single-link cluster of cross-frame annotations by class + IoU.

    ``refs`` is a flat list across all images in a group. The clusterer
    requires same ``class_name`` AND IoU ≥ threshold to merge; matches across
    different *images* are preferred (we never merge two annotations from the
    *same* image — that's already aav4's job).

    Anchors are picked highest-confidence first so a cluster's centroid is
    biased toward the strongest evidence.
    """
    # Sort by confidence desc; stable on (image_id, candidate_id) for determinism.
    sorted_refs = sorted(
        refs,
        key=lambda r: (
            -r.annotation.confidence,
            r.image_id,
            r.annotation.candidate_id,
        ),
    )

    clusters: list[DetectionCluster] = []
    consumed: set[tuple[str, str]] = set()  # (image_id, candidate_id)

    for anchor in sorted_refs:
        anchor_key = (anchor.image_id, anchor.annotation.candidate_id)
        if anchor_key in consumed:
            continue
        consumed.add(anchor_key)
        cluster = DetectionCluster(
            cluster_id=f"{group_id}::{len(clusters)}" if group_id else f"cluster_{len(clusters)}",
            class_name=anchor.annotation.class_name,
            members=[anchor],
        )

        for other in sorted_refs:
            other_key = (other.image_id, other.annotation.candidate_id)
            if other_key in consumed:
                continue
            if other.image_id == anchor.image_id:
                continue  # one annotation per image per cluster
            if other.annotation.class_name != anchor.annotation.class_name:
                continue
            if any(other.image_id == m.image_id for m in cluster.members):
                continue  # already have an annotation from that image
            if iou(anchor.annotation.bbox, other.annotation.bbox) < iou_threshold:
                continue
            cluster.members.append(other)
            consumed.add(other_key)

        clusters.append(cluster)

    return clusters
