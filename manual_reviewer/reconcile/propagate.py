"""Orchestrate cross-frame static-object propagation via SAM3-DART image-mode.

For each frame group:
  1. Collect all finalize annotations across the group.
  2. Cluster cross-frame annotations of the same class at IoU ≥ T.
  3. For every cluster with ≥ ``min_positive_frames`` positive members,
     iterate the *missing* images in the group and ask SAM3-DART to refine
     the cluster's canonical box on that image.
  4. Accept the propagation if the refined mask passes both:
       - ``mask_score ≥ accept_score``
       - IoU(seed, refined) ≥ ``accept_iou`` (refined box stays at the seed
         location — i.e., the object really is there, not somewhere else).
  5. Bundle accepted/rejected propagations per image as ``ReconcileResult``.

This file contains pure orchestration — no I/O, no HTTP. The Sam3Client
protocol is the only injection point. Real runs use ``Sam3HttpClient``;
tests pass a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from data_miner.auto_annotation_v4.configs.contracts import (
    FinalAnnotation,
    PropagationVote,
    ReconciledDetection,
    ReconcileResult,
)

from .clustering import (
    AnnotationRef,
    DetectionCluster,
    build_clusters,
    iou,
)
from .sam3_client import RefineResponse, Sam3Client

__all__ = [
    "PropagationConfig",
    "ImageContext",
    "reconcile_group",
]


@dataclass(frozen=True)
class PropagationConfig:
    """Tunables for the reconciler. Sane defaults for static-camera footage."""

    cluster_iou_threshold: float = 0.5
    """Same-class boxes from different frames merge above this IoU."""

    min_positive_frames: int = 2
    """Cluster must appear in ≥ N positive frames before propagating."""

    accept_score: float = 0.5
    """SAM3-DART mask_score floor — below this the object isn't there."""

    accept_iou: float = 0.7
    """IoU(seed, refined) floor — below this the refined box drifted too
    far from the seed location to be the same object."""

    refine_threshold: float = 0.5
    """Threshold passed to SAM3-DART's /refine endpoint (its internal
    binarization cutoff). Independent of accept_score."""


@dataclass(frozen=True)
class ImageContext:
    """Per-image inputs to the reconciler."""

    image_id: str
    image_path: str
    final_annotations: list[FinalAnnotation]


def reconcile_group(
    group_id: str,
    images: list[ImageContext],
    *,
    client: Sam3Client,
    config: PropagationConfig | None = None,
) -> dict[str, ReconcileResult]:
    """Run cross-frame propagation for one group; return per-image results.

    The returned dict keys are image_ids; every image in the group gets an
    entry, even if it received no propagations (empty ``propagated`` list).
    Callers can persist the empty rows or filter them — the writer handles
    both cases.
    """
    cfg = config or PropagationConfig()

    # Index for fast missing-frame lookup
    by_id: Mapping[str, ImageContext] = {im.image_id: im for im in images}

    # Flatten cross-frame annotations into the cluster pool
    refs: list[AnnotationRef] = []
    for im in images:
        for ann in im.final_annotations:
            refs.append(AnnotationRef(image_id=im.image_id, annotation=ann))

    clusters = build_clusters(
        refs,
        iou_threshold=cfg.cluster_iou_threshold,
        group_id=group_id,
    )

    # Per-image result accumulators (all images get an entry).
    per_image_propagated: dict[str, list[ReconciledDetection]] = {
        im.image_id: [] for im in images
    }
    per_image_rejected: dict[str, list[ReconciledDetection]] = {
        im.image_id: [] for im in images
    }

    for cluster in clusters:
        positives = cluster.positive_image_ids
        if len(positives) < cfg.min_positive_frames:
            continue
        seed_bbox = cluster.canonical_bbox()
        votes = [
            PropagationVote(
                image_id=m.image_id,
                candidate_id=m.annotation.candidate_id,
                score=m.annotation.confidence,
                source_model=m.annotation.source_model,
            )
            for m in cluster.members
        ]

        missing_image_ids = sorted(set(by_id) - positives)
        for target_id in missing_image_ids:
            target = by_id[target_id]
            try:
                resp: RefineResponse = client.refine(
                    image_path=target.image_path,
                    bbox=seed_bbox,
                    threshold=cfg.refine_threshold,
                )
            except Exception:
                # Network/server failure — record as rejection so the audit
                # row still names the cluster, and move on.
                resp = RefineResponse(box=None, score=0.0)

            if resp.box is None:
                continue  # SAM3-DART returned nothing — silently skip

            refined_iou = iou(seed_bbox, resp.box)
            propagated_id = f"{cluster.cluster_id}@{target_id}"
            detection = ReconciledDetection(
                candidate_id=propagated_id,
                class_name=cluster.class_name,
                class_id=cluster.class_id,
                bbox=resp.box,
                seed_bbox=seed_bbox,
                mask_score=resp.score,
                seed_iou=refined_iou,
                cluster_id=cluster.cluster_id,
                votes=votes,
            )

            if (
                resp.score >= cfg.accept_score
                and refined_iou >= cfg.accept_iou
            ):
                per_image_propagated[target_id].append(detection)
            else:
                per_image_rejected[target_id].append(detection)

    return {
        image_id: ReconcileResult(
            image_id=image_id,
            group_id=group_id,
            propagated=per_image_propagated[image_id],
            rejected=per_image_rejected[image_id],
        )
        for image_id in by_id
    }
