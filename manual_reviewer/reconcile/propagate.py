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

import logging
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

logger = logging.getLogger(__name__)

__all__ = [
    "PropagationConfig",
    "ImageContext",
    "MAX_CONSECUTIVE_TRANSPORT_ERRORS",
    "reconcile_group",
]


MAX_CONSECUTIVE_TRANSPORT_ERRORS = 5


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

    max_consecutive_transport_errors: int = MAX_CONSECUTIVE_TRANSPORT_ERRORS
    """Bail with RuntimeError after N consecutive transport-error refines
    in the same group — a flapping SAM endpoint shouldn't silently flood
    the audit log with synthetic rejected rows."""


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

    consecutive_transport_errors = 0
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
            transport_error = False
            try:
                resp: RefineResponse = client.refine(
                    image_path=target.image_path,
                    bbox=seed_bbox,
                    threshold=cfg.refine_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                # Network/server failure: log loudly so a flapping connection
                # doesn't bias the system toward false negatives without
                # operator signal, and record a synthetic rejection so the
                # audit row still names the cluster + target frame.
                logger.warning(
                    "reconcile_group: SAM3-DART refine failed for cluster=%s target=%s: %s",
                    cluster.cluster_id,
                    target_id,
                    exc,
                )
                resp = RefineResponse(box=None, score=0.0)
                transport_error = True

            if transport_error:
                consecutive_transport_errors += 1
                if consecutive_transport_errors >= cfg.max_consecutive_transport_errors:
                    raise RuntimeError(
                        f"reconcile_group: aborting group={group_id} after "
                        f"{consecutive_transport_errors} consecutive SAM transport "
                        f"errors (cluster={cluster.cluster_id}, target={target_id}); "
                        f"the endpoint appears to be down."
                    )
            else:
                consecutive_transport_errors = 0

            if resp.box is None:
                if transport_error:
                    # Synthetic rejected row: bbox==seed_bbox (best we can
                    # do without a SAM response), score=0, seed_iou=0.
                    # ``reject_reason="transport_error"`` is the structured
                    # signal that this row is an infra failure rather than
                    # a real "object not present" rejection. The
                    # ``#transport_error`` candidate_id suffix is preserved
                    # for legacy string-grep callers.
                    per_image_rejected[target_id].append(
                        ReconciledDetection(
                            candidate_id=f"{cluster.cluster_id}@{target_id}#transport_error",
                            class_name=cluster.class_name,
                            class_id=cluster.class_id,
                            bbox=seed_bbox,
                            seed_bbox=seed_bbox,
                            mask_score=0.0,
                            seed_iou=0.0,
                            cluster_id=cluster.cluster_id,
                            votes=votes,
                            reject_reason="transport_error",
                        )
                    )
                # No-mask (non-transport): silent skip — neither propagated nor rejected.
                continue

            # Acceptance-IoU is measured against the best-matching real
            # member, not the (potentially mid-air) cluster centroid.
            refined_iou = max(
                iou(m.annotation.bbox, resp.box) for m in cluster.members
            )
            propagated_id = f"{cluster.cluster_id}@{target_id}"
            score_ok = resp.score >= cfg.accept_score
            iou_ok = refined_iou >= cfg.accept_iou
            if score_ok and iou_ok:
                reject_reason = None
            elif not score_ok:
                reject_reason = "below_score"
            else:
                reject_reason = "below_iou"
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
                reject_reason=reject_reason,
            )

            if score_ok and iou_ok:
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
