"""Cross-frame static-object propagation for the manual review workflow.

Reads aav4 finalize annotations across grouped frames, asks SAM3-DART (image
mode) whether a detection seen in some frames is also present in frames that
the pipeline missed, and writes ``Stage.RECONCILE`` rows back to pipeline.db.
"""

from .clustering import (
    AnnotationRef,
    DetectionCluster,
    build_clusters,
    iou,
)
from .grouping import DEFAULT_CLIP_REGEX, group_images
from .propagate import (
    ImageContext,
    PropagationConfig,
    reconcile_group,
)
from .propagate_static import (
    CosineGenerator,
    CropEncoder,
    Match,
    PropagateStaticConfig,
    PropagateStaticSummary,
    Reconciler,
    Seed,
    Verdict,
    propagate_static,
    verdict_to_ls_region,
)
from .sam3_client import (
    DEFAULT_SAM3_1_REFINE_URL,
    DEFAULT_SAM3_1_TRACK_URL,
    DEFAULT_SAM3_DART_REFINE_URL,
    RefineResponse,
    Sam3Client,
    Sam3HttpClient,
    Sam3OneHttpClient,
)

__all__ = [
    "AnnotationRef",
    "CosineGenerator",
    "CropEncoder",
    "DEFAULT_CLIP_REGEX",
    "DEFAULT_SAM3_1_REFINE_URL",
    "DEFAULT_SAM3_1_TRACK_URL",
    "DEFAULT_SAM3_DART_REFINE_URL",
    "DetectionCluster",
    "ImageContext",
    "Match",
    "PropagateStaticConfig",
    "PropagateStaticSummary",
    "PropagationConfig",
    "Reconciler",
    "RefineResponse",
    "Sam3Client",
    "Sam3HttpClient",
    "Sam3OneHttpClient",
    "Seed",
    "Verdict",
    "build_clusters",
    "group_images",
    "iou",
    "propagate_static",
    "reconcile_group",
    "verdict_to_ls_region",
]
