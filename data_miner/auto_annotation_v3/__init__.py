"""Auto Annotation V3 — 3-stage pipeline with LitServe + Redis Streams."""

from .config import AutoAnnotationV3Config, load_config
from .pipeline import AutoAnnotationPipelineV3, run_pipeline
from .contracts import (
    Candidate,
    BoundingBox,
    FinalAnnotation,
    DetectResult,
    EvaluateResult,
    RefineResult,
)

__all__ = [
    "AutoAnnotationV3Config",
    "load_config",
    "AutoAnnotationPipelineV3",
    "run_pipeline",
    "Candidate",
    "BoundingBox",
    "FinalAnnotation",
    "DetectResult",
    "EvaluateResult",
    "RefineResult",
]
