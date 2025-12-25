"""Models package - Deep learning model wrappers."""

from .base import BaseModel, load_image, create_batch_iterator
from .siglip_model import SigLIPModel
from .dinov3_model import DINOv3Model
from .detector_models import (
    BaseDetector,
    Detection,
    DetectionResult,
    Florence2Detector,
    GroundingDINODetector,
    MoondreamDetector,
    get_detector,
)

__all__ = [
    # Base utilities
    "BaseModel",
    "load_image",
    "create_batch_iterator",
    # Model wrappers
    "SigLIPModel",
    "DINOv3Model",
    # Detectors
    "BaseDetector",
    "Detection",
    "DetectionResult",
    "Florence2Detector",
    "GroundingDINODetector",
    "MoondreamDetector",
    "get_detector",
]
