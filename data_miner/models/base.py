"""
Base Model Utilities

Common utilities and base classes shared across all model wrappers.
Centralizes repetitive code for maintainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from ..logging import get_logger
from ..utils.device import clear_gpu_cache, resolve_device

logger = get_logger(__name__)


# =============================================================================
# Image Loading Utility
# =============================================================================


def load_image(image: Union[Path, str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Convert various image inputs to PIL Image.

    Supports:
    - PIL Image objects
    - numpy arrays (assumes BGR from OpenCV)
    - file paths (str or Path)

    Args:
        image: Input image in various formats

    Returns:
        PIL Image in RGB mode

    Raises:
        ValueError: If image type is not supported
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        # Assume BGR from OpenCV, convert to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        return Image.fromarray(image).convert("RGB")
    elif isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def load_images_batch(
    images: list[Union[Path, str, Image.Image, np.ndarray]],
) -> list[Image.Image]:
    """Load a batch of images to PIL format."""
    return [load_image(img) for img in images]


# =============================================================================
# Base Model Class
# =============================================================================


class BaseModel(ABC):
    """
    Abstract base class for all model wrappers.

    Provides common functionality:
    - Load/unload lifecycle
    - Image loading utilities
    """
    def __init__(self):
        """Initialize base model."""
        self.model = None
        self.processor = None  # Generic name for processor/tokenizer
        self._loaded = False
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    @abstractmethod
    def load(self) -> None:
        """Load the model. Must be implemented by subclasses."""
        pass

    def unload(self) -> None:
        """Unload model and processor to free memory."""
        model_name = self.__class__.__name__

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        self._loaded = False
        clear_gpu_cache()
        logger.info(f"{model_name} unloaded")

    def _load_image(
        self, image: Union[Path, str, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various image inputs to PIL Image."""
        return load_image(image)

    def _load_images_batch(
        self,
        images: list[Union[Path, str, Image.Image, np.ndarray]],
    ) -> list[Image.Image]:
        """Load a batch of images."""
        return load_images_batch(images)
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            self.load()
    def __enter__(self):
        """Context manager entry - load model."""
        self.load()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def create_batch_iterator(
    items: list,
    batch_size: int,
    show_progress: bool = False,
    desc: str = "Processing",
):
    """
    Create a batch iterator with optional progress bar.
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        show_progress: Whether to show tqdm progress
        desc: Description for progress bar
    Yields:
        Tuples of (start_idx, batch)
    """
    from tqdm import tqdm

    iterator = range(0, len(items), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, unit="batch", leave=False)

    for start_idx in iterator:
        batch = items[start_idx : start_idx + batch_size]
        yield start_idx, batch


# =============================================================================
# Model Loading with Fallback
# =============================================================================

def load_model_with_fallback(
    model_ids: list[str],
    loader_func,
    model_name: str = "model",
) -> tuple:
    """
    Try loading models in order, returning first successful load.
    Args:
        model_ids: List of model IDs to try in order
        loader_func: Function that takes model_id and returns (model, processor)
        model_name: Name for logging

    Returns:
        Tuple of (model, processor, actual_model_id)

    Raises:
        RuntimeError: If all models fail to load
    """
    for model_id in model_ids:
        try:
            logger.info(f"Attempting to load {model_name}: {model_id}")
            model, processor = loader_func(model_id)
            logger.info(f"Successfully loaded {model_name}: {model_id}")
            return model, processor, model_id
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")
            continue
    raise RuntimeError(f"Failed to load any {model_name}. Tried: {model_ids}")


@dataclass
class Detection:
    """A single detection result."""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized 0-1
    label: str
    confidence: float
    caption: Optional[str] = None  # Optional caption for the detection


@dataclass
class DetectionResult:
    """Detection results for an image."""
    image_path: Optional[Path]
    detections: list[Detection] = field(default_factory=list)
    width: int = 0
    height: int = 0
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "width": self.width,
            "height": self.height,
            "detections": [
                {
                    "bbox": list(d.bbox),
                    "bbox_abs": [
                        int(d.bbox[0] * self.width),
                        int(d.bbox[1] * self.height),
                        int(d.bbox[2] * self.width),
                        int(d.bbox[3] * self.height),
                    ],
                    "label": d.label,
                    "confidence": d.confidence,
                    "caption": d.caption,
                }
                for d in self.detections
            ],
        }


class BaseDetector(ABC):
    """Abstract base class for detection models."""
    def __init__(self, device_map: str = "auto"):
        self.device_map = resolve_device(device_map)
        self.model = None
        self.processor = None
        self._loaded = False
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass
    @abstractmethod
    def detect(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        prompt: str,
        confidence_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Detect objects in an image.
        Args:
            image: Input image
            prompt: Text prompt describing what to detect
            confidence_threshold: Minimum confidence for detections
        Returns:
            DetectionResult with list of detections
        """
        pass

    def _load_image(
        self, image: Union[Path, str, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various image inputs to PIL Image."""
        return load_image(image)
