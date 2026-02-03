"""Inference engine protocol and base class."""

from typing import List, Protocol, Tuple, runtime_checkable
from pathlib import Path


@runtime_checkable
class InferenceEngine(Protocol):
    """
    Protocol for model inference engines.
    
    Any model that implements this protocol can be used for prediction generation.
    """
    
    def load_model(self) -> None:
        """Load the model weights and prepare for inference."""
        ...
    
    def infer_image(
        self, 
        img_path: str, 
        threshold: float = 0.001
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Run inference on a single image.
        
        Args:
            img_path: Path to the image file.
            threshold: Confidence threshold for filtering detections.
        
        Returns:
            Tuple of (bboxes, scores, class_ids) where:
                - bboxes: List of [x1, y1, w, h] in COCO format (xywh)
                - scores: List of confidence scores
                - class_ids: List of class IDs
        """
        ...
    
    def resolution_tag(self) -> str:
        """
        Get a string tag representing the model resolution.
        
        Returns:
            String like "560_560" or "default"
        """
        ...
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        ...


class BaseInferenceEngine:
    """
    Base class for inference engines with common functionality.
    
    Subclasses should implement:
        - load_model()
        - infer_image()
        - resolution_tag()
    """
    
    def __init__(self):
        self._model_loaded = False
        self.latencies: List[float] = []
    
    @property
    def is_loaded(self) -> bool:
        return self._model_loaded
    
    def load_model(self) -> None:
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def infer_image(
        self, 
        img_path: str, 
        threshold: float = 0.001
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        raise NotImplementedError("Subclasses must implement infer_image()")
    
    def resolution_tag(self) -> str:
        return "default"
    
    def get_average_latency(self) -> float:
        """Get average inference latency in seconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
