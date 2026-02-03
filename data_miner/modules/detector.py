"""
Object Detector Module

Runs open-set object detection on frames using configurable detector backends.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from ..config import DetectionConfig, DetectorType
from ..models.detector_models import BaseDetector, DetectionResult, get_detector
from ..utils.io import ensure_dir, save_json
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionBatchResult:
    """Result of batch detection operation."""
    total_frames: int
    frames_with_detections: int
    total_detections: int
    results: list[DetectionResult] = field(default_factory=list)


class ObjectDetector:
    """
    Object detector with configurable backends.
    
    Supports multiple detection models:
    - DINO-X (highest accuracy)
    - Moondream3 (VQA + detection)
    - Florence-2 (multi-task)
    - Grounding DINO (stable)
    
    Example:
        >>> config = DetectionConfig(detector=DetectorType.FLORENCE2)
        >>> detector = ObjectDetector(config)
        >>> results = detector.detect_batch(frame_paths, "glass door")
    """
    
    def __init__(self, config: DetectionConfig, device_map: str = "auto"):
        """
        Initialize object detector.
        
        Args:
            config: Detection configuration
            device_map: Device: 'auto', 'cuda', 'cuda:0', 'cpu'
        """
        self.config = config
        self.device_map = device_map
        self.detector: Optional[BaseDetector] = None
        ensure_dir(config.output_dir)
        
        if config.save_visualizations:
            ensure_dir(config.output_dir / "visualizations")
    
    def _load_detector(self) -> None:
        """Load the configured detector model."""
        if self.detector is not None:
            return
        
        model_id = self.config.model_ids.get(self.config.detector.value)
        
        self.detector = get_detector(
            detector_type=self.config.detector,
            model_id=model_id,
            device_map=self.device_map,
        )
        self.detector.load()
    
    def detect_single(
        self,
        image_path: Path,
        prompt: str,
        save_visualization: bool = True,
    ) -> DetectionResult:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to image
            prompt: Detection prompt (what to look for)
            save_visualization: Save image with bounding boxes
            
        Returns:
            DetectionResult
        """
        self._load_detector()
        
        result = self.detector.detect(
            image=image_path,
            prompt=prompt,
            confidence_threshold=self.config.confidence_threshold,
        )
        
        # Save visualization if requested
        if save_visualization and self.config.save_visualizations and result.detections:
            self._save_visualization(image_path, result)
        
        return result
    
    def _save_visualization(
        self,
        image_path: Path,
        result: DetectionResult,
    ) -> Path:
        """Draw bounding boxes and save visualization."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image for visualization: {image_path}")
            return None
        
        height, width = image.shape[:2]
        
        # Draw each detection
        for det in result.detections:
            # Convert normalized bbox to absolute coordinates
            x1 = int(det.bbox[0] * width)
            y1 = int(det.bbox[1] * height)
            x2 = int(det.bbox[2] * width)
            y2 = int(det.bbox[3] * height)
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.label}: {det.confidence:.2f}"
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1,
            )
            
            # Text
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )
        
        # Save
        vis_path = self.config.output_dir / "visualizations" / f"{image_path.stem}_det.jpg"
        cv2.imwrite(str(vis_path), image)
        
        return vis_path
    
    def detect_batch(
        self,
        image_paths: list[Path],
        prompt: str,
        show_progress: bool = True,
    ) -> DetectionBatchResult:
        """
        Run detection on multiple images.
        
        Args:
            image_paths: List of image paths
            prompt: Detection prompt
            show_progress: Show progress bar
            
        Returns:
            DetectionBatchResult with all results
        """
        if not image_paths:
            return DetectionBatchResult(
                total_frames=0,
                frames_with_detections=0,
                total_detections=0,
            )
        
        logger.info(f"Running detection on {len(image_paths)} images")
        self._load_detector()
        
        results = []
        frames_with_detections = 0
        total_detections = 0
        
        iterator = image_paths
        if show_progress:
            iterator = tqdm(iterator, desc="Detecting objects", unit="image")
        
        for image_path in iterator:
            try:
                result = self.detect_single(
                    image_path=image_path,
                    prompt=prompt,
                    save_visualization=self.config.save_visualizations,
                )
                results.append(result)
                
                if result.detections:
                    frames_with_detections += 1
                    total_detections += len(result.detections)
                    
            except Exception as e:
                logger.error(f"Detection failed for {image_path}: {e}")
                results.append(DetectionResult(
                    image_path=image_path,
                    detections=[],
                ))
        
        # Save all results as JSON
        annotations = {
            "prompt": prompt,
            "detector": self.config.detector.value,
            "confidence_threshold": self.config.confidence_threshold,
            "images": [r.to_dict() for r in results],
        }
        
        annotations_path = self.config.output_dir / "annotations.json"
        save_json(annotations, annotations_path)
        logger.info(f"Saved annotations to {annotations_path}")
        
        batch_result = DetectionBatchResult(
            total_frames=len(image_paths),
            frames_with_detections=frames_with_detections,
            total_detections=total_detections,
            results=results,
        )
        
        logger.info(
            f"Detection complete: {frames_with_detections}/{len(image_paths)} frames "
            f"with {total_detections} detections"
        )
        
        return batch_result
    
    def unload_model(self) -> None:
        """Unload detector to free memory."""
        if self.detector is not None:
            self.detector.unload()
            self.detector = None
