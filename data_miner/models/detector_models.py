"""
Detector Models Module

Unified interface for multiple open-set detection models:
- DINO-X (highest accuracy)
- Moondream3 (VQA + detection)
- Florence-2 (multi-task)
- Grounding DINO (stable)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from ..config import DetectorType
from ..logging import get_logger
from ..utils.device import clear_gpu_cache, get_model_device
from .base import BaseDetector, Detection, DetectionResult

logger = get_logger(__name__)



class Florence2Detector(BaseDetector):
    """
    Florence-2 detector for open-vocabulary detection.
    
    Supports multiple detection modes via task prompts.
    """
    
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device: str = "auto",
    ):
        super().__init__(device)
        self.model_id = model_id
        self.model = None
        self.processor = None
    
    def load(self) -> None:
        if self._loaded:
            return
        
        logger.info(f"Loading Florence-2: {self.model_id}")
        
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        torch_dtype = torch.float16 if self.device_map != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Florence-2 loaded on {self.device_map}")
    
    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        clear_gpu_cache()
    
    @torch.no_grad()
    def detect(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        prompt: str,
        confidence_threshold: float = 0.3,
    ) -> DetectionResult:
        if not self._loaded:
            self.load()
        
        pil_image = self._load_image(image)
        width, height = pil_image.size
        
        # Use phrase grounding task for open-vocabulary detection
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        text_input = f"{task_prompt} {prompt}"
        
        inputs = self.processor(
            text=text_input,
            images=pil_image,
            return_tensors="pt",
        )
        device = get_model_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3,
        )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        
        # Parse the output
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(width, height),
        )
        
        detections = []
        
        if task_prompt in parsed:
            result = parsed[task_prompt]
            bboxes = result.get("bboxes", [])
            labels = result.get("labels", [])
            
            for bbox, label in zip(bboxes, labels):
                # Normalize bbox to 0-1 range
                x1, y1, x2, y2 = bbox
                normalized_bbox = (
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                )
                
                detections.append(Detection(
                    bbox=normalized_bbox,
                    label=label,
                    confidence=1.0,  # Florence-2 doesn't provide confidence
                ))
        
        return DetectionResult(
            image_path=Path(image) if isinstance(image, (str, Path)) else None,
            detections=detections,
            width=width,
            height=height,
        )


class GroundingDINODetector(BaseDetector):
    """
    Grounding DINO detector for text-conditioned detection.
    """
    
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device_map: str = "auto",
    ):
        super().__init__(device_map)
        self.model_id = model_id
    
    def load(self) -> None:
        if self._loaded:
            return
        
        logger.info(f"Loading Grounding DINO: {self.model_id}")
        
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        torch_dtype = torch.float16 if self.device_map != "cpu" else torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Grounding DINO loaded on {self.device_map}")
    
    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        clear_gpu_cache()
    
    @torch.no_grad()
    def detect(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        prompt: str,
        confidence_threshold: float = 0.3,
    ) -> DetectionResult:
        if not self._loaded:
            self.load()
        
        pil_image = self._load_image(image)
        width, height = pil_image.size
        
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        )
        device = get_model_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[(height, width)],
        )[0]
        
        detections = []
        
        for bbox, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            x1, y1, x2, y2 = bbox.cpu().numpy()
            # Normalize to 0-1
            normalized_bbox = (
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            )
            
            detections.append(Detection(
                bbox=normalized_bbox,
                label=label,
                confidence=float(score.cpu()),
            ))
        
        return DetectionResult(
            image_path=Path(image) if isinstance(image, (str, Path)) else None,
            detections=detections,
            width=width,
            height=height,
        )


class MoondreamDetector(BaseDetector):
    """
    Moondream detector for open-vocabulary detection with VQA capabilities.
    
    Attempts to use Moondream3 first, falls back to Moondream2.
    """
    
    def __init__(self, device_map: str = "auto"):
        super().__init__(device_map)
        from .moondream import MoonDreamHelper
        
        self.model_id = model_id
        self.tokenizer = None
        self._actual_model_id = None
    
    def load(self) -> None:
        if self._loaded:
            return
        
        # Try Moondream3 first, fallback to Moondream2
        models_to_try = [
            "moondream/moondream3-preview",
            "vikhyatk/moondream2",
            self.model_id,
        ]
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        for model_id in models_to_try:
            try:
                logger.info(f"Attempting to load: {model_id}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
                torch_dtype = torch.float16 if self.device_map != "cpu" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map=self.device_map,
                    torch_dtype=torch_dtype,
                )
                self.model.eval()
                self._actual_model_id = model_id
                self._loaded = True
                logger.info(f"Successfully loaded: {model_id} on {self.device_map}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load {model_id}: {e}")
                continue
        
        raise RuntimeError("Failed to load any Moondream model")
    
    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        clear_gpu_cache()
    
    @torch.no_grad()
    def detect(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        prompt: str,
        confidence_threshold: float = 0.3,
    ) -> DetectionResult:
        if not self._loaded:
            self.load()
        
        pil_image = self._load_image(image)
        width, height = pil_image.size
        
        detections = []
        
        # Use Moondream's detect method if available
        if hasattr(self.model, "detect"):
            try:
                result = self.model.detect(pil_image, prompt)
                
                if result and "objects" in result:
                    for obj in result["objects"]:
                        bbox = obj.get("bbox", obj.get("box", [0, 0, 0, 0]))
                        if len(bbox) == 4:
                            detections.append(Detection(
                                bbox=tuple(bbox),
                                label=obj.get("label", prompt),
                                confidence=obj.get("confidence", 1.0),
                            ))
                
            except Exception as e:
                logger.warning(f"Moondream detect failed: {e}, falling back to query")
        
        # Fallback: Use query to get detection info
        if not detections:
            try:
                query = f"Detect and locate all instances of: {prompt}. For each, provide the bounding box coordinates."
                
                if hasattr(self.model, "query"):
                    response = self.model.query(pil_image, query)
                elif hasattr(self.model, "answer_question"):
                    response = self.model.answer_question(
                        self.model.encode_image(pil_image),
                        query,
                        self.tokenizer,
                    )
                else:
                    response = None
                
                if response:
                    logger.debug(f"Moondream response: {response}")
                    # Parse response for bounding boxes (model-specific)
                    # This is a fallback and may not always work
                    
            except Exception as e:
                logger.warning(f"Moondream query failed: {e}")
        
        return DetectionResult(
            image_path=Path(image) if isinstance(image, (str, Path)) else None,
            detections=detections,
            width=width,
            height=height,
        )


def get_detector(
    detector_type: DetectorType,
    model_id: Optional[str] = None,
    device_map: str = "auto",
) -> BaseDetector:
    """
    Factory function to get the appropriate detector.
    
    Args:
        detector_type: Type of detector to create
        model_id: Optional custom model ID
        device_map: Device: 'auto', 'cuda', 'cuda:0', 'cpu'
        
    Returns:
        BaseDetector instance
    """
    detectors = {
        DetectorType.MOONDREAM3: (MoondreamDetector, "vikhyatk/moondream2"),
        DetectorType.FLORENCE2: (Florence2Detector, "microsoft/Florence-2-large"),
        DetectorType.GROUNDING_DINO: (GroundingDINODetector, "IDEA-Research/grounding-dino-base"),
    }
    
    detector_class, default_model = detectors.get(
        detector_type,
        (Florence2Detector, "microsoft/Florence-2-large"),
    )
    
    return detector_class(
        model_id=model_id or default_model,
        device_map=device_map,
    )
