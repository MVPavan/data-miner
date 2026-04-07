from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ..config import ClassPackConfig
from ..contracts import BoundingBox, Candidate, ReviewDecision
from ..registry import register_adapter
from ..utils import clamp
from .base import AnnotationAdapter


@register_adapter("grounding_dino")
class GroundingDINOAdapter(AnnotationAdapter):
    capabilities = {"proposal"}

    def __init__(self, name, config):
        super().__init__(name, config)
        self.model = None
        self.processor = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        model_id = self.config.model_id or "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def propose(self, image: Image.Image, class_pack: ClassPackConfig, expression: str, params: dict[str, Any]) -> list[Candidate]:
        self._ensure_loaded()
        text = " . ".join(class_pack.names())
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=float(params.get("box_threshold", 0.25)),
            text_threshold=float(params.get("text_threshold", 0.2)),
            target_sizes=[image.size[::-1]],
        )[0]
        width, height = image.size
        candidates: list[Candidate] = []
        for index, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"]), start=1):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            candidates.append(
                Candidate(
                    candidate_id=f"{self.name}:{class_pack.name}:{index}",
                    class_name=class_pack.name,
                    label=str(label),
                    source_model=self.name,
                    expression=expression,
                    bbox=BoundingBox(
                        x1=clamp(x1 / width),
                        y1=clamp(y1 / height),
                        x2=clamp(x2 / width),
                        y2=clamp(y2 / height),
                    ),
                    score=float(score),
                )
            )
        return candidates

    def refine(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any], request: ReviewDecision | None = None) -> Candidate | None:
        return None

    def verify(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any]) -> ReviewDecision:
        raise NotImplementedError