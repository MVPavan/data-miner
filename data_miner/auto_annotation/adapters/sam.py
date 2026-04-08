from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from ..config import ClassPackConfig
from ..contracts import BoundingBox, Candidate, ReviewDecision
from ..registry import register_adapter
from ..utils import bbox_to_pixels, clamp
from .base import AnnotationAdapter


def _pick_best(results: dict[str, Any], image_size: tuple[int, int]) -> tuple[BoundingBox, float] | None:
    boxes = results.get("boxes")
    scores = results.get("scores")
    if boxes is None or scores is None or len(boxes) == 0:
        return None
    width, height = image_size
    index = int(torch.argmax(scores).item()) if torch.is_tensor(scores) else max(range(len(scores)), key=lambda idx: scores[idx])
    box = boxes[index]
    score = float(scores[index])
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    return (
        BoundingBox(
            x1=clamp(x1 / width),
            y1=clamp(y1 / height),
            x2=clamp(x2 / width),
            y2=clamp(y2 / height),
        ),
        score,
    )


@register_adapter("sam")
class SAMAdapter(AnnotationAdapter):
    capabilities = {"proposal", "refinement"}

    def __init__(self, name, config):
        super().__init__(name, config)
        self.model = None
        self.processor = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        model_id = self.config.model_id or "facebook/sam3"
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32).to(self.device).eval()

    def _post_process(self, image: Image.Image, inputs: dict[str, Any], outputs: Any, threshold: float) -> dict[str, Any]:
        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

    def propose(self, image: Image.Image, class_pack: ClassPackConfig, expression: str, params: dict[str, Any]) -> list[Candidate]:
        self._ensure_loaded()
        inputs = self.processor(images=image, text=expression, return_tensors="pt")
        inputs = inputs.to(device=self.device, dtype=self.model.dtype)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self._post_process(image, inputs, outputs, float(params.get("threshold", 0.5)))
        boxes = results.get("boxes")
        scores = results.get("scores")
        candidates: list[Candidate] = []
        if boxes is None or scores is None:
            return candidates
        width, height = image.size
        for index, (box, score) in enumerate(zip(boxes, scores), start=1):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            candidates.append(
                Candidate(
                    candidate_id=f"{self.name}:{class_pack.name}:{index}",
                    class_name=class_pack.name,
                    label=class_pack.name,
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
        self._ensure_loaded()
        pixel_box = list(bbox_to_pixels(candidate.bbox, image.size))
        inputs = self.processor(
            images=image,
            input_boxes=[[pixel_box]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(device=self.device, dtype=self.model.dtype)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self._post_process(image, inputs, outputs, float(params.get("threshold", 0.5)))
        best = _pick_best(results, image.size)
        if best is None:
            return None
        box, score = best
        return candidate.model_copy(
            update={
                "bbox": box,
                "score": max(candidate.score, score),
                "source_model": f"{candidate.source_model}+{self.name}",
                "status": "refined",
                "notes": [*candidate.notes, "sam_refined"],
            }
        )

    def verify(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any]) -> ReviewDecision:
        raise NotImplementedError