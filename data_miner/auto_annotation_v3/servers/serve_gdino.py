"""LitServe server for GroundingDINO zero-shot object detection.

Subclasses DetectorServerBase — uniform DetectorRequest/DetectorResponse wire.
Per-class iteration lives here, not in the pipeline: the client sends the
full class list in ``DetectorRequest.prompts`` and this server runs one
forward pass per prompt (GDINO degrades on joint multi-class) and
concatenates the results into a single DetectorResponse.

Launch:
    python serve_gdino.py --port 3001 --device 0 --max-batch-size 8
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from ..contracts import (
    DetectorRequest,
    DetectorResponse,
    PreparedInput,
    RawPrediction,
)
from .base import DetectorServerBase, run_server

logger = logging.getLogger(__name__)

_MODEL_ID = "IDEA-Research/grounding-dino-base"


class GDINOApi(DetectorServerBase):
    """GroundingDINO server — loops per prompt internally."""

    def _load_model(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info("Loading GroundingDINO processor %s", _MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(_MODEL_ID)
        logger.info("Loading GroundingDINO model onto %s", self.device)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)
            .to(self.device)
            .eval()
        )

    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        # Build per-prompt processor inputs up-front. Each carries its own
        # input_ids used by post_process to map detections back to the prompt.
        per_prompt = []
        for prompt in req.prompts:
            text = f"{prompt.strip()} ."  # GDINO post-process requires trailing "."
            inputs = self.processor(images=image, text=text, return_tensors="pt")
            per_prompt.append({"prompt": prompt, "inputs": inputs, "text": text})
        w, h = image.size
        return PreparedInput(
            image=image,
            processor_inputs=per_prompt,
            image_size=(w, h),
            prompts=list(req.prompts),
            threshold=req.threshold,
        )

    def _run_one_request(self, item: PreparedInput) -> RawPrediction:
        per_prompt_outputs = []
        for entry in item.processor_inputs:
            moved = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in entry["inputs"].items()
            }
            with torch.no_grad():
                outputs = self.model(**moved)
            per_prompt_outputs.append({
                "prompt": entry["prompt"],
                "outputs": outputs,
                "input_ids": moved["input_ids"],
            })
        return RawPrediction(
            outputs=per_prompt_outputs,
            inputs=None,
            image_size=item.image_size,
            prompts=item.prompts,
            threshold=item.threshold,
        )

    def _to_response(self, result: RawPrediction) -> DetectorResponse:
        w, h = result.image_size
        threshold = result.threshold if result.threshold is not None else 0.25

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for entry in result.outputs:
            prompt = entry["prompt"]
            post = self.processor.post_process_grounded_object_detection(
                entry["outputs"],
                entry["input_ids"],
                threshold=threshold,
                text_threshold=0.2,
                target_sizes=[(h, w)],
            )[0]
            raw_boxes = post["boxes"].cpu().tolist()
            scores = post["scores"].cpu().tolist()
            # Return the prompt we were given as the label — caller-uniform,
            # no label-mapping needed on the client side.
            all_boxes.extend(self._norm_box_px(b, w, h) for b in raw_boxes)
            all_scores.extend(float(s) for s in scores)
            all_labels.extend([prompt] * len(raw_boxes))

        return DetectorResponse(
            boxes=all_boxes, scores=all_scores, labels=all_labels
        )


if __name__ == "__main__":
    run_server(GDINOApi, default_port=3001)
