"""LitServe server for OWLv2 (OWL-ViT v2) zero-shot object detection.

Subclasses DetectorServerBase — uniform DetectorRequest/DetectorResponse wire.
Unlike GDINO/Falcon/SAM3, OWLv2 natively handles multi-class in a single
forward pass, so no per-prompt loop inside _run_one_request — all prompts
are wrapped as ``[f"a photo of a {p}"]`` and processed jointly.

Launch:
    python serve_owlvit2.py --port 3004 --device 0 --max-batch-size 8
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from ..contracts import (
    DetectorRequest,
    DetectorResponse,
    PreparedInput,
    RawPrediction,
)
from .base import DetectorServerBase, run_server

logger = logging.getLogger(__name__)

_MODEL_ID = "google/owlv2-base-patch16-ensemble"
_DEFAULT_THRESHOLD = 0.1


class OWLv2Api(DetectorServerBase):
    """OWLv2 server — native multi-class, one forward pass per request."""

    def _load_model(self) -> None:
        logger.info("Loading OWLv2 processor %s", _MODEL_ID)
        self.processor = Owlv2Processor.from_pretrained(_MODEL_ID)
        torch_dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        logger.info("Loading OWLv2 model onto %s dtype=%s", self.device, torch_dtype)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(_MODEL_ID, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )

    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        text_queries = [f"a photo of a {p}" for p in req.prompts]
        # OWLv2 expects nested: [[q0, q1, ...]] (one set per image).
        inputs = self.processor(
            text=[text_queries], images=image, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        w, h = image.size
        target_sizes = torch.tensor([[h, w]], device=self.device)
        return PreparedInput(
            image=image,
            processor_inputs=inputs,
            image_size=(w, h),
            prompts=list(req.prompts),
            threshold=req.threshold,
            extras={"target_sizes": target_sizes, "text_queries": text_queries},
        )

    def _run_one_request(self, item: PreparedInput) -> RawPrediction:
        with torch.no_grad():
            outputs = self.model(**item.processor_inputs)
        return RawPrediction(
            outputs=outputs,
            inputs=None,
            image_size=item.image_size,
            prompts=item.prompts,
            threshold=item.threshold,
            extras=item.extras,
        )

    def _to_response(self, result: RawPrediction) -> DetectorResponse:
        w, h = result.image_size
        threshold = result.threshold if result.threshold is not None else _DEFAULT_THRESHOLD
        post = self.processor.post_process_grounded_object_detection(
            outputs=result.outputs,
            target_sizes=result.extras["target_sizes"],
            threshold=threshold,
        )[0]

        raw_boxes = post.get("boxes")
        scores_t = post.get("scores")
        label_idxs = post.get("labels")
        if raw_boxes is None or len(raw_boxes) == 0:
            return DetectorResponse(boxes=[], scores=[], labels=[])

        scores_list = scores_t.cpu().tolist() if torch.is_tensor(scores_t) else list(scores_t)
        label_list = label_idxs.cpu().tolist() if torch.is_tensor(label_idxs) else list(label_idxs)

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []
        for box_t, score, idx in zip(raw_boxes, scores_list, label_list):
            x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
            all_boxes.append(self._norm_box_px([x1, y1, x2, y2], w, h))
            all_scores.append(float(score))
            i = int(idx)
            # Echo back the ORIGINAL prompt (not the "a photo of a X" wrapper)
            # so client-side label matching is uniform across detectors.
            all_labels.append(
                result.prompts[i] if 0 <= i < len(result.prompts) else "object"
            )
        return DetectorResponse(boxes=all_boxes, scores=all_scores, labels=all_labels)


if __name__ == "__main__":
    run_server(OWLv2Api, default_port=3004)
