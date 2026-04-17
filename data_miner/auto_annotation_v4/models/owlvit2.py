"""Pure inference model for OWLv2 (OWL-ViT v2) zero-shot object detection.

Wraps ``google/owlv2-base-patch16-ensemble`` from HuggingFace transformers.
No LitServe dependency — the LitAPI wrapper in model_servers/ calls these
methods.

Unlike GDINO/Falcon/SAM3, OWLv2 natively handles multi-class in a single
forward pass, so no per-prompt loop — all prompts are wrapped as
``"a photo of a {prompt}"`` and processed jointly.

Quirks:
  - Prompts are wrapped with ``"a photo of a ..."`` prefix for better
    zero-shot performance (standard OWL-ViT practice).
  - Labels in the response echo back the ORIGINAL prompt string (not the
    wrapped version) for uniform client-side matching.
  - Default threshold is 0.1 (lower than GDINO due to OWLv2's score
    calibration).
  - Uses ``float16`` on CUDA (not bfloat16) to match v3 behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from ..configs.wire import DetectorResponse, PreparedInput, RawPrediction
from .base import BaseDetectorModel, normalize_box

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"
_DEFAULT_THRESHOLD = 0.1


class OWLv2Model(BaseDetectorModel):
    """OWLv2 detector — native multi-class, one forward pass per request.

    Attributes (populated by ``load``):
        processor: HuggingFace ``Owlv2Processor``.
        model: ``Owlv2ForObjectDetection`` on the target device.
        device: Torch device string.
        dtype: Torch dtype for inference (float16 on CUDA, float32 on CPU).
    """

    def load(self, device: str, model_id: str = _DEFAULT_MODEL_ID,
             **options: Any) -> None:
        """Load OWLv2 processor and model onto *device*.

        Args:
            device: Torch device string (``"cuda:0"``, ``"cpu"``, etc.).
            model_id: HuggingFace model identifier.
            **options: Unused — reserved for forward-compat.
        """
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device
        torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32
        self.dtype = torch_dtype

        logger.info("Loading OWLv2 processor %s", model_id)
        self.processor = Owlv2Processor.from_pretrained(model_id)

        logger.info("Loading OWLv2 model onto %s dtype=%s", device, torch_dtype)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_id, torch_dtype=torch_dtype)
            .to(device)
            .eval()
        )

    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Build processor inputs with wrapped text queries.

        All prompts are wrapped as ``"a photo of a {prompt}"`` and passed
        as a nested list (one set per image) to the OWLv2 processor. Inputs
        are pre-moved to the model device.

        Args:
            image: RGB PIL image.
            prompts: List of class-name strings to detect.
            threshold: Optional confidence threshold override (default 0.1).

        Returns:
            ``PreparedInput`` with processor inputs and target sizes tensor.
        """
        import torch

        text_queries = [f"a photo of a {p}" for p in prompts]
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
            prompts=list(prompts),
            threshold=threshold,
            extras={"target_sizes": target_sizes, "text_queries": text_queries},
        )

    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run a single forward pass over all prompts jointly.

        Args:
            prepared: Result of ``prepare()``.

        Returns:
            ``RawPrediction`` with raw model outputs and extras for postprocess.
        """
        import torch

        with torch.no_grad():
            outputs = self.model(**prepared.processor_inputs)
        return RawPrediction(
            outputs=outputs,
            inputs=None,
            image_size=prepared.image_size,
            prompts=prepared.prompts,
            threshold=prepared.threshold,
            extras=prepared.extras,
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Post-process OWLv2 outputs into normalized boxes/scores/labels.

        Uses the processor's ``post_process_grounded_object_detection``.
        Labels echo back the ORIGINAL prompt (not the ``"a photo of a ..."``
        wrapper) for uniform client-side matching.

        Args:
            raw: Result of ``infer()``.

        Returns:
            ``DetectorResponse`` with detections from the joint forward pass.
        """
        import torch

        w, h = raw.image_size
        threshold = raw.threshold if raw.threshold is not None else _DEFAULT_THRESHOLD
        post = self.processor.post_process_grounded_object_detection(
            outputs=raw.outputs,
            target_sizes=raw.extras["target_sizes"],
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
            all_boxes.append(normalize_box([x1, y1, x2, y2], w, h))
            all_scores.append(float(score))
            i = int(idx)
            # Echo back the ORIGINAL prompt (not the "a photo of a X" wrapper)
            # so client-side label matching is uniform across detectors.
            all_labels.append(
                raw.prompts[i] if 0 <= i < len(raw.prompts) else "object"
            )
        return DetectorResponse(boxes=all_boxes, scores=all_scores, labels=all_labels)
