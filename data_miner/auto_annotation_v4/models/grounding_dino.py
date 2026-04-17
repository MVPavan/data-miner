"""Pure inference model for GroundingDINO zero-shot object detection.

Wraps ``IDEA-Research/grounding-dino-base`` via ``GDINOBatchPredictor`` from
``gdino_batch.py``. No LitServe dependency — the LitAPI wrapper in
model_servers/ calls these methods.

GroundingDINO degrades when multiple classes are combined in a single
forward pass, so the batch predictor keeps per-prompt isolation but stacks
all N prompts (and optionally B images) into a single GPU call by expanding
the image tensor to (B*N, 3, H, W) and tokenising all prompts together.

Quirks:
  - ``text_threshold=0.2`` is hardcoded (matching v3 behaviour); the main
    ``threshold`` controls the box confidence cutoff.
  - The batch predictor returns per-prompt result dicts; boxes already live
    in pixel space on CPU, so postprocess only normalises and relabels.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from ..configs.wire import DetectorResponse, PreparedInput, RawPrediction
from .base import BaseDetectorModel, normalize_box

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"
_DEFAULT_THRESHOLD = 0.25
_TEXT_THRESHOLD = 0.2


class GDINOModel(BaseDetectorModel):
    """GroundingDINO detector — batched N-prompt forward via GDINOBatchPredictor.

    Attributes (populated by ``load``):
        predictor: :class:`GDINOBatchPredictor` instance.
        device: Torch device string (e.g. ``"cuda:0"``).
    """

    def load(self, device: str, model_id: str = _DEFAULT_MODEL_ID,
             **options: Any) -> None:
        """Load GroundingDINO batch predictor onto *device*.

        Args:
            device: Torch device string (``"cuda:0"``, ``"cpu"``, etc.).
            model_id: HuggingFace model identifier.
            **options: Unused — reserved for forward-compat.
        """
        from .gdino_batch import GDINOBatchPredictor

        self.device = device
        logger.info("Loading GDINOBatchPredictor (%s) onto %s", model_id, device)
        self.predictor = GDINOBatchPredictor(model_id=model_id, device=device)

    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Pass-through preprocessing — the batch predictor handles encoding.

        Args:
            image: RGB PIL image.
            prompts: List of class-name strings to detect.
            threshold: Optional box-confidence threshold override.

        Returns:
            ``PreparedInput`` carrying the image and prompt list.
        """
        w, h = image.size
        return PreparedInput(
            image=image,
            processor_inputs={"image": image},
            image_size=(w, h),
            prompts=list(prompts),
            threshold=threshold,
        )

    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run a single-image N-prompt batched forward pass.

        Args:
            prepared: Result of ``prepare()``.

        Returns:
            ``RawPrediction`` whose ``outputs`` is the per-prompt list from
            :meth:`GDINOBatchPredictor.predict`.
        """
        threshold = (
            prepared.threshold if prepared.threshold is not None else _DEFAULT_THRESHOLD
        )
        per_prompt_results = self.predictor.predict(
            prepared.image,
            prepared.prompts,
            threshold=float(threshold),
            text_threshold=_TEXT_THRESHOLD,
        )
        return RawPrediction(
            outputs=per_prompt_results,
            inputs=None,
            image_size=prepared.image_size,
            prompts=prepared.prompts,
            threshold=prepared.threshold,
        )

    def infer_batch(self, prepareds: list[PreparedInput]) -> list[RawPrediction]:
        """Run a multi-image N-prompt batched forward pass.

        All images in the batch must share the same prompt list (the
        pipeline case — one set of classes per job). When prompts diverge
        across items, falls back to per-item :meth:`infer`.

        Args:
            prepareds: Batch of prepared inputs (same prompts preferred).

        Returns:
            List of ``RawPrediction`` in the same order as *prepareds*.
        """
        if not prepareds:
            return []

        first_prompts = tuple(prepareds[0].prompts)
        first_threshold = prepareds[0].threshold
        homogeneous = all(
            tuple(p.prompts) == first_prompts and p.threshold == first_threshold
            for p in prepareds
        )
        if not homogeneous:
            return [self.infer(p) for p in prepareds]

        threshold = (
            first_threshold if first_threshold is not None else _DEFAULT_THRESHOLD
        )
        images = [p.image for p in prepareds]
        per_image = self.predictor.predict_images(
            images,
            list(first_prompts),
            threshold=float(threshold),
            text_threshold=_TEXT_THRESHOLD,
        )
        raws: list[RawPrediction] = []
        for p, per_prompt_results in zip(prepareds, per_image):
            raws.append(RawPrediction(
                outputs=per_prompt_results,
                inputs=None,
                image_size=p.image_size,
                prompts=p.prompts,
                threshold=p.threshold,
            ))
        return raws

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Normalise per-prompt pixel boxes to [0, 1] and relabel to prompts.

        The batch predictor returns a list of dicts (one per prompt) with
        ``boxes`` / ``scores`` tensors already on CPU. We normalise the
        pixel boxes and echo back the original prompt string for each box.

        Args:
            raw: Result of ``infer()`` or one element of ``infer_batch()``.

        Returns:
            ``DetectorResponse`` with concatenated detections across prompts.
        """
        import torch

        w, h = raw.image_size
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for entry in raw.outputs:
            prompt = entry["prompt"]
            boxes = entry["boxes"]
            scores = entry["scores"]
            boxes_list = (
                boxes.cpu().tolist() if torch.is_tensor(boxes) else list(boxes)
            )
            scores_list = (
                scores.cpu().tolist() if torch.is_tensor(scores) else list(scores)
            )
            all_boxes.extend(normalize_box(b, w, h) for b in boxes_list)
            all_scores.extend(float(s) for s in scores_list)
            all_labels.extend([prompt] * len(boxes_list))

        return DetectorResponse(
            boxes=all_boxes, scores=all_scores, labels=all_labels
        )
