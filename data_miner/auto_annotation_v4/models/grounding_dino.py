"""Pure inference model for GroundingDINO zero-shot object detection.

Wraps ``IDEA-Research/grounding-dino-base`` from HuggingFace transformers.
No LitServe dependency — the LitAPI wrapper in model_servers/ calls these
methods.

GroundingDINO degrades when multiple classes are combined in a single
forward pass, so this model loops per-prompt internally and concatenates
the results. Each prompt gets a trailing ``.`` appended (required by the
post-process tokenizer logic).

Quirks:
  - The processor's ``post_process_grounded_object_detection`` requires the
    ``input_ids`` from each prompt's encoding to map detected sub-strings
    back to text labels.
  - ``text_threshold=0.2`` is hardcoded (matching v3 behaviour); the main
    ``threshold`` controls the box confidence cutoff.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from ..configs.wire import DetectorResponse, PreparedInput, RawPrediction
from .base import BaseDetectorModel, normalize_box

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


class GDINOModel(BaseDetectorModel):
    """GroundingDINO detector — per-prompt loop, concatenated results.

    Attributes (populated by ``load``):
        processor: HuggingFace ``AutoProcessor`` for image/text pre-processing.
        model: ``AutoModelForZeroShotObjectDetection`` on the target device.
        device: Torch device string (e.g. ``"cuda:0"``).
        dtype: Torch dtype for inference.
    """

    def load(self, device: str, model_id: str = _DEFAULT_MODEL_ID,
             **options: Any) -> None:
        """Load GroundingDINO processor and model onto *device*.

        Args:
            device: Torch device string (``"cuda:0"``, ``"cpu"``, etc.).
            model_id: HuggingFace model identifier.
            **options: Unused — reserved for forward-compat.
        """
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

        logger.info("Loading GroundingDINO processor %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)

        logger.info("Loading GroundingDINO model onto %s", device)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(device)
            .eval()
        )

    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Build per-prompt processor inputs.

        Each prompt is suffixed with `` .`` (required by GDINO post-process)
        and individually encoded through the processor. The per-prompt
        ``input_ids`` are preserved for post-processing.

        Args:
            image: RGB PIL image.
            prompts: List of class-name strings to detect.
            threshold: Optional box-confidence threshold override.

        Returns:
            ``PreparedInput`` with ``processor_inputs`` as a list of per-prompt
            dicts containing ``prompt``, ``inputs``, and ``text``.
        """
        per_prompt = []
        for prompt in prompts:
            text = f"{prompt.strip()} ."
            inputs = self.processor(images=image, text=text, return_tensors="pt")
            per_prompt.append({"prompt": prompt, "inputs": inputs, "text": text})
        w, h = image.size
        return PreparedInput(
            image=image,
            processor_inputs=per_prompt,
            image_size=(w, h),
            prompts=list(prompts),
            threshold=threshold,
        )

    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run one forward pass per prompt and gather raw outputs.

        Each prompt's inputs are moved to the model device and run through
        the model independently. The raw ``outputs`` and ``input_ids`` are
        preserved for ``postprocess``.

        Args:
            prepared: Result of ``prepare()``.

        Returns:
            ``RawPrediction`` with per-prompt outputs list.
        """
        import torch

        per_prompt_outputs = []
        for entry in prepared.processor_inputs:
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
            image_size=prepared.image_size,
            prompts=prepared.prompts,
            threshold=prepared.threshold,
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Post-process per-prompt outputs into normalized boxes/scores/labels.

        Uses the processor's ``post_process_grounded_object_detection`` with
        ``text_threshold=0.2``. Boxes are normalized to [0, 1] range. Labels
        echo back the original prompt string for uniform client-side matching.

        Args:
            raw: Result of ``infer()``.

        Returns:
            ``DetectorResponse`` with concatenated detections across all prompts.
        """
        w, h = raw.image_size
        threshold = raw.threshold if raw.threshold is not None else 0.25

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for entry in raw.outputs:
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
            all_boxes.extend(normalize_box(b, w, h) for b in raw_boxes)
            all_scores.extend(float(s) for s in scores)
            all_labels.extend([prompt] * len(raw_boxes))

        return DetectorResponse(
            boxes=all_boxes, scores=all_scores, labels=all_labels
        )
