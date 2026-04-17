"""Pure inference model for Falcon-Perception zero-shot detection / segmentation.

Wraps ``tiiuae/Falcon-Perception`` via the ``falcon_perception`` library.
No LitServe dependency — the LitAPI wrapper in model_servers/ calls these
methods.

Per-class iteration happens inside this model (joint multi-class conflates
everything as "person"). The caller sends the full class list in ``prompts``
and this model runs one generate per prompt, then concatenates results.

Quirks:
  - Falcon produces no per-detection confidence score; all detections get
    ``score=1.0``.
  - When the model supports segmentation, RLE masks are decoded via
    ``pycocotools`` to extract tighter bounding boxes. Falls back to
    coordinate-based bboxes from the raw output.
  - ``torch._dynamo.config.suppress_errors`` is set to True at load time to
    avoid compile failures on some models.

Default parameters (matching v3):
  - task: "segmentation" (falls back to "detection" if model lacks it)
  - dtype: "bfloat16"
  - max_length: 4096, max_new_tokens: 2048
  - min_dimension: 256, max_dimension: 512
  - seed: 42
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from ..configs.wire import DetectorResponse, PreparedInput, RawPrediction
from .base import BaseDetectorModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "tiiuae/Falcon-Perception"

_DEFAULT_PARAMS: dict[str, Any] = {
    "task": "segmentation",
    "dtype": "bfloat16",
    "max_length": 4096,
    "min_dimension": 256,
    "max_dimension": 512,
    "max_new_tokens": 2048,
    "seed": 42,
}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _center_size_to_xyxy(xc: float, yc: float, bw: float, bh: float) -> tuple[float, ...]:
    """Convert centre+size format to (x1, y1, x2, y2) clamped to [0,1]."""
    return (_clamp(xc - bw / 2), _clamp(yc - bh / 2),
            _clamp(xc + bw / 2), _clamp(yc + bh / 2))


def _mask_rle_to_xyxy(mask_rle: dict) -> tuple[float, ...] | None:
    """Decode a COCO RLE mask and return its bounding box in [0,1] coords."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    if not isinstance(mask_rle, dict):
        return None
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not isinstance(size, list) or len(size) != 2 or counts is None:
        return None
    try:
        decoded = mask_utils.decode({
            "size": size,
            "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
        })
    except Exception:
        return None
    if decoded is None or decoded.size == 0:
        return None
    fg = np.argwhere(decoded.astype(bool))
    if fg.size == 0:
        return None
    ys, xs = fg[:, 0], fg[:, 1]
    img_h, img_w = decoded.shape[:2]
    return (_clamp(float(xs.min()) / img_w),
            _clamp(float(ys.min()) / img_h),
            _clamp(float(xs.max() + 1) / img_w),
            _clamp(float(ys.max() + 1) / img_h))


class FalconModel(BaseDetectorModel):
    """Falcon-Perception detector — per-prompt loop, concatenated results.

    Attributes (populated by ``load``):
        model: The loaded Falcon model.
        tokenizer: Falcon tokenizer.
        model_args: Model configuration arguments.
        engine: ``BatchInferenceEngine`` used for generation.
        stop_token_ids: Token IDs that stop generation.
        device: Torch device string.
        dtype: Torch dtype for inference.
    """

    def load(self, device: str, model_id: str = _DEFAULT_MODEL_ID,
             **options: Any) -> None:
        """Load Falcon-Perception model, tokenizer, and batch engine.

        Args:
            device: Torch device string.
            model_id: HuggingFace model identifier.
            **options: Unused — reserved for forward-compat.
        """
        import torch
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        from falcon_perception import load_and_prepare_model, setup_torch_config
        from falcon_perception.batch_inference import BatchInferenceEngine

        self.device = device
        self.dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

        setup_torch_config()
        model, tokenizer, model_args = load_and_prepare_model(
            hf_model_id=model_id,
            device=device,
            dtype=_DEFAULT_PARAMS["dtype"],
            compile=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.engine = BatchInferenceEngine(model, tokenizer)
        self.stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
        self._supports_segmentation = bool(model_args.do_segmentation)
        logger.info(
            "Falcon ready | supports_segmentation=%s", self._supports_segmentation
        )

    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Build LLM prompts and batch inputs per class.

        For each prompt, constructs a task-specific LLM prompt and runs
        ``process_batch_and_generate`` to produce tokenized inputs ready
        for the engine's generate step.

        Args:
            image: RGB PIL image.
            prompts: List of class-name strings to detect.
            threshold: Unused by Falcon (no score-based filtering).

        Returns:
            ``PreparedInput`` with per-prompt batch inputs and the resolved task.
        """
        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import process_batch_and_generate

        task = _DEFAULT_PARAMS["task"]
        if task == "segmentation" and not self._supports_segmentation:
            task = "detection"

        per_prompt = []
        for prompt in prompts:
            llm_prompt = build_prompt_for_task(prompt, task)
            batch_inputs = process_batch_and_generate(
                self.tokenizer, [(image, llm_prompt)],
                max_length=_DEFAULT_PARAMS["max_length"],
                min_dimension=_DEFAULT_PARAMS["min_dimension"],
                max_dimension=_DEFAULT_PARAMS["max_dimension"],
            )
            per_prompt.append({"prompt": prompt, "batch_inputs": batch_inputs})
        w, h = image.size
        return PreparedInput(
            image=image,
            processor_inputs=per_prompt,
            image_size=(w, h),
            prompts=list(prompts),
            threshold=threshold,
            extras={"task": task},
        )

    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run generation loop — one generate call per prompt.

        Each prompt's batch inputs are moved to the device and passed to
        the engine's ``generate`` method. The auxiliary outputs (containing
        bounding-box and mask data) are collected per prompt.

        Args:
            prepared: Result of ``prepare()``.

        Returns:
            ``RawPrediction`` with per-prompt auxiliary outputs.
        """
        import torch

        task = prepared.extras["task"]
        per_prompt_outputs = []
        for entry in prepared.processor_inputs:
            batch_inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in entry["batch_inputs"].items()
            }
            _, aux_out = self.engine.generate(
                **batch_inputs,
                max_new_tokens=_DEFAULT_PARAMS["max_new_tokens"],
                temperature=0.0,
                stop_token_ids=self.stop_token_ids,
                seed=_DEFAULT_PARAMS["seed"],
                task=task,
            )
            per_prompt_outputs.append({"prompt": entry["prompt"], "aux": aux_out[0]})
        return RawPrediction(
            outputs=per_prompt_outputs,
            inputs=None,
            image_size=prepared.image_size,
            prompts=prepared.prompts,
            threshold=prepared.threshold,
            extras={"task": task},
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Decode RLE masks and extract bounding boxes.

        For each detection, prefers the mask-derived bbox (from RLE decode)
        over coordinate-based bbox. All detections get ``score=1.0`` since
        Falcon provides no per-detection confidence.

        Args:
            raw: Result of ``infer()``.

        Returns:
            ``DetectorResponse`` with concatenated detections across all prompts.
        """
        from falcon_perception.visualization_utils import pair_bbox_entries

        task = raw.extras["task"]
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for entry in raw.outputs:
            prompt = entry["prompt"]
            aux = entry["aux"]
            paired = pair_bbox_entries(aux.bboxes_raw)
            masks_rle = aux.masks_rle if task == "segmentation" else []

            for idx, b in enumerate(paired):
                mask_rle = masks_rle[idx] if idx < len(masks_rle) else None
                mask_bbox = _mask_rle_to_xyxy(mask_rle) if mask_rle else None
                coord_bbox = None
                xy = b.get("x"), b.get("y")
                hw = b.get("h"), b.get("w")
                if all(v is not None for v in (*xy, *hw)):
                    try:
                        coord_bbox = _center_size_to_xyxy(
                            float(xy[0]), float(xy[1]), float(hw[1]), float(hw[0])
                        )
                    except (TypeError, ValueError):
                        coord_bbox = None
                primary = mask_bbox or coord_bbox
                if primary is None:
                    continue
                x1, y1, x2, y2 = primary
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(1.0)
                all_labels.append(prompt)

        return DetectorResponse(boxes=all_boxes, scores=all_scores, labels=all_labels)
