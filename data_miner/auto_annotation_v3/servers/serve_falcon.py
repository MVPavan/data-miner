"""LitServe server for Falcon-Perception zero-shot detection / segmentation.

Subclasses DetectorServerBase — uniform DetectorRequest/DetectorResponse wire.
Per-class iteration lives here (joint multi-class conflates everything as
"person"). Client sends full class list in DetectorRequest.prompts; server
runs one generate per prompt and concatenates.

Launch:
    python serve_falcon.py --port 3002 --device 1 --max-batch-size 4
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
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

_MODEL_ID = "tiiuae/Falcon-Perception"

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


def _center_size_to_xyxy(xc, yc, bw, bh):
    return (_clamp(xc - bw / 2), _clamp(yc - bh / 2),
            _clamp(xc + bw / 2), _clamp(yc + bh / 2))


def _mask_rle_to_xyxy(mask_rle: dict):
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


class FalconApi(DetectorServerBase):
    """Falcon-Perception server — loops per prompt internally."""

    def _load_model(self) -> None:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        from falcon_perception import load_and_prepare_model, setup_torch_config
        from falcon_perception.batch_inference import BatchInferenceEngine

        setup_torch_config()
        model, tokenizer, model_args = load_and_prepare_model(
            hf_model_id=_MODEL_ID,
            device=self.device,
            dtype=_DEFAULT_PARAMS["dtype"],
            compile=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.engine = BatchInferenceEngine(model, tokenizer)
        self.stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
        self._supports_segmentation = bool(model_args.do_segmentation)
        logger.info("Falcon ready | supports_segmentation=%s", self._supports_segmentation)

    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import process_batch_and_generate

        task = _DEFAULT_PARAMS["task"]
        if task == "segmentation" and not self._supports_segmentation:
            task = "detection"

        per_prompt = []
        for prompt in req.prompts:
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
            prompts=list(req.prompts),
            threshold=req.threshold,
            extras={"task": task},
        )

    def _run_one_request(self, item: PreparedInput) -> RawPrediction:
        task = item.extras["task"]
        per_prompt_outputs = []
        for entry in item.processor_inputs:
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
            image_size=item.image_size,
            prompts=item.prompts,
            threshold=item.threshold,
            extras={"task": task},
        )

    def _to_response(self, result: RawPrediction) -> DetectorResponse:
        from falcon_perception.visualization_utils import pair_bbox_entries

        task = result.extras["task"]
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for entry in result.outputs:
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
                all_scores.append(1.0)  # Falcon has no per-detection score
                all_labels.append(prompt)

        return DetectorResponse(boxes=all_boxes, scores=all_scores, labels=all_labels)


if __name__ == "__main__":
    run_server(FalconApi, default_port=3002)
