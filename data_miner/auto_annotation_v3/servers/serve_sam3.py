"""LitServe server for SAM3 (Segment-Anything Model 3) — dual-mode.

Single endpoint, two wire shapes:

  * Proposal — DetectorRequest → DetectorResponse (base class path).
    Per-class iteration: one forward pass per prompt inside _run_one_request.

  * Refine — SAM3RefineRequest → SAM3RefineResponse. Discriminated by the
    presence of ``bbox`` in the payload; handled by an override of
    decode_request / encode_response that bypasses the proposal hooks.

Launch:
    python serve_sam3.py --port 3003 --device 2 --max-batch-size 8
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from ..contracts import (
    DetectorRequest,
    DetectorResponse,
    PreparedInput,
    RawPrediction,
    SAM3RefineRequest,
    SAM3RefineResponse,
)
from .base import DetectorServerBase, run_server

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/sam3"


class _RefinePrepared(BaseModel):
    """Carrier for refine-mode decode→predict handoff (sibling of PreparedInput)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    inputs: object
    image_size: tuple[int, int]
    threshold: float


class _RefineRaw(BaseModel):
    """Carrier for refine-mode predict→encode handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    outputs: object
    inputs: object
    image_size: tuple[int, int]
    threshold: float


class SAM3Api(DetectorServerBase):
    """SAM3 server: proposal uses DetectorServerBase hooks; refine overrides."""

    # ------------------------------------------------------------------
    # Base-class lifecycle: load model once; proposal uses hooks below.
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import Sam3Model, Sam3Processor

        logger.info("Loading SAM3 processor %s", _MODEL_ID)
        self.processor = Sam3Processor.from_pretrained(_MODEL_ID)
        torch_dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        logger.info("Loading SAM3 model onto %s dtype=%s", self.device, torch_dtype)
        self.model = (
            Sam3Model.from_pretrained(_MODEL_ID, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )

    # ------------------------------------------------------------------
    # Mode dispatcher: 'bbox' → refine; otherwise → proposal via base.
    # ------------------------------------------------------------------

    def decode_request(self, request: dict, **kwargs):
        if "bbox" in request:
            return self._decode_refine(request)
        return super().decode_request(request)

    def predict(self, batch, **kwargs):
        return [
            self._run_refine(item) if isinstance(item, _RefinePrepared)
            else self._run_one_request(item)
            for item in batch
        ]

    def encode_response(self, output, **kwargs):
        if isinstance(output, _RefineRaw):
            return self._to_refine_response(output)
        return self._to_response(output)

    # ------------------------------------------------------------------
    # Proposal-mode hooks (DetectorServerBase contract).
    # ------------------------------------------------------------------

    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        # Build one processor input per prompt (SAM3 degrades on joint multi-class).
        per_prompt = []
        for prompt in req.prompts:
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(device=self.device, dtype=self.model.dtype)
            per_prompt.append({"prompt": prompt, "inputs": inputs})
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
            with torch.no_grad():
                outputs = self.model(**entry["inputs"])
            per_prompt_outputs.append({
                "prompt": entry["prompt"],
                "outputs": outputs,
                "inputs": entry["inputs"],
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
        threshold = result.threshold if result.threshold is not None else 0.5

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for entry in result.outputs:
            prompt = entry["prompt"]
            outputs = entry["outputs"]
            inputs = entry["inputs"]
            post = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            raw_boxes = post.get("boxes")
            scores_t = post.get("scores")
            if raw_boxes is None or scores_t is None or len(raw_boxes) == 0:
                continue
            scores_list = (
                scores_t.cpu().tolist() if torch.is_tensor(scores_t) else list(scores_t)
            )
            for b, s in zip(raw_boxes, scores_list):
                x1, y1, x2, y2 = [float(v) for v in b.tolist()]
                all_boxes.append(self._norm_box_px([x1, y1, x2, y2], w, h))
                all_scores.append(float(s))
                all_labels.append(prompt)

        return DetectorResponse(boxes=all_boxes, scores=all_scores, labels=all_labels)

    # ------------------------------------------------------------------
    # Refine-mode path (sibling to proposal; skips base hooks).
    # ------------------------------------------------------------------

    def _decode_refine(self, request: dict) -> _RefinePrepared:
        req = SAM3RefineRequest.model_validate(request)
        try:
            image = Image.open(req.image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{req.image_path}': {exc}") from exc
        w, h = image.size
        pixel_box = [req.bbox[0] * w, req.bbox[1] * h, req.bbox[2] * w, req.bbox[3] * h]

        if req.points:
            input_points = [[[p[0], p[1]] for p in req.points]]
            input_labels = [[int(p[2]) for p in req.points]]
            inputs = self.processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=[[pixel_box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device=self.device, dtype=self.model.dtype)
        else:
            inputs = self.processor(
                images=image,
                input_boxes=[[pixel_box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device=self.device, dtype=self.model.dtype)

        return _RefinePrepared(inputs=inputs, image_size=(w, h), threshold=req.threshold)

    def _run_refine(self, item: _RefinePrepared) -> _RefineRaw:
        with torch.no_grad():
            outputs = self.model(**item.inputs)
        return _RefineRaw(
            outputs=outputs,
            inputs=item.inputs,
            image_size=item.image_size,
            threshold=item.threshold,
        )

    def _to_refine_response(self, result: _RefineRaw) -> SAM3RefineResponse:
        w, h = result.image_size
        post = self.processor.post_process_instance_segmentation(
            result.outputs,
            threshold=result.threshold,
            mask_threshold=0.5,
            target_sizes=result.inputs.get("original_sizes").tolist(),
        )[0]
        raw_boxes = post.get("boxes")
        scores_t = post.get("scores")
        if raw_boxes is None or scores_t is None or len(raw_boxes) == 0:
            return SAM3RefineResponse(box=None, score=0.0)
        scores_list = (
            scores_t.cpu().tolist() if torch.is_tensor(scores_t) else list(scores_t)
        )
        best_idx = int(max(range(len(scores_list)), key=lambda i: scores_list[i]))
        x1, y1, x2, y2 = [float(v) for v in raw_boxes[best_idx].tolist()]
        return SAM3RefineResponse(
            box=self._norm_box_px([x1, y1, x2, y2], w, h),
            score=float(scores_list[best_idx]),
        )


if __name__ == "__main__":
    run_server(SAM3Api, default_port=3003)
