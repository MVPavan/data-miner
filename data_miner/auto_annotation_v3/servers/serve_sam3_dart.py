"""LitServe server for SAM3 powered by DART — dual-mode, single model load.

Replaces the per-class fan-out in ``serve_sam3.py`` with DART's
``Sam3MultiClassPredictorFast`` (M3/M4). One ``Sam3Image`` model, built with
``enable_inst_interactivity=True``, drives BOTH paths:

  * Proposal — DetectorRequest → DetectorResponse. Single forward pass over the
    full prompt list; labels returned mapped back to the request's prompts via
    DART's ``class_ids`` index.

  * Refine — SAM3RefineRequest → SAM3RefineResponse. Goes through DART's
    ``Sam3Image.predict_inst`` (uses the shared detector backbone + the
    SAM3InteractiveImagePredictor that ``enable_inst_interactivity`` loads from
    the same ``sam3.pt`` checkpoint).

Discriminator: presence of ``bbox`` in the payload (matches serve_sam3.py).

Config knobs (forwarded from ``DetectorConfig.options`` via launch_all.py):
    detection_only (bool): M4 when True (box-NMS, no masks), M3 when False.
        Default True — matched spike parity and slightly higher proposal IoU.
    presence_threshold (float): early-exit threshold in DART's decoder
        presence head. Default 0.05.

Launch standalone:
    python -m data_miner.auto_annotation_v3.servers.serve_sam3_dart \
        --port 3013 --device 0 --detection-only --presence-threshold 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

import numpy as np
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
from .base import DetectorServerBase

logger = logging.getLogger(__name__)

# DART lives under scratchpad/; prepend to sys.path once at import time.
_DART_ROOT = (
    Path(__file__).resolve().parents[3] / "scratchpad" / "DART"
)
if str(_DART_ROOT) not in sys.path:
    sys.path.insert(0, str(_DART_ROOT))


class _RefinePrepared(BaseModel):
    """Carrier for refine-mode decode→predict handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    image: object
    pixel_box: list[float]
    points: list[list[float]] | None
    image_size: tuple[int, int]


class _RefineRaw(BaseModel):
    """Carrier for refine-mode predict→encode handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    mask: object               # numpy bool 2-D
    mask_score: float
    image_size: tuple[int, int]


class SAM3DartApi(DetectorServerBase):
    """SAM3 via DART: proposal uses base hooks; refine overrides.

    Instance attributes populated by launch_all via pre-CLI assignment:
        _detection_only (bool)
        _presence_threshold (float)
    Defaults applied in _load_model if not set.
    """

    _detection_only: bool = True
    _presence_threshold: float = 0.05

    # ------------------------------------------------------------------
    # One-time model load
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
        from sam3.model.sam3_image_processor import Sam3Processor as DartProcessor

        # DART's _setup_device_and_mode only matches bare "cuda"; build with
        # that and move explicitly so sub-modules (inst_interactive) follow.
        logger.info("Building DART Sam3Image with inst_interactivity=True …")
        self.model = build_sam3_image_model(
            device="cuda", eval_mode=True, enable_inst_interactivity=True
        ).to(self.device)
        assert self.model.inst_interactive_predictor is not None, (
            "inst_interactive_predictor missing — enable_inst_interactivity=True "
            "should have loaded the tracker half of sam3.pt."
        )

        logger.info(
            "Wrapping with Sam3MultiClassPredictorFast "
            "(detection_only=%s, presence_threshold=%s)",
            self._detection_only, self._presence_threshold,
        )
        self.predictor = Sam3MultiClassPredictorFast(
            self.model,
            device=self.device,
            use_fp16=True,
            presence_threshold=self._presence_threshold,
            detection_only=self._detection_only,
        )
        self.dart_processor = DartProcessor(self.model, device=self.device)

        # set_classes mutates predictor state; serialize access.
        self._class_lock = threading.Lock()
        self._current_classes: tuple[str, ...] | None = None

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
    # Proposal-mode hooks
    # ------------------------------------------------------------------

    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        w, h = image.size
        # Capture the image in processor_inputs for the forward pass.
        return PreparedInput(
            image=image,
            processor_inputs={"image": image},
            image_size=(w, h),
            prompts=list(req.prompts),
            threshold=req.threshold,
        )

    def _run_one_request(self, item: PreparedInput) -> RawPrediction:
        # Ensure classes are set (DART caches text embeddings keyed on the
        # exact class list; re-setting with the same tuple is a no-op thanks
        # to our own equality check).
        prompt_key = tuple(item.prompts)
        with self._class_lock:
            if self._current_classes != prompt_key:
                self.predictor.set_classes(list(prompt_key))
                self._current_classes = prompt_key

            threshold = item.threshold if item.threshold is not None else 0.5
            state = self.predictor.set_image(item.image)
            res = self.predictor.predict(
                state,
                confidence_threshold=float(threshold),
                nms_threshold=0.7,
            )
        return RawPrediction(
            outputs=res,
            inputs=None,
            image_size=item.image_size,
            prompts=item.prompts,
            threshold=item.threshold,
        )

    def _to_response(self, result: RawPrediction) -> DetectorResponse:
        w, h = result.image_size
        res = result.outputs
        if res is None or "scores" not in res or len(res["scores"]) == 0:
            return DetectorResponse(boxes=[], scores=[], labels=[])

        def _tolist(x):
            return x.detach().cpu().tolist() if torch.is_tensor(x) else list(x)

        boxes_px = _tolist(res["boxes"])
        scores = _tolist(res["scores"])
        class_ids = _tolist(res["class_ids"])

        prompts = result.prompts
        out_boxes: list[list[float]] = []
        out_scores: list[float] = []
        out_labels: list[str] = []
        for bx, sc, cid in zip(boxes_px, scores, class_ids):
            idx = int(cid)
            if idx < 0 or idx >= len(prompts):
                # Should not happen — DART's class_ids indexes the prompt list
                # we sent via set_classes. Log and skip defensively.
                logger.warning("DART returned class_id=%s outside prompts range", idx)
                continue
            x1, y1, x2, y2 = [float(v) for v in bx]
            out_boxes.append(self._norm_box_px([x1, y1, x2, y2], w, h))
            out_scores.append(float(sc))
            out_labels.append(prompts[idx])
        return DetectorResponse(boxes=out_boxes, scores=out_scores, labels=out_labels)

    # ------------------------------------------------------------------
    # Refine-mode path
    # ------------------------------------------------------------------

    def _decode_refine(self, request: dict) -> _RefinePrepared:
        req = SAM3RefineRequest.model_validate(request)
        try:
            image = Image.open(req.image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{req.image_path}': {exc}") from exc
        w, h = image.size
        pixel_box = [
            req.bbox[0] * w, req.bbox[1] * h,
            req.bbox[2] * w, req.bbox[3] * h,
        ]
        return _RefinePrepared(
            image=image,
            pixel_box=pixel_box,
            points=req.points,
            image_size=(w, h),
        )

    def _run_refine(self, item: _RefinePrepared) -> _RefineRaw:
        # predict_inst needs a fresh inference_state per image; set_image on
        # the processor is cheap (single backbone pass).
        box_np = np.array(item.pixel_box, dtype=np.float32)[None, :]
        pc = pl = None
        if item.points:
            pc = np.array([[p[0], p[1]] for p in item.points], dtype=np.float32)
            pl = np.array([int(p[2]) for p in item.points], dtype=np.int64)

        with torch.inference_mode():
            inference_state = self.dart_processor.set_image(item.image)
            masks, iou_preds, _low = self.model.predict_inst(
                inference_state,
                point_coords=pc, point_labels=pl,
                box=box_np,
                multimask_output=False,
            )
        if torch.is_tensor(masks):
            mask = (masks[0].detach().cpu().numpy() > 0)
        else:
            arr = np.asarray(masks)
            mask = (arr[0] > 0) if arr.ndim == 3 else (arr > 0)
        if torch.is_tensor(iou_preds):
            iou_preds = iou_preds.detach().cpu().numpy()
        iou_arr = np.asarray(iou_preds).flatten()
        mask_score = float(iou_arr[0]) if iou_arr.size else 0.0
        return _RefineRaw(mask=mask, mask_score=mask_score, image_size=item.image_size)

    def _to_refine_response(self, result: _RefineRaw) -> SAM3RefineResponse:
        w, h = result.image_size
        mask = result.mask
        ys, xs = np.where(mask)
        if xs.size == 0:
            return SAM3RefineResponse(box=None, score=0.0)
        box_px = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        return SAM3RefineResponse(
            box=self._norm_box_px(box_px, w, h),
            score=result.mask_score,
        )


# ---------------------------------------------------------------------------
# Entry point — custom argparse so we can accept the DART-specific flags
# forwarded by launch_all.py from DetectorConfig.options.
# ---------------------------------------------------------------------------

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


if __name__ == "__main__":
    import litserve as ls

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="SAM3 (DART) LitServe server")
    ap.add_argument("--port", type=int, default=3013)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--batch-timeout", type=float, default=0.05)
    ap.add_argument("--detection-only", type=_str2bool, default=True,
                    help="M4 when True (box-NMS, no masks), M3 when False.")
    ap.add_argument("--presence-threshold", type=float, default=0.05)
    args = ap.parse_args()

    api = SAM3DartApi()
    api._detection_only = args.detection_only
    api._presence_threshold = args.presence_threshold

    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=1,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
