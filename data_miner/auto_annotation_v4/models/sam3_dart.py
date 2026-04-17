"""Pure inference model for SAM3 powered by DART — dual-mode, single model load.

Wraps DART's ``Sam3Image`` model with ``Sam3MultiClassPredictorFast`` for
proposal-mode detection. No LitServe dependency — the LitAPI wrapper in
model_servers/ calls these methods.

Supports two modes from a single model load:

  * **Proposal** — ``prepare`` / ``infer`` / ``postprocess``.
    Single forward pass over the full prompt list; labels returned mapped
    back to the request's prompts via DART's ``class_ids`` index.

  * **Refine** — ``prepare_refine`` / ``infer_refine`` / ``postprocess_refine``.
    Uses DART's ``Sam3Image.predict_inst`` with the shared detector backbone
    + the SAM3InteractiveImagePredictor loaded from the same ``sam3.pt``
    checkpoint (via ``enable_inst_interactivity=True``).

Config knobs (passed via ``load(options=...)``):
    detection_only (bool): M4 when True (box-NMS, no masks), M3 when False.
        Default True.
    presence_threshold (float): Early-exit threshold in DART's decoder
        presence head. Default 0.05.

Quirks:
  - DART lives under ``scratchpad/DART/``; its path is prepended to
    ``sys.path`` at load time.
  - ``set_classes`` mutates predictor state; access is serialized with a
    threading lock.
  - DART's ``_setup_device_and_mode`` only matches bare ``"cuda"``; the
    model is built with ``device="cuda"`` then moved explicitly.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from ..configs.wire import (
    DetectorResponse,
    PreparedInput,
    RawPrediction,
    SAM3RefineResponse,
)
from .base import BaseDetectorModel, normalize_box

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal hand-off models for refine mode
# ---------------------------------------------------------------------------

class _RefinePrepared(BaseModel):
    """Carrier for refine-mode prepare->infer handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    image: object
    pixel_box: list[float]
    points: list[list[float]] | None
    image_size: tuple[int, int]


class _RefineRaw(BaseModel):
    """Carrier for refine-mode infer->postprocess handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    mask: object               # numpy bool 2-D
    mask_score: float
    image_size: tuple[int, int]


class SAM3DartModel(BaseDetectorModel):
    """SAM3 via DART: proposal-mode and refine-mode from a single model load.

    Attributes (populated by ``load``):
        model: DART ``Sam3Image`` model instance.
        predictor: ``Sam3MultiClassPredictorFast`` wrapper for proposal mode.
        dart_processor: ``Sam3Processor`` for refine-mode image encoding.
        device: Torch device string.
        dtype: Torch dtype for inference.
    """

    def load(self, device: str, model_id: str = "",
             **options: Any) -> None:
        """Build DART Sam3Image with inst_interactivity and multiclass wrapper.

        Prepends DART's scratchpad path to ``sys.path``, builds the model
        with ``enable_inst_interactivity=True``, and wraps it with
        ``Sam3MultiClassPredictorFast`` for proposal mode.

        Args:
            device: Torch device string (``"cuda:0"``, ``"cpu"``, etc.).
            model_id: Unused — DART uses a local checkpoint, not HuggingFace.
            **options:
                detection_only (bool): M4 (True) or M3 (False). Default True.
                presence_threshold (float): DART presence head threshold.
                    Default 0.05.
        """
        import torch

        # DART lives under scratchpad/; prepend to sys.path once.
        dart_root = Path(__file__).resolve().parents[3] / "scratchpad" / "DART"
        if str(dart_root) not in sys.path:
            sys.path.insert(0, str(dart_root))

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
        from sam3.model.sam3_image_processor import Sam3Processor as DartProcessor

        self.device = device
        self.dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

        detection_only = options.get("detection_only", True)
        presence_threshold = options.get("presence_threshold", 0.05)

        # DART's _setup_device_and_mode only matches bare "cuda"; build with
        # that and move explicitly so sub-modules (inst_interactive) follow.
        logger.info("Building DART Sam3Image with inst_interactivity=True ...")
        self.model = build_sam3_image_model(
            device="cuda", eval_mode=True, enable_inst_interactivity=True
        ).to(device)
        assert self.model.inst_interactive_predictor is not None, (
            "inst_interactive_predictor missing — enable_inst_interactivity=True "
            "should have loaded the tracker half of sam3.pt."
        )

        logger.info(
            "Wrapping with Sam3MultiClassPredictorFast "
            "(detection_only=%s, presence_threshold=%s)",
            detection_only, presence_threshold,
        )
        self.predictor = Sam3MultiClassPredictorFast(
            self.model,
            device=device,
            use_fp16=True,
            presence_threshold=presence_threshold,
            detection_only=detection_only,
        )
        self.dart_processor = DartProcessor(self.model, device=device)

        # set_classes mutates predictor state; serialize access.
        self._class_lock = threading.Lock()
        self._current_classes: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # Proposal-mode: prepare / infer / postprocess
    # ------------------------------------------------------------------

    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Preprocess image for DART proposal-mode detection.

        Minimal preprocessing — the image is passed through as-is; DART's
        predictor handles internal resizing/normalization.

        Args:
            image: RGB PIL image.
            prompts: List of class-name strings to detect.
            threshold: Confidence threshold override (default 0.5).

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
        """Set classes and run DART prediction.

        Updates the class list (no-op if unchanged), encodes the image
        through the backbone, and runs the decoder with NMS.

        Args:
            prepared: Result of ``prepare()``.

        Returns:
            ``RawPrediction`` with DART's result dict.
        """
        prompt_key = tuple(prepared.prompts)
        with self._class_lock:
            if self._current_classes != prompt_key:
                self.predictor.set_classes(list(prompt_key))
                self._current_classes = prompt_key

            threshold = prepared.threshold if prepared.threshold is not None else 0.5
            state = self.predictor.set_image(prepared.image)
            res = self.predictor.predict(
                state,
                confidence_threshold=float(threshold),
                nms_threshold=0.7,
            )
        return RawPrediction(
            outputs=res,
            inputs=None,
            image_size=prepared.image_size,
            prompts=prepared.prompts,
            threshold=prepared.threshold,
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Post-process DART results into normalized boxes/scores/labels.

        Maps ``class_ids`` back to the original prompt strings. Boxes are
        normalized from pixel coordinates to [0, 1] range.

        Args:
            raw: Result of ``infer()``.

        Returns:
            ``DetectorResponse`` with detections relabelled to prompt strings.
        """
        import torch

        w, h = raw.image_size
        res = raw.outputs
        if res is None or "scores" not in res or len(res["scores"]) == 0:
            return DetectorResponse(boxes=[], scores=[], labels=[])

        def _tolist(x):
            return x.detach().cpu().tolist() if torch.is_tensor(x) else list(x)

        boxes_px = _tolist(res["boxes"])
        scores = _tolist(res["scores"])
        class_ids = _tolist(res["class_ids"])

        prompts = raw.prompts
        out_boxes: list[list[float]] = []
        out_scores: list[float] = []
        out_labels: list[str] = []
        for bx, sc, cid in zip(boxes_px, scores, class_ids):
            idx = int(cid)
            if idx < 0 or idx >= len(prompts):
                logger.warning("DART returned class_id=%s outside prompts range", idx)
                continue
            x1, y1, x2, y2 = [float(v) for v in bx]
            out_boxes.append(normalize_box([x1, y1, x2, y2], w, h))
            out_scores.append(float(sc))
            out_labels.append(prompts[idx])
        return DetectorResponse(boxes=out_boxes, scores=out_scores, labels=out_labels)

    # ------------------------------------------------------------------
    # Refine-mode: prepare_refine / infer_refine / postprocess_refine
    # ------------------------------------------------------------------

    def prepare_refine(self, image: Image.Image, bbox: list[float],
                       points: list[list[float]] | None = None,
                       threshold: float = 0.5) -> _RefinePrepared:
        """Prepare inputs for SAM3 refine-mode (mask -> bbox tightening).

        Converts the normalized bbox to pixel coordinates for DART's
        ``predict_inst``.

        Args:
            image: RGB PIL image.
            bbox: Normalized [x1, y1, x2, y2] bounding box in [0, 1] range.
            points: Optional list of ``[[x, y, label], ...]`` point prompts
                (pixel coords with positive=1 / negative=0 labels).
            threshold: Unused in refine — kept for API symmetry.

        Returns:
            ``_RefinePrepared`` carrier with pixel-space box and image.
        """
        w, h = image.size
        pixel_box = [
            bbox[0] * w, bbox[1] * h,
            bbox[2] * w, bbox[3] * h,
        ]
        return _RefinePrepared(
            image=image,
            pixel_box=pixel_box,
            points=points,
            image_size=(w, h),
        )

    def infer_refine(self, prepared: _RefinePrepared) -> _RefineRaw:
        """Run SAM3's interactive predict_inst for mask refinement.

        Sets a fresh inference state per image (single backbone pass) and
        runs ``predict_inst`` with the provided box and optional point
        prompts.

        Args:
            prepared: Result of ``prepare_refine()``.

        Returns:
            ``_RefineRaw`` with the decoded boolean mask and IoU score.
        """
        import torch

        box_np = np.array(prepared.pixel_box, dtype=np.float32)[None, :]
        pc = pl = None
        if prepared.points:
            pc = np.array([[p[0], p[1]] for p in prepared.points], dtype=np.float32)
            pl = np.array([int(p[2]) for p in prepared.points], dtype=np.int64)

        with torch.inference_mode():
            inference_state = self.dart_processor.set_image(prepared.image)
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
        return _RefineRaw(mask=mask, mask_score=mask_score, image_size=prepared.image_size)

    def postprocess_refine(self, raw: _RefineRaw) -> SAM3RefineResponse:
        """Convert refine-mode mask to a normalized bounding box.

        Finds the foreground extent of the boolean mask and returns the
        tight bounding box in normalized [0, 1] coordinates.

        Args:
            raw: Result of ``infer_refine()``.

        Returns:
            ``SAM3RefineResponse`` with the refined box and mask IoU score.
        """
        w, h = raw.image_size
        mask = raw.mask
        ys, xs = np.where(mask)
        if xs.size == 0:
            return SAM3RefineResponse(box=None, score=0.0)
        box_px = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        return SAM3RefineResponse(
            box=normalize_box(box_px, w, h),
            score=raw.mask_score,
        )
