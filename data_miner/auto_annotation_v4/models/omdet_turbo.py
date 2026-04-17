"""Pure inference model for OmDet-Turbo open-vocabulary object detection.

Wraps ``omlab/omdet-turbo-swin-tiny-hf`` via ``OmDetTurboBatchPredictor``
from ``omdet_turbo_batch.py``. No LitServe dependency — the LitAPI wrapper
in model_servers/ calls these methods.

OmDet-Turbo natively handles multi-class detection in a single forward pass
(similar to OWLv2, unlike per-prompt GDINO). Uses the batch predictor which
splits the forward pass into reusable stages: vision encoding, text
embedding, and decoding.

Quirks:
  - ``set_classes`` mutates predictor state; access is serialized with a
    threading lock (same pattern as SAM3/DART).
  - Default threshold is 0.3 (calibrated for OmDet-Turbo's score range).
  - Uses ``float16`` on CUDA (not bfloat16).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from PIL import Image

from ..configs.wire import DetectorResponse, PreparedInput, RawPrediction
from .base import BaseDetectorModel, normalize_box

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "omlab/omdet-turbo-swin-tiny-hf"
_DEFAULT_THRESHOLD = 0.3


class OmDetTurboModel(BaseDetectorModel):
    """OmDet-Turbo detector — native multi-class, split forward pass.

    Attributes (populated by ``load``):
        predictor: ``OmDetTurboBatchPredictor`` wrapping the HuggingFace model.
        device: Torch device string.
        dtype: Torch dtype for inference.
    """

    def load(self, device: str, model_id: str = _DEFAULT_MODEL_ID,
             **options: Any) -> None:
        """Load OmDet-Turbo batch predictor onto *device*.

        Args:
            device: Torch device string (``"cuda:0"``, ``"cpu"``, etc.).
            model_id: HuggingFace model identifier.
            **options: Unused — reserved for forward-compat.
        """
        import torch
        from .omdet_turbo_batch import OmDetTurboBatchPredictor

        self.device = device
        self.dtype = torch.float16 if "cuda" in str(device) else torch.float32

        logger.info("Loading OmDet-Turbo predictor (%s) ...", model_id)
        self.predictor = OmDetTurboBatchPredictor(
            model_id=model_id,
            device=device,
            dtype=self.dtype,
        )

        # set_classes mutates predictor state; serialize access.
        self._class_lock = threading.Lock()
        self._current_classes: tuple[str, ...] | None = None

    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Minimal preprocessing — pass image through for batch predictor.

        The batch predictor handles all internal resizing/normalization.

        Args:
            image: RGB PIL image.
            prompts: List of class-name strings to detect.
            threshold: Optional confidence threshold override (default 0.3).

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
        """Set classes and run OmDet-Turbo prediction.

        Updates the class list (no-op if unchanged), then runs the full
        set_images + predict_batch pipeline for a single image.

        Args:
            prepared: Result of ``prepare()``.

        Returns:
            ``RawPrediction`` with the batch predictor's result dict.
        """
        prompt_key = tuple(prepared.prompts)
        with self._class_lock:
            if self._current_classes != prompt_key:
                self.predictor.set_classes(list(prompt_key))
                self._current_classes = prompt_key

            threshold = prepared.threshold if prepared.threshold is not None else _DEFAULT_THRESHOLD
            results = self.predictor.predict_images(
                [prepared.image],
                threshold=float(threshold),
                nms_threshold=0.5,
            )

        return RawPrediction(
            outputs=results[0] if results else None,
            inputs=None,
            image_size=prepared.image_size,
            prompts=prepared.prompts,
            threshold=prepared.threshold,
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Post-process OmDet-Turbo results into normalized boxes/scores/labels.

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
        if res is None or len(res.get("scores", [])) == 0:
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
                logger.warning(
                    "OmDet-Turbo returned class_id=%s outside prompts range", idx
                )
                continue
            x1, y1, x2, y2 = [float(v) for v in bx]
            out_boxes.append(normalize_box([x1, y1, x2, y2], w, h))
            out_scores.append(float(sc))
            out_labels.append(prompts[idx])

        return DetectorResponse(
            boxes=out_boxes, scores=out_scores, labels=out_labels
        )
