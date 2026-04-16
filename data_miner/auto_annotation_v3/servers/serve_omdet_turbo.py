"""LitServe server for OmDet-Turbo open-vocabulary object detection.

Subclasses DetectorServerBase — uniform DetectorRequest/DetectorResponse wire.
OmDet-Turbo natively handles multi-class detection in a single forward pass,
so all prompts are processed jointly (similar to OWLv2, unlike per-prompt
GDINO).  Uses ``OmDetTurboBatchPredictor`` from ``omdet_batch.py`` which
splits the forward pass for efficient batching.

Config knobs (forwarded from ``DetectorConfig.options`` via launch_all.py):
    model_id (str): HuggingFace model identifier.
        Default "omlab/omdet-turbo-swin-tiny-hf".

Launch standalone:
    python -m data_miner.auto_annotation_v3.servers.serve_omdet_turbo \
        --port 3005 --device 0 --max-batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

import torch
from PIL import Image

from ..contracts import (
    DetectorRequest,
    DetectorResponse,
    PreparedInput,
    RawPrediction,
)
from .base import DetectorServerBase

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "omlab/omdet-turbo-swin-tiny-hf"
_DEFAULT_THRESHOLD = 0.3


class OmDetTurboApi(DetectorServerBase):
    """OmDet-Turbo server — native multi-class, split forward pass.

    Instance attributes populated by launch_all via pre-CLI assignment:
        _model_id (str): HuggingFace model ID.
    """

    _model_id: str = _DEFAULT_MODEL_ID

    # ------------------------------------------------------------------
    # One-time model load
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from ..omdet_batch import OmDetTurboBatchPredictor

        logger.info("Loading OmDet-Turbo predictor (%s) …", self._model_id)
        self.predictor = OmDetTurboBatchPredictor(
            model_id=self._model_id,
            device=self.device,
            dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
        )

        # set_classes mutates predictor state; serialize access.
        self._class_lock = threading.Lock()
        self._current_classes: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # Proposal-mode hooks
    # ------------------------------------------------------------------

    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        w, h = image.size
        return PreparedInput(
            image=image,
            processor_inputs={"image": image},
            image_size=(w, h),
            prompts=list(req.prompts),
            threshold=req.threshold,
        )

    def _run_one_request(self, item: PreparedInput) -> RawPrediction:
        prompt_key = tuple(item.prompts)
        with self._class_lock:
            if self._current_classes != prompt_key:
                self.predictor.set_classes(list(prompt_key))
                self._current_classes = prompt_key

            threshold = item.threshold if item.threshold is not None else _DEFAULT_THRESHOLD
            results = self.predictor.predict_images(
                [item.image],
                threshold=float(threshold),
                nms_threshold=0.5,
            )

        return RawPrediction(
            outputs=results[0] if results else None,
            inputs=None,
            image_size=item.image_size,
            prompts=item.prompts,
            threshold=item.threshold,
        )

    def _to_response(self, result: RawPrediction) -> DetectorResponse:
        w, h = result.image_size
        res = result.outputs
        if res is None or len(res.get("scores", [])) == 0:
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
                logger.warning(
                    "OmDet-Turbo returned class_id=%s outside prompts range", idx
                )
                continue
            x1, y1, x2, y2 = [float(v) for v in bx]
            out_boxes.append(self._norm_box_px([x1, y1, x2, y2], w, h))
            out_scores.append(float(sc))
            out_labels.append(prompts[idx])

        return DetectorResponse(
            boxes=out_boxes, scores=out_scores, labels=out_labels
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import litserve as ls

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="OmDet-Turbo LitServe server")
    ap.add_argument("--port", type=int, default=3005)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--batch-timeout", type=float, default=0.05)
    ap.add_argument("--model-id", type=str, default=_DEFAULT_MODEL_ID)
    args = ap.parse_args()

    api = OmDetTurboApi()
    api._model_id = args.model_id

    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=1,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
