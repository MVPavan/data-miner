"""LitAPI wrapper for SAM3 via DART -- dual-mode (proposal + refine).

Proposal mode: standard DetectorRequest -> DetectorResponse flow via
the base class hooks, delegating to SAM3DartModel.

Refine mode: SAM3RefineRequest -> SAM3RefineResponse.  Detected by the
presence of ``bbox`` in the payload.  Handled via dedicated decode/predict/
encode overrides that call SAM3DartModel's refine methods.

All model logic lives in ``models/sam3_dart.py``; this file only routes.
"""

from __future__ import annotations

import logging

from PIL import Image

from ..configs.wire import (
    DetectorRequest,
    SAM3RefineRequest,
    SAM3RefineResponse,
)
from ..models.sam3_dart import SAM3DartModel
from .base import DetectorServerBase

logger = logging.getLogger(__name__)

# Sentinel type tag so predict/encode can distinguish modes.
_REFINE_TAG = "__refine__"


class SAM3DartApi(DetectorServerBase):
    """SAM3 (DART) dual-mode server -- routes proposal vs refine.

    Instance attributes optionally set before setup():
        _detection_only (bool): M4 when True, M3 when False.  Default True.
        _presence_threshold (float): DART presence-head threshold.  Default 0.05.
    """

    model_id = "sam3_dart"
    _detection_only: bool = True
    _presence_threshold: float = 0.05

    def setup(self, device: str) -> None:
        self.model = SAM3DartModel()
        self.model.load(
            device, self.model_id,
            detection_only=self._detection_only,
            presence_threshold=self._presence_threshold,
        )

    # ------------------------------------------------------------------
    # Mode dispatcher
    # ------------------------------------------------------------------

    def decode_request(self, request: dict, **kwargs):
        if "bbox" in request:
            return self._decode_refine(request)
        return super().decode_request(request, **kwargs)

    def predict(self, batch, **kwargs):
        results = []
        for item in batch:
            if isinstance(item, dict) and item.get("__mode__") == _REFINE_TAG:
                results.append(self.model.refine(item))
            else:
                results.append(self.model.infer(item))
        return results

    def encode_response(self, output, **kwargs):
        if isinstance(output, SAM3RefineResponse):
            return output.model_dump()
        resp = self.model.postprocess(output)
        return resp.model_dump()

    # ------------------------------------------------------------------
    # Refine-mode decode
    # ------------------------------------------------------------------

    def _decode_refine(self, request: dict) -> dict:
        """Parse refine request and package for model.refine()."""
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
        return {
            "__mode__": _REFINE_TAG,
            "image": image,
            "pixel_box": pixel_box,
            "points": req.points,
            "image_size": (w, h),
        }


# ---------------------------------------------------------------------------
# Entry point -- custom argparse for DART-specific flags
# ---------------------------------------------------------------------------

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    import argparse
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


if __name__ == "__main__":
    import argparse
    import sys

    import litserve as ls

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="SAM3 (DART) LitServe server")
    ap.add_argument("--port", type=int, default=3013)
    ap.add_argument("--gpu", default="cuda:0")
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
        accelerator="gpu",
        devices=[args.gpu],
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
