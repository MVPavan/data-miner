"""DetectorServerBase — abstract LitServe base for every detector server.

All detector servers share the same four-method LitAPI contract, image
loading, device/dtype selection, and Pydantic validation at the wire
boundary. Subclasses implement four hooks:

    _load_model()                             one-time setup
    _prepare(image, req)    -> PreparedInput
    _run_one_request(item)  -> RawPrediction
    _to_response(result)    -> DetectorResponse

Wire contract is uniform: DetectorRequest → DetectorResponse. Servers
whose wire shape diverges (e.g. SAM3 refine) override decode_request /
encode_response directly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import litserve as ls
import torch
from PIL import Image

from ..contracts import (
    DetectorRequest,
    DetectorResponse,
    PreparedInput,
    RawPrediction,
)

logger = logging.getLogger(__name__)


class DetectorServerBase(ls.LitAPI, ABC):
    """Shared template-method skeleton for detector LitServe servers."""

    # ------------------------------------------------------------------
    # LitServe hooks (concrete; dispatch to abstract hooks below)
    # ------------------------------------------------------------------

    def setup(self, device: str) -> None:
        self.device = device
        self.dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
        self._load_model()
        logger.info("%s ready on %s (dtype=%s)", type(self).__name__, device, self.dtype)

    def decode_request(self, request: dict) -> PreparedInput:
        req = DetectorRequest.model_validate(request)
        try:
            image = Image.open(req.image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{req.image_path}': {exc}") from exc
        return self._prepare(image, req)

    def predict(self, batch: list[PreparedInput]) -> list[RawPrediction]:
        # Default: sequential per-item inference. Subclasses can override to
        # do tensor-stacked batching if their inputs permit it.
        return [self._run_one_request(item) for item in batch]

    def encode_response(self, output: RawPrediction, **kwargs):
        # Return annotation intentionally omitted: FastAPI would treat it as
        # a response_model and choke on the forward ref (from __future__
        # annotations in contracts.py). LitServe's format_encoded_response
        # isinstance-checks BaseModel and calls model_dump_json() itself.
        return self._to_response(output)

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model(self) -> None:
        """Load processor/model/engine onto self.device; runs once on startup."""

    @abstractmethod
    def _prepare(self, image: Image.Image, req: DetectorRequest) -> PreparedInput:
        """Build processor inputs; return the intermediate Pydantic model."""

    @abstractmethod
    def _run_one_request(self, item: PreparedInput) -> RawPrediction:
        """Run forward pass(es) for ONE request. Subclass decides loop vs
        native-multi-prompt internally."""

    @abstractmethod
    def _to_response(self, result: RawPrediction) -> DetectorResponse:
        """Post-process outputs → DetectorResponse."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    @classmethod
    def _norm_box_px(cls, box_px: list[float], w: int, h: int) -> list[float]:
        x1, y1, x2, y2 = box_px
        return [
            cls._clamp01(x1 / w),
            cls._clamp01(y1 / h),
            cls._clamp01(x2 / w),
            cls._clamp01(y2 / h),
        ]


# ---------------------------------------------------------------------------
# Entry-point helper — lets every server file share identical CLI boilerplate
# ---------------------------------------------------------------------------


def run_server(api_cls: type[DetectorServerBase], default_port: int) -> None:
    """Parse standard CLI args and start the LitServe server."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description=f"{api_cls.__name__} LitServe server")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout", type=float, default=0.05)
    args = parser.parse_args()

    api = api_cls()
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=1,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
