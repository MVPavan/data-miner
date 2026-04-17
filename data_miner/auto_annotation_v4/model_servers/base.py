"""DetectorServerBase -- LitAPI base class for all detector model servers.

Subclasses set ``self.model`` (a :class:`BaseDetectorModel`) in ``setup()``
and optionally override ``decode_request`` / ``predict`` / ``encode_response``
for custom routing (e.g. SAM3-DART dual mode).

Wire contract is uniform: ``DetectorRequest`` -> ``DetectorResponse``.
All inference logic lives in ``models/``; servers only parse, delegate, and
format.
"""

from __future__ import annotations

import logging

import litserve as ls
from PIL import Image

from ..configs.wire import DetectorRequest, DetectorResponse, PreparedInput, RawPrediction

logger = logging.getLogger(__name__)


class DetectorServerBase(ls.LitAPI):
    """Shared LitAPI skeleton for detector model servers.

    Subclasses create ``self.model`` (a ``BaseDetectorModel``) in ``setup()``
    and inherit the default four-hook flow.  Override individual hooks only
    when the model needs non-standard dispatch (e.g. dual-mode SAM3-DART).
    """

    model_id: str = ""

    # ------------------------------------------------------------------
    # LitServe hooks
    # ------------------------------------------------------------------

    def setup(self, device: str) -> None:
        """Load model onto *device*.  Subclasses must create ``self.model``."""
        raise NotImplementedError

    def decode_request(self, request: dict, **kwargs) -> PreparedInput:
        """Parse wire request, load image, delegate to ``model.prepare()``."""
        req = DetectorRequest.model_validate(request)
        try:
            image = Image.open(req.image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{req.image_path}': {exc}") from exc
        return self.model.prepare(image, req.prompts, req.threshold)

    def predict(self, batch: list, **kwargs) -> list:
        """Run inference.  Default: sequential per-item."""
        return [self.model.infer(item) for item in batch]

    def encode_response(self, output, **kwargs):
        """Convert model output to wire response."""
        resp = self.model.postprocess(output)
        return resp.model_dump()


# ---------------------------------------------------------------------------
# CLI entry-point helper
# ---------------------------------------------------------------------------


def run_server(api_cls: type[DetectorServerBase], default_port: int = 3001,
               default_model_id: str = "") -> None:
    """Parse standard CLI args and start a LitServer."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description=f"{api_cls.__name__} LitServe server")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--model-id", default=default_model_id)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout", type=float, default=0.05)
    args = parser.parse_args()

    api = api_cls()
    if args.model_id:
        api.model_id = args.model_id

    server = ls.LitServer(
        api,
        accelerator="gpu",
        devices=[args.gpu],
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
