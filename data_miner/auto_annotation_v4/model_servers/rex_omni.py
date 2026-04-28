"""LitAPI wrapper for Rex-Omni.

Thin protocol shim. All inference logic is in
:class:`data_miner.auto_annotation_v4.models.rex_omni.RexOmniModel`.

Wire is the standard ``DetectorRequest`` / ``DetectorResponse``. Server runs
``max_batch_size=1`` because Rex-Omni's autoregressive decoder is sequence-
length bound, not batch-bound — multi-image batching does not improve
throughput in practice and complicates session ownership.

Knobs (set on the api instance before ``setup()``, mirrors sam3_1 style):
    backend (str):           "transformers" or "vllm"  (default "transformers").
    max_new_tokens (int|None): forwarded to the wrapper if set.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models.rex_omni import RexOmniModel
from .base import DetectorServerBase

logger = logging.getLogger(__name__)


class RexOmniApi(DetectorServerBase):
    """Rex-Omni LitServe server. Delegates to :class:`RexOmniModel`."""

    model_id = "IDEA-Research/Rex-Omni"
    _backend: str = "transformers"
    _max_new_tokens: int | None = None

    def setup(self, device: str) -> None:
        self.model = RexOmniModel()
        options: dict[str, Any] = {"backend": self._backend}
        if self._max_new_tokens is not None:
            options["max_new_tokens"] = self._max_new_tokens
        self.model.load(device, self.model_id, **options)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    import litserve as ls

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Rex-Omni LitServe server")
    ap.add_argument("--port", type=int, default=3015)
    ap.add_argument("--gpu", default="cuda:0")
    ap.add_argument("--model-id", default="IDEA-Research/Rex-Omni")
    ap.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Rex-Omni inference backend",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the wrapper's default decode length",
    )
    args = ap.parse_args()

    api = RexOmniApi()
    api.model_id = args.model_id
    api._backend = args.backend
    api._max_new_tokens = args.max_new_tokens

    # max_batch_size=1: Rex-Omni decoding is autoregressive — batching adds
    # latency without throughput gain at typical sequence lengths.
    server = ls.LitServer(
        api,
        accelerator="gpu",
        devices=[args.gpu],
        max_batch_size=1,
    )
    server.run(port=args.port)
