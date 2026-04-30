"""LitAPI wrapper for SAM 3.1 — image refine + text→detect + video tracker.

Mode dispatch on request shape (mirrors ``sam3_dart.py`` style):

  * Has ``seeds`` list                → :meth:`SAM3OneModel.track`
  * Has ``bbox`` field                → :meth:`SAM3OneModel.refine`
  * Has ``prompts`` list and ``image_path`` → :meth:`SAM3OneModel.text_detect`

Used exclusively by ``manual_reviewer``. Auto-pipeline detectors live on
their own server; this file deliberately does not import or modify
``sam3_dart``.

All inference logic lives in :class:`data_miner.auto_annotation_v4.models.sam3_1.SAM3OneModel`.
This file is the HTTP/LitServe seam only.
"""

from __future__ import annotations

import logging
from typing import Any

from ..configs.wire import (
    DetectorRequest,
    DetectorResponse,
    SAM3ClickMaskRequest,
    SAM3ClickMaskResponse,
    SAM3RefineRequest,
    SAM3RefineResponse,
    SAM3VideoTrackRequest,
    SAM3VideoTrackResponse,
    SAM3VisualPromptRequest,
    SAM3VisualPromptResponse,
)
from ..models.sam3_1 import SAM3OneModel
from .base import DetectorServerBase

logger = logging.getLogger(__name__)


_REFINE_TAG = "__sam3_1_refine__"
_TRACK_TAG = "__sam3_1_track__"
_TEXT_TAG = "__sam3_1_text__"
_CLICK_TAG = "__sam3_1_click__"
_VISUAL_TAG = "__sam3_1_visual__"


class SAM3OneApi(DetectorServerBase):
    """SAM 3.1 server — refine / text-detect / video-track on one port.

    Instance attributes optionally set before ``setup()``:
        _checkpoint_path (str | None): pass-through to SAM 3.1's video
            predictor (default: download from HF).
        _bpe_path (str | None): pass-through to SAM 3.1's tokenizer.
        _apply_temporal_disambiguation (bool): SAM 3.1 video predictor flag.
    """

    model_id = "sam3_1"
    _checkpoint_path: str | None = None
    _bpe_path: str | None = None
    _apply_temporal_disambiguation: bool = True

    def setup(self, device: str) -> None:
        self.model = SAM3OneModel()
        options: dict[str, Any] = {
            "apply_temporal_disambiguation": self._apply_temporal_disambiguation,
        }
        if self._checkpoint_path is not None:
            options["checkpoint_path"] = self._checkpoint_path
        if self._bpe_path is not None:
            options["bpe_path"] = self._bpe_path
        self.model.load(device, self.model_id, **options)

    # ------------------------------------------------------------------
    # Mode dispatcher
    # ------------------------------------------------------------------

    def decode_request(self, request: dict, **kwargs):
        if "seeds" in request:
            req = SAM3VideoTrackRequest.model_validate(request)
            return {"__mode__": _TRACK_TAG, "request": req}
        if "exemplar_boxes_norm" in request:
            req_visual = SAM3VisualPromptRequest.model_validate(request)
            return {"__mode__": _VISUAL_TAG, "request": req_visual}
        if "bbox" in request:
            req_refine = SAM3RefineRequest.model_validate(request)
            return {"__mode__": _REFINE_TAG, "request": req_refine}
        if "point" in request:
            req_click = SAM3ClickMaskRequest.model_validate(request)
            return {"__mode__": _CLICK_TAG, "request": req_click}
        # Fall through to text→detect proposal mode. The wire is
        # DetectorRequest; SAM 3.1 takes file paths so we keep image_path
        # as the canonical key (rather than going through PIL like the
        # base class does for in-memory pipelines).
        req_det = DetectorRequest.model_validate(request)
        return {"__mode__": _TEXT_TAG, "request": req_det}

    def predict(self, batch, **kwargs):
        # Sessions inside SAM 3.1 are stateful; LitServe batched mode would
        # interleave sessions across workers. We force max_batch_size=1
        # (servers.yaml) so ``batch`` is always a single dict here. If a
        # user overrides batch size we still serialize per item.
        single = not isinstance(batch, list)
        items = [batch] if single else batch

        results: list[Any] = []
        for item in items:
            mode = item.get("__mode__")
            req = item["request"]
            if mode == _REFINE_TAG:
                results.append(self._do_refine(req))
            elif mode == _TRACK_TAG:
                results.append(self._do_track(req))
            elif mode == _TEXT_TAG:
                results.append(self._do_text_detect(req))
            elif mode == _CLICK_TAG:
                results.append(self._do_click_mask(req))
            elif mode == _VISUAL_TAG:
                results.append(self._do_visual_prompt(req))
            else:
                raise RuntimeError(f"unhandled mode: {mode!r}")

        return results[0] if single else results

    def encode_response(self, output, **kwargs):
        if isinstance(
            output,
            (
                SAM3RefineResponse,
                SAM3VideoTrackResponse,
                SAM3ClickMaskResponse,
                SAM3VisualPromptResponse,
                DetectorResponse,
            ),
        ):
            return output.model_dump()
        raise TypeError(f"unexpected output type: {type(output)!r}")

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    def _do_refine(self, req: SAM3RefineRequest) -> SAM3RefineResponse:
        return self.model.refine(
            image_path=req.image_path,
            bbox_norm=req.bbox,
            threshold=req.threshold,
        )

    def _do_track(self, req: SAM3VideoTrackRequest) -> SAM3VideoTrackResponse:
        return self.model.track(
            resource_path=req.resource_path,
            seeds=req.seeds,
            propagation_direction=req.propagation_direction,
            max_frames=req.max_frames,
            return_masks=req.return_masks,
        )

    def _do_text_detect(self, req: DetectorRequest) -> DetectorResponse:
        return self.model.text_detect(
            image_path=req.image_path,
            prompts=req.prompts,
            threshold=req.threshold,
        )

    def _do_click_mask(self, req: SAM3ClickMaskRequest) -> SAM3ClickMaskResponse:
        return self.model.click_mask(
            image_path=req.image_path,
            point_norm=req.point,
            point_label=req.point_label,
            threshold=req.threshold,
        )

    def _do_visual_prompt(
        self, req: SAM3VisualPromptRequest
    ) -> SAM3VisualPromptResponse:
        return self.model.visual_prompt(
            image_path=req.image_path,
            exemplar_boxes_norm=req.exemplar_boxes_norm,
            exemplar_labels=req.exemplar_labels or None,
            threshold=req.threshold,
            max_results=req.max_results,
        )


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

    ap = argparse.ArgumentParser(description="SAM 3.1 LitServe server")
    ap.add_argument("--port", type=int, default=3014)
    ap.add_argument("--gpu", default="cuda:0")
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--bpe-path", default=None)
    ap.add_argument(
        "--no-temporal-disambiguation",
        action="store_true",
        help="Disable SAM 3.1's temporal disambiguation",
    )
    args = ap.parse_args()

    api = SAM3OneApi()
    api._checkpoint_path = args.checkpoint_path
    api._bpe_path = args.bpe_path
    api._apply_temporal_disambiguation = not args.no_temporal_disambiguation

    # LitServer's devices= wants int(s), not "cuda:N" strings. Accept either.
    gpu_str = str(args.gpu)
    device_idx = int(gpu_str.split(":", 1)[1]) if gpu_str.startswith("cuda:") else int(gpu_str)

    # max_batch_size=1: SAM 3.1 sessions are stateful, batching would
    # interleave session ids across workers.
    server = ls.LitServer(
        api,
        accelerator="gpu",
        devices=[device_idx],
        max_batch_size=1,
    )
    server.run(port=args.port)
