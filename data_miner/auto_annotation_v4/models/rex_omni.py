"""Pure inference wrapper for Rex-Omni — autoregressive MLLM detector.

Rex-Omni (IDEA-Research) is a 3B Qwen2.5-VL-based detector that emits
bounding boxes as autoregressive token sequences. Used by the aav4 detect
stage when a user opts in via ``runtime.detect_models: [rex_omni]``.
manual_reviewer does **not** use this model — it standardises on SAM 3.1.

Wire contract is the standard ``DetectorRequest`` / ``DetectorResponse``.
The model is class-conditioned: the detect stage sends the full prompt list
in one request, Rex-Omni runs the full vocabulary in a single generate, and
we return the union of detections labeled by the prompt that produced them.

Latency note: 1-3 s per image at batch=1. Decoding is sequence-length
bound, not batch-size bound, so multi-image batching does not help. The
LitServe wrapper sets ``max_batch_size=1`` accordingly.

The ``rex_omni`` package is imported lazily so this file remains importable
on machines without the model installed (CI / dev boxes / tests).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict

from ..configs.wire import DetectorResponse, PreparedInput, RawPrediction
from .base import BaseDetectorModel, clamp01

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_ID = "IDEA-Research/Rex-Omni"
_DEFAULT_BACKEND = "transformers"
_DEFAULT_TASK = "detection"


# ---------------------------------------------------------------------------
# Internal hand-off models
# ---------------------------------------------------------------------------


class _RexPrepared(BaseModel):
    """Carrier for prepare → infer handoff (PIL image + prompts)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    image: Any  # PIL.Image
    prompts: list[str]
    image_size: tuple[int, int]


class _RexRaw(BaseModel):
    """Carrier for infer → postprocess handoff.

    ``predictions`` is the upstream ``extracted_predictions`` mapping —
    ``{category_label: [{"type": "box", "coords": [x0, y0, x1, y1]}, ...]}``.
    Coordinates are in pixel (absolute) space.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    predictions: dict[str, list[dict[str, Any]]]
    image_size: tuple[int, int]
    prompts: list[str]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _pixel_xyxy_to_norm(coords: list[float], w: int, h: int) -> list[float] | None:
    """Convert Rex-Omni's pixel [x0,y0,x1,y1] to normalized [0,1] coords.

    Rex-Omni emits absolute pixel coordinates; we normalise into the same
    [0, 1] range used by every other detector in aav4. Returns ``None`` for
    degenerate / unparseable boxes so the caller can drop them silently.
    """
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(coords[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    nx1 = clamp01(x1 / w)
    ny1 = clamp01(y1 / h)
    nx2 = clamp01(x2 / w)
    ny2 = clamp01(y2 / h)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RexOmniModel(BaseDetectorModel):
    """Rex-Omni wrapper — single per-image inference, all prompts in one call.

    Attributes (populated by :meth:`load`):
        wrapper: ``rex_omni.RexOmniWrapper`` instance.
        device: torch device string.
        backend: "transformers" or "vllm".
        max_new_tokens: forwarded to the wrapper's inference config.

    The Rex-Omni autoregressive decoder is stateful, so concurrent calls
    are serialized via :attr:`_lock`.
    """

    def __init__(self) -> None:
        self.wrapper: Any = None
        self.device: str = "cpu"
        self.backend: str = _DEFAULT_BACKEND
        self.max_new_tokens: int | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseDetectorModel hooks
    # ------------------------------------------------------------------

    def load(
        self,
        device: str,
        model_id: str = _DEFAULT_MODEL_ID,
        **options: Any,
    ) -> None:
        """Lazy-import ``rex_omni.RexOmniWrapper`` and instantiate.

        Args:
            device: Torch device string (``"cuda:0"``, ``"cpu"`` …). Forwarded
                to the wrapper unchanged; Rex-Omni picks its own dtype.
            model_id: HuggingFace identifier (default: ``IDEA-Research/Rex-Omni``).
            **options:
                backend (str): ``"transformers"`` or ``"vllm"``. Default
                    ``"transformers"``.
                max_new_tokens (int): override the wrapper's default decode
                    length. Increase only if you observe truncated boxes.
        """
        from rex_omni import RexOmniWrapper  # type: ignore[import-not-found]

        self.device = device
        self.backend = options.get("backend") or _DEFAULT_BACKEND
        self.max_new_tokens = options.get("max_new_tokens")

        logger.info(
            "Loading Rex-Omni (model_id=%s, device=%s, backend=%s)",
            model_id,
            device,
            self.backend,
        )
        # Newer wrapper versions accept ``device``; older ones derive it
        # from torch internals. Pass it via kwargs so an older wrapper just
        # picks up the right bind via env without raising.
        wrapper_kwargs: dict[str, Any] = {
            "model_path": model_id,
            "backend": self.backend,
        }
        try:
            self.wrapper = RexOmniWrapper(**wrapper_kwargs, device=device)
        except TypeError:
            self.wrapper = RexOmniWrapper(**wrapper_kwargs)

    def prepare(
        self,
        image: Image.Image,
        prompts: list[str],
        threshold: float | None = None,
    ) -> PreparedInput:
        """Pass-through preprocessing.

        Rex-Omni's wrapper does its own image preprocessing internally, so
        the prepared payload just carries the PIL image and prompt list.
        ``threshold`` is unused — Rex-Omni doesn't emit a per-box score.
        """
        if image is None:
            raise ValueError("prepare() requires a PIL.Image")
        if not prompts:
            return PreparedInput(
                image=image,
                processor_inputs=None,
                image_size=image.size,
                prompts=[],
                threshold=threshold,
                extras={},
            )
        return PreparedInput(
            image=image,
            processor_inputs=None,
            image_size=image.size,
            prompts=list(prompts),
            threshold=threshold,
            extras={},
        )

    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run one Rex-Omni generate call across the full prompt list.

        The wrapper accepts a list of category strings as ``categories`` and
        runs a single autoregressive pass — much more efficient than looping
        per prompt. Result is the wrapper's ``extracted_predictions`` dict.
        """
        if self.wrapper is None:
            raise RuntimeError("RexOmniModel.load() must be called before infer()")
        if not prepared.prompts:
            return RawPrediction(
                outputs=_RexRaw(
                    predictions={},
                    image_size=prepared.image_size,
                    prompts=[],
                ),
                inputs=None,
                image_size=prepared.image_size,
                prompts=[],
                threshold=prepared.threshold,
                extras={},
            )

        kwargs: dict[str, Any] = {
            "images": prepared.image,
            "task": _DEFAULT_TASK,
            "categories": list(prepared.prompts),
        }
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens

        with self._lock:
            results = self.wrapper.inference(**kwargs)

        # Rex-Omni returns list-of-results when given a single image.
        if isinstance(results, list) and results:
            first = results[0]
        elif isinstance(results, dict):
            first = results
        else:
            first = {}
        predictions_raw = (
            first.get("extracted_predictions") if isinstance(first, dict) else None
        ) or {}
        if not isinstance(predictions_raw, dict):
            predictions_raw = {}

        return RawPrediction(
            outputs=_RexRaw(
                predictions=predictions_raw,
                image_size=prepared.image_size,
                prompts=list(prepared.prompts),
            ),
            inputs=None,
            image_size=prepared.image_size,
            prompts=list(prepared.prompts),
            threshold=prepared.threshold,
            extras={},
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Flatten Rex-Omni's per-class predictions into wire DetectorResponse.

        Rex-Omni does not emit per-box confidences; every detection is
        scored at 1.0 (matches the convention used by FalconModel).
        """
        outputs = raw.outputs
        if not isinstance(outputs, _RexRaw):
            raise TypeError(f"unexpected raw payload: {type(outputs)!r}")

        w, h = outputs.image_size
        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []

        for category, items in outputs.predictions.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") and item["type"] != "box":
                    # Rex-Omni can emit polygons/keypoints in other tasks;
                    # detection task should only return boxes.
                    continue
                coords = item.get("coords")
                bbox_norm = _pixel_xyxy_to_norm(coords, w, h)
                if bbox_norm is None:
                    continue
                boxes.append(bbox_norm)
                scores.append(1.0)
                labels.append(str(category))

        return DetectorResponse(boxes=boxes, scores=scores, labels=labels)


__all__ = ["RexOmniModel"]
