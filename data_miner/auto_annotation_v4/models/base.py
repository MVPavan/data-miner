"""Shared base class and helpers for all detector models.

Every model implements load/prepare/infer/postprocess with zero LitServe
dependency. LitAPI wrappers in model_servers/ call these methods.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from PIL import Image

from ..configs.wire import PreparedInput, RawPrediction, DetectorResponse


def clamp01(v: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))


def normalize_box(box_px: list[float], w: int, h: int) -> list[float]:
    """Normalize pixel [x1,y1,x2,y2] to [0,1] range."""
    x1, y1, x2, y2 = box_px
    return [clamp01(x1/w), clamp01(y1/h), clamp01(x2/w), clamp01(y2/h)]


class BaseDetectorModel(ABC):
    """Pure inference model — no HTTP, no LitServe, no checkpoint awareness.

    Subclasses implement model-specific loading, preprocessing, forward
    pass, and postprocessing. All methods are synchronous (GPU-bound).
    """

    @abstractmethod
    def load(self, device: str, model_id: str, **options: Any) -> None:
        """One-time model + processor initialization onto device."""

    @abstractmethod
    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None = None) -> PreparedInput:
        """Preprocess image + prompts for inference."""

    @abstractmethod
    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run model forward pass. Returns raw outputs (tensors, etc.)."""

    @abstractmethod
    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Convert raw outputs to normalized boxes/scores/labels."""
