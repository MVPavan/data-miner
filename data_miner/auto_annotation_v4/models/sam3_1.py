"""Pure inference wrapper for SAM 3.1 — image-mode + video-tracker modes.

This is the canonical SAM 3.1 model used by ``manual_reviewer`` (reconciler,
ML backend, future tracker workflows). aav4's auto pipeline keeps using the
existing ``sam3_dart`` model — this file deliberately does not touch it.

Three capabilities exposed:

  * **refine** (image mode) — bbox → refined bbox + score. Drop-in for the
    SAM3RefineRequest/Response wire that ``sam3_dart`` already speaks, so
    reconciler clients work with either server.
  * **text_detect** (image mode) — text prompt → list of boxes + scores.
    Used by the manual_reviewer ML backend's text→detect path (Phase 2).
  * **track** (video mode) — seeds (bbox / text / points) on specific frames
    of a video / JPEG-folder, propagated forward+backward. Used by the
    tracker-based reconciler upgrade.

The video predictor is the single inference engine — SAM 3.1 supports
"single image as one-frame video", so we never load the standalone image
predictor. Sessions are owned by the model, not the caller — every public
method does ``start_session`` → seed → propagate → ``close_session`` so the
RPC layer stays stateless.

The ``sam3`` package is imported lazily inside :meth:`load` so the wrapper
is importable on machines without SAM 3.1 installed (tests, CI, dev boxes).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from ..configs.wire import (
    DetectorResponse,
    PreparedInput,
    RawPrediction,
    SAM3ClickMaskResponse,
    SAM3RefineResponse,
    SAM3VideoTrackFrameOutput,
    SAM3VideoTrackObjectOutput,
    SAM3VideoTrackResponse,
    SAM3VideoTrackSeed,
)
from .base import BaseDetectorModel, clamp01

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal hand-off models
# ---------------------------------------------------------------------------


class _ImagePrepared(BaseModel):
    """Carrier for proposal-mode (text→detect) prepare→infer handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    image_path: str
    prompts: list[str]
    threshold: float | None
    image_size: tuple[int, int]


class _ImageRaw(BaseModel):
    """Carrier for proposal-mode infer→postprocess handoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    boxes: list[list[float]]
    scores: list[float]
    labels: list[str]
    image_size: tuple[int, int]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight bbox (x1, y1, x2, y2) in pixel coords for a 2-D bool mask.

    Returns ``None`` for an empty mask.
    """
    if mask is None or mask.size == 0:
        return None
    if mask.ndim != 2:
        # ``Sam3VideoPredictor`` masks are sometimes (1, H, W) with a leading
        # batch axis — squeeze it so callers don't have to know.
        mask = np.asarray(mask).squeeze()
        if mask.ndim != 2:
            return None
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


def _norm_xyxy(box_px: tuple[float, float, float, float], w: int, h: int) -> list[float]:
    return [
        clamp01(box_px[0] / w),
        clamp01(box_px[1] / h),
        clamp01(box_px[2] / w),
        clamp01(box_px[3] / h),
    ]


def _denorm_xyxy(box_norm: list[float], w: int, h: int) -> list[float]:
    return [box_norm[0] * w, box_norm[1] * h, box_norm[2] * w, box_norm[3] * h]


def _denorm_points(points: list[list[float]], w: int, h: int) -> np.ndarray:
    return np.asarray(
        [[p[0] * w, p[1] * h] for p in points],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class SAM3OneModel(BaseDetectorModel):
    """SAM 3.1 wrapper covering image-mode refine + text→detect + video track.

    Single :class:`Sam3VideoPredictor` instance handles all three modes; the
    predictor's session API is the single inference seam. Sessions are
    short-lived (one per public call), guarded by an instance lock so
    concurrent LitServe workers don't race on shared model state.
    """

    def __init__(self) -> None:
        self._predictor: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseDetectorModel hooks
    # ------------------------------------------------------------------

    def load(self, device: str, model_id: str, **options: Any) -> None:
        """Construct the SAM 3.1 video predictor.

        Lazy import keeps the file importable without ``sam3`` on the path.
        ``checkpoint_path`` and ``bpe_path`` can be passed via *options*; both
        default to SAM 3.1's HuggingFace download.
        """
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        kwargs: dict[str, Any] = {}
        for key in ("checkpoint_path", "bpe_path", "apply_temporal_disambiguation"):
            if key in options:
                kwargs[key] = options[key]

        logger.info("Loading SAM 3.1 (model_id=%s, device=%s, options=%s)", model_id, device, kwargs)
        self._predictor = Sam3VideoPredictor(**kwargs)
        self._device = device

    def prepare(
        self,
        image: Image.Image,
        prompts: list[str],
        threshold: float | None = None,
    ) -> PreparedInput:
        """Image-mode text→detect preprocessing.

        SAM 3.1 takes file paths rather than in-memory images, so the image
        is buffered to a tempfile and the path is carried through.
        """
        if image is None:
            raise ValueError("prepare() requires a PIL.Image")
        # The base class hands us a PIL.Image; SAM 3.1's session API expects
        # a path. Caller can also pre-stage an image_path in extras to skip
        # the temp file write.
        return PreparedInput(
            image=image,
            processor_inputs=None,
            image_size=image.size,
            prompts=prompts,
            threshold=threshold,
            extras={},
        )

    def infer(self, prepared: PreparedInput) -> RawPrediction:
        raise NotImplementedError(
            "SAM3OneModel.infer is not used directly; call text_detect()"
            " from the LitAPI server (it owns the image_path)"
        )

    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        # The server packages text_detect() output through this hook so
        # encode_response stays uniform across detectors.
        outputs = raw.outputs
        if not isinstance(outputs, _ImageRaw):
            raise TypeError(f"unexpected raw payload: {type(outputs)!r}")
        return DetectorResponse(
            boxes=outputs.boxes,
            scores=outputs.scores,
            labels=outputs.labels,
        )

    # ------------------------------------------------------------------
    # Public modes
    # ------------------------------------------------------------------

    def refine(
        self,
        image_path: str,
        bbox_norm: list[float],
        *,
        threshold: float = 0.5,
    ) -> SAM3RefineResponse:
        """Single-image bbox refinement.

        Starts a one-frame "video" session at ``image_path``, seeds it with
        the bbox, propagates (which on a 1-frame video is a no-op beyond
        running the encoder), and reads back the refined mask's tight bbox
        plus the predictor's score. Drop-in for SAM3-DART's /refine wire.
        """
        with self._lock:
            session_id = self._start(image_path)
            try:
                w, h = self._image_size(image_path)
                bbox_px = np.asarray([_denorm_xyxy(bbox_norm, w, h)], dtype=np.float32)
                self._predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        bounding_boxes=bbox_px,
                        bounding_box_labels=np.asarray([1], dtype=np.int32),
                        obj_id=1,
                    )
                )
                outputs = self._collect_propagation(session_id)
            finally:
                self._close(session_id)

        frame0 = outputs.get(0) or {}
        obj = self._first_object(frame0)
        if obj is None:
            return SAM3RefineResponse(box=None, score=0.0)
        mask = obj.get("mask")
        score = float(obj.get("score") or 0.0)
        if mask is None:
            return SAM3RefineResponse(box=None, score=score)
        bbox_px_refined = _mask_to_bbox(np.asarray(mask))
        if bbox_px_refined is None:
            return SAM3RefineResponse(box=None, score=score)
        if score < threshold:
            return SAM3RefineResponse(box=None, score=score)
        return SAM3RefineResponse(box=_norm_xyxy(bbox_px_refined, w, h), score=score)

    def click_mask(
        self,
        image_path: str,
        point_norm: list[float],
        *,
        point_label: int = 1,
        threshold: float = 0.5,
        return_mask_rle: bool = False,
    ) -> SAM3ClickMaskResponse:
        """Single-image point→mask inference for the LS ML backend.

        Seeds a one-frame session with one point prompt, propagates, and
        returns the resulting mask's tight bbox plus optionally the full
        RLE. Output below ``threshold`` is suppressed (returns ``bbox=None``).
        """
        if len(point_norm) != 2:
            raise ValueError("point_norm must be [x, y]")
        with self._lock:
            session_id = self._start(image_path)
            try:
                w, h = self._image_size(image_path)
                point_px = _denorm_points([point_norm], w, h)
                self._predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        points=point_px,
                        point_labels=np.asarray([int(point_label)], dtype=np.int32),
                        obj_id=1,
                    )
                )
                outputs = self._collect_propagation(session_id)
            finally:
                self._close(session_id)

        frame0 = outputs.get(0) or {}
        obj = self._first_object(frame0)
        if obj is None:
            return SAM3ClickMaskResponse(bbox=None, mask_rle=None, score=0.0)
        score = float(obj.get("score") or 0.0)
        if score < threshold:
            return SAM3ClickMaskResponse(bbox=None, mask_rle=None, score=score)
        bbox_px = self._object_bbox_px(obj)
        bbox_norm = _norm_xyxy(bbox_px, w, h) if bbox_px else None
        mask_rle = obj.get("mask_rle") if return_mask_rle else None
        return SAM3ClickMaskResponse(bbox=bbox_norm, mask_rle=mask_rle, score=score)

    def text_detect(
        self,
        image_path: str,
        prompts: list[str],
        *,
        threshold: float | None = None,
    ) -> DetectorResponse:
        """Image-mode text→detect.

        For each prompt, runs an independent session+add_prompt+propagate,
        accumulates the per-prompt boxes/scores, and labels them with the
        original prompt string. Multi-prompt batching across the model is
        not exposed by the upstream session API in v3.1; we serialize.
        """
        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []
        w, h = self._image_size(image_path)

        with self._lock:
            for prompt in prompts:
                session_id = self._start(image_path)
                try:
                    self._predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=prompt,
                        )
                    )
                    outputs = self._collect_propagation(session_id)
                finally:
                    self._close(session_id)

                for obj in (outputs.get(0) or {}).get("objects", []):
                    score = float(obj.get("score") or 0.0)
                    if threshold is not None and score < threshold:
                        continue
                    bbox_px = self._object_bbox_px(obj)
                    if bbox_px is None:
                        continue
                    boxes.append(_norm_xyxy(bbox_px, w, h))
                    scores.append(score)
                    labels.append(prompt)

        return DetectorResponse(boxes=boxes, scores=scores, labels=labels)

    def track(
        self,
        resource_path: str,
        seeds: list[SAM3VideoTrackSeed],
        *,
        propagation_direction: str = "both",
        max_frames: int | None = None,
        return_masks: bool = False,
    ) -> SAM3VideoTrackResponse:
        """Video-mode multi-object tracker.

        ``resource_path`` is a video file, JPEG folder, or single image.
        ``seeds`` is one or more prompts each tagged with an ``obj_id``;
        SAM 3.1 propagates them across all frames in the resource (or the
        first ``max_frames`` if set). The full per-frame output is collected
        and returned in a single response — no streaming on the client.
        """
        w, h = self._image_size_or_default(resource_path)

        with self._lock:
            session_id = self._start(resource_path)
            try:
                for seed in seeds:
                    self._predictor.handle_request(
                        request=self._seed_to_request(session_id, seed, w, h)
                    )
                outputs = self._collect_propagation(
                    session_id,
                    propagation_direction=propagation_direction,
                    max_frame_num_to_track=max_frames,
                )
            finally:
                self._close(session_id)

        frames: list[SAM3VideoTrackFrameOutput] = []
        for frame_idx in sorted(outputs):
            frame_data = outputs[frame_idx] or {}
            objs: list[SAM3VideoTrackObjectOutput] = []
            for obj in frame_data.get("objects", []):
                bbox_px = self._object_bbox_px(obj)
                bbox_norm = _norm_xyxy(bbox_px, w, h) if bbox_px else None
                mask_rle = obj.get("mask_rle") if return_masks else None
                objs.append(
                    SAM3VideoTrackObjectOutput(
                        obj_id=int(obj.get("obj_id") or 0),
                        bbox=bbox_norm,
                        mask_rle=mask_rle,
                        score=float(obj.get("score") or 0.0),
                    )
                )
            frames.append(SAM3VideoTrackFrameOutput(frame_index=frame_idx, objects=objs))

        return SAM3VideoTrackResponse(frames=frames)

    # ------------------------------------------------------------------
    # Predictor session helpers
    # ------------------------------------------------------------------

    def _require_predictor(self) -> Any:
        if self._predictor is None:
            raise RuntimeError("SAM3OneModel.load() must be called before inference")
        return self._predictor

    def _start(self, resource_path: str) -> str:
        resp = self._require_predictor().handle_request(
            request=dict(type="start_session", resource_path=resource_path)
        )
        sid = resp.get("session_id")
        if not sid:
            raise RuntimeError("SAM 3.1 start_session returned no session_id")
        return sid

    def _close(self, session_id: str) -> None:
        try:
            self._require_predictor().handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        except Exception as exc:  # noqa: BLE001 — cleanup best-effort
            logger.warning("close_session failed for %s: %s", session_id, exc)

    def _collect_propagation(
        self,
        session_id: str,
        *,
        propagation_direction: str = "both",
        max_frame_num_to_track: int | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Drain ``propagate_in_video`` into ``{frame_index: outputs}``."""
        request = dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=propagation_direction,
        )
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = max_frame_num_to_track

        collected: dict[int, dict[str, Any]] = {}
        for response in self._require_predictor().handle_stream_request(request=request):
            idx = int(response.get("frame_index", -1))
            if idx < 0:
                continue
            collected[idx] = response.get("outputs") or {}
        return collected

    def _seed_to_request(
        self,
        session_id: str,
        seed: SAM3VideoTrackSeed,
        w: int,
        h: int,
    ) -> dict[str, Any]:
        request: dict[str, Any] = dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=seed.frame_index,
            obj_id=seed.obj_id,
        )
        if seed.bbox is not None:
            request["bounding_boxes"] = np.asarray(
                [_denorm_xyxy(seed.bbox, w, h)], dtype=np.float32
            )
            request["bounding_box_labels"] = np.asarray([1], dtype=np.int32)
        if seed.text is not None:
            request["text"] = seed.text
        if seed.points is not None:
            request["points"] = _denorm_points(seed.points, w, h)
            labels = seed.point_labels or [1] * len(seed.points)
            request["point_labels"] = np.asarray(labels, dtype=np.int32)
        return request

    @staticmethod
    def _first_object(frame_output: dict[str, Any]) -> dict[str, Any] | None:
        objs = frame_output.get("objects")
        if not objs:
            return None
        return objs[0]

    @staticmethod
    def _object_bbox_px(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
        # Predictor outputs vary by build: sometimes ``bbox_px``, sometimes
        # ``mask`` only. Try the bbox first, fall back to mask.
        if "bbox_px" in obj and obj["bbox_px"] is not None:
            x1, y1, x2, y2 = obj["bbox_px"]
            return float(x1), float(y1), float(x2), float(y2)
        mask = obj.get("mask")
        if mask is not None:
            return _mask_to_bbox(np.asarray(mask))
        return None

    @staticmethod
    def _image_size(image_path: str) -> tuple[int, int]:
        with Image.open(image_path) as im:
            return im.size

    def _image_size_or_default(self, resource_path: str) -> tuple[int, int]:
        """``resource_path`` can be a folder/MP4 — fall back to the first
        frame's size when ``Image.open`` rejects the path."""
        try:
            return self._image_size(resource_path)
        except Exception:
            from pathlib import Path

            p = Path(resource_path)
            if p.is_dir():
                jpegs = sorted(p.glob("*.jpg"))
                if jpegs:
                    return self._image_size(str(jpegs[0]))
            # Caller asked us to track; we'll use 1.0-normalized coords as
            # a degenerate fallback so the response shape is still valid.
            return (1, 1)


__all__ = ["SAM3OneModel"]
