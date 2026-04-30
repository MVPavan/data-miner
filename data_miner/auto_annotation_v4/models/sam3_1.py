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
    SAM3VisualPromptResponse,
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


def _xyxy_norm_to_xywh_norm(box_norm: list[float]) -> list[float]:
    """Convert ``[x1, y1, x2, y2]`` in [0,1] to ``[x, y, w, h]`` in [0,1].

    SAM 3.1's ``Sam3VideoPredictor.add_prompt(bounding_boxes=...)`` requires
    normalized xywh — see ``sam3_video_inference.py:891``
    ``assert (boxes_xywh <= 1).all()``. We keep aav4's external contract on
    xyxy (which `BoundingBox` uses) and convert at the seam.
    """
    x1, y1, x2, y2 = box_norm
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _xyxy_norm_to_cxcywh_norm(box_norm: list[float]) -> list[float]:
    """Convert ``[x1, y1, x2, y2]`` in [0,1] to ``[cx, cy, w, h]`` in [0,1].

    SAM 3.1's ``Sam3Processor.add_geometric_prompt`` expects the exemplar
    box in normalized cxcywh — see
    ``sam3/model/sam3_image_processor.py:128`` docstring.
    """
    x, y, bw, bh = _xyxy_norm_to_xywh_norm(box_norm)
    return [x + bw / 2.0, y + bh / 2.0, bw, bh]


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
        # Standalone image-mode SAM 3 model with the SAM-1-style interactive
        # head; used for click→mask. Distinct from ``self._predictor``
        # (video tracker) because the video session API can't issue a
        # single-image click prompt without prior cached propagation.
        self._image_model: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseDetectorModel hooks
    # ------------------------------------------------------------------

    def load(self, device: str, model_id: str, **options: Any) -> None:
        """Construct the SAM 3.1 video predictor + the image-mode model.

        Lazy import keeps the file importable without ``sam3`` on the path.
        ``checkpoint_path`` and ``bpe_path`` can be passed via *options*; both
        default to SAM 3.1's HuggingFace download.

        Two model objects are loaded from the same checkpoint:

        * ``self._predictor`` — ``Sam3VideoPredictor`` for refine / text /
          tracker, where the session API is the right shape.
        * ``self._image_model`` — ``Sam3Image(enable_inst_interactivity=True)``
          for click → mask. The video-predictor session API can't issue a
          point prompt on a fresh single-image session; this model has the
          SAM-1-style interactive head bolted onto the SAM 3 image branch.

        Disable the second model by passing ``enable_image_predictor=False``.
        """
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        kwargs: dict[str, Any] = {}
        for key in ("checkpoint_path", "bpe_path", "apply_temporal_disambiguation"):
            if key in options:
                kwargs[key] = options[key]

        logger.info("Loading SAM 3.1 (model_id=%s, device=%s, options=%s)", model_id, device, kwargs)
        self._predictor = Sam3VideoPredictor(**kwargs)
        self._device = device

        if options.get("enable_image_predictor", True):
            from sam3.model_builder import build_sam3_image_model

            img_kwargs: dict[str, Any] = {"enable_inst_interactivity": True}
            for key in ("checkpoint_path", "bpe_path"):
                if key in options:
                    img_kwargs[key] = options[key]
            logger.info("Loading SAM 3.1 image model for click→mask (options=%s)", img_kwargs)
            self._image_model = build_sam3_image_model(**img_kwargs)

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
                # SAM 3.1 wants normalized xywh, not pixel xyxy.
                bbox_xywh_norm = np.asarray(
                    [_xyxy_norm_to_xywh_norm(bbox_norm)], dtype=np.float32
                )
                self._predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        bounding_boxes=bbox_xywh_norm,
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
        if self._image_model is None:
            raise RuntimeError(
                "click_mask requires the SAM 3 image model. Reload SAM3OneModel "
                "without ``enable_image_predictor=False``."
            )

        # Lazy import — keeps the module importable on machines without sam3.
        from sam3.model.sam3_image_processor import Sam3Processor
        import torch

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            w, h = image.size

        point_px = np.asarray(
            [[point_norm[0] * w, point_norm[1] * h]], dtype=np.float32
        )
        labels = np.asarray([int(point_label)], dtype=np.int32)

        with self._lock:
            processor = Sam3Processor(self._image_model, device=self._device)
            with torch.inference_mode():
                inference_state = processor.set_image(image)
                masks, scores, _ = self._image_model.predict_inst(
                    inference_state,
                    point_coords=point_px,
                    point_labels=labels,
                    box=None,
                    multimask_output=True,
                )

        if isinstance(masks, torch.Tensor):
            masks_np = masks.float().detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        if isinstance(scores, torch.Tensor):
            scores_np = scores.float().detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)

        if masks_np.size == 0:
            return SAM3ClickMaskResponse(bbox=None, mask_rle=None, score=0.0)
        # ``predict_inst`` returns CxHxW with C=3 when multimask_output=True.
        # Pick the highest-score mask.
        best_idx = int(np.argmax(scores_np))
        best_mask = masks_np[best_idx] > 0.0
        best_score = float(scores_np[best_idx])
        if best_score < threshold:
            return SAM3ClickMaskResponse(bbox=None, mask_rle=None, score=best_score)
        bbox_px = _mask_to_bbox(best_mask)
        if bbox_px is None:
            return SAM3ClickMaskResponse(bbox=None, mask_rle=None, score=best_score)
        return SAM3ClickMaskResponse(
            bbox=_norm_xyxy(bbox_px, w, h),
            mask_rle=None,
            score=best_score,
        )

    def visual_prompt(
        self,
        image_path: str,
        exemplar_boxes_norm: list[list[float]],
        *,
        exemplar_labels: list[int] | None = None,
        threshold: float = 0.4,
        max_results: int = 50,
    ) -> SAM3VisualPromptResponse:
        """Box-prompt grounding — feed exemplar(s) through the SAM 3.1
        grounding head and return all matching instances in the same image.

        Backed by ``Sam3Processor.add_geometric_prompt``. Exemplars are
        accepted as normalized [x1, y1, x2, y2] (aav4 convention); the
        processor wants normalized cxcywh, conversion happens at the seam.

        ``exemplar_labels`` follow SAM convention: 1 = positive, 0 =
        negative. Defaults to all positives. Multiple exemplars are each
        appended to ``state["geometric_prompt"]`` and trigger a forward
        pass — the LAST pass uses the accumulated N-box prompt so the
        result is the correct N-exemplar grounding output, but at the
        cost of N forward passes (the upstream processor exposes no
        public API to set N boxes before a single forward). v1 only
        sends one exemplar so this asymmetry is dormant.

        Output is filtered by ``threshold`` (the processor's
        ``confidence_threshold``) and capped at ``max_results``.
        """
        if not exemplar_boxes_norm:
            return SAM3VisualPromptResponse(boxes_norm=[], scores=[])
        if self._image_model is None:
            raise RuntimeError(
                "visual_prompt requires the SAM 3 image model. Reload "
                "SAM3OneModel without enable_image_predictor=False."
            )

        from sam3.model.sam3_image_processor import Sam3Processor
        import torch

        labels = list(exemplar_labels) if exemplar_labels else [1] * len(
            exemplar_boxes_norm
        )
        if len(labels) != len(exemplar_boxes_norm):
            raise ValueError(
                "exemplar_labels length must match exemplar_boxes_norm"
            )

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            w, h = image.size

        with self._lock:
            processor = Sam3Processor(
                self._image_model,
                device=self._device,
                confidence_threshold=float(threshold),
            )
            with torch.inference_mode():
                state = processor.set_image(image)
                for box_norm, label_int in zip(exemplar_boxes_norm, labels):
                    cxcywh = _xyxy_norm_to_cxcywh_norm(box_norm)
                    state = processor.add_geometric_prompt(
                        box=cxcywh,
                        label=bool(label_int),
                        state=state,
                    )
            boxes_t = state.get("boxes")
            scores_t = state.get("scores")

        if boxes_t is None or scores_t is None:
            return SAM3VisualPromptResponse(boxes_norm=[], scores=[])
        # SAM 3.1 returns BFloat16 tensors on GPU; cast to float32 before
        # .numpy() since NumPy has no BF16 dtype.
        if isinstance(boxes_t, torch.Tensor):
            boxes_px = boxes_t.detach().to(torch.float32).cpu().numpy()
        else:
            boxes_px = np.asarray(boxes_t)
        if isinstance(scores_t, torch.Tensor):
            scores_np = scores_t.detach().to(torch.float32).cpu().numpy()
        else:
            scores_np = np.asarray(scores_t)
        if boxes_px.size == 0:
            return SAM3VisualPromptResponse(boxes_norm=[], scores=[])

        # Sort highest-confidence first, cap at max_results.
        order = np.argsort(-scores_np)
        boxes_norm: list[list[float]] = []
        scores_out: list[float] = []
        for idx in order[:max_results]:
            box_px = boxes_px[idx]
            if len(box_px) != 4:
                continue
            x1 = clamp01(float(box_px[0]) / w)
            y1 = clamp01(float(box_px[1]) / h)
            x2 = clamp01(float(box_px[2]) / w)
            y2 = clamp01(float(box_px[3]) / h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_norm.append([x1, y1, x2, y2])
            scores_out.append(float(scores_np[idx]))

        return SAM3VisualPromptResponse(boxes_norm=boxes_norm, scores=scores_out)

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

                for obj in self._frame_objects(outputs.get(0) or {}):
                    score = float(obj.get("score") or 0.0)
                    if threshold is not None and score < threshold:
                        continue
                    bbox_px = self._object_bbox_px(obj, w, h)
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
            for obj in self._frame_objects(frame_data):
                bbox_px = self._object_bbox_px(obj, w, h)
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
    def _frame_objects(frame_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize SAM 3.1's parallel-array frame output to objects.

        Upstream returns ``out_obj_ids`` / ``out_probs`` / ``out_boxes_xywh``
        / ``out_binary_masks`` as parallel arrays of length N. ``out_boxes_xywh``
        is in normalized [0,1] xywh; ``out_binary_masks`` is ``(N, H, W)``
        pixel-space binary masks.
        """
        # Upstream emits these as numpy arrays; ``or []`` would raise on
        # ndarray ambiguity. Probe for None / empty explicitly.
        raw_ids = frame_output.get("out_obj_ids")
        obj_ids = list(raw_ids) if raw_ids is not None else []
        raw_probs = frame_output.get("out_probs")
        probs = np.asarray(raw_probs) if raw_probs is not None else np.asarray([])
        raw_boxes = frame_output.get("out_boxes_xywh")
        boxes = np.asarray(raw_boxes) if raw_boxes is not None else np.zeros((0, 4))
        masks = frame_output.get("out_binary_masks")
        out: list[dict[str, Any]] = []
        for i, oid in enumerate(obj_ids):
            entry: dict[str, Any] = {
                "obj_id": int(oid),
                "score": float(probs[i]) if i < len(probs) else 0.0,
            }
            if i < len(boxes):
                entry["bbox_xywh_norm"] = boxes[i].tolist()
            if masks is not None and i < len(masks):
                entry["mask"] = masks[i]
            out.append(entry)
        return out

    @staticmethod
    def _first_object(frame_output: dict[str, Any]) -> dict[str, Any] | None:
        objs = SAM3OneModel._frame_objects(frame_output)
        return objs[0] if objs else None

    @staticmethod
    def _object_bbox_px(obj: dict[str, Any], w: int, h: int) -> tuple[float, float, float, float] | None:
        """Return tight pixel bbox for the predictor object.

        Prefers the upstream-emitted ``bbox_xywh_norm`` (cheap, exact); falls
        back to recomputing from the mask when the bbox is absent (older builds
        or empty boxes).
        """
        xywh = obj.get("bbox_xywh_norm")
        if xywh is not None and len(xywh) == 4:
            x, y, bw, bh = xywh
            if bw > 0 and bh > 0:
                return (
                    float(x) * w,
                    float(y) * h,
                    float(x + bw) * w,
                    float(y + bh) * h,
                )
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
