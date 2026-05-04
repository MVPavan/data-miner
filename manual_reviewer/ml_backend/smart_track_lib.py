"""Tracker-driven cross-frame propagation for static objects.

Pipeline:

  1. User draws a smart Rectangle (the seed) over a static object.
  2. We find sibling tasks in the same LS project by clip-prefix match
     on ``data.image_id`` (e.g. ``Caifu_Center_Fewer_2_f00516`` and
     ``Caifu_Center_Fewer_2_f00645`` share the prefix
     ``Caifu_Center_Fewer_2``).
  3. We assemble those images into a JPEG-folder symlink directory
     (seed at index 0, siblings at 1..N) and call SAM 3.1's
     ``/track`` mode forward from frame 0.
  4. We filter the response — only frames where the propagated bbox
     is close to the seed (motion <= ``motion_thresh``) and confident
     (score >= ``score_thresh``) survive. This is the "static-only"
     filter: a moving object's tracker output drifts spatially or drops
     out, both of which we reject.
  5. Surviving propagations are POSTed as predictions to the matching
     sibling tasks via the LS REST API.

No model imports here — this file orchestrates the SAM 3.1 HTTP client
and the LS REST client. Everything else lives in
:mod:`manual_reviewer.reconcile.sam3_client`,
:mod:`manual_reviewer.ml_backend.ls_rest`,
:mod:`manual_reviewer.ml_backend.ls_payload`.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from data_miner.auto_annotation_v4.configs.wire import (
    SAM3VideoTrackResponse,
    SAM3VideoTrackSeed,
)

from manual_reviewer.ml_backend.ls_payload import norm_box_to_ls_region
from manual_reviewer.ml_backend.ls_rest import LSRestClient
from manual_reviewer.pipeline_io.clip_id import clip_prefix

logger = logging.getLogger(__name__)

__all__ = [
    "PropagateResult",
    "Sibling",
    "TrackerLikeClient",
    "build_jpeg_folder",
    "filter_track_response",
    "find_siblings",
    "propagate_via_tracker",
]


def _append_to_latest_or_post(
    ls_rest: LSRestClient,
    *,
    task_id: int,
    region: dict[str, Any],
    score: float,
    model_version: str,
) -> int | None:
    """Append ``region`` to the sibling task's latest existing prediction
    (PATCH), or POST a fresh prediction when none exist.

    Why merge instead of create-new: LS's labeler shows ONE prediction at
    a time on the canvas (the active one in the predictions dropdown).
    Posting a separate ``sam3_1_track`` prediction means the user opens
    the sibling, sees only the propagated motorcycle, hits Submit, and
    the existing YOLO regions never make it into the annotation. Live
    failure on project 9 task 222 (2026-05-04). Merging into the latest
    prediction puts every region — YOLO + propagated — on a single
    canvas the user can submit in one shot.

    Returns the prediction id (existing PATCHed or new POST), or None
    on failure. The propagated region carries ``meta.source="smart_track"``
    + ``meta.from_image`` so attribution is preserved even though the
    container prediction's ``model_version`` may say "dataset_selection_yolo".
    """
    existing = ls_rest.list_predictions(task_id)
    if existing:
        latest = max(existing, key=lambda p: p.get("created_at") or "")
        new_result = list(latest.get("result") or []) + [region]
        # Take max of existing score and new region's score so the
        # task-list summary reflects the higher confidence.
        try:
            new_score = max(float(latest.get("score") or 0.0), float(score))
        except (TypeError, ValueError):
            new_score = float(score)
        ok = ls_rest.patch_prediction(
            int(latest["id"]), result=new_result, score=new_score,
        )
        return int(latest["id"]) if ok else None
    return ls_rest.post_prediction(
        task_id=task_id,
        result=[region],
        score=score,
        model_version=model_version,
    )


class TrackerLikeClient(Protocol):
    """Protocol for the SAM 3.1 client surface used by the tracker route.

    Mirrors :meth:`manual_reviewer.reconcile.sam3_client.Sam3OneHttpClient.track`.
    """

    def track(
        self,
        *,
        resource_path: str,
        seeds: list[SAM3VideoTrackSeed],
        propagation_direction: str = "both",
        max_frames: int | None = None,
        return_masks: bool = False,
    ) -> SAM3VideoTrackResponse: ...


@dataclass
class Sibling:
    """One LS task that's a candidate target for propagation."""

    task_id: int
    image_id: str
    image_path: str


@dataclass
class PropagateResult:
    """Outcome of one ``propagate_via_tracker`` call.

    ``written_task_ids`` is the list of LS task IDs that received a
    new prediction. ``siblings_total`` is how many sibling frames we
    *considered* — useful for reporting "tracked across N frames"
    even when most got filtered out.
    """

    siblings_total: int = 0
    propagated: int = 0
    rejected_motion: int = 0
    rejected_score: int = 0
    rejected_missing: int = 0
    written_task_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JPEG folder builder
# ---------------------------------------------------------------------------


def build_jpeg_folder(
    seed_image_path: str,
    sibling_image_paths: Iterable[str],
    *,
    parent: Path | str | None = None,
) -> Path:
    """Create a temp directory of symlinks ordered seed→siblings.

    SAM 3.1's video predictor expects an MP4, JPEG folder, or single
    image at ``resource_path``. JPEG folders are loaded in sorted-name
    order, so we name files ``00000.jpg``, ``00001.jpg``, ... with the
    seed at index 0. Caller must :func:`shutil.rmtree` the returned path.
    """
    base_dir = Path(parent) if parent else None
    tmp = Path(tempfile.mkdtemp(prefix="sam3_track_", dir=base_dir))
    paths = [seed_image_path, *sibling_image_paths]
    for idx, src in enumerate(paths):
        dst = tmp / f"{idx:05d}.jpg"
        try:
            os.symlink(src, dst)
        except OSError as exc:
            logger.warning("symlink %s → %s failed: %s", src, dst, exc)
    return tmp


# ---------------------------------------------------------------------------
# Sibling discovery
# ---------------------------------------------------------------------------


def find_siblings(
    ls_rest: LSRestClient,
    *,
    project_id: int,
    seed_image_id: str,
    max_siblings: int = 100,
    assigned_to: str | None = None,
) -> list[Sibling]:
    """Iterate the project's tasks and return clip-mates of the seed.

    Tasks must carry ``data.image_id`` and ``data.image_path`` — both
    are written by ``scripts/build_tasks.py``. Self is excluded by
    ``image_id`` equality, not by task_id (defensive against duplicate
    task imports for the same image).

    When ``assigned_to`` is set, only tasks whose ``data.assigned_to``
    matches are included. This is the per-user containment filter:
    prevents reviewer A's smart_track from writing predictions onto
    reviewer B's frames in a shared LS project. Tasks without an
    assignment field are skipped under this mode (no implicit fallthrough).

    Sorted by ``image_id`` for deterministic ordering — frame index is
    embedded in the suffix so this is also a frame-ordered sort, which
    is what SAM 3.1's video predictor expects.
    """
    target_prefix = clip_prefix(seed_image_id)
    siblings: list[Sibling] = []
    for task in ls_rest.iter_project_tasks(project_id):
        data = task.get("data") or {}
        image_id = data.get("image_id")
        image_path = data.get("image_path")
        if not isinstance(image_id, str) or not isinstance(image_path, str):
            continue
        if image_id == seed_image_id:
            continue
        if clip_prefix(image_id) != target_prefix:
            continue
        if assigned_to is not None and data.get("assigned_to") != assigned_to:
            continue
        task_id = task.get("id")
        if not isinstance(task_id, int):
            continue
        siblings.append(
            Sibling(task_id=task_id, image_id=image_id, image_path=image_path)
        )
    siblings.sort(key=lambda s: s.image_id)
    return siblings[:max_siblings]


# ---------------------------------------------------------------------------
# Response filter
# ---------------------------------------------------------------------------


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def filter_track_response(
    response: SAM3VideoTrackResponse,
    *,
    seed_bbox: list[float],
    score_thresh: float = 0.5,
    motion_thresh: float = 0.05,
) -> tuple[dict[int, tuple[list[float], float]], dict[str, int]]:
    """Keep one bbox per frame: closest-to-seed AND static-enough.

    For each frame > 0 in the response (frame 0 is the seed itself):

      * iterate every detected object,
      * accept if ``score >= score_thresh`` AND
        center-distance from the seed bbox <= ``motion_thresh``,
      * if multiple objects survive, pick the one with highest score,
      * if none survive, the frame is dropped.

    The motion threshold doubles as a temporal-disambiguation filter
    — when SAM 3.1 auto-detects multiple instances on a strong-prior
    class (people), unrelated detections at other locations are
    rejected because their centers are far from the seed.

    Returns ``(per_frame, stats)`` where ``per_frame`` maps frame index
    to ``(bbox_norm, score)`` and ``stats`` counts rejected frames by
    cause (for logging).
    """
    per_frame: dict[int, tuple[list[float], float]] = {}
    stats = {"motion": 0, "score": 0, "missing": 0}

    for frame in response.frames:
        if frame.frame_index <= 0:
            continue  # frame 0 = seed; nothing to propagate
        best: tuple[list[float], float] | None = None
        had_any = False
        had_motion_ok = False
        for obj in frame.objects:
            if obj.bbox is None:
                continue
            had_any = True
            dist = _center_distance(seed_bbox, obj.bbox)
            if dist > motion_thresh:
                continue
            had_motion_ok = True
            if obj.score < score_thresh:
                continue
            if best is None or obj.score > best[1]:
                best = (list(obj.bbox), float(obj.score))
        if best is not None:
            per_frame[frame.frame_index] = best
        elif not had_any:
            stats["missing"] += 1
        elif not had_motion_ok:
            stats["motion"] += 1
        else:
            stats["score"] += 1

    return per_frame, stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def propagate_via_tracker(
    *,
    sam3_client: TrackerLikeClient,
    ls_rest: LSRestClient,
    project_id: int,
    seed_image_path: str,
    seed_image_id: str,
    seed_bbox: list[float],
    seed_label: str,
    current_task_id: int | None = None,
    score_thresh: float = 0.5,
    motion_thresh: float = 0.05,
    max_siblings: int = 100,
    model_version: str = "sam3_1_track",
    assigned_to: str | None = None,
) -> PropagateResult:
    """End-to-end propagation: find siblings, track, filter, write.

    Returns the :class:`PropagateResult` regardless of partial failure.
    A few sibling writes failing doesn't abort the rest. Cleanup of the
    JPEG-folder is unconditional (``finally`` block).

    ``assigned_to`` constrains sibling discovery to a single reviewer's
    slice of the project (see :func:`find_siblings`).
    """
    siblings = find_siblings(
        ls_rest,
        project_id=project_id,
        seed_image_id=seed_image_id,
        max_siblings=max_siblings,
        assigned_to=assigned_to,
    )
    result = PropagateResult(siblings_total=len(siblings))
    if not siblings:
        logger.info(
            "smart_track: no siblings found for seed %s in project %s",
            seed_image_id,
            project_id,
        )
        return result

    folder = build_jpeg_folder(seed_image_path, [s.image_path for s in siblings])
    try:
        seeds = [SAM3VideoTrackSeed(obj_id=1, frame_index=0, bbox=list(seed_bbox))]
        try:
            response = sam3_client.track(
                resource_path=str(folder),
                seeds=seeds,
                propagation_direction="forward",
                return_masks=False,
            )
        except Exception as exc:  # noqa: BLE001 — never crash the LS predict()
            logger.warning("smart_track: SAM 3.1 /track failed: %s", exc)
            return result

        per_frame, stats = filter_track_response(
            response,
            seed_bbox=list(seed_bbox),
            score_thresh=score_thresh,
            motion_thresh=motion_thresh,
        )
        result.rejected_motion = stats["motion"]
        result.rejected_score = stats["score"]
        result.rejected_missing = stats["missing"]

        # frame_index in tracker output = position in the JPEG folder.
        # Frame 0 is the seed, frames 1..N map to siblings[0..N-1].
        for frame_idx, (bbox, score) in sorted(per_frame.items()):
            sib_idx = frame_idx - 1
            if not 0 <= sib_idx < len(siblings):
                continue
            sib = siblings[sib_idx]
            region = norm_box_to_ls_region(
                bbox,
                seed_label,
                score=score,
                extra_meta={
                    "source": "smart_track",
                    "model_version": model_version,
                    "from_image": seed_image_id,
                    "from_task": current_task_id,
                    "outcome": "propagated",
                },
            )
            if region is None:
                continue
            pid = _append_to_latest_or_post(
                ls_rest,
                task_id=sib.task_id,
                region=region,
                score=score,
                model_version=model_version,
            )
            if pid is not None:
                result.propagated += 1
                result.written_task_ids.append(sib.task_id)
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    logger.info(
        "smart_track: seed=%s siblings=%d propagated=%d "
        "rejected(motion=%d, score=%d, missing=%d)",
        seed_image_id,
        result.siblings_total,
        result.propagated,
        result.rejected_motion,
        result.rejected_score,
        result.rejected_missing,
    )
    return result


