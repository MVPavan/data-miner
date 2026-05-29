"""Run SAM 3.1 video tracker over a folder of MP4s with periodic re-seeding.

Per-video pipeline:

  1. ffprobe → fps, n_frames.
  2. Extract every frame to a temp JPEG folder (00000.jpg ...).
  3. Walk windows of --seed-stride-sec seconds. For each window:
       a. Build a symlink JPEG folder for the window's frames.
       b. SAM3-DART (port 3013) on frame 0 with ALL class prompts →
          per-class NMS-deduplicated bboxes [(class, bbox_xyxy, score)].
       c. For each detected bbox, send /track with one bbox seed at
          frame 0 → SAM 3.1 propagates that exact instance through the
          window. Each detection = its own obj_id.
  4. Stitch per-class detections across windows via IoU at the seam frame
     so a continuously visible object becomes one logical track_id.
  5. Write tracks.json + summary.csv per video.

Why DART for seeding:
DART is the codebase's high-confidence detector (auto_accept threshold 0.85,
sole `allowed_source_models` for the canonical class registry). It has
built-in per-class NMS (iou=0.7) and presence-head early exit — strict upgrade
over SAM 3.1 text_detect for finding "where are the objects in this frame".

Why bbox seeds (not text seeds):
DART already gives us per-instance bboxes — passing them as bbox seeds
preserves multi-instance counts and gives the tracker a precise anchor.
Text seeds re-run grounding inside SAM 3.1, losing DART's NMS quality.

Usage:
    python -m data_miner.auto_annotation_v4.scripts.track_videos_sam3 \\
        --videos-dir /media/data_2/datasets/datasets_pavan/FLPJ_TEST \\
        --config output/auto_annotation_v4/datatang_val_detect/config.yaml \\
        --out-dir output/sam3_track \\
        --sam3-url http://localhost:3015/predict \\
        --dart-url http://localhost:3013/predict
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_miner.auto_annotation_v4.configs.wire import (
    DetectorRequest,
    DetectorResponse,
    SAM3VideoTrackResponse,
    SAM3VideoTrackSeed,
)
from manual_reviewer.reconcile.sam3_client import Sam3OneHttpClient

import requests

logger = logging.getLogger("track_videos_sam3")

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassEntry:
    name: str       # canonical class name (matches classes.txt)
    yolo_id: int    # class_registry[name].id
    prompt: str     # first entry in class_registry[name].prompts


def load_classes(config_path: Path) -> list[ClassEntry]:
    """Read class_registry from a job's config.yaml (JSON-shaped)."""
    with config_path.open() as f:
        cfg = json.load(f)
    registry = cfg.get("class_registry") or {}
    out: list[ClassEntry] = []
    for name, body in registry.items():
        prompts = body.get("prompts") or [name]
        out.append(ClassEntry(name=name, yolo_id=int(body["id"]), prompt=prompts[0]))
    out.sort(key=lambda c: c.yolo_id)
    return out


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> tuple[float, int, float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,duration",
        "-of", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)["streams"][0]
    num, den = data["r_frame_rate"].split("/")
    fps = float(num) / max(float(den), 1.0)
    duration = float(data.get("duration") or 0.0)
    n_frames_str = data.get("nb_read_frames")
    if n_frames_str and n_frames_str.isdigit():
        n_frames = int(n_frames_str)
    else:
        n_frames = max(1, int(round(duration * fps)))
    return fps, n_frames, duration


def extract_all_frames(video: Path, dst_dir: Path) -> int:
    """Extract every frame to dst_dir/00000.jpg ... and return count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-q:v", "2",
        "-start_number", "0",
        str(dst_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(dst_dir.glob("*.jpg")))


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Window symlink folder
# ---------------------------------------------------------------------------


def build_window_folder(
    src_frames_dir: Path,
    window_start: int,
    window_end: int,
    dst_dir: Path,
) -> int:
    """Symlink frames [window_start, window_end) → dst_dir/00000.jpg ...

    Symlink targets are absolute — the SAM 3.1 server resolves relative
    targets from the symlink's directory, not its own cwd.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_abs = src_frames_dir.resolve()
    n = 0
    for i in range(window_start, window_end):
        src = src_abs / f"{i:05d}.jpg"
        if not src.exists():
            break
        dst = dst_dir / f"{n:05d}.jpg"
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
        n += 1
    return n


# ---------------------------------------------------------------------------
# Per-window tracking
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """One detected/tracked object instance within a window."""
    instance_id: int            # unique within window: monotonically assigned
    class_name: str
    frames: dict[int, tuple[list[float], float]] = field(default_factory=dict)
    """abs_frame → (bbox_norm_xyxy, score)"""


def track_one_class_in_window(
    client: Sam3OneHttpClient,
    window_dir: Path,
    window_start: int,
    cls: ClassEntry,
    seed_bboxes: list[tuple[list[float], float]],
    *,
    track_score_thresh: float,
    next_instance_id: int,
) -> list[Instance]:
    """One bbox seed per DART detection → SAM 3.1 propagates each instance.

    seed_bboxes: list of (bbox_norm_xyxy, score) from DART for this class.
    Each becomes its own obj_id; the tracker propagates them in one /track
    call. Returns one Instance per obj_id with absolute frame numbers.
    """
    if not seed_bboxes:
        return []
    seeds = [
        SAM3VideoTrackSeed(obj_id=i, frame_index=0, bbox=bbox)
        for i, (bbox, _score) in enumerate(seed_bboxes)
    ]
    response = client.track(
        resource_path=str(window_dir),
        seeds=seeds,
        propagation_direction="forward",
        max_frames=None,
        return_masks=False,
    )
    by_obj: dict[int, dict[int, tuple[list[float], float]]] = defaultdict(dict)
    for frame in response.frames:
        abs_frame = frame.frame_index + window_start
        for obj in frame.objects:
            if obj.bbox is None or obj.score < track_score_thresh:
                continue
            by_obj[obj.obj_id][abs_frame] = (list(obj.bbox), float(obj.score))
    instances: list[Instance] = []
    for oid in sorted(by_obj):
        if not by_obj[oid]:
            continue
        instances.append(Instance(
            instance_id=next_instance_id,
            class_name=cls.name,
            frames=dict(by_obj[oid]),
        ))
        next_instance_id += 1
    return instances


def dart_detect_all_classes(
    dart_url: str,
    keyframe_jpg: Path,
    classes: list[ClassEntry],
    *,
    detect_thresh: float,
    timeout: float,
) -> dict[str, list[tuple[list[float], float]]]:
    """One sam3_dart call for all classes → class_name → [(bbox, score), ...].

    DART returns per-class NMS-deduplicated bboxes (multi-instance native).
    """
    prompt_to_class = {c.prompt: c for c in classes}
    req = DetectorRequest(
        image_path=str(keyframe_jpg),
        prompts=[c.prompt for c in classes],
        threshold=detect_thresh,
    )
    try:
        resp = requests.post(dart_url, json=req.model_dump(), timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("sam3_dart failed for %s: %s", keyframe_jpg, exc)
        return {}
    body = DetectorResponse.model_validate(resp.json())
    out: dict[str, list[tuple[list[float], float]]] = defaultdict(list)
    for bbox, score, label in zip(body.boxes, body.scores, body.labels):
        cls = prompt_to_class.get(label)
        if cls is None:
            continue
        out[cls.name].append(([float(b) for b in bbox], float(score)))
    return dict(out)


# ---------------------------------------------------------------------------
# Stitching across windows
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    class_name: str
    yolo_id: int
    frames: dict[int, tuple[list[float], float]] = field(default_factory=dict)
    seed_windows: list[int] = field(default_factory=list)


def stitch_instances(
    instances_by_window: list[tuple[int, list[Instance]]],
    yolo_id_by_class: dict[str, int],
    *,
    iou_thresh: float = 0.4,
) -> list[Track]:
    """Greedy left-to-right merge across windows by class+IoU at seam frame.

    For each pair of adjacent (window_i, window_i+1) instances of the same
    class: if window_i's last frame bbox overlaps window_i+1's first frame
    bbox with IoU >= thresh, merge them as one track. Otherwise both keep
    their own track_ids.
    """
    by_class: dict[str, list[tuple[int, Instance]]] = defaultdict(list)
    for wi, instances in instances_by_window:
        for inst in instances:
            by_class[inst.class_name].append((wi, inst))

    tracks: list[Track] = []
    for cls_name, items in by_class.items():
        items.sort(key=lambda x: (x[0], min(x[1].frames)))
        # parent[i] = j means item i is in same track as item j
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj

        # Compare each item against later items from the same or next window.
        for i, (wi, inst_i) in enumerate(items):
            i_last = max(inst_i.frames)
            i_last_box = inst_i.frames[i_last][0]
            for j in range(i + 1, len(items)):
                wj, inst_j = items[j]
                if wj == wi:
                    continue
                if wj > wi + 1:
                    break  # past one-window gap
                j_first = min(inst_j.frames)
                j_first_box = inst_j.frames[j_first][0]
                if j_first - i_last > 5:  # frame gap too big
                    continue
                if iou_xyxy(i_last_box, j_first_box) >= iou_thresh:
                    union(i, j)

        # Build tracks per group
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(items)):
            groups[find(i)].append(i)
        for root, members in groups.items():
            members.sort()
            tr = Track(
                track_id=len(tracks) + 1,
                class_name=cls_name,
                yolo_id=yolo_id_by_class.get(cls_name, -1),
            )
            for m in members:
                wi, inst = items[m]
                tr.seed_windows.append(wi)
                for fr, (bb, sc) in inst.frames.items():
                    if fr in tr.frames:
                        # Prefer fresher (later window) seed on overlap
                        existing_wi = tr.seed_windows[-1]
                        if wi >= existing_wi:
                            tr.frames[fr] = (bb, sc)
                    else:
                        tr.frames[fr] = (bb, sc)
            tr.seed_windows = sorted(set(tr.seed_windows))
            tracks.append(tr)
    return tracks


# ---------------------------------------------------------------------------
# Per-video runner
# ---------------------------------------------------------------------------


def run_one_video(
    video: Path,
    classes: list[ClassEntry],
    out_dir: Path,
    *,
    client: Sam3OneHttpClient,
    dart_url: str,
    http_timeout: float,
    seed_stride_sec: float,
    detect_thresh: float,
    track_score_thresh: float,
    iou_merge_thresh: float,
    overwrite: bool,
    keep_temp_frames: bool,
) -> dict[str, Any] | None:
    video_stem = video.stem
    video_out = out_dir / video_stem
    tracks_path = video_out / "tracks.json"
    if tracks_path.exists() and not overwrite:
        logger.info("[%s] tracks.json exists, skipping (use --overwrite to redo)", video_stem)
        return json.loads(tracks_path.read_text())

    logger.info("[%s] probing", video_stem)
    try:
        fps, n_frames, duration = probe_video(video)
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] ffprobe failed: %s", video_stem, exc)
        return None
    logger.info("[%s] fps=%.2f n_frames=%d duration=%.2fs",
                video_stem, fps, n_frames, duration)

    stride = max(1, int(round(fps * seed_stride_sec)))
    boundaries: list[tuple[int, int]] = []
    s = 0
    while s < n_frames:
        e = min(s + stride, n_frames)
        boundaries.append((s, e))
        s = e
    logger.info("[%s] %d windows of %d frames (%.1fs each)",
                video_stem, len(boundaries), stride, stride / fps)

    video_out.mkdir(parents=True, exist_ok=True)
    frames_dir = video_out / ".frames"

    if frames_dir.exists() and not overwrite:
        n_existing = len(list(frames_dir.glob("*.jpg")))
        if n_existing != n_frames:
            shutil.rmtree(frames_dir)
    if not frames_dir.exists():
        logger.info("[%s] extracting frames…", video_stem)
        n_extracted = extract_all_frames(video, frames_dir)
        logger.info("[%s] extracted %d frames", video_stem, n_extracted)
        if n_extracted == 0:
            logger.error("[%s] no frames extracted", video_stem)
            return None
        n_frames = min(n_frames, n_extracted)

    yolo_id_by_class = {c.name: c.yolo_id for c in classes}
    instances_by_window: list[tuple[int, list[Instance]]] = []
    next_instance_id = 1
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix=f"sam3track_{video_stem.replace(' ', '_')}_") as tmp_str:
        tmp_root = Path(tmp_str)
        for wi, (ws, we) in enumerate(boundaries):
            wdir = tmp_root / f"w{wi:04d}"
            n_in_window = build_window_folder(frames_dir, ws, we, wdir)
            if n_in_window == 0:
                continue

            t_w = time.monotonic()
            keyframe_jpg = wdir / "00000.jpg"
            dart_dets = dart_detect_all_classes(
                dart_url, keyframe_jpg, classes,
                detect_thresh=detect_thresh, timeout=http_timeout,
            )
            window_instances: list[Instance] = []
            class_by_name = {c.name: c for c in classes}
            n_dart = sum(len(v) for v in dart_dets.values())
            for cls_name, dets in dart_dets.items():
                cls = class_by_name.get(cls_name)
                if cls is None:
                    continue
                try:
                    new_insts = track_one_class_in_window(
                        client, wdir, ws, cls, dets,
                        track_score_thresh=track_score_thresh,
                        next_instance_id=next_instance_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] window %d %s /track failed: %s",
                                   video_stem, wi, cls.name, exc)
                    continue
                window_instances.extend(new_insts)
                next_instance_id += len(new_insts)

            instances_by_window.append((wi, window_instances))
            n_classes_hit = len({i.class_name for i in window_instances})
            logger.info("[%s] window %d/%d frames=%d dart_dets=%d active_cls=%d hit_cls=%d (%d tracked insts) (%.1fs)",
                        video_stem, wi + 1, len(boundaries), n_in_window,
                        n_dart, len(dart_dets), n_classes_hit,
                        len(window_instances), time.monotonic() - t_w)

    if not keep_temp_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    tracks = stitch_instances(instances_by_window, yolo_id_by_class,
                              iou_thresh=iou_merge_thresh)
    elapsed = time.monotonic() - t_start

    payload_tracks = []
    for tr in tracks:
        frames_sorted = sorted(tr.frames)
        scores = [tr.frames[f][1] for f in frames_sorted]
        payload_tracks.append({
            "track_id": tr.track_id,
            "class": tr.class_name,
            "yolo_id": tr.yolo_id,
            "n_frames": len(frames_sorted),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "first_frame": frames_sorted[0],
            "last_frame": frames_sorted[-1],
            "seed_windows": tr.seed_windows,
            "frames": [
                {"frame": f, "bbox": tr.frames[f][0], "score": tr.frames[f][1]}
                for f in frames_sorted
            ],
        })

    payload = {
        "video": video.name,
        "fps": fps,
        "n_frames": n_frames,
        "duration_sec": duration,
        "seed_stride_sec": seed_stride_sec,
        "n_windows": len(boundaries),
        "n_tracks": len(payload_tracks),
        "tracking_seconds": round(elapsed, 1),
        "tracks": payload_tracks,
    }
    tracks_path.write_text(json.dumps(payload, indent=2))

    summary_path = video_out / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "class", "yolo_id", "n_frames", "mean_score",
                    "first_frame", "last_frame", "seed_windows"])
        for tr in payload_tracks:
            w.writerow([
                tr["track_id"], tr["class"], tr["yolo_id"], tr["n_frames"],
                f"{tr['mean_score']:.3f}",
                tr["first_frame"], tr["last_frame"],
                "|".join(str(s) for s in tr["seed_windows"]),
            ])
    logger.info("[%s] %d tracks across %d windows (%.1fs total) → %s",
                video_stem, len(payload_tracks), len(boundaries),
                elapsed, tracks_path)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--videos-dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path,
                   help="Job config.yaml with class_registry "
                        "(e.g. datatang_val_detect/config.yaml)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sam3-url", default="http://localhost:3014/predict",
                   help="SAM 3.1 video tracker endpoint")
    p.add_argument("--dart-url", default="http://localhost:3013/predict",
                   help="SAM3-DART detector endpoint (used for keyframe seeding)")
    p.add_argument("--seed-stride-sec", type=float, default=2.0,
                   help="re-seed every N seconds (window length)")
    p.add_argument("--detect-thresh", type=float, default=0.3,
                   help="DART score threshold for keyframe detections")
    p.add_argument("--track-score-thresh", type=float, default=0.3,
                   help="drop tracker outputs below this score per frame")
    p.add_argument("--iou-merge-thresh", type=float, default=0.4,
                   help="IoU at seam frame to merge same-class tracks across windows")
    p.add_argument("--http-timeout", type=float, default=600.0,
                   help="HTTP timeout for /track per window")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N videos (0=all)")
    p.add_argument("--video-glob", default="*", help="filename pattern")
    p.add_argument("--keep-temp-frames", action="store_true",
                   help="don't delete the per-video .frames/ extraction dir")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        logger.error("ffmpeg/ffprobe not found on PATH")
        return 2

    classes = load_classes(args.config)
    logger.info("Loaded %d classes from %s", len(classes), args.config)

    videos = sorted(
        p for p in args.videos_dir.glob(args.video_glob)
        if p.suffix.lower() in VIDEO_EXTS
    )
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        logger.error("No videos found in %s", args.videos_dir)
        return 1
    logger.info("Found %d videos in %s", len(videos), args.videos_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = Sam3OneHttpClient(
        url=args.sam3_url,
        track_url=args.sam3_url,
        text_url=args.sam3_url,
        timeout=args.http_timeout,
    )

    global_summary_path = args.out_dir / "summary.csv"
    global_rows: list[list[Any]] = []
    failures: list[str] = []

    t_start = time.monotonic()
    for i, video in enumerate(videos, 1):
        logger.info("=" * 70)
        logger.info("(%d/%d) %s", i, len(videos), video.name)
        try:
            payload = run_one_video(
                video, classes, args.out_dir,
                client=client,
                dart_url=args.dart_url,
                http_timeout=args.http_timeout,
                seed_stride_sec=args.seed_stride_sec,
                detect_thresh=args.detect_thresh,
                track_score_thresh=args.track_score_thresh,
                iou_merge_thresh=args.iou_merge_thresh,
                overwrite=args.overwrite,
                keep_temp_frames=args.keep_temp_frames,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] unexpected failure: %s", video.stem, exc)
            failures.append(video.name)
            continue

        if payload is None:
            failures.append(video.name)
            continue

        for tr in payload.get("tracks", []):
            global_rows.append([
                video.name, tr["track_id"], tr["class"], tr["yolo_id"],
                tr["n_frames"], f"{tr['mean_score']:.3f}",
                tr["first_frame"], tr["last_frame"],
            ])

    with global_summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "class", "yolo_id", "n_frames",
                    "mean_score", "first_frame", "last_frame"])
        w.writerows(global_rows)

    elapsed = time.monotonic() - t_start
    logger.info("=" * 70)
    logger.info("Done. %d/%d videos succeeded in %.1fs (%.1fs avg). %d failures.",
                len(videos) - len(failures), len(videos), elapsed,
                elapsed / max(len(videos), 1), len(failures))
    if failures:
        logger.warning("Failures: %s", failures)
    logger.info("Global summary: %s", global_summary_path)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
