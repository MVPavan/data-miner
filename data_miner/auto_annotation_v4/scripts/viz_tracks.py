"""Render an annotated MP4 from a tracks.json produced by track_videos_sam3.py.

For each frame in the source video, draw every active track's bbox with a
class-coloured rectangle + label "<class>:<track_id> (score)". Frame numbers
in tracks.json are absolute (0-indexed) and match ffmpeg's start_number=0
extraction.

Usage:
    python -m data_miner.auto_annotation_v4.scripts.viz_tracks \\
        --video /media/.../video.mp4 \\
        --tracks output/sam3_track/<stem>/tracks.json \\
        --out output/sam3_track/<stem>/annotated.mp4

Defaults to writing annotated.mp4 next to tracks.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("viz_tracks")

# 23 distinguishable colors (re-used across track ids; class drives color)
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ff00ff", "#00ff00", "#ff8800",
]


def color_for_class(class_name: str, class_to_idx: dict[str, int]) -> str:
    return _PALETTE[class_to_idx.get(class_name, 0) % len(_PALETTE)]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def extract_frames(video: Path, dst_dir: Path) -> tuple[int, int, float]:
    """Extract video to dst_dir/00000.jpg ... and return (n, fps_num, fps_den_unused)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-q:v", "2",
        "-start_number", "0",
        str(dst_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n = len(list(dst_dir.glob("*.jpg")))
    return n, 0, 0.0


def annotate_frames(
    frames_dir: Path,
    tracks: list[dict],
    classes_in_order: list[str],
    n_frames: int,
    *,
    line_width: int | None = None,
    font_size: int | None = None,
) -> None:
    """Mutate frames_dir in-place: draw bboxes per track on each frame.

    line_width / font_size auto-scale from image min(w, h):
      width = clamp(min_dim / 500, 1, 4)
      font  = clamp(min_dim / 50, 9, 18)
    Override with explicit args.
    """
    class_to_idx = {c: i for i, c in enumerate(classes_in_order)}
    per_frame: dict[int, list[tuple[int, str, list[float], float]]] = defaultdict(list)
    for tr in tracks:
        cls = tr["class"]
        tid = tr["track_id"]
        for f in tr["frames"]:
            per_frame[f["frame"]].append((tid, cls, f["bbox"], f["score"]))

    # Probe one frame to size font (constant across frames; same resolution).
    sample_path = frames_dir / "00000.jpg"
    sw, sh = Image.open(sample_path).size if sample_path.exists() else (1280, 720)
    min_dim = min(sw, sh)
    auto_line = max(1, min(4, round(min_dim / 500)))
    auto_font = max(9, min(18, round(min_dim / 50)))
    lw = line_width if line_width is not None else auto_line
    fs = font_size if font_size is not None else auto_font
    font_label = _load_font(fs)
    label_pad = 1
    label_h = fs + 2 * label_pad
    logger.info("annotation: img=%dx%d, line_width=%d, font_size=%d", sw, sh, lw, fs)

    for fr in range(n_frames):
        path = frames_dir / f"{fr:05d}.jpg"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for tid, cls, bbox, score in per_frame.get(fr, []):
            x1, y1, x2, y2 = bbox
            box_px = (x1 * w, y1 * h, x2 * w, y2 * h)
            color = color_for_class(cls, class_to_idx)
            draw.rectangle(box_px, outline=color, width=lw)
            label = f"{cls}#{tid} {score:.2f}"
            tx = box_px[0] + 1
            ty = max(0.0, box_px[1] - label_h)
            try:
                bbox_text = draw.textbbox((tx, ty), label, font=font_label)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except Exception:  # noqa: BLE001
                tw, th = len(label) * (fs // 2), fs
            draw.rectangle(
                (tx - label_pad, ty,
                 tx + tw + label_pad, ty + th + 2 * label_pad),
                fill=color,
            )
            draw.text((tx, ty + label_pad), label, fill="black", font=font_label)
        img.save(path, quality=88)


def encode_video(frames_dir: Path, fps: float, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps}",
        "-start_number", "0",
        "-i", str(frames_dir / "%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--tracks", required=True, type=Path,
                   help="tracks.json from track_videos_sam3.py")
    p.add_argument("--out", type=Path,
                   help="output mp4 path (default: alongside tracks.json)")
    p.add_argument("--line-width", type=int, default=None,
                   help="bbox stroke width in px (default: auto from resolution)")
    p.add_argument("--font-size", type=int, default=None,
                   help="label font size in px (default: auto from resolution)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found")
        return 2

    payload = json.loads(args.tracks.read_text())
    fps = float(payload["fps"])
    n_frames = int(payload["n_frames"])
    tracks = payload["tracks"]
    classes_in_order = sorted({t["class"] for t in tracks})
    out_path = args.out or args.tracks.parent / "annotated.mp4"

    logger.info("video=%s fps=%.2f n_frames=%d tracks=%d → %s",
                args.video.name, fps, n_frames, len(tracks), out_path)

    with tempfile.TemporaryDirectory(prefix="viz_tracks_") as td:
        tmp = Path(td)
        logger.info("extracting frames…")
        n_extracted, _, _ = extract_frames(args.video, tmp)
        logger.info("extracted %d frames", n_extracted)
        logger.info("annotating frames…")
        annotate_frames(
            tmp, tracks, classes_in_order, min(n_frames, n_extracted),
            line_width=args.line_width, font_size=args.font_size,
        )
        logger.info("encoding mp4…")
        encode_video(tmp, fps, out_path)
    logger.info("done → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
