"""Utility functions: bbox math, image drawing, YOLO export."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw, ImageFont

from .config import ClassPackConfig
from .contracts import BoundingBox, Candidate, FinalAnnotation

# ---------------------------------------------------------------------------
# Bbox math
# ---------------------------------------------------------------------------


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return 0.0 if union <= 0 else inter / union


def bbox_to_pixels(
    box: BoundingBox, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    w, h = size
    return (
        int(clamp(box.x1) * w),
        int(clamp(box.y1) * h),
        int(clamp(box.x2) * w),
        int(clamp(box.y2) * h),
    )


def pixels_to_bbox(
    coords: tuple[int, int, int, int], size: tuple[int, int]
) -> BoundingBox:
    w, h = size
    x1, y1, x2, y2 = coords
    return BoundingBox(
        x1=clamp(x1 / w), y1=clamp(y1 / h), x2=clamp(x2 / w), y2=clamp(y2 / h)
    )


# ---------------------------------------------------------------------------
# Image drawing
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
]


def draw_candidates_numbered(
    image: Image.Image,
    candidates: list[Candidate],
    class_colors: bool = True,
) -> Image.Image:
    """Draw all candidates on image with numbered labels. For VLM screening."""
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, cand in enumerate(candidates):
        color = _COLORS[idx % len(_COLORS)] if class_colors else (255, 0, 0)
        px = bbox_to_pixels(cand.bbox, rendered.size)
        draw.rectangle(px, outline=color, width=3)
        label = f"[{idx}] {cand.label} ({cand.score:.2f})"
        draw.text((px[0], max(0, px[1] - 16)), label, fill=color, font=font)
    return rendered


def draw_single_candidate(image: Image.Image, candidate: Candidate) -> Image.Image:
    """Draw one candidate with its bbox highlighted."""
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    px = bbox_to_pixels(candidate.bbox, rendered.size)
    draw.rectangle(px, outline=(255, 0, 0), width=3)
    draw.text((px[0], max(0, px[1] - 16)), candidate.label, fill=(255, 0, 0))
    return rendered


def crop_candidate(
    image: Image.Image, candidate: Candidate, padding: float = 0.08
) -> Image.Image:
    """Crop around a candidate with relative padding."""
    w, h = image.size
    box = candidate.bbox
    px = int(box.width * w * padding)
    py = int(box.height * h * padding)
    x1, y1, x2, y2 = bbox_to_pixels(box, image.size)
    return image.crop(
        (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))
    )


# ---------------------------------------------------------------------------
# Image encoding for VLM
# ---------------------------------------------------------------------------


def pil_to_data_url(image: Image.Image, max_size: int = 1024) -> str:
    """Encode PIL image as base64 data URL, resizing if needed."""
    img = image.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def pil_to_png_bytes(image: Image.Image, max_size: int = 1024) -> bytes:
    """Encode PIL image as PNG bytes, resizing if needed."""
    img = image.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------


def annotation_to_yolo_line(annotation: FinalAnnotation, class_index: int) -> str:
    box = annotation.bbox
    return f"{class_index} {box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}"


# ---------------------------------------------------------------------------
# Class alias resolution
# ---------------------------------------------------------------------------


def normalize_class_alias(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def build_class_alias_map(class_packs: list[ClassPackConfig]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for cp in class_packs:
        canonical = cp.name
        for alias in cp.all_names():
            alias_map[normalize_class_alias(alias)] = canonical
    return alias_map


def resolve_canonical_class_name(
    value: str | None, class_packs: list[ClassPackConfig]
) -> str | None:
    if not value:
        return None
    return build_class_alias_map(class_packs).get(normalize_class_alias(value))
