"""
Annotation Validator — Qwen3.5-27B via vLLM (OpenAI SDK)

Validates bounding box annotations for class correctness and bbox quality.
Draws red bbox on image, sends to VLM for single-pass evaluation.

Usage:
    # Single image + annotation
    python validator.py --image frame.jpg --annotation frame.txt

    # Batch from folder
    python validator.py --image-dir /path/to/images --annotation-dir /path/to/labels --output-dir results/

    # Custom classes
    python validator.py --image-dir imgs/ --annotation-dir labels/ --output-dir out/ \
        --classes "forklift,pallet_jack" --class-descriptions "A forklift: powered industrial truck with forks for lifting pallets. A pallet jack: manual or electric low-profile jack for moving pallets on ground level."
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("LLM_API_KEY", "dummy")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-9B")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PROMPT_VERSION = "v1"

# Scoring thresholds (3-tier: keep / fix / discard)
THRESHOLD_DISCARD = 0.4
THRESHOLD_FIX = 0.75

# Default classes
DEFAULT_CLASSES = "forklift,pallet_jack"
DEFAULT_CLASS_DESCRIPTIONS = """\
- FORKLIFT: A powered industrial truck with a vertical MAST and two horizontal FORKS that raise and lower. \
Key features: upright mast/lifting mechanism, counterweight body at the rear, overhead guard/cage protecting the operator, \
large rear-steer wheels. Can be sit-down or stand-up. Always has a visible mast structure even when forks are lowered. \
Common confusion: do NOT classify reach trucks without a counterweight body, or AGVs/robots as forklifts.

- PALLET JACK (also called pallet truck): A LOW-PROFILE wheeled device for moving pallets at GROUND LEVEL. \
Key features: two flat forks that slide under pallets, small steer wheels at the handle end, load rollers at the fork tips, \
a long steering handle/tiller. NO mast, NO lifting mechanism beyond a few inches of hydraulic raise. \
Manual pallet jacks have a pump handle; electric (powered) pallet jacks have a motor housing at the handle end. \
Common confusion: do NOT classify forklifts with lowered forks as pallet jacks — check for a mast."""


def parse_class_list(classes: str) -> list[str]:
    """Parse comma-separated class string into list."""
    return [c.strip() for c in classes.split(",")]


def build_rating_prompt(expected_class: str, all_classes: list[str], class_descriptions: str) -> str:
    """Build the rating prompt for a specific expected class."""
    class_names = ", ".join(all_classes)

    return f"""Look at the RED box in this image. What object is inside it?

STEP 1: Identify the object inside the red box. Be specific — say exactly what you see (e.g. "person", "shelf", "wall", "{all_classes[0]}", "{all_classes[-1]}").
STEP 2: Check if it matches the label "{expected_class}".
STEP 3: Rate how well the red box covers that object.

DEFINITIONS:
{class_descriptions}

RETURN only this JSON:
{{"detected_class": "what_is_in_the_box", "class_match": true/false, "bbox_score": 0.XX}}

- "detected_class": the actual object inside the red box. NOT what you think should be there.
- "class_match": true only if detected_class is "{expected_class}"
- "bbox_score": 0.0-0.4 unusable, 0.4-0.75 needs adjustment, 0.75-1.0 good tight coverage"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_annotations(txt_path: Path) -> list[tuple]:
    """
    Load YOLO format annotations from txt file.

    Returns list of (class_id, x_center, y_center, width, height).
    """
    annotations = []
    if not txt_path.exists():
        return annotations
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, w, h = (float(p) for p in parts[1:5])
                annotations.append((class_id, x_center, y_center, w, h))
    return annotations


def yolo_to_pixel(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Convert YOLO normalized bbox to pixel coords (x1, y1, x2, y2). Returns None if degenerate."""
    x_center, y_center, w, h = bbox
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, x2 = max(0, x1), min(img_w - 1, x2)
    y1, y2 = max(0, y1), min(img_h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_bbox_on_image(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> tuple[Image.Image, bool]:
    """
    Draw bounding box on image.

    Args:
        image: PIL Image.
        bbox: (x_center, y_center, width, height) in normalized YOLO coordinates (0-1).

    Returns:
        (annotated_image, is_valid_bbox)
    """
    coords = yolo_to_pixel(bbox, *image.size)
    if coords is None:
        return image, False

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(coords, outline=color, width=width)
    return img_copy, True


# Category -> color for debug visualization
CATEGORY_COLORS = {
    "keep": (0, 200, 0),       # green
    "fix": (255, 165, 0),      # orange
    "discard": (255, 0, 0),    # red
}


def save_debug_image(
    image: Image.Image,
    results: list[dict],
    output_path: Path,
):
    """Save image with all bboxes drawn, color-coded by category."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    img_w, img_h = image.size

    for r in results:
        coords = yolo_to_pixel(tuple(r["bbox"]), img_w, img_h)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        cat = r.get("category", "discard")
        color = CATEGORY_COLORS.get(cat, (128, 128, 128))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f'{r.get("detected_class", "?")} {r.get("bbox_score", 0):.2f} [{cat}]'
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 14), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 14), label, fill=(255, 255, 255))

    img_copy.save(output_path, quality=90)
    print(f"Debug image saved: {output_path}")


def image_to_base64(image: Image.Image) -> str:
    """Encode PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_output(text: str) -> dict:
    """Parse JSON output with robust extraction. Handles <think> blocks and markdown fences."""
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    text = re.sub(r"```json\s*|```\s*", "", text)
    for prefix in ["Here is the JSON:", "Here's the JSON:", "JSON:", "Output:"]:
        text = text.replace(prefix, "")
    text = text.strip()

    # Extract first balanced {...} containing "score"
    start = text.find("{")
    if start != -1:
        depth, end = 0, start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = text[start:end]
        if "bbox_score" in candidate or "score" in candidate:
            try:
                data = json.loads(candidate)
                # Accept both "bbox_score" and "score" keys
                raw_score = data.get("bbox_score", data.get("score", -1))
                score = float(raw_score)
                data["bbox_score"] = max(0.0, min(1.0, score)) if score >= 0 else -1
                return data
            except (json.JSONDecodeError, ValueError):
                pass

    # Fallback: regex extraction
    score_match = re.search(r"\"?(?:bbox_)?score\"?\s*[:=]\s*[\"']?([\d.]+)", text)
    if score_match:
        score = max(0.0, min(1.0, float(score_match.group(1))))
        return {"class_match": None, "detected_class": "unknown", "bbox_score": score}

    print(f"  Parse failed: {text[:200]}")
    return {"class_match": None, "detected_class": "unknown", "bbox_score": -1}


def derive_category(
    bbox_score: float,
    class_match: bool | None,
    detected_class: str,
    all_classes: list[str],
) -> str:
    """
    Derive 3-tier category: keep / fix / discard.

    Decision matrix:
      class_match=True  + bbox_score >= 0.75 → keep
      class_match=True  + bbox_score >= 0.4  → fix  (adjust bbox)
      class_match=True  + bbox_score <  0.4  → discard (unusable bbox)
      class_match=False + valid class + bbox_score >= 0.4 → fix (relabel + maybe adjust bbox)
      class_match=False + valid class + bbox_score <  0.4 → discard (wrong label AND bad bbox)
      class_match=False + NOT valid class → discard (not our object)
      bbox_score < 0 (parse error) → discard
    """
    if bbox_score < 0:
        return "discard"

    valid_classes_lower = {c.lower().strip() for c in all_classes}
    detected_lower = detected_class.lower().strip()

    if class_match is False:
        # Not any of our classes → discard
        if detected_lower not in valid_classes_lower:
            return "discard"
        # Valid class but wrong label → fix if bbox is usable
        return "fix" if bbox_score >= THRESHOLD_DISCARD else "discard"

    # class_match is True (or None from parse error)
    if bbox_score >= THRESHOLD_FIX:
        return "keep"
    elif bbox_score >= THRESHOLD_DISCARD:
        return "fix"
    else:
        return "discard"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_annotation(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    expected_class: str,
    all_classes: list[str],
    class_descriptions: str,
    thinking: bool = False,
) -> dict:
    """
    Validate a single bbox annotation via vLLM.

    Draws the red bbox on the image, encodes to base64, sends to model.
    The prompt tells the model what class the annotation claims, so it can
    catch misclassifications (e.g. forklift labeled as pallet_jack).
    """
    t0 = time.perf_counter()

    annotated_image, bbox_valid = draw_bbox_on_image(image, bbox)
    if not bbox_valid:
        return {
            "class_match": None,
            "detected_class": "unknown",
            "bbox_score": -1,
            "category": "discard",
            "latency": 0,
        }

    b64 = image_to_base64(annotated_image)

    extra_body = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": build_rating_prompt(expected_class, all_classes, class_descriptions)},
                ],
            },
        ],
        max_tokens=512 if thinking else 256,
        temperature=1.0 if thinking else 0.7,
        top_p=0.95 if thinking else 0.8,
        presence_penalty=1.5,
        extra_body=extra_body,
    )

    raw = response.choices[0].message.content.strip()
    latency = time.perf_counter() - t0

    result = parse_json_output(raw)
    class_match = result.get("class_match")
    detected_class = result.get("detected_class", "unknown")
    bbox_score = result.get("bbox_score", -1)
    category = derive_category(bbox_score, class_match, detected_class, all_classes)

    return {
        "class_match": class_match,
        "detected_class": detected_class,
        "bbox_score": bbox_score,
        "category": category,
        "latency": latency,
    }


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_single_image(
    image_path: Path,
    annotation_path: Path,
    all_classes: list[str],
    class_descriptions: str,
    thinking: bool = False,
    debug_dir: Path | None = None,
) -> tuple[list[dict], int, int]:
    """Validate all annotations for a single image. Returns (results, img_w, img_h)."""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    annotations = load_annotations(annotation_path)

    results = []
    for i, (class_id, x_center, y_center, w, h) in enumerate(annotations):
        expected_class = all_classes[class_id] if class_id < len(all_classes) else f"unknown_class_{class_id}"
        rating = validate_annotation(
            image, (x_center, y_center, w, h),
            expected_class, all_classes, class_descriptions, thinking,
        )
        rating["bbox_index"] = i
        rating["class_id"] = class_id
        rating["expected_class"] = expected_class
        rating["bbox"] = [x_center, y_center, w, h]
        results.append(rating)

    if debug_dir and results:
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_debug_image(image, results, debug_dir / f"{image_path.stem}.jpg")

    return results, img_w, img_h


def process_folder(
    image_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    all_classes: list[str],
    class_descriptions: str,
    thinking: bool = False,
    max_workers: int = 4,
):
    """Process all images and rate their annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = output_dir / "sidecar_json"
    scored_ann_dir = output_dir / "scored_annotations"
    sidecar_dir.mkdir(exist_ok=True)
    scored_ann_dir.mkdir(exist_ok=True)

    # Collect image files
    image_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Filter to only images that have annotation files
    items = []
    missing_ann = 0
    for img_file in image_files:
        ann_path = annotation_dir / f"{img_file.stem}.txt"
        if ann_path.exists():
            items.append((img_file, ann_path))
        else:
            missing_ann += 1

    print(f"Processing {len(items)} images ({missing_ann} skipped, no annotation file)")

    category_counts = {"keep": 0, "fix": 0, "discard": 0}
    total_annotations = 0
    latencies = []

    debug_img_dir = output_dir / "debug_images"

    def _process_one(img_file: Path, ann_path: Path):
        return img_file, process_single_image(img_file, ann_path, all_classes, class_descriptions, thinking, debug_img_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, img, ann): img
            for img, ann in items
        }

        with tqdm(total=len(items), desc="Rating annotations") as pbar:
            for future in as_completed(futures):
                img_file = futures[future]
                try:
                    img_file, (results, img_w, img_h) = future.result()

                    for r in results:
                        cat = r["category"]
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                        total_annotations += 1
                        latencies.append(r["latency"])

                    # Sidecar JSON
                    sidecar_data = {
                        "image_path": str(img_file),
                        "img_w": img_w,
                        "img_h": img_h,
                        "model_id": MODEL,
                        "prompt_version": PROMPT_VERSION,
                        "annotations": [
                            {
                                "bbox_index": r["bbox_index"],
                                "class_id": r["class_id"],
                                "expected_class": r["expected_class"],
                                "bbox": r["bbox"],
                                "class_match": r["class_match"],
                                "detected_class": r["detected_class"],
                                "bbox_score": r["bbox_score"],
                                "category": r["category"],
                            }
                            for r in results
                        ],
                    }
                    with open(sidecar_dir / f"{img_file.stem}.json", "w") as f:
                        json.dump(sidecar_data, f, indent=2)

                    # Scored annotations (YOLO + bbox_score + category + detected_class)
                    lines = []
                    for r in results:
                        x_c, y_c, w, h = r["bbox"]
                        lines.append(
                            f"{r['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {r['bbox_score']:.3f} {r['category']} {r['detected_class']}"
                        )
                    with open(scored_ann_dir / f"{img_file.stem}.txt", "w") as f:
                        f.write("\n".join(lines))

                except Exception as e:
                    print(f"\nError processing {img_file.name}: {e}")

                pbar.update(1)

    # Summary
    print_summary(category_counts, total_annotations, missing_ann, len(image_files), latencies, output_dir)


def print_summary(
    category_counts: dict,
    total: int,
    missing: int,
    num_images: int,
    latencies: list[float],
    output_dir: Path,
):
    """Print and save summary."""
    print(f"\n{'=' * 50}\nANNOTATION VALIDATION SUMMARY\n{'=' * 50}")
    print(f"Total annotations rated: {total} | Missing ann files: {missing}")
    for cat in ["keep", "fix", "discard"]:
        cnt = category_counts.get(cat, 0)
        pct = 100 * cnt / max(1, total)
        print(f"  {cat:14}: {cnt:5} ({pct:5.1f}%)")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nAvg latency: {avg:.3f}s ({1 / avg:.1f} ratings/sec)")

    summary = {
        "model_id": MODEL,
        "prompt_version": PROMPT_VERSION,
        "total_images": num_images,
        "missing_annotations": missing,
        "total_annotations": total,
        "category_counts": category_counts,
        "thresholds": {
            "discard": THRESHOLD_DISCARD,
            "fix": THRESHOLD_FIX,
        },
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Annotation validator using Qwen3.5-27B via vLLM"
    )

    # Input modes
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--annotation", help="Single annotation .txt path (YOLO format)")
    parser.add_argument("--image-dir", help="Directory of images for batch processing")
    parser.add_argument("--annotation-dir", help="Directory of YOLO .txt annotations")

    # Output
    parser.add_argument("--output-dir", "-o", default="validation_results", help="Output directory (default: validation_results)")

    # Class config
    parser.add_argument("--classes", default=DEFAULT_CLASSES, help=f"Comma-separated valid classes (default: {DEFAULT_CLASSES})")
    parser.add_argument("--class-descriptions", default=DEFAULT_CLASS_DESCRIPTIONS, help="Text describing valid classes for the model")

    # Inference
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (slower, better on edge cases)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests to vLLM (default: 4)")

    args = parser.parse_args()

    all_classes = parse_class_list(args.classes)

    # --- Single image mode ---
    if args.image:
        image_path = Path(args.image)
        ann_path = Path(args.annotation) if args.annotation else image_path.with_suffix(".txt")
        if not ann_path.exists():
            print(f"Annotation file not found: {ann_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Validating {image_path.name} against {BASE_URL}")
        print(f"Model: {MODEL} | Classes: {args.classes}\n")

        debug_dir = Path(args.output_dir) / "debug_images"
        results, _, _ = process_single_image(
            image_path, ann_path, all_classes, args.class_descriptions, args.thinking, debug_dir,
        )
        for r in results:
            print(json.dumps(r, indent=2))
        return

    # --- Batch folder mode ---
    if args.image_dir:
        image_dir = Path(args.image_dir)
        annotation_dir = Path(args.annotation_dir) if args.annotation_dir else image_dir
        output_dir = Path(args.output_dir)

        print(f"Validating annotations against {BASE_URL}")
        print(f"Model: {MODEL} | Classes: {args.classes}")
        print(f"Thinking: {args.thinking} | Workers: {args.workers}\n")

        process_folder(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            all_classes=all_classes,
            class_descriptions=args.class_descriptions,
            thinking=args.thinking,
            max_workers=args.workers,
        )
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
