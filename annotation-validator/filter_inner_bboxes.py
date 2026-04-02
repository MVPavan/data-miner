"""
Filter bounding boxes from YOLO annotation files.

Modes:
  inner  — Remove bboxes contained within larger bboxes (with edge tolerance)
  bad    — Split out bad bboxes (too tall, too wide, too narrow, edge-touching)

Usage:
    # Filter inner bboxes
    python filter_inner_bboxes.py --mode inner \
        --input-dir /path/to/conf_nms \
        --output-dir /path/to/conf_nms_outer

    # Filter bad bboxes
    python filter_inner_bboxes.py --mode bad \
        --input-dir /path/to/conf_nms_outer \
        --output-dir /path/to/conf_nms_filtered \
        --bad-dir /path/to/conf_nms_bad
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# YOLO parsing helpers
# ---------------------------------------------------------------------------


def parse_yolo_line(line: str) -> tuple | None:
    """Parse a YOLO annotation line. Returns (class_id, x_c, y_c, w, h, *rest) or None."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    rest = parts[5:]
    return class_id, x_c, y_c, w, h, rest


def yolo_to_edges(x_c, y_c, w, h):
    """Convert YOLO center format to (x1, y1, x2, y2) edges."""
    return x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2


def format_yolo_line(bbox: tuple) -> str:
    """Format bbox back to YOLO annotation line."""
    class_id, x_c, y_c, w, h, rest = bbox
    parts = [str(class_id), f"{x_c:.6f}", f"{y_c:.6f}", f"{w:.6f}", f"{h:.6f}"]
    parts.extend(rest)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Inner bbox filtering
# ---------------------------------------------------------------------------


def is_inside(inner, outer, tolerance=0.05):
    """Check if inner bbox is contained within outer bbox (with tolerance)."""
    _, ix_c, iy_c, iw, ih, _ = inner
    _, ox_c, oy_c, ow, oh, _ = outer

    ix1, iy1, ix2, iy2 = yolo_to_edges(ix_c, iy_c, iw, ih)
    ox1, oy1, ox2, oy2 = yolo_to_edges(ox_c, oy_c, ow, oh)

    tol_x = tolerance * ow
    tol_y = tolerance * oh

    return (
        ix1 >= ox1 - tol_x
        and iy1 >= oy1 - tol_y
        and ix2 <= ox2 + tol_x
        and iy2 <= oy2 + tol_y
    )


def filter_inner_bboxes(bboxes: list[tuple], tolerance: float = 0.05) -> list[tuple]:
    """Remove bboxes that are inside larger bboxes."""
    if len(bboxes) <= 1:
        return bboxes

    sorted_bboxes = sorted(bboxes, key=lambda b: b[3] * b[4], reverse=True)

    keep = []
    for i, bbox in enumerate(sorted_bboxes):
        contained = False
        for j in range(i):
            if is_inside(bbox, sorted_bboxes[j], tolerance):
                contained = True
                break
        if not contained:
            keep.append(bbox)

    return keep


def process_inner_file(input_path: Path, output_path: Path, tolerance: float) -> tuple[int, int]:
    """Process a single annotation file for inner bbox filtering."""
    bboxes = []
    with open(input_path) as f:
        for line in f:
            parsed = parse_yolo_line(line)
            if parsed:
                bboxes.append(parsed)

    original = len(bboxes)
    filtered = filter_inner_bboxes(bboxes, tolerance)

    with open(output_path, "w") as f:
        for bbox in filtered:
            f.write(format_yolo_line(bbox) + "\n")

    return original, len(filtered)


# ---------------------------------------------------------------------------
# Bad bbox filtering
# ---------------------------------------------------------------------------


class BadBboxConfig(BaseModel):
    """Configuration for bad bbox detection."""

    max_height: float = 0.9
    """Bbox normalized height above this is bad."""

    max_width: float = 0.5
    """Bbox normalized width above this is bad."""

    min_width: float = 0.01
    """Bbox normalized width below this is bad."""

    edge_threshold: float = 0.02
    """How close to image edge (0-1) to count as 'touching'. Bbox touching any two edges is bad."""

    max_area: float = 0.75
    """Bbox normalized area (w*h) above this is bad."""


def is_bad_bbox(bbox: tuple, config: BadBboxConfig) -> bool:
    """
    Check if a bbox is bad based on config rules.

    Bad if ANY of:
      - height > max_height
      - width > max_width
      - width < min_width
      - bbox touches 2+ edges of the image
    """
    _, x_c, y_c, w, h, _ = bbox
    x1, y1, x2, y2 = yolo_to_edges(x_c, y_c, w, h)

    tall, wide, narrow, edge_touch = False, False, False, False
    # Height too tall
    if h > config.max_height:
        tall = True

    # Width too wide or too narrow
    if w > config.max_width:
        wide = True
    elif w < config.min_width:
        narrow = True

    # Touching edges (within threshold of 0.0 or 1.0)
    t = config.edge_threshold
    edges_touched = 0
    if x1 <= t:
        edges_touched += 1  # left
    if y1 <= t:
        edges_touched += 1  # top
    if x2 >= 1.0 - t:
        edges_touched += 1  # right
    if y2 >= 1.0 - t:
        edges_touched += 1  # bottom
    if edges_touched >= 2:
        edge_touch = True

    # Area too large
    large_area = (w * h) > config.max_area

    mask = (tall and wide and edge_touch) or (tall and narrow) or large_area
    return mask


def filter_bad_bboxes(input_dir: Path, output_dir: Path, bad_dir: Path, config: BadBboxConfig):
    """Split annotations into good and bad bboxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"Processing {len(txt_files)} annotation files")
    print(f"Config: {config.model_dump()}")

    total = 0
    kept = 0
    bad = 0
    reason_counts = {"tall": 0, "wide": 0, "narrow": 0, "edge_touch": 0, "large_area": 0}

    for txt_file in tqdm(txt_files, desc="Filtering bad bboxes"):
        bboxes = []
        with open(txt_file) as f:
            for line in f:
                parsed = parse_yolo_line(line)
                if parsed:
                    bboxes.append(parsed)

        good_bboxes = []
        bad_bboxes = []
        for b in bboxes:
            if is_bad_bbox(b, config):
                bad_bboxes.append(b)
                # Track reasons
                _, x_c, y_c, w, h, _ = b
                x1, y1, x2, y2 = yolo_to_edges(x_c, y_c, w, h)
                if h > config.max_height:
                    reason_counts["tall"] += 1
                if w > config.max_width:
                    reason_counts["wide"] += 1
                if w < config.min_width:
                    reason_counts["narrow"] += 1
                if w * h > config.max_area:
                    reason_counts["large_area"] += 1
                t = config.edge_threshold
                edges = sum([x1 <= t, y1 <= t, x2 >= 1.0 - t, y2 >= 1.0 - t])
                if edges >= 2:
                    reason_counts["edge_touch"] += 1
            else:
                good_bboxes.append(b)

        total += len(bboxes)
        kept += len(good_bboxes)
        bad += len(bad_bboxes)

        if good_bboxes:
            with open(output_dir / txt_file.name, "w") as f:
                f.write("\n".join(format_yolo_line(b) for b in good_bboxes) + "\n")

        if bad_bboxes:
            with open(bad_dir / txt_file.name, "w") as f:
                f.write("\n".join(format_yolo_line(b) for b in bad_bboxes) + "\n")

    print(f"\nDone: {total} bboxes → {kept} kept, {bad} bad ({100*bad/max(1,total):.1f}%)")
    print(f"Reasons (bbox can match multiple):")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:14}: {cnt:5}")


# ---------------------------------------------------------------------------
# Clean YOLO dataset creation
# ---------------------------------------------------------------------------


def create_clean_dataset(
    images_dir: Path,
    sidecar_dir: Path,
    output_dir: Path,
    all_classes: list[str],
    bad_config: BadBboxConfig,
):
    """
    Create a clean YOLO dataset from validation results.

    - Skips entire image if any bbox is bad
    - Keeps: keep + fix (relabel, adjust_bbox, relabel+bbox) annotations
    - Discards: discard annotations
    - For relabel/relabel+bbox: uses detected_class as the label
    - Symlinks images, writes new label txts
    """
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {c.lower().strip(): i for i, c in enumerate(all_classes)}

    json_files = sorted(sidecar_dir.glob("*.json"))
    print(f"Processing {len(json_files)} sidecar JSONs")
    print(f"Classes: {class_to_id}")
    print(f"Bad bbox config: {bad_config.model_dump()}")

    total_images = 0
    kept_images = 0
    skipped_bad = 0
    skipped_empty = 0
    total_anns = 0
    kept_anns = 0
    relabeled = 0

    for jf in tqdm(json_files, desc="Building clean dataset"):
        with open(jf) as f:
            data = json.load(f)

        total_images += 1
        img_path = Path(data["image_path"])

        # Check if image exists
        if not img_path.exists():
            # Try finding in images_dir
            img_path = images_dir / f"{jf.stem}.jpg"
            if not img_path.exists():
                continue

        annotations = data.get("annotations", [])

        # Check if any bbox is bad — skip entire image
        has_bad = False
        for ann in annotations:
            bbox_tuple = (ann["class_id"], *ann["bbox"], [])
            if is_bad_bbox(bbox_tuple, bad_config):
                has_bad = True
                break
        if has_bad:
            skipped_bad += 1
            continue

        # Filter to keep + fix only
        clean_lines = []
        for ann in annotations:
            total_anns += 1
            cat = ann.get("category", "discard")
            sub = ann.get("sub_category", "")

            if cat == "discard":
                continue

            # Determine class label
            if sub in ("relabel", "relabel+bbox"):
                # Use detected_class instead of expected
                det = ann.get("detected_class", "").lower().strip()
                cid = class_to_id.get(det)
                if cid is None:
                    continue  # detected class not in our class list
                relabeled += 1
            else:
                # Use expected class (original label is correct)
                exp = ann.get("expected_class", "").lower().strip()
                cid = class_to_id.get(exp)
                if cid is None:
                    continue

            x_c, y_c, w, h = ann["bbox"]
            clean_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            kept_anns += 1

        if not clean_lines:
            skipped_empty += 1
            continue

        # Symlink image
        link_path = img_dir / img_path.name
        if not link_path.exists():
            link_path.symlink_to(img_path.resolve())

        # Write labels
        with open(lbl_dir / f"{jf.stem}.txt", "w") as f:
            f.write("\n".join(clean_lines) + "\n")

        kept_images += 1

    print(f"\n{'=' * 60}\nCLEAN DATASET SUMMARY\n{'=' * 60}")
    print(f"Images: {total_images} total → {kept_images} kept, {skipped_bad} skipped (bad bbox), {skipped_empty} skipped (no valid anns)")
    print(f"Annotations: {total_anns} total → {kept_anns} kept, {relabeled} relabeled")
    print(f"Classes: {all_classes}")
    print(f"Output: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Filter bboxes from YOLO annotations")
    parser.add_argument("--mode", choices=["inner", "bad", "clean"], default="inner", help="Filter mode")
    parser.add_argument("--input-dir", required=True, help="Input annotations directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for filtered annotations")

    # Inner mode
    parser.add_argument("--tolerance", type=float, default=0.05, help="Edge tolerance for inner mode (default: 0.05)")

    # Bad mode
    parser.add_argument("--bad-dir", help="Directory for bad bboxes (bad mode only)")
    parser.add_argument("--max-height", type=float, default=0.9, help="Max bbox height (default: 0.9)")
    parser.add_argument("--max-width", type=float, default=0.5, help="Max bbox width (default: 0.5)")
    parser.add_argument("--min-width", type=float, default=0.01, help="Min bbox width (default: 0.01)")
    parser.add_argument("--edge-threshold", type=float, default=0.02, help="Edge touch threshold (default: 0.02)")
    parser.add_argument("--max-area", type=float, default=0.75, help="Max bbox area as fraction of image (default: 0.75)")

    # Clean mode
    parser.add_argument("--images-dir", help="Images directory (clean mode)")
    parser.add_argument("--sidecar-dir", help="Sidecar JSON directory (clean mode)")
    parser.add_argument("--classes", default="forklift,pallet_jack", help="Comma-separated class names (clean mode)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "inner":
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(input_dir.glob("*.txt"))
        print(f"Processing {len(txt_files)} annotation files")
        print(f"Tolerance: {args.tolerance} (fraction of outer box size)")

        total_original = 0
        total_filtered = 0
        files_changed = 0

        for txt_file in tqdm(txt_files, desc="Filtering inner"):
            original, kept_count = process_inner_file(txt_file, output_dir / txt_file.name, args.tolerance)
            total_original += original
            total_filtered += kept_count
            if kept_count < original:
                files_changed += 1

        removed = total_original - total_filtered
        print(f"\nDone: {total_original} bboxes → {total_filtered} kept, {removed} removed ({100*removed/max(1,total_original):.1f}%)")
        print(f"Files changed: {files_changed} / {len(txt_files)}")

    elif args.mode == "bad":
        if not args.bad_dir:
            parser.error("--bad-dir is required for bad mode")
        config = BadBboxConfig(
            max_height=args.max_height,
            max_width=args.max_width,
            min_width=args.min_width,
            edge_threshold=args.edge_threshold,
            max_area=args.max_area,
        )
        filter_bad_bboxes(input_dir, output_dir, Path(args.bad_dir), config)

    elif args.mode == "clean":
        if not args.images_dir or not args.sidecar_dir:
            parser.error("--images-dir and --sidecar-dir are required for clean mode")
        bad_config = BadBboxConfig(
            max_height=args.max_height,
            max_width=args.max_width,
            min_width=args.min_width,
            edge_threshold=args.edge_threshold,
            max_area=args.max_area,
        )
        all_classes = [c.strip() for c in args.classes.split(",")]
        create_clean_dataset(
            Path(args.images_dir), Path(args.sidecar_dir), output_dir, all_classes, bad_config,
        )


if __name__ == "__main__":
    main()
