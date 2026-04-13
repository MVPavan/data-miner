"""
Filter and clean YOLO bounding box annotations.

Modes:
  inner — Remove bboxes contained within larger bboxes
  bad   — Split out bad bboxes (tall, wide, narrow, edge-touching, large area)
  clean — Create clean YOLO training dataset from validation sidecar JSONs

Usage:
    python filter_inner_bboxes.py inner --input-dir ... --output-dir ...
    python filter_inner_bboxes.py bad   --input-dir ... --output-dir ... --bad-dir ...
    python filter_inner_bboxes.py clean --images-dir ... --sidecar-dir ... --output-dir ...
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class InnerFilterConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    tolerance: float = 0.05
    """Edge tolerance as fraction of outer box size."""


class BadBboxConfig(BaseModel):
    max_height: float = 0.9
    max_width: float = 0.5
    min_width: float = 0.01
    edge_threshold: float = 0.02
    max_area: float = 0.75


class BadFilterConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    bad_dir: Path
    bbox: BadBboxConfig = BadBboxConfig()


class CleanDatasetConfig(BaseModel):
    images_dir: Path
    sidecar_dir: Path
    output_dir: Path
    classes: list[str] = ["forklift", "pallet_jack"]
    bbox: BadBboxConfig = BadBboxConfig()


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------


def parse_yolo_line(line: str) -> tuple | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), parts[5:]


def yolo_to_edges(x_c, y_c, w, h):
    return x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2


def format_yolo_line(bbox: tuple) -> str:
    class_id, x_c, y_c, w, h, rest = bbox
    parts = [str(class_id), f"{x_c:.6f}", f"{y_c:.6f}", f"{w:.6f}", f"{h:.6f}"]
    parts.extend(rest)
    return " ".join(parts)


def load_bboxes(path: Path) -> list[tuple]:
    bboxes = []
    with open(path) as f:
        for line in f:
            parsed = parse_yolo_line(line)
            if parsed:
                bboxes.append(parsed)
    return bboxes


def write_bboxes(path: Path, bboxes: list[tuple]):
    with open(path, "w") as f:
        f.write("\n".join(format_yolo_line(b) for b in bboxes) + "\n")


# ---------------------------------------------------------------------------
# Inner bbox filtering
# ---------------------------------------------------------------------------


def is_inside(inner, outer, tolerance=0.05):
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


def remove_inner_bboxes(bboxes: list[tuple], tolerance: float) -> list[tuple]:
    if len(bboxes) <= 1:
        return bboxes
    sorted_bboxes = sorted(bboxes, key=lambda b: b[3] * b[4], reverse=True)
    keep = []
    for i, bbox in enumerate(sorted_bboxes):
        if not any(is_inside(bbox, sorted_bboxes[j], tolerance) for j in range(i)):
            keep.append(bbox)
    return keep


def run_inner_filter(config: InnerFilterConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(config.input_dir.glob("*.txt"))
    print(f"Processing {len(txt_files)} files | tolerance={config.tolerance}")

    total, kept, changed = 0, 0, 0
    for txt_file in tqdm(txt_files, desc="Filtering inner"):
        bboxes = load_bboxes(txt_file)
        filtered = remove_inner_bboxes(bboxes, config.tolerance)
        total += len(bboxes)
        kept += len(filtered)
        if len(filtered) < len(bboxes):
            changed += 1
        write_bboxes(config.output_dir / txt_file.name, filtered)

    removed = total - kept
    print(f"\nDone: {total} → {kept} kept, {removed} removed ({100*removed/max(1,total):.1f}%)")
    print(f"Files changed: {changed} / {len(txt_files)}")


# ---------------------------------------------------------------------------
# Bad bbox filtering
# ---------------------------------------------------------------------------


def is_bad_bbox(bbox: tuple, config: BadBboxConfig) -> bool:
    _, x_c, y_c, w, h, _ = bbox
    x1, y1, x2, y2 = yolo_to_edges(x_c, y_c, w, h)

    tall = h > config.max_height
    wide = w > config.max_width
    narrow = w < config.min_width
    large_area = (w * h) > config.max_area

    t = config.edge_threshold
    edge_touch = sum([x1 <= t, y1 <= t, x2 >= 1.0 - t, y2 >= 1.0 - t]) >= 2

    return (tall and wide and edge_touch) or (tall and narrow) or large_area


def classify_bad_reasons(bbox: tuple, config: BadBboxConfig) -> dict[str, bool]:
    _, x_c, y_c, w, h, _ = bbox
    x1, y1, x2, y2 = yolo_to_edges(x_c, y_c, w, h)
    t = config.edge_threshold
    return {
        "tall": h > config.max_height,
        "wide": w > config.max_width,
        "narrow": w < config.min_width,
        "large_area": (w * h) > config.max_area,
        "edge_touch": sum([x1 <= t, y1 <= t, x2 >= 1.0 - t, y2 >= 1.0 - t]) >= 2,
    }


def run_bad_filter(config: BadFilterConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.bad_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(config.input_dir.glob("*.txt"))
    print(f"Processing {len(txt_files)} files | {config.bbox.model_dump()}")

    total, kept_count, bad_count = 0, 0, 0
    reason_counts = {"tall": 0, "wide": 0, "narrow": 0, "edge_touch": 0, "large_area": 0}

    for txt_file in tqdm(txt_files, desc="Filtering bad"):
        bboxes = load_bboxes(txt_file)
        good, bad = [], []
        for b in bboxes:
            if is_bad_bbox(b, config.bbox):
                bad.append(b)
                for reason, hit in classify_bad_reasons(b, config.bbox).items():
                    if hit:
                        reason_counts[reason] += 1
            else:
                good.append(b)

        total += len(bboxes)
        kept_count += len(good)
        bad_count += len(bad)

        if good:
            write_bboxes(config.output_dir / txt_file.name, good)
        if bad:
            write_bboxes(config.bad_dir / txt_file.name, bad)

    print(f"\nDone: {total} → {kept_count} kept, {bad_count} bad ({100*bad_count/max(1,total):.1f}%)")
    print(f"Reasons (can overlap):")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:14}: {cnt:5}")


# ---------------------------------------------------------------------------
# Clean YOLO dataset
# ---------------------------------------------------------------------------


def run_clean_dataset(config: CleanDatasetConfig):
    img_dir = config.output_dir / "images"
    lbl_dir = config.output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {c.lower().strip(): i for i, c in enumerate(config.classes)}
    json_files = sorted(config.sidecar_dir.glob("*.json"))
    print(f"Processing {len(json_files)} sidecars | classes={class_to_id}")
    print(f"Bad bbox config: {config.bbox.model_dump()}")

    total_images, kept_images, skipped_bad, skipped_empty = 0, 0, 0, 0
    total_anns, kept_anns, relabeled = 0, 0, 0

    for jf in tqdm(json_files, desc="Building clean dataset"):
        with open(jf) as f:
            data = json.load(f)

        total_images += 1
        img_path = Path(data["image_path"])
        if not img_path.exists():
            img_path = config.images_dir / f"{jf.stem}.jpg"
            if not img_path.exists():
                continue

        annotations = data.get("annotations", [])

        # Skip image if any bbox is bad
        if any(is_bad_bbox((a["class_id"], *a["bbox"], []), config.bbox) for a in annotations):
            skipped_bad += 1
            continue

        # Build clean labels
        clean_lines = []
        for ann in annotations:
            total_anns += 1
            cat = ann.get("category", "discard")
            sub = ann.get("sub_category", "")

            if cat == "discard":
                continue

            if sub in ("relabel", "relabel+bbox"):
                cid = class_to_id.get(ann.get("detected_class", "").lower().strip())
                if cid is None:
                    continue
                relabeled += 1
            else:
                cid = class_to_id.get(ann.get("expected_class", "").lower().strip())
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
    print(f"Images: {total_images} → {kept_images} kept, {skipped_bad} bad, {skipped_empty} empty")
    print(f"Annotations: {total_anns} → {kept_anns} kept, {relabeled} relabeled")
    print(f"Output: {config.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python filter_inner_bboxes.py <mode> [options]")
        print("Modes: inner, bad, clean")
        print("  inner --input-dir DIR --output-dir DIR [--tolerance 0.05]")
        print("  bad   --input-dir DIR --output-dir DIR --bad-dir DIR [--max-height 0.9] ...")
        print("  clean --images-dir DIR --sidecar-dir DIR --output-dir DIR [--classes forklift,pallet_jack] ...")
        sys.exit(0)

    mode = sys.argv[1]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")

    if mode == "inner":
        parser.add_argument("--input-dir", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--tolerance", type=float, default=0.05)
        args = parser.parse_args()
        run_inner_filter(InnerFilterConfig(
            input_dir=args.input_dir, output_dir=args.output_dir, tolerance=args.tolerance,
        ))

    elif mode == "bad":
        parser.add_argument("--input-dir", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--bad-dir", required=True)
        parser.add_argument("--max-height", type=float, default=0.9)
        parser.add_argument("--max-width", type=float, default=0.5)
        parser.add_argument("--min-width", type=float, default=0.01)
        parser.add_argument("--edge-threshold", type=float, default=0.02)
        parser.add_argument("--max-area", type=float, default=0.75)
        args = parser.parse_args()
        run_bad_filter(BadFilterConfig(
            input_dir=args.input_dir, output_dir=args.output_dir, bad_dir=args.bad_dir,
            bbox=BadBboxConfig(
                max_height=args.max_height, max_width=args.max_width, min_width=args.min_width,
                edge_threshold=args.edge_threshold, max_area=args.max_area,
            ),
        ))

    elif mode == "clean":
        parser.add_argument("--images-dir", required=True)
        parser.add_argument("--sidecar-dir", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--classes", default="forklift,pallet_jack")
        parser.add_argument("--max-height", type=float, default=0.9)
        parser.add_argument("--max-width", type=float, default=0.5)
        parser.add_argument("--min-width", type=float, default=0.01)
        parser.add_argument("--edge-threshold", type=float, default=0.02)
        parser.add_argument("--max-area", type=float, default=0.75)
        args = parser.parse_args()
        run_clean_dataset(CleanDatasetConfig(
            images_dir=args.images_dir, sidecar_dir=args.sidecar_dir, output_dir=args.output_dir,
            classes=[c.strip() for c in args.classes.split(",")],
            bbox=BadBboxConfig(
                max_height=args.max_height, max_width=args.max_width, min_width=args.min_width,
                edge_threshold=args.edge_threshold, max_area=args.max_area,
            ),
        ))

    else:
        print(f"Unknown mode: {mode}. Use: inner, bad, clean")
        sys.exit(1)


if __name__ == "__main__":
    main()
