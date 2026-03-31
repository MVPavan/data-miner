#!/usr/bin/env python3
"""
Pipeline for fl_pj detections (class-agnostic):
1. Confidence filter (conf >= 0.25) per model
2. Intra-model class-agnostic NMS (IoU=0.5) — higher conf class wins
3. Normalize confidence per model globally (divide by model's global max)
4. Cross-model class-agnostic NMS (IoU=0.5) on combined normalized results
"""

import os
import sys
from pathlib import Path

from tqdm import tqdm


def iou_single(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2):
    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter
    return inter / union if union > 0 else 0


def parse_line(line: str) -> tuple | None:
    """Parse a YOLO line into (cls, xc, yc, w, h, conf). Returns None on failure."""
    parts = line.split()
    n = len(parts)
    if n == 6:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
    elif n == 5:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), 0.9
    return None


def to_yolo(det: tuple) -> str:
    return f"{det[0]} {det[1]:.6f} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f} {det[5]:.6f}"


def nms(dets: list[tuple], iou_thresh: float = 0.5, class_agnostic: bool = True) -> list[tuple]:
    """NMS. det = (cls, xc, yc, w, h, conf).
    class_agnostic=True: all boxes compete, higher conf wins regardless of class.
    class_agnostic=False: NMS runs independently per class.
    """
    if len(dets) <= 1:
        return dets

    if class_agnostic:
        groups = {"all": list(dets)}
    else:
        groups: dict[int, list[tuple]] = {}
        for d in dets:
            groups.setdefault(d[0], []).append(d)

    keep = []
    for group_dets in groups.values():
        group_dets.sort(key=lambda d: d[5], reverse=True)
        boxes = [(d[1] - d[3] / 2, d[2] - d[4] / 2, d[1] + d[3] / 2, d[2] + d[4] / 2) for d in group_dets]
        suppressed = set()
        for i in range(len(group_dets)):
            if i in suppressed:
                continue
            keep.append(group_dets[i])
            bi = boxes[i]
            for j in range(i + 1, len(group_dets)):
                if j in suppressed:
                    continue
                bj = boxes[j]
                if iou_single(bi[0], bi[1], bi[2], bi[3], bj[0], bj[1], bj[2], bj[3]) >= iou_thresh:
                    suppressed.add(j)
    return keep


def load_yolo_fast(folder: str | Path) -> dict[str, list[tuple]]:
    """Load YOLO detections using fast line parsing."""
    dets = {}
    folder = Path(folder)
    files = list(folder.glob("*.txt"))
    for f in tqdm(files, desc=f"  Loading {folder.parent.name}", file=sys.stderr):
        stem = f.stem
        img_dets = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    parsed = parse_line(line)
                    if parsed:
                        img_dets.append(parsed)
        dets[stem] = img_dets
    return dets


def save_yolo(dets: dict[str, list[tuple]], out_dir: str | Path):
    """Save detections to YOLO txt files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img, img_dets in tqdm(dets.items(), desc=f"  Saving to {out_dir.name}", file=sys.stderr):
        text = "\n".join(to_yolo(d) for d in img_dets)
        (out_dir / f"{img}.txt").write_text(text)


def count_dets(dets: dict[str, list[tuple]]) -> int:
    return sum(len(d) for d in dets.values())


def apply_nms(dets: dict[str, list[tuple]], iou_thresh: float = 0.5, class_agnostic: bool = True) -> dict[str, list[tuple]]:
    """Apply NMS per image."""
    return {img: nms(img_dets, iou_thresh, class_agnostic) for img, img_dets in dets.items()}


def apply_conf_filter(dets: dict[str, list[tuple]], conf_thresh: float = 0.25) -> dict[str, list[tuple]]:
    """Filter detections below confidence threshold."""
    return {img: [d for d in img_dets if d[5] >= conf_thresh] for img, img_dets in dets.items()}


def get_global_max_conf(dets: dict[str, list[tuple]]) -> float:
    """Get max confidence across all images."""
    max_conf = 0.0
    for img_dets in dets.values():
        for d in img_dets:
            if d[5] > max_conf:
                max_conf = d[5]
    return max_conf


def normalize_conf(dets: dict[str, list[tuple]], global_max: float) -> dict[str, list[tuple]]:
    """Normalize confidence by dividing by global max."""
    if global_max <= 0:
        return dets
    result = {}
    for img, img_dets in dets.items():
        result[img] = [
            (d[0], d[1], d[2], d[3], d[4], d[5] / global_max)
            for d in img_dets
        ]
    return result


def main():
    src_base = Path("/data/datasets/data_miner_datasets/forklift_palletjack_v1/detections/fl_pj")
    models = ["grounding_dino", "owlvit", "sam3"]
    iou_thresh = 0.5
    conf_thresh = 0.25
    class_agnostic = True  # True: class-agnostic NMS for all, False: per-class NMS for all

    suffix = "agnostic" if class_agnostic else "per_class"
    out_base = Path(f"/data/datasets/data_miner_datasets/forklift_palletjack_v1/detections/fl_pj_{suffix}")
    print(f"Mode: {'class-agnostic' if class_agnostic else 'per-class'} NMS")
    print(f"Output: {out_base}")

    model_results = {}

    # Steps 1-3: per model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model}")
        print(f"{'='*60}")

        pred_dir = src_base / model / "pred_txt"
        dets = load_yolo_fast(pred_dir)
        print(f"  Loaded: {count_dets(dets)} dets across {len(dets)} images")

        # Step 1: Confidence filter
        dets_filtered = apply_conf_filter(dets, conf_thresh)
        conf_dir = out_base / model / "conf"
        save_yolo(dets_filtered, conf_dir)
        print(f"  After conf filter (>={conf_thresh}): {count_dets(dets_filtered)} dets -> saved to {conf_dir}")

        # Step 2: Intra-model NMS
        dets_nms = apply_nms(dets_filtered, iou_thresh, class_agnostic)
        nms_dir = out_base / model / "conf_nms"
        save_yolo(dets_nms, nms_dir)
        print(f"  After NMS (IoU={iou_thresh}): {count_dets(dets_nms)} dets -> saved to {nms_dir}")

        # Step 3: Normalize confidence
        global_max = get_global_max_conf(dets_nms)
        dets_norm = normalize_conf(dets_nms, global_max)
        norm_dir = out_base / model / "conf_nms_norm"
        save_yolo(dets_norm, norm_dir)
        print(f"  Global max conf: {global_max:.6f}")
        print(f"  After normalization: {count_dets(dets_norm)} dets -> saved to {norm_dir}")

        model_results[model] = dets_norm

    # Step 4: Cross-model NMS
    print(f"\n{'='*60}")
    print(f"Cross-model NMS ({'class-agnostic' if class_agnostic else 'per-class'})")
    print(f"{'='*60}")

    all_images = set()
    for dets in model_results.values():
        all_images.update(dets.keys())

    combined = {}
    for img in all_images:
        all_dets = []
        for dets in model_results.values():
            all_dets.extend(dets.get(img, []))
        combined[img] = all_dets

    print(f"  Combined: {count_dets(combined)} dets across {len(combined)} images")

    merged = apply_nms(combined, iou_thresh, class_agnostic)
    merged_dir = out_base / "cross_model_nms"
    save_yolo(merged, merged_dir)
    print(f"  After cross-model NMS (IoU={iou_thresh}): {count_dets(merged)} dets -> saved to {merged_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
