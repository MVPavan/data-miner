#!/usr/bin/env python3
"""Cross-model detection merger with size-aware NMS."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Det:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    conf: float
    model: str

    @property
    def xyxy(self):
        return (
            self.xc - self.w / 2,
            self.yc - self.h / 2,
            self.xc + self.w / 2,
            self.yc + self.h / 2,
        )

    @property
    def area(self):
        return self.w * self.h

    def to_yolo(self):
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f} {self.conf:.4f}"


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    )
    return inter / union if union > 0 else 0


def contained(inner, outer, thresh=0.85):
    """Check if inner box is mostly inside outer."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (inter / inner_area) >= thresh if inner_area > 0 else False


def size_aware_nms(dets, iou_thresh=0.5, size_ratio_thresh=0.6):
    """NMS preserving boxes with different sizes or containment."""
    if len(dets) <= 1:
        return dets

    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep, suppressed = [], set()

    for i, di in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(di)

        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            dj = dets[j]

            if iou(di.xyxy, dj.xyxy) < iou_thresh:
                continue

            ratio = (
                min(di.area, dj.area) / max(di.area, dj.area)
                if max(di.area, dj.area) > 0
                else 1
            )
            if ratio < size_ratio_thresh:
                small, large = (
                    (dj.xyxy, di.xyxy) if dj.area < di.area else (di.xyxy, dj.xyxy)
                )
                if contained(small, large):
                    continue

            suppressed.add(j)

    return keep


def load_yolo(folder, model, default_conf=0.9, conf_thresh=0.0):
    """Load YOLO detections using numpy."""
    dets = {}
    for f in Path(folder).glob("*.txt"):
        try:
            data = np.loadtxt(f, ndmin=2)
            if data.size == 0:
                dets[f.stem] = []
                continue
            # Add default conf if missing (5 cols -> 6 cols)
            if data.shape[1] == 5:
                # print(f"Adding default conf for {f}")
                data = np.hstack([data, np.full((len(data), 1), default_conf)])
            # Filter by confidence
            data = data[data[:, 5] >= conf_thresh]
            # default all classes to 0
            data[:, 0] = 0
            dets[f.stem] = [
                Det(int(r[0]), r[1], r[2], r[3], r[4], r[5], model) for r in data
            ]
        except Exception:
            dets[f.stem] = []
    return dets


def merge(model_dets, iou_thresh=0.5, size_ratio_thresh=0.6):
    """Cross-model NMS per image per class."""
    all_imgs = set().union(*[d.keys() for d in model_dets.values()])
    merged = {}

    for img in all_imgs:
        all_d = [d for m in model_dets.values() for d in m.get(img, [])]
        by_cls = {}
        for d in all_d:
            by_cls.setdefault(d.cls, []).append(d)

        merged[img] = []
        for cls_dets in by_cls.values():
            merged[img].extend(size_aware_nms(cls_dets, iou_thresh, size_ratio_thresh))

    return merged


def save_yolo(merged, out_dir, conf_thresh=0.0):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for img, dets in merged.items():
        (Path(out_dir) / f"{img}.txt").write_text(
            "\n".join(d.to_yolo() for d in dets if d.conf >= conf_thresh)
        )

    # total dets above threshold
    total_dets = sum(
        len([d for d in dets if d.conf >= conf_thresh]) for dets in merged.values()
    )
    print(f"Total dets above {conf_thresh}: {total_dets}")


if __name__ == "__main__":
    folders = {
        "moondream": (
            "output/projects/delivery_pov_v1/moondream/frames_filtered_v2_dedup/pred_txt_merged_filtered",
            0.9,
            0.3,
        ),
        "sam3": (
            "output/projects/delivery_pov_v1/sam3/frames_filtered_v2_dedup/pred_txt",
            0.9,
            0.5,
        ),
        "gdino": (
            "output/projects/delivery_pov_v1/gdino/frames_filtered_v2_dedup/pred_txt",
            0.9,
            0.3,
        ),
        "owlvit": (
            "output/projects/delivery_pov_v1/owlvit/frames_filtered_v2_dedup/pred_txt",
            0.9,
            0.3,
        ),
    }

    model_dets = {}
    for name, (path, def_conf, thresh) in folders.items():
        if Path(path).exists():
            model_dets[name] = load_yolo(path, name, def_conf, thresh)
            print(f"{name}: {sum(len(d) for d in model_dets[name].values())} dets")

    merged = merge(model_dets, iou_thresh=0.3, size_ratio_thresh=0.0001)
    print(f"Merged: {sum(len(d) for d in merged.values())} dets")
    save_yolo(
        merged,
        "output/projects/delivery_pov_v1/merged_detections_03iou_06conf/frames_filtered_v2_dedup",
        conf_thresh=0.6,
    )
