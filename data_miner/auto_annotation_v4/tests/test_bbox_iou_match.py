"""Single benchmark: batched-vs-per-item bbox IoU match rate for GDINO and SAM3-DART.

For each model, runs the same batch through `model.infer_batch([...])` and
`[model.infer(p) for p in ...]` then pairs detections (same label, highest
IoU), and reports:
  - match rate relative to each side (percentage of detections that pair up)
  - Dice/F1 (symmetric match rate)
  - mean IoU of paired pairs

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python -m data_miner.auto_annotation_v4.tests.test_bbox_iou_match
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from PIL import Image

REPO    = Path("/media/data_2/vlm/code/data_miner")
SAMPLES = REPO / "output" / "sample" / "fl_pj_sample"

CLASSES = ["person", "forklift", "pallet jack", "box",
           "cart", "truck", "door", "shelf"]

B = 4           # batch size
IOU_THR = 0.5   # a detection pairs if same label & IoU >= this


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def _pair_ious(resp_a, resp_b) -> list[float]:
    """Greedy pair by label + best IoU. Returns IoUs of accepted pairs (IoU >= IOU_THR)."""
    used: set[int] = set()
    out: list[float] = []
    for i in range(len(resp_a.boxes)):
        best_j, best_iou = -1, -1.0
        for j in range(len(resp_b.boxes)):
            if j in used or resp_a.labels[i] != resp_b.labels[j]:
                continue
            iou = _iou(resp_a.boxes[i], resp_b.boxes[j])
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= IOU_THR:
            out.append(best_iou)
            used.add(best_j)
    return out


def _load_images() -> list[Image.Image]:
    paths = sorted(SAMPLES.glob("*.jpg"))[:B]
    assert len(paths) >= B, f"Need {B} images in {SAMPLES}, found {len(paths)}"
    return [Image.open(p).convert("RGB") for p in paths]


def benchmark(model, name: str, threshold: float) -> None:
    images = _load_images()
    prepared = [model.prepare(img, CLASSES, threshold=threshold) for img in images]

    batched  = model.infer_batch(prepared)
    per_item = [model.infer(p) for p in prepared]

    paired_ious: list[float] = []
    n_batched = n_per_item = 0
    for b_raw, p_raw in zip(batched, per_item):
        b = model.postprocess(b_raw)
        p = model.postprocess(p_raw)
        n_batched  += len(b.boxes)
        n_per_item += len(p.boxes)
        paired_ious.extend(_pair_ious(b, p))

    n_pair = len(paired_ious)
    rate_batched  = 100.0 * n_pair / n_batched  if n_batched  else 0.0
    rate_per_item = 100.0 * n_pair / n_per_item if n_per_item else 0.0
    dice = 200.0 * n_pair / (n_batched + n_per_item) if (n_batched + n_per_item) else 0.0
    mean_iou = sum(paired_ious) / n_pair if n_pair else 0.0
    n_iou99  = sum(1 for x in paired_ious if x >= 0.99)
    n_iou999 = sum(1 for x in paired_ious if x >= 0.999)

    print(f"\n── {name} ───────────────────────────────────────────────")
    print(f"  B={B}, N={len(CLASSES)}, threshold={threshold}, IoU pairing @ >= {IOU_THR}")
    print(f"  Detections — batched: {n_batched}, per-item: {n_per_item}")
    print(f"  Matched pairs:          {n_pair}")
    print(f"  Match rate (of batched):  {rate_batched:.2f}%")
    print(f"  Match rate (of per-item): {rate_per_item:.2f}%")
    print(f"  Dice/F1 (symmetric):      {dice:.2f}%")
    print(f"  Mean IoU of matched:      {mean_iou:.4f}")
    print(f"  IoU >= 0.99:   {n_iou99}/{n_pair} ({100.0*n_iou99/n_pair if n_pair else 0:.1f}%)")
    print(f"  IoU >= 0.999:  {n_iou999}/{n_pair} ({100.0*n_iou999/n_pair if n_pair else 0:.1f}%)")


def _reset_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    from data_miner.auto_annotation_v4.models.grounding_dino import GDINOModel
    from data_miner.auto_annotation_v4.models.sam3_dart import SAM3DartModel

    print("Loading GDINOModel ...")
    gdino = GDINOModel()
    gdino.load(device="cuda:0", model_id="IDEA-Research/grounding-dino-base")
    benchmark(gdino, "GDINO", threshold=0.25)
    del gdino
    _reset_cuda()

    print("\nLoading SAM3DartModel ...")
    sam3 = SAM3DartModel()
    sam3.load(device="cuda:0", model_id="sam3_dart",
              detection_only=True, presence_threshold=0.05)
    benchmark(sam3, "SAM3-DART", threshold=0.5)
    del sam3
    _reset_cuda()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
