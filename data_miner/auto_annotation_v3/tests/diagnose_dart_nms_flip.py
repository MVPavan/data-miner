"""Inspect raw detections on 9_fPtt5zRpA_002730.jpg for M1-M4 to find the flip."""
import sys
from pathlib import Path
REPO = Path("/media/data_2/vlm/code/data_miner")
sys.path.insert(0, str(REPO / "scratchpad" / "DART"))

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

CLASSES = ["person", "forklift", "pallet jack"]
IMG = REPO / "output/sample/fl_pj_sample/9_fPtt5zRpA_002730.jpg"

model = build_sam3_image_model(device="cuda", eval_mode=True)
img = Image.open(IMG).convert("RGB")

configs = {
    "M1_baseline_masks":  Sam3MultiClassPredictor(model, device="cuda"),
    "M2_baseline_detonly": Sam3MultiClassPredictor(model, device="cuda", detection_only=True),
    "M3_fast_masks":       Sam3MultiClassPredictorFast(model, device="cuda", use_fp16=True, presence_threshold=0.05),
    "M4_fast_detonly":     Sam3MultiClassPredictorFast(model, device="cuda", use_fp16=True, presence_threshold=0.05, detection_only=True),
}

per_mode = {}
for name, pred in configs.items():
    pred.set_classes(CLASSES)
    st = pred.set_image(img)
    res = pred.predict(st, confidence_threshold=0.3, nms_threshold=0.7)
    dets = []
    for i in range(len(res["scores"])):
        b = res["boxes"][i].tolist()
        dets.append({
            "cls": res["class_names"][i],
            "score": round(res["scores"][i].item(), 3),
            "box": [round(x, 1) for x in b],
        })
    # sort by box center for stable comparison
    dets.sort(key=lambda d: ((d["box"][0]+d["box"][2])/2, (d["box"][1]+d["box"][3])/2))
    per_mode[name] = dets
    print(f"\n{name}: {len(dets)} detections")
    for d in dets:
        print(f"  {d['cls']:14s} score={d['score']:.3f}  box={d['box']}")

# Pairwise IoU to find flips
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-6)

print("\n--- Potential flips (IoU>0.5 between M1-group and M2-group with different class) ---")
for d1 in per_mode["M1_baseline_masks"]:
    for d2 in per_mode["M2_baseline_detonly"]:
        if iou(d1["box"], d2["box"]) > 0.5 and d1["cls"] != d2["cls"]:
            print(f"  M1 {d1['cls']:12s} {d1['score']:.3f} {d1['box']}  <-->  "
                  f"M2 {d2['cls']:12s} {d2['score']:.3f} {d2['box']}  IoU={iou(d1['box'], d2['box']):.3f}")

print("\n--- M3 vs M4 same check ---")
for d1 in per_mode["M3_fast_masks"]:
    for d2 in per_mode["M4_fast_detonly"]:
        if iou(d1["box"], d2["box"]) > 0.5 and d1["cls"] != d2["cls"]:
            print(f"  M3 {d1['cls']:12s} {d1['score']:.3f} {d1['box']}  <-->  "
                  f"M4 {d2['cls']:12s} {d2['score']:.3f} {d2['box']}  IoU={iou(d1['box'], d2['box']):.3f}")
