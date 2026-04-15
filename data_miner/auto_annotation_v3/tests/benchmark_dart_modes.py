"""Benchmark DART modes on aa v3 sample images.

Loads SAM3 once, then runs each predictor configuration on the sample set.
Saves per-mode annotated images, per-image latency, detection counts, and a
bbox-from-mask vs box-head agreement check.

Run from repo root:
    python scratchpad/DART/test_aav3_modes.py
"""
import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO = Path("/media/data_2/vlm/code/data_miner")
DART = REPO / "scratchpad" / "DART"
SAMPLES = REPO / "output" / "sample" / "fl_pj_sample"
OUT = REPO / "output" / "dart_tests"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DART))

import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

sys.path.insert(0, str(DART))
from demo_multiclass import annotate_image

CLASSES = ["person", "forklift", "pallet jack"]
DEVICE = "cuda"
CONF = 0.3
NMS = 0.7
WARMUP = 1


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_images() -> List[tuple]:
    imgs = sorted(SAMPLES.glob("*.jpg"))
    return [(p.stem, Image.open(p).convert("RGB")) for p in imgs]


def run_mode(mode_name: str, predictor, images, save_dir: Path) -> Dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    predictor.set_classes(CLASSES)

    # Warmup on first image
    w_img = images[0][1]
    for _ in range(WARMUP):
        st = predictor.set_image(w_img)
        predictor.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
    sync()

    per_image = []
    for stem, img in images:
        sync(); t0 = time.perf_counter()
        st = predictor.set_image(img)
        sync(); t_bb = (time.perf_counter() - t0) * 1000

        sync(); t0 = time.perf_counter()
        res = predictor.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
        sync(); t_pred = (time.perf_counter() - t0) * 1000

        n_det = len(res["scores"])
        has_masks = res["masks"] is not None

        # Save annotated image
        if has_masks:
            ann = annotate_image(img, res, CLASSES)
        else:
            ann = annotate_image(img, res, CLASSES, mask_alpha=0.0)
        ann.save(save_dir / f"{stem}.jpg", quality=90)

        per_image.append({
            "image": stem,
            "backbone_ms": round(t_bb, 1),
            "predict_ms": round(t_pred, 1),
            "total_ms": round(t_bb + t_pred, 1),
            "n_det": n_det,
            "has_masks": has_masks,
        })

    mean_bb = np.mean([p["backbone_ms"] for p in per_image])
    mean_pr = np.mean([p["predict_ms"] for p in per_image])
    mean_tot = np.mean([p["total_ms"] for p in per_image])
    total_det = sum(p["n_det"] for p in per_image)

    return {
        "mode": mode_name,
        "mean_backbone_ms": round(mean_bb, 1),
        "mean_predict_ms": round(mean_pr, 1),
        "mean_total_ms": round(mean_tot, 1),
        "total_detections": total_det,
        "has_masks": per_image[0]["has_masks"],
        "per_image": per_image,
    }


def bbox_from_masks_check(predictor, images) -> Dict:
    """For the full-mask fast predictor, compare mask-tight bbox vs box-head bbox."""
    from torchvision.ops import box_iou
    predictor.set_classes(CLASSES)
    st = predictor.set_image(images[0][1])
    predictor.predict(st, confidence_threshold=CONF, nms_threshold=NMS)  # warmup
    sync()

    all_ious = []
    per_image = []
    for stem, img in images:
        st = predictor.set_image(img)
        res = predictor.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
        if res["masks"] is None or len(res["scores"]) == 0:
            continue
        boxes_head = res["boxes"].cpu()  # (K, 4) xyxy pixel
        masks = res["masks"].cpu()  # (K, H, W) bool
        tight_boxes = []
        for m in masks:
            ys, xs = torch.where(m)
            if len(xs) == 0:
                tight_boxes.append(torch.zeros(4))
                continue
            tight_boxes.append(torch.tensor(
                [xs.min().item(), ys.min().item(),
                 xs.max().item(), ys.max().item()], dtype=torch.float32
            ))
        tight_boxes = torch.stack(tight_boxes)
        # Per-row IoU: diagonal of box_iou
        ious = torch.diag(box_iou(boxes_head, tight_boxes))
        per_image.append({
            "image": stem,
            "n_det": len(ious),
            "mean_iou": round(ious.mean().item(), 4),
            "min_iou": round(ious.min().item(), 4),
        })
        all_ious.append(ious)

    all_ious = torch.cat(all_ious) if all_ious else torch.tensor([])
    summary = {
        "n_total_dets": len(all_ious),
        "mean_iou": round(all_ious.mean().item(), 4) if len(all_ious) else None,
        "min_iou": round(all_ious.min().item(), 4) if len(all_ious) else None,
        "pct_above_0.95": round((all_ious > 0.95).float().mean().item() * 100, 1) if len(all_ious) else None,
        "pct_above_0.90": round((all_ious > 0.90).float().mean().item() * 100, 1) if len(all_ious) else None,
        "per_image": per_image,
    }
    return summary


def main():
    print(f"Loading SAM3 model on {DEVICE} (this triggers HF download on first run)...")
    t0 = time.perf_counter()
    model = build_sam3_image_model(device=DEVICE, eval_mode=True)
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    images = load_images()
    print(f"Found {len(images)} images in {SAMPLES}")
    print(f"Classes: {CLASSES}\n")

    results = {}

    configs = [
        ("M1_baseline_seq",          lambda: Sam3MultiClassPredictor(model, device=DEVICE)),
        ("M2_baseline_seq_detonly",  lambda: Sam3MultiClassPredictor(model, device=DEVICE, detection_only=True)),
        ("M3_fast_batched_fp16",     lambda: Sam3MultiClassPredictorFast(model, device=DEVICE, use_fp16=True, presence_threshold=0.05)),
        ("M4_fast_batched_detonly",  lambda: Sam3MultiClassPredictorFast(model, device=DEVICE, use_fp16=True, presence_threshold=0.05, detection_only=True)),
        ("M5_fast_shared_encoder",   lambda: Sam3MultiClassPredictorFast(model, device=DEVICE, use_fp16=True, presence_threshold=0.05, shared_encoder=True, generic_prompt="warehouse")),
        ("M6_singlepass_cosine",     lambda: Sam3MultiClassPredictorFast(model, device=DEVICE, use_fp16=True, single_pass=True, class_method="cosine")),
        ("M7_singlepass_attention",  lambda: Sam3MultiClassPredictorFast(model, device=DEVICE, use_fp16=True, single_pass=True, class_method="attention")),
    ]

    for name, factory in configs:
        print(f"--- {name} ---")
        try:
            pred = factory()
            r = run_mode(name, pred, images, OUT / name)
            results[name] = r
            print(f"  bb={r['mean_backbone_ms']}ms  predict={r['mean_predict_ms']}ms  "
                  f"total={r['mean_total_ms']}ms  dets={r['total_detections']}  masks={r['has_masks']}")
        except Exception as e:
            print(f"  FAILED: {e!r}")
            results[name] = {"mode": name, "error": repr(e)}
        del pred
        gc.collect(); torch.cuda.empty_cache()

    # M8: torch.compile + fast batched — separate because compile is slow
    print("--- M8_fast_compile_reduceoverhead ---")
    try:
        pred = Sam3MultiClassPredictorFast(
            model, device=DEVICE, use_fp16=True, presence_threshold=0.05,
            compile_mode="reduce-overhead",
        )
        r = run_mode("M8_fast_compile_reduceoverhead", pred, images, OUT / "M8_fast_compile_reduceoverhead")
        results["M8_fast_compile_reduceoverhead"] = r
        print(f"  bb={r['mean_backbone_ms']}ms  predict={r['mean_predict_ms']}ms  total={r['mean_total_ms']}ms")
        del pred
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e!r}")
        results["M8_fast_compile_reduceoverhead"] = {"mode": "M8_fast_compile_reduceoverhead", "error": repr(e)}

    # Bbox-from-mask check using M3 config
    print("--- BBOX_FROM_MASK_CHECK (using fast batched full-mask) ---")
    try:
        pred = Sam3MultiClassPredictorFast(model, device=DEVICE, use_fp16=True, presence_threshold=0.05)
        results["bbox_from_mask"] = bbox_from_masks_check(pred, images)
        print(f"  n_dets={results['bbox_from_mask']['n_total_dets']}  "
              f"mean_iou={results['bbox_from_mask']['mean_iou']}  "
              f"min_iou={results['bbox_from_mask']['min_iou']}  "
              f">0.95: {results['bbox_from_mask']['pct_above_0.95']}%  "
              f">0.90: {results['bbox_from_mask']['pct_above_0.90']}%")
    except Exception as e:
        print(f"  FAILED: {e!r}")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT/'results.json'}")


if __name__ == "__main__":
    main()
