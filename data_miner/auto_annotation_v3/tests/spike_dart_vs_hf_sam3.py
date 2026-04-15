"""Spike: DART-native SAM3 vs HF transformers SAM3 — proposal + refine parity.

Two questions:
  1. Does DART's `Sam3MultiClassPredictorFast` (M3/M4) produce the same proposal
     boxes as HF per-class fan-out (serve_sam3.py's current proposal path)?
  2. Does DART's `SAM3InteractiveImagePredictor.predict(box=...)` (enabled via
     `build_sam3_image_model(enable_inst_interactivity=True)`) produce the same
     refined box/mask as HF `Sam3Processor(input_boxes=...)` + `Sam3Model`?

If both answer yes on fl_pj_sample, a single DART model can drive both proposal
AND refine — no HF model load needed in the unified server.

Run (both models in-process, separate GPUs — servers need NOT be running):
    python -m data_miner.auto_annotation_v3.tests.spike_dart_vs_hf_sam3

    # Pin GPUs explicitly:
    python -m data_miner.auto_annotation_v3.tests.spike_dart_vs_hf_sam3 \
        --dart-gpu 0 --hf-gpu 1

Bars:
    Proposal parity  : ≥95% match rate, min matched IoU ≥ 0.90 (looser than
                       the raw-vs-litserve 0.98 because DART's NMS differs)
    Refine parity    : mean matched IoU ≥ 0.90; report min/max for inspection

Emits tests/spike_dart_vs_hf_sam3_results.json for post-hoc analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DART_ROOT = Path("/media/data_2/vlm/code/data_miner/scratchpad/DART")
SAMPLES_DIR = Path("/media/data_2/vlm/code/data_miner/output/sample/fl_pj_sample")
RESULTS_PATH = Path(__file__).with_name("spike_dart_vs_hf_sam3_results.json")

CLASSES = ["person", "forklift", "pallet jack"]
PROPOSAL_THRESHOLD = 0.5
PROPOSAL_NMS = 0.7


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dart-gpu", type=int, default=0)
    ap.add_argument("--hf-gpu", type=int, default=1)
    ap.add_argument("--n-images", type=int, default=8, help="how many fl_pj samples")
    ap.add_argument("--detection-only", action="store_true",
                    help="run DART with detection_only=True (M4 instead of M3)")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_pair(a_boxes, a_labels, b_boxes, b_labels, iou_thr=0.5):
    """Pair boxes by max-IoU greedy within same-label. Returns (pairs, u_a, u_b).
    pairs = list of (iou, a_idx, b_idx)."""
    used = [False] * len(b_boxes)
    pairs = []
    for i, (ab, al) in enumerate(zip(a_boxes, a_labels)):
        best, best_j = 0.0, -1
        for j, (bb, bl) in enumerate(zip(b_boxes, b_labels)):
            if used[j] or al != bl:
                continue
            v = iou_xyxy(ab, bb)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > iou_thr:
            used[best_j] = True
            pairs.append((best, i, best_j))
    u_a = len(a_boxes) - len(pairs)
    u_b = sum(1 for u in used if not u)
    return pairs, u_a, u_b


def mask_to_bbox(mask_2d) -> list[float] | None:
    """mask_2d: HxW bool/0-1. Returns [x1,y1,x2,y2] in pixels, or None if empty."""
    if hasattr(mask_2d, "cpu"):
        mask_2d = mask_2d.cpu().numpy()
    ys, xs = np.where(mask_2d > 0)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# ----------------------------------------------------------------------------
# HF path
# ----------------------------------------------------------------------------

def hf_proposal_per_class(hf_model, hf_proc, image, device, dtype):
    """Replicates serve_sam3.py _to_response: per-prompt loop + aggregate."""
    import torch
    w, h = image.size
    all_boxes, all_scores, all_labels = [], [], []
    for cls in CLASSES:
        inputs = hf_proc(images=image, text=cls, return_tensors="pt").to(
            device=device, dtype=dtype
        )
        with torch.no_grad():
            outputs = hf_model(**inputs)
        post = hf_proc.post_process_instance_segmentation(
            outputs,
            threshold=PROPOSAL_THRESHOLD,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        boxes, scores = post.get("boxes"), post.get("scores")
        if boxes is None or scores is None or len(boxes) == 0:
            continue
        for bx, sc in zip(boxes, scores.cpu().tolist()):
            x1, y1, x2, y2 = [float(v) for v in bx.tolist()]
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(float(sc))
            all_labels.append(cls)
    return all_boxes, all_scores, all_labels


def hf_refine_box(hf_model, hf_proc, image, pixel_box, device, dtype):
    """Replicates serve_sam3.py _to_refine_response for a single box."""
    import torch
    inputs = hf_proc(
        images=image,
        input_boxes=[[pixel_box]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        outputs = hf_model(**inputs)
    post = hf_proc.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    boxes, scores = post.get("boxes"), post.get("scores")
    if boxes is None or scores is None or len(boxes) == 0:
        return None, 0.0, None
    scores_list = scores.cpu().tolist()
    best = int(max(range(len(scores_list)), key=lambda i: scores_list[i]))
    x1, y1, x2, y2 = [float(v) for v in boxes[best].tolist()]
    masks = post.get("masks")
    mask = masks[best] if masks is not None and len(masks) > best else None
    return [x1, y1, x2, y2], float(scores_list[best]), mask


# ----------------------------------------------------------------------------
# DART path
# ----------------------------------------------------------------------------

def dart_proposal(predictor, image):
    """Run Sam3MultiClassPredictorFast; return (boxes, scores, labels)."""
    st = predictor.set_image(image)
    res = predictor.predict(
        st, confidence_threshold=PROPOSAL_THRESHOLD, nms_threshold=PROPOSAL_NMS
    )
    if len(res["scores"]) == 0:
        return [], [], []
    boxes = res["boxes"].cpu().tolist()
    scores = res["scores"].cpu().tolist() if hasattr(res["scores"], "cpu") \
        else list(res["scores"])
    class_ids = res["class_ids"].cpu().tolist() if hasattr(res["class_ids"], "cpu") \
        else list(res["class_ids"])
    labels = [CLASSES[int(c)] for c in class_ids]
    return boxes, scores, labels


def dart_refine_box(dart_model, dart_processor, image, pixel_box):
    """Refine a box prompt via Sam3Image.predict_inst (reuses detector backbone).

    Follows DART's eval_cocoseg.py pattern:
        inference_state = processor.set_image(image)
        masks, scores, _ = model.predict_inst(inference_state, box=box[None,:], ...)
    Returns (mask_tight_box, iou_pred, mask_2d).
    """
    import torch
    with torch.inference_mode():
        inference_state = dart_processor.set_image(image)
        box_arr = np.array(pixel_box, dtype=np.float32)[None, :]  # (1, 4)
        masks, iou_preds, _low = dart_model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=box_arr,
            multimask_output=False,
        )
    # masks shape: (C, H, W) torch or numpy; C=1 with multimask_output=False.
    if isinstance(masks, torch.Tensor):
        mask = masks[0].detach().cpu().numpy() > 0
    else:
        mask = masks[0] > 0 if masks.ndim == 3 else masks > 0
    bbox = mask_to_bbox(mask)
    if isinstance(iou_preds, torch.Tensor):
        iou_preds = iou_preds.detach().cpu().numpy()
    iou_pred = float(np.asarray(iou_preds).flatten()[0]) if np.asarray(iou_preds).size else 0.0
    return bbox, iou_pred, mask


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    # Don't pin CUDA_VISIBLE_DEVICES — we need both GPUs visible. Use explicit
    # device strings instead.
    import torch
    from PIL import Image

    dart_device = f"cuda:{args.dart_gpu}"
    hf_device = f"cuda:{args.hf_gpu}"

    print(f"DART on {dart_device}, HF on {hf_device}")

    # --- DART ---
    sys.path.insert(0, str(DART_ROOT))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    print(f"Loading DART Sam3Image (enable_inst_interactivity=True) on {dart_device} …")
    t0 = time.time()
    # DART's _setup_device_and_mode only matches bare "cuda"; pass that, then
    # .to(dart_device) explicitly so sub-modules (incl. inst_interactive) move.
    dart_model = build_sam3_image_model(
        device="cuda", eval_mode=True, enable_inst_interactivity=True
    )
    dart_model = dart_model.to(dart_device)
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"inst_predictor={dart_model.inst_interactive_predictor is not None}")

    dart_predictor = Sam3MultiClassPredictorFast(
        dart_model,
        device=dart_device,
        use_fp16=True,
        presence_threshold=0.05,
        detection_only=args.detection_only,
    )
    dart_predictor.set_classes(CLASSES)
    assert dart_model.inst_interactive_predictor is not None, \
        "inst_interactive_predictor not built!"

    # DART has its own Sam3Processor (same name as HF's but different class)
    # — used to pre-compute inference_state for predict_inst.
    from sam3.model.sam3_image_processor import Sam3Processor as DartProcessor
    dart_processor = DartProcessor(dart_model, device=dart_device)

    # --- HF ---
    from transformers import Sam3Model, Sam3Processor
    print(f"Loading HF Sam3Model on {hf_device} (fp16) …")
    t0 = time.time()
    hf_proc = Sam3Processor.from_pretrained("facebook/sam3")
    hf_dtype = torch.float16
    hf_model = (
        Sam3Model.from_pretrained("facebook/sam3", torch_dtype=hf_dtype)
        .to(hf_device)
        .eval()
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # --- Load images ---
    images = sorted(SAMPLES_DIR.glob("*.jpg"))[: args.n_images]
    print(f"Running on {len(images)} images from {SAMPLES_DIR.name}")

    results = {
        "config": {
            "dart_gpu": args.dart_gpu,
            "hf_gpu": args.hf_gpu,
            "detection_only": args.detection_only,
            "n_images": len(images),
            "proposal_threshold": PROPOSAL_THRESHOLD,
            "proposal_nms": PROPOSAL_NMS,
            "classes": CLASSES,
        },
        "per_image": [],
    }

    total_proposal_pairs = []
    total_refine_pairs = []
    for idx, img_path in enumerate(images):
        stem = img_path.stem
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        print(f"\n[{idx+1}/{len(images)}] {stem}")

        # ---------- PROPOSAL ----------
        t0 = time.time()
        hf_boxes, hf_scores, hf_labels = hf_proposal_per_class(
            hf_model, hf_proc, image, hf_device, hf_dtype
        )
        t_hf_prop = time.time() - t0

        t0 = time.time()
        dart_boxes, dart_scores, dart_labels = dart_proposal(dart_predictor, image)
        t_dart_prop = time.time() - t0

        p_pairs, p_u_hf, p_u_dart = greedy_pair(
            hf_boxes, hf_labels, dart_boxes, dart_labels, iou_thr=0.5
        )
        p_mean = float(np.mean([p[0] for p in p_pairs])) if p_pairs else 0.0
        p_min = float(np.min([p[0] for p in p_pairs])) if p_pairs else 0.0
        total_proposal_pairs.extend(p[0] for p in p_pairs)

        print(f"  PROPOSAL  hf={len(hf_boxes)}({t_hf_prop:.2f}s) "
              f"dart={len(dart_boxes)}({t_dart_prop:.2f}s) "
              f"paired={len(p_pairs)} u_hf={p_u_hf} u_dart={p_u_dart} "
              f"mean_iou={p_mean:.3f} min_iou={p_min:.3f}")

        # ---------- REFINE (use HF top-1 per class as box prompts) ----------
        # Pick one box per class (the highest-score HF proposal) to refine.
        # Skips classes with zero HF proposals.
        refine_results = []
        per_class_top = {}
        for bx, sc, lb in zip(hf_boxes, hf_scores, hf_labels):
            if lb not in per_class_top or sc > per_class_top[lb][1]:
                per_class_top[lb] = (bx, sc)

        for cls, (box, _sc) in per_class_top.items():
            try:
                t0 = time.time()
                hf_rb, hf_rs, hf_rmask = hf_refine_box(
                    hf_model, hf_proc, image, box, hf_device, hf_dtype
                )
                t_hf = time.time() - t0

                t0 = time.time()
                dart_rb, dart_iou, _dart_mask = dart_refine_box(
                    dart_model, dart_processor, image, box
                )
                t_dart = time.time() - t0

                refine_iou = (
                    iou_xyxy(hf_rb, dart_rb)
                    if hf_rb is not None and dart_rb is not None
                    else 0.0
                )
                # Also compare HF-refined-box to DART-refined-box against the
                # original input box (how much each "refined" the input).
                hf_shift = (
                    iou_xyxy(box, hf_rb) if hf_rb is not None else 0.0
                )
                dart_shift = (
                    iou_xyxy(box, dart_rb) if dart_rb is not None else 0.0
                )
                refine_results.append({
                    "class": cls,
                    "input_box": box,
                    "hf_refined": hf_rb,
                    "dart_refined": dart_rb,
                    "hf_vs_dart_iou": refine_iou,
                    "hf_vs_input_iou": hf_shift,
                    "dart_vs_input_iou": dart_shift,
                    "hf_ms": round(t_hf * 1000, 1),
                    "dart_ms": round(t_dart * 1000, 1),
                })
                total_refine_pairs.append(refine_iou)
                print(f"  REFINE[{cls:<12}] hf={hf_rb} dart={dart_rb} "
                      f"IoU(hf,dart)={refine_iou:.3f} "
                      f"hf_shift_vs_input={hf_shift:.3f} "
                      f"dart_shift_vs_input={dart_shift:.3f} "
                      f"t_hf={t_hf*1000:.0f}ms t_dart={t_dart*1000:.0f}ms")
            except Exception as exc:
                print(f"  REFINE[{cls}] FAILED: {exc!r}")
                refine_results.append({"class": cls, "error": repr(exc)})

        results["per_image"].append({
            "image": stem,
            "proposal": {
                "hf_n": len(hf_boxes),
                "dart_n": len(dart_boxes),
                "paired": len(p_pairs),
                "unmatched_hf": p_u_hf,
                "unmatched_dart": p_u_dart,
                "mean_iou": p_mean,
                "min_iou": p_min,
                "hf_s": t_hf_prop,
                "dart_s": t_dart_prop,
            },
            "refine": refine_results,
        })

    # -------- Summary --------
    prop_arr = np.array(total_proposal_pairs) if total_proposal_pairs else np.array([])
    ref_arr = np.array(total_refine_pairs) if total_refine_pairs else np.array([])
    summary = {
        "proposal": {
            "n_pairs": int(prop_arr.size),
            "mean_iou": float(prop_arr.mean()) if prop_arr.size else 0.0,
            "min_iou": float(prop_arr.min()) if prop_arr.size else 0.0,
            "pct_above_0.9": float((prop_arr > 0.9).mean() * 100) if prop_arr.size else 0.0,
        },
        "refine": {
            "n_pairs": int(ref_arr.size),
            "mean_iou": float(ref_arr.mean()) if ref_arr.size else 0.0,
            "min_iou": float(ref_arr.min()) if ref_arr.size else 0.0,
            "pct_above_0.9": float((ref_arr > 0.9).mean() * 100) if ref_arr.size else 0.0,
            "pct_above_0.8": float((ref_arr > 0.8).mean() * 100) if ref_arr.size else 0.0,
        },
    }
    results["summary"] = summary
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Proposal: {summary['proposal']}")
    print(f"Refine:   {summary['refine']}")
    print(f"\nSaved → {RESULTS_PATH}")

    # Green-light conditions
    prop_ok = (
        summary["proposal"]["n_pairs"] > 0
        and summary["proposal"]["mean_iou"] >= 0.85
    )
    refine_ok = (
        summary["refine"]["n_pairs"] > 0
        and summary["refine"]["mean_iou"] >= 0.85
    )
    print(f"\nProposal parity {'✅' if prop_ok else '❌'}  "
          f"(mean_iou ≥ 0.85 on paired)")
    print(f"Refine parity   {'✅' if refine_ok else '❌'}  "
          f"(mean_iou ≥ 0.85)")
    print("→ Unified DART server viable" if (prop_ok and refine_ok)
          else "→ Keep HF refine path; use DART for proposal only")

    return 0


if __name__ == "__main__":
    sys.exit(main())
