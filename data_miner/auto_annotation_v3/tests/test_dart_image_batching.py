"""Tests for multi-image batching in Sam3MultiClassPredictorBatch.

Validates that batched inference produces identical results to sequential
single-image inference, and benchmarks throughput gains.

Run from repo root:
    python data_miner/auto_annotation_v3/tests/test_dart_image_batching.py
"""

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

REPO = Path("/media/data_2/vlm/code/data_miner")
SAMPLES = REPO / "output" / "sample" / "fl_pj_sample"

# Add DART to sys.path so sam3 package is importable
_DART_ROOT = Path(__file__).resolve().parents[3] / "scratchpad" / "DART"
if str(_DART_ROOT) not in sys.path:
    sys.path.insert(0, str(_DART_ROOT))

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from data_miner.auto_annotation_v3.dart_batch import Sam3MultiClassPredictorBatch

DEVICE = "cuda"
CLASSES = ["person", "forklift", "pallet jack"]
CONF = 0.3
NMS = 0.7
WARMUP = 2


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_images():
    paths = sorted(SAMPLES.glob("*.jpg"))
    return [(p.stem, Image.open(p).convert("RGB")) for p in paths]


def per_det_iou(boxes_a, boxes_b):
    """Compute per-detection IoU between paired boxes (both xyxy format)."""
    from torchvision.ops import box_iou
    if len(boxes_a) == 0:
        return torch.tensor([])
    # Diagonal of the IoU matrix gives per-pair IoU
    return torch.diag(box_iou(boxes_a, boxes_b))


# ---------------------------------------------------------------
# Test 1: Parity — single-image API vs batch API
# ---------------------------------------------------------------
def test_parity_detection_only(model, images):
    """Compare single predict() vs batch predict_images() in detection_only mode."""
    print("\n=== Test 1a: Parity (detection_only=True) ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    imgs = [img for _, img in images]

    # Single-image sequential results
    single_results = []
    for img in imgs:
        state = pred.set_image(img)
        r = pred.predict(state, confidence_threshold=CONF, nms_threshold=NMS)
        single_results.append(r)

    # Batch results — process in sub-batches of 4 to manage GPU memory
    SUB_BATCH = 4
    batch_results = []
    for start in range(0, len(imgs), SUB_BATCH):
        chunk = imgs[start : start + SUB_BATCH]
        batch_results.extend(
            pred.predict_images(chunk, confidence_threshold=CONF, nms_threshold=NMS)
        )

    assert len(batch_results) == len(single_results), (
        f"Length mismatch: batch={len(batch_results)}, single={len(single_results)}"
    )

    all_pass = True
    for i, (sr, br) in enumerate(zip(single_results, batch_results)):
        stem = images[i][0]
        n_single = len(sr["scores"])
        n_batch = len(br["scores"])

        # Detection count must match
        if n_single != n_batch:
            print(f"  FAIL {stem}: n_det single={n_single} batch={n_batch}")
            all_pass = False
            continue

        if n_single == 0:
            print(f"  OK   {stem}: 0 detections (both)")
            continue

        # Sort both by score descending for stable comparison
        s_idx = sr["scores"].argsort(descending=True)
        b_idx = br["scores"].argsort(descending=True)

        s_boxes = sr["boxes"][s_idx].cpu().float()
        b_boxes = br["boxes"][b_idx].cpu().float()
        s_scores = sr["scores"][s_idx].cpu().float()
        b_scores = br["scores"][b_idx].cpu().float()
        s_cls = sr["class_ids"][s_idx].cpu()
        b_cls = br["class_ids"][b_idx].cpu()

        # FP16 batching introduces small numerical differences — use relaxed tolerances
        # Boxes: atol=2px covers sub-pixel FP16 rounding across different batch sizes
        # Scores: atol=5e-3 covers sigmoid(fp16) accumulation differences
        boxes_close = torch.allclose(s_boxes, b_boxes, atol=2.0, rtol=1e-3)
        scores_close = torch.allclose(s_scores, b_scores, atol=5e-3)
        cls_match = torch.equal(s_cls, b_cls)

        box_diff = (s_boxes - b_boxes).abs().max().item()
        score_diff = (s_scores - b_scores).abs().max().item()
        ious = per_det_iou(s_boxes, b_boxes)
        mean_iou = ious.mean().item() * 100
        min_iou = ious.min().item() * 100

        if boxes_close and scores_close and cls_match:
            print(f"  OK   {stem}: {n_single} dets, "
                  f"IoU={mean_iou:.4f}% (min={min_iou:.4f}%), "
                  f"max_box_diff={box_diff:.3f}px, max_score_diff={score_diff:.6f}")
        else:
            print(f"  FAIL {stem}: boxes_close={boxes_close}, "
                  f"scores_close={scores_close}, cls_match={cls_match}")
            print(f"       IoU={mean_iou:.4f}% (min={min_iou:.4f}%), "
                  f"max box diff: {box_diff:.3f}px, max score diff: {score_diff:.6f}")
            if not cls_match:
                mismatches = (s_cls != b_cls).sum().item()
                print(f"       class mismatches: {mismatches}/{n_single}")
            all_pass = False

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 1a {status} ---")
    return all_pass


def test_parity_with_masks(model, images):
    """Compare single vs batch with mask generation (detection_only=False)."""
    print("\n=== Test 1b: Parity (with masks, B=2) ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=False,
    )
    pred.set_classes(CLASSES)

    # Use first 2 images for mask test (masks are expensive)
    imgs = [img for _, img in images[:2]]

    single_results = []
    for img in imgs:
        state = pred.set_image(img)
        r = pred.predict(state, confidence_threshold=CONF, nms_threshold=NMS)
        single_results.append(r)

    batch_results = pred.predict_images(
        imgs, confidence_threshold=CONF, nms_threshold=NMS,
    )

    all_pass = True
    for i, (sr, br) in enumerate(zip(single_results, batch_results)):
        stem = images[i][0]
        n_single = len(sr["scores"])
        n_batch = len(br["scores"])

        if n_single != n_batch:
            print(f"  FAIL {stem}: n_det single={n_single} batch={n_batch}")
            all_pass = False
            continue

        if n_single == 0:
            print(f"  OK   {stem}: 0 detections (both)")
            continue

        # Sort by score
        s_idx = sr["scores"].argsort(descending=True)
        b_idx = br["scores"].argsort(descending=True)

        s_boxes = sr["boxes"][s_idx].cpu().float()
        b_boxes = br["boxes"][b_idx].cpu().float()
        s_scores = sr["scores"][s_idx].cpu().float()
        b_scores = br["scores"][b_idx].cpu().float()

        boxes_close = torch.allclose(s_boxes, b_boxes, atol=1.0, rtol=1e-3)
        scores_close = torch.allclose(s_scores, b_scores, atol=1e-3)

        # Check masks
        masks_ok = True
        if sr["masks"] is not None and br["masks"] is not None:
            s_masks = sr["masks"][s_idx].cpu()
            b_masks = br["masks"][b_idx].cpu()
            # Masks should be nearly identical (binary, so check exact match rate)
            match_rate = (s_masks == b_masks).float().mean().item()
            if match_rate < 0.99:
                masks_ok = False
                print(f"  WARN {stem}: mask match rate = {match_rate:.4f}")
        elif (sr["masks"] is None) != (br["masks"] is None):
            masks_ok = False

        if boxes_close and scores_close and masks_ok:
            box_diff = (s_boxes - b_boxes).abs().max().item()
            score_diff = (s_scores - b_scores).abs().max().item()
            mask_info = ""
            if sr["masks"] is not None:
                s_m = sr["masks"][s_idx].cpu()
                b_m = br["masks"][b_idx].cpu()
                mask_info = f", mask_match={(s_m == b_m).float().mean().item():.4f}"
            print(f"  OK   {stem}: {n_single} dets, "
                  f"box_diff={box_diff:.3f}px, score_diff={score_diff:.6f}{mask_info}")
        else:
            print(f"  FAIL {stem}: boxes={boxes_close}, scores={scores_close}, masks={masks_ok}")
            all_pass = False

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 1b {status} ---")
    return all_pass


# ---------------------------------------------------------------
# Test 2: Batch size 1 regression
# ---------------------------------------------------------------
def test_batch_size_one(model, images):
    """predict_images([single_img]) must match predict_image(single_img)."""
    print("\n=== Test 2: Batch size 1 regression ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    img = images[0][1]
    stem = images[0][0]

    single_result = pred.predict_image(
        img, confidence_threshold=CONF, nms_threshold=NMS,
    )
    batch_result = pred.predict_images(
        [img], confidence_threshold=CONF, nms_threshold=NMS,
    )[0]

    n_s = len(single_result["scores"])
    n_b = len(batch_result["scores"])

    if n_s != n_b:
        print(f"  FAIL: n_det single={n_s} batch={n_b}")
        return False

    if n_s == 0:
        print(f"  OK: 0 detections (both)")
        return True

    s_idx = single_result["scores"].argsort(descending=True)
    b_idx = batch_result["scores"].argsort(descending=True)

    s_boxes = single_result["boxes"][s_idx].cpu().float()
    b_boxes = batch_result["boxes"][b_idx].cpu().float()
    s_scores = single_result["scores"][s_idx].cpu().float()
    b_scores = batch_result["scores"][b_idx].cpu().float()

    boxes_ok = torch.allclose(s_boxes, b_boxes, atol=1.0, rtol=1e-3)
    scores_ok = torch.allclose(s_scores, b_scores, atol=1e-3)
    cls_ok = torch.equal(
        single_result["class_ids"][s_idx].cpu(),
        batch_result["class_ids"][b_idx].cpu(),
    )

    if boxes_ok and scores_ok and cls_ok:
        box_diff = (s_boxes - b_boxes).abs().max().item()
        score_diff = (s_scores - b_scores).abs().max().item()
        print(f"  OK   {stem}: {n_s} dets, "
              f"box_diff={box_diff:.3f}px, score_diff={score_diff:.6f}")
        print("  --- Test 2 PASSED ---")
        return True
    else:
        print(f"  FAIL: boxes={boxes_ok}, scores={scores_ok}, cls={cls_ok}")
        print("  --- Test 2 FAILED ---")
        return False


# ---------------------------------------------------------------
# Test 3: Variable image sizes
# ---------------------------------------------------------------
def test_variable_sizes(model, images):
    """Images of different original sizes should produce correctly scaled outputs."""
    print("\n=== Test 3: Variable image sizes ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    # Create images of different sizes via PIL resize
    img_orig = images[0][1]
    w0, h0 = img_orig.size

    sizes = [(640, 480), (1920, 1080), (800, 600), (w0, h0)]
    resized_imgs = []
    for w, h in sizes:
        resized_imgs.append(img_orig.resize((w, h), Image.BILINEAR))

    batch_results = pred.predict_images(
        resized_imgs, confidence_threshold=CONF, nms_threshold=NMS,
    )

    all_pass = True
    for idx, ((w, h), result) in enumerate(zip(sizes, batch_results)):
        n_det = len(result["scores"])
        if n_det == 0:
            print(f"  OK   {w}x{h}: 0 detections")
            continue

        boxes = result["boxes"].cpu().float()
        # Boxes should be within image bounds (with small tolerance for rounding)
        tol = 5.0
        x_ok = (boxes[:, 0] >= -tol).all() and (boxes[:, 2] <= w + tol).all()
        y_ok = (boxes[:, 1] >= -tol).all() and (boxes[:, 3] <= h + tol).all()

        if x_ok and y_ok:
            print(f"  OK   {w}x{h}: {n_det} dets, boxes within image bounds")
        else:
            print(f"  FAIL {w}x{h}: {n_det} dets, boxes OUT OF BOUNDS")
            print(f"       x range: [{boxes[:, 0].min():.1f}, {boxes[:, 2].max():.1f}] (expected [0, {w}])")
            print(f"       y range: [{boxes[:, 1].min():.1f}, {boxes[:, 3].max():.1f}] (expected [0, {h}])")
            all_pass = False

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 3 {status} ---")
    return all_pass


# ---------------------------------------------------------------
# Test 4: Latency benchmark
# ---------------------------------------------------------------
def test_latency(model, images):
    """Benchmark sequential vs batched throughput."""
    print("\n=== Test 4: Latency benchmark ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    img = images[0][1]

    # Warmup
    for _ in range(WARMUP):
        st = pred.set_image(img)
        pred.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
    for _ in range(WARMUP):
        pred.predict_images([img] * 4, confidence_threshold=CONF, nms_threshold=NMS)
    sync()

    # Sequential baseline
    N_RUNS = 5
    batch_sizes = [1, 2, 4, 8]

    # Single-image sequential timing
    seq_times = []
    for _ in range(N_RUNS):
        sync()
        t0 = time.perf_counter()
        st = pred.set_image(img)
        pred.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
        sync()
        seq_times.append((time.perf_counter() - t0) * 1000)
    seq_ms = np.median(seq_times)
    print(f"  Single-image sequential: {seq_ms:.1f} ms (median of {N_RUNS})")

    print(f"\n  {'B':>3} | {'Batch ms':>10} | {'Seq equiv':>10} | {'Speedup':>8} | {'Per-img':>8}")
    print(f"  {'---':>3} | {'----------':>10} | {'----------':>10} | {'--------':>8} | {'--------':>8}")

    for B in batch_sizes:
        imgs_batch = [img] * B
        batch_times = []
        for _ in range(N_RUNS):
            sync()
            t0 = time.perf_counter()
            pred.predict_images(imgs_batch, confidence_threshold=CONF, nms_threshold=NMS)
            sync()
            batch_times.append((time.perf_counter() - t0) * 1000)
        batch_ms = np.median(batch_times)
        seq_equiv = seq_ms * B
        speedup = seq_equiv / batch_ms if batch_ms > 0 else float("inf")
        per_img = batch_ms / B
        print(f"  {B:>3} | {batch_ms:>10.1f} | {seq_equiv:>10.1f} | {speedup:>7.2f}x | {per_img:>7.1f}")

    # Check that B=4 gives measurable throughput improvement (>1.1x)
    # Note: actual speedup is highly GPU-dependent. RTX 3090 may see ~6x,
    # older/smaller GPUs may see 1.2-2x due to memory bandwidth limits.
    sync()
    b4_times = []
    for _ in range(N_RUNS):
        sync()
        t0 = time.perf_counter()
        pred.predict_images([img] * 4, confidence_threshold=CONF, nms_threshold=NMS)
        sync()
        b4_times.append((time.perf_counter() - t0) * 1000)
    b4_ms = np.median(b4_times)
    seq_4 = seq_ms * 4
    speedup_4 = seq_4 / b4_ms

    passed = speedup_4 > 1.1
    print(f"\n  B=4 speedup: {speedup_4:.2f}x (threshold: >1.1x, any measurable gain)")
    status = "PASSED" if passed else "FAILED"
    print(f"  --- Test 4 {status} ---")
    return passed


# ---------------------------------------------------------------
# Test 5: Multiple class configurations
# ---------------------------------------------------------------
def test_multi_class_configs(model, images):
    """Test with different numbers of classes."""
    print("\n=== Test 5: Multiple class configurations ===")

    configs = [
        (["person"], "1 class"),
        (["person", "forklift"], "2 classes"),
        (["person", "forklift", "pallet jack"], "3 classes"),
    ]

    imgs = [img for _, img in images[:4]]
    all_pass = True

    for class_names, label in configs:
        pred = Sam3MultiClassPredictorBatch(
            model, device=DEVICE, use_fp16=True,
            presence_threshold=0.05, detection_only=True,
        )
        pred.set_classes(class_names)

        # Single sequential
        single_results = []
        for img in imgs:
            st = pred.set_image(img)
            r = pred.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
            single_results.append(r)

        # Batch
        batch_results = pred.predict_images(
            imgs, confidence_threshold=CONF, nms_threshold=NMS,
        )

        match = True
        for i, (sr, br) in enumerate(zip(single_results, batch_results)):
            if len(sr["scores"]) != len(br["scores"]):
                match = False
                break
            if len(sr["scores"]) == 0:
                continue
            s_idx = sr["scores"].argsort(descending=True)
            b_idx = br["scores"].argsort(descending=True)
            if not torch.allclose(
                sr["scores"][s_idx].cpu().float(),
                br["scores"][b_idx].cpu().float(),
                atol=1e-3,
            ):
                match = False
                break

        total_single = sum(len(r["scores"]) for r in single_results)
        total_batch = sum(len(r["scores"]) for r in batch_results)
        status = "OK" if match else "FAIL"
        print(f"  {status}  {label}: single_dets={total_single}, batch_dets={total_batch}")
        if not match:
            all_pass = False

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 5 {status} ---")
    return all_pass


# ---------------------------------------------------------------
# Test 6: Detailed per-image per-class breakdown
# ---------------------------------------------------------------
def test_detailed_breakdown(model, images):
    """Print detailed per-image per-class detection comparison."""
    print("\n=== Test 6: Detailed per-image breakdown (detection_only) ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    imgs = [img for _, img in images]

    # Warmup
    st = pred.set_image(imgs[0])
    pred.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
    pred.predict_images(imgs[:2], confidence_threshold=CONF, nms_threshold=NMS)
    sync()

    # Sequential
    single_results = []
    sync()
    t0 = time.perf_counter()
    for img in imgs:
        st = pred.set_image(img)
        r = pred.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
        single_results.append(r)
    sync()
    seq_total_ms = (time.perf_counter() - t0) * 1000

    # Batch (sub-batches of 4)
    SUB_BATCH = 4
    sync()
    t0 = time.perf_counter()
    batch_results = []
    for start in range(0, len(imgs), SUB_BATCH):
        chunk = imgs[start : start + SUB_BATCH]
        batch_results.extend(
            pred.predict_images(chunk, confidence_threshold=CONF, nms_threshold=NMS)
        )
    sync()
    batch_total_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Sequential total: {seq_total_ms:.1f} ms ({len(imgs)} images)")
    print(f"  Batch total:      {batch_total_ms:.1f} ms ({len(imgs)} images, sub-batch={SUB_BATCH})")
    print(f"  Speedup:          {seq_total_ms / batch_total_ms:.2f}x")

    print(f"\n  {'Image':<35} | {'Mode':<6} | {'#det':>4} | {'Classes'}")
    print(f"  {'-'*35} | {'-'*6} | {'-'*4} | {'-'*40}")

    all_match = True
    for i, (sr, br) in enumerate(zip(single_results, batch_results)):
        stem = images[i][0][:32]

        def class_summary(r):
            if len(r["scores"]) == 0:
                return "none"
            counts = {}
            for name in r["class_names"]:
                counts[name] = counts.get(name, 0) + 1
            return ", ".join(f"{n}:{c}" for n, c in sorted(counts.items()))

        s_summary = class_summary(sr)
        b_summary = class_summary(br)
        match = len(sr["scores"]) == len(br["scores"])
        if match and len(sr["scores"]) > 0:
            s_idx = sr["scores"].argsort(descending=True)
            b_idx = br["scores"].argsort(descending=True)
            match = torch.equal(
                sr["class_ids"][s_idx].cpu(),
                br["class_ids"][b_idx].cpu(),
            )

        mark = " " if match else "*"
        print(f" {mark}{stem:<35} | single | {len(sr['scores']):>4} | {s_summary}")
        print(f" {mark}{'':<35} | batch  | {len(br['scores']):>4} | {b_summary}")
        if not match:
            all_match = False

    if not all_match:
        print("\n  * = mismatch between single and batch")

    status = "PASSED" if all_match else "FAILED"
    print(f"\n  --- Test 6 {status} ---")
    return all_match


# ---------------------------------------------------------------
# Test 7: Cross-image independence
# ---------------------------------------------------------------
def test_cross_image_independence(model, images):
    """Image A's results must be identical regardless of batch companion."""
    print("\n=== Test 7: Cross-image independence ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    imgA = images[0][1]
    imgB = images[1][1]
    imgC = images[3][1]  # deliberately skip index 2 for diversity

    rAB = pred.predict_images([imgA, imgB], confidence_threshold=CONF, nms_threshold=NMS)
    rAC = pred.predict_images([imgA, imgC], confidence_threshold=CONF, nms_threshold=NMS)
    rBA = pred.predict_images([imgB, imgA], confidence_threshold=CONF, nms_threshold=NMS)

    all_pass = True

    # A's results from [A,B] vs [A,C]
    nAB, nAC = len(rAB[0]["scores"]), len(rAC[0]["scores"])
    if nAB != nAC:
        print(f"  FAIL: A paired with B gives {nAB} dets, A paired with C gives {nAC}")
        all_pass = False
    elif nAB > 0:
        sAB = rAB[0]["scores"].cpu().float()
        sAC = rAC[0]["scores"].cpu().float()
        bAB = rAB[0]["boxes"].cpu().float()
        bAC = rAC[0]["boxes"].cpu().float()
        score_diff = (sAB - sAC).abs().max().item()
        box_diff = (bAB - bAC).abs().max().item()
        ious = per_det_iou(bAB, bAC)
        min_iou = ious.min().item() * 100
        ok = score_diff < 1e-6 and box_diff < 0.01
        status = "OK" if ok else "FAIL"
        print(f"  {status}  A in [A,B] vs [A,C]: {nAB} dets, "
              f"score_diff={score_diff:.8f}, box_diff={box_diff:.4f}px, min_IoU={min_iou:.4f}%")
        if not ok:
            all_pass = False

    # A's results from [A,B] vs [B,A] (position 0 vs position 1)
    nAB0, nBA1 = len(rAB[0]["scores"]), len(rBA[1]["scores"])
    if nAB0 != nBA1:
        print(f"  FAIL: A at pos 0 gives {nAB0} dets, A at pos 1 gives {nBA1}")
        all_pass = False
    elif nAB0 > 0:
        sAB0 = rAB[0]["scores"].cpu().float()
        sBA1 = rBA[1]["scores"].cpu().float()
        bAB0 = rAB[0]["boxes"].cpu().float()
        bBA1 = rBA[1]["boxes"].cpu().float()
        score_diff = (sAB0 - sBA1).abs().max().item()
        box_diff = (bAB0 - bBA1).abs().max().item()
        ious = per_det_iou(bAB0, bBA1)
        min_iou = ious.min().item() * 100
        ok = score_diff < 1e-6 and box_diff < 0.01
        status = "OK" if ok else "FAIL"
        print(f"  {status}  A at pos 0 vs pos 1 (order invariance): {nAB0} dets, "
              f"score_diff={score_diff:.8f}, box_diff={box_diff:.4f}px, min_IoU={min_iou:.4f}%")
        if not ok:
            all_pass = False

    # B's results from [A,B] vs [B,A] (position 1 vs position 0)
    nAB1, nBA0 = len(rAB[1]["scores"]), len(rBA[0]["scores"])
    if nAB1 != nBA0:
        print(f"  FAIL: B at pos 1 gives {nAB1} dets, B at pos 0 gives {nBA0}")
        all_pass = False
    elif nAB1 > 0:
        sAB1 = rAB[1]["scores"].cpu().float()
        sBA0 = rBA[0]["scores"].cpu().float()
        bAB1 = rAB[1]["boxes"].cpu().float()
        bBA0 = rBA[0]["boxes"].cpu().float()
        score_diff = (sAB1 - sBA0).abs().max().item()
        box_diff = (bAB1 - bBA0).abs().max().item()
        ious = per_det_iou(bAB1, bBA0)
        min_iou = ious.min().item() * 100
        ok = score_diff < 1e-6 and box_diff < 0.01
        status = "OK" if ok else "FAIL"
        print(f"  {status}  B at pos 1 vs pos 0 (order invariance): {nAB1} dets, "
              f"score_diff={score_diff:.8f}, box_diff={box_diff:.4f}px, min_IoU={min_iou:.4f}%")
        if not ok:
            all_pass = False

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 7 {status} ---")
    return all_pass


# ---------------------------------------------------------------
# Test 8: Zero-detection image in batch
# ---------------------------------------------------------------
def test_zero_detection_in_batch(model, images):
    """A blank image producing 0 dets shouldn't corrupt a real image's results."""
    print("\n=== Test 8: Zero-detection image in batch ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    real_img = images[0][1]
    # Create a blank white image — should produce 0 detections
    blank_img = Image.new("RGB", (800, 600), color=(255, 255, 255))

    # Baseline: real image alone
    single_state = pred.set_image(real_img)
    single_result = pred.predict(
        single_state, confidence_threshold=CONF, nms_threshold=NMS,
    )

    # Batch: [blank, real]
    batch_results = pred.predict_images(
        [blank_img, real_img], confidence_threshold=CONF, nms_threshold=NMS,
    )

    blank_result = batch_results[0]
    batched_real_result = batch_results[1]

    all_pass = True

    # Blank should have 0 detections
    n_blank = len(blank_result["scores"])
    if n_blank == 0:
        print(f"  OK   blank image: 0 detections (expected)")
    else:
        print(f"  WARN blank image: {n_blank} detections (unexpected but not fatal)")

    # Real image results should match single-image baseline
    n_single = len(single_result["scores"])
    n_batched = len(batched_real_result["scores"])

    if n_single != n_batched:
        print(f"  FAIL real image: single={n_single} dets, batched={n_batched} dets")
        all_pass = False
    elif n_single > 0:
        s_idx = single_result["scores"].argsort(descending=True)
        b_idx = batched_real_result["scores"].argsort(descending=True)
        s_boxes = single_result["boxes"][s_idx].cpu().float()
        b_boxes = batched_real_result["boxes"][b_idx].cpu().float()
        s_scores = single_result["scores"][s_idx].cpu().float()
        b_scores = batched_real_result["scores"][b_idx].cpu().float()

        score_diff = (s_scores - b_scores).abs().max().item()
        ious = per_det_iou(s_boxes, b_boxes)
        min_iou = ious.min().item() * 100

        ok = torch.allclose(s_scores, b_scores, atol=5e-3) and min_iou > 99.0
        status = "OK" if ok else "FAIL"
        print(f"  {status}  real image: {n_single} dets, "
              f"min_IoU={min_iou:.4f}%, score_diff={score_diff:.6f}")
        if not ok:
            all_pass = False
    else:
        print(f"  OK   real image: 0 detections (both)")

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 8 {status} ---")
    return all_pass


# ---------------------------------------------------------------
# Test 9: Parity against parent class (ground truth baseline)
# ---------------------------------------------------------------
def test_parity_vs_parent_class(model, images):
    """Compare batch subclass against parent Sam3MultiClassPredictorFast directly."""
    print("\n=== Test 9: Parity vs parent class (ground truth) ===")

    # Parent class (ground truth)
    parent_pred = Sam3MultiClassPredictorFast(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    parent_pred.set_classes(CLASSES)

    # Batch subclass
    batch_pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    batch_pred.set_classes(CLASSES)

    imgs = [img for _, img in images[:4]]

    # Parent sequential results
    parent_results = []
    for img in imgs:
        st = parent_pred.set_image(img)
        parent_results.append(
            parent_pred.predict(st, confidence_threshold=CONF, nms_threshold=NMS)
        )

    # Batch results
    batch_results = batch_pred.predict_images(
        imgs, confidence_threshold=CONF, nms_threshold=NMS,
    )

    all_pass = True
    for i, (pr, br) in enumerate(zip(parent_results, batch_results)):
        stem = images[i][0][:32]
        n_p = len(pr["scores"])
        n_b = len(br["scores"])

        if n_p != n_b:
            print(f"  FAIL {stem}: parent={n_p} dets, batch={n_b} dets")
            all_pass = False
            continue

        if n_p == 0:
            print(f"  OK   {stem}: 0 detections (both)")
            continue

        p_idx = pr["scores"].argsort(descending=True)
        b_idx = br["scores"].argsort(descending=True)
        p_boxes = pr["boxes"][p_idx].cpu().float()
        b_boxes = br["boxes"][b_idx].cpu().float()
        p_scores = pr["scores"][p_idx].cpu().float()
        b_scores = br["scores"][b_idx].cpu().float()

        score_diff = (p_scores - b_scores).abs().max().item()
        ious = per_det_iou(p_boxes, b_boxes)
        mean_iou = ious.mean().item() * 100
        min_iou = ious.min().item() * 100
        cls_match = torch.equal(pr["class_ids"][p_idx].cpu(), br["class_ids"][b_idx].cpu())

        ok = min_iou > 99.0 and score_diff < 5e-3 and cls_match
        status = "OK" if ok else "FAIL"
        print(f"  {status}  {stem}: {n_p} dets, "
              f"IoU={mean_iou:.4f}% (min={min_iou:.4f}%), "
              f"score_diff={score_diff:.6f}, cls_match={cls_match}")
        if not ok:
            all_pass = False

    status = "PASSED" if all_pass else "FAILED"
    print(f"  --- Test 9 {status} ---")
    return all_pass


# ---------------------------------------------------------------
# Test 10: GPU memory stability
# ---------------------------------------------------------------
def test_gpu_memory_stability(model, images):
    """Repeated predict_images calls should not leak GPU memory."""
    print("\n=== Test 10: GPU memory stability ===")

    pred = Sam3MultiClassPredictorBatch(
        model, device=DEVICE, use_fp16=True,
        presence_threshold=0.05, detection_only=True,
    )
    pred.set_classes(CLASSES)

    imgs = [img for _, img in images[:2]]

    # Warmup
    pred.predict_images(imgs, confidence_threshold=CONF, nms_threshold=NMS)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    baseline_mem = torch.cuda.memory_allocated()

    N_ITERS = 10
    for _ in range(N_ITERS):
        pred.predict_images(imgs, confidence_threshold=CONF, nms_threshold=NMS)

    torch.cuda.synchronize()
    final_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    growth_mb = (final_mem - baseline_mem) / (1024 * 1024)
    peak_mb = peak_mem / (1024 * 1024)

    # Allow small growth (< 10 MB) from PyTorch allocator rounding
    passed = abs(growth_mb) < 10.0
    status = "OK" if passed else "FAIL"
    print(f"  {status}  After {N_ITERS} iterations: "
          f"memory growth={growth_mb:+.1f} MB, peak={peak_mb:.0f} MB")

    status = "PASSED" if passed else "FAILED"
    print(f"  --- Test 10 {status} ---")
    return passed


def main():
    print("Loading SAM3 model...")
    t0 = time.perf_counter()
    model = build_sam3_image_model(device=DEVICE, eval_mode=True)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

    images = load_images()
    print(f"Loaded {len(images)} images from {SAMPLES}")
    print(f"Classes: {CLASSES}")

    results = {}
    results["parity_det"] = test_parity_detection_only(model, images)
    results["parity_masks"] = test_parity_with_masks(model, images)
    results["batch_1"] = test_batch_size_one(model, images)
    results["var_sizes"] = test_variable_sizes(model, images)
    results["multi_class"] = test_multi_class_configs(model, images)
    results["detailed"] = test_detailed_breakdown(model, images)
    results["cross_image"] = test_cross_image_independence(model, images)
    results["zero_det"] = test_zero_detection_in_batch(model, images)
    results["vs_parent"] = test_parity_vs_parent_class(model, images)
    results["mem_stable"] = test_gpu_memory_stability(model, images)
    results["latency"] = test_latency(model, images)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<20}: {status}")

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")

    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
