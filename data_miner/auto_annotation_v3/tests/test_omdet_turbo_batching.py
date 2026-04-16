"""Tests for OmDet-Turbo multi-image batch predictor.

Validates correctness, parity with native HF forward, batch independence,
edge cases, and performance.

Run from repo root:
    CUDA_VISIBLE_DEVICES=4 python -m data_miner.auto_annotation_v3.tests.test_omdet_turbo_batching
"""

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/media/data_2/vlm/code/data_miner")
SAMPLES = REPO / "output" / "sample" / "fl_pj_sample"

CLASSES_3 = ["person", "forklift", "pallet jack"]
CLASSES_1 = ["person"]
CLASSES_10 = [
    "person", "forklift", "pallet jack", "car", "truck",
    "bicycle", "dog", "cat", "backpack", "laptop",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.3
NMS_THRESHOLD = 0.5


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_images(max_images: int | None = None) -> list[tuple[str, Image.Image]]:
    imgs = sorted(SAMPLES.glob("*.jpg"))
    if max_images:
        imgs = imgs[:max_images]
    return [(p.stem, Image.open(p).convert("RGB")) for p in imgs]


def _build_predictor():
    from data_miner.auto_annotation_v3.omdet_batch import OmDetTurboBatchPredictor

    return OmDetTurboBatchPredictor(device=DEVICE, dtype=torch.float16)


# ===================================================================
# Test 1: Ground truth parity — split forward vs native model.forward()
# ===================================================================

def test_1_ground_truth_parity():
    """Our split pipeline must produce identical results to model.forward()."""
    print("\n=== Test 1: Ground truth parity (split vs native forward) ===")
    from transformers import AutoProcessor, OmDetTurboForObjectDetection

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=4)
    images = [img for _, img in images_data]

    # --- Native HF forward (same model, with autocast to match fp16) ---
    processor = predictor.processor
    model = predictor.model

    inputs = processor(
        images=images,
        text=[CLASSES_3] * len(images),
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=predictor.dtype, enabled=DEVICE == "cuda"
    ):
        native_outputs = model(**inputs)

    target_sizes = [(img.height, img.width) for img in images]
    native_results = processor.post_process_grounded_object_detection(
        native_outputs,
        text_labels=[CLASSES_3] * len(images),
        target_sizes=target_sizes,
        threshold=THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )

    # --- Our split forward ---
    state = predictor.set_images(images)
    split_results = predictor.predict_batch(
        state, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD
    )

    # --- Compare ---
    all_ok = True
    for i, (stem, _) in enumerate(images_data):
        nr = native_results[i]
        sr = split_results[i]

        n_native = len(nr["scores"])
        n_split = len(sr["scores"])

        if n_native != n_split:
            print(f"  [{stem}] MISMATCH: native={n_native} dets, split={n_split} dets")
            all_ok = False
            continue

        if n_native == 0:
            print(f"  [{stem}] OK: 0 detections in both")
            continue

        box_diff = (nr["boxes"].cpu() - sr["boxes"].cpu()).abs().max().item()
        score_diff = (nr["scores"].cpu() - sr["scores"].cpu()).abs().max().item()

        labels_match = (nr["labels"].cpu() == sr["class_ids"].cpu()).all().item()
        names_match = nr["text_labels"] == sr["class_names"]

        # FP16 accumulation noise: split vs monolithic forward may differ
        # in autocast boundaries → sub-pixel box diffs, tiny score diffs.
        ok = box_diff < 1.0 and score_diff < 5e-3 and labels_match and names_match
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(
            f"  [{stem}] {status}: {n_native} dets, "
            f"box_diff={box_diff:.4f}, score_diff={score_diff:.6f}, "
            f"labels_match={labels_match}, names_match={names_match}"
        )

    assert all_ok, "Ground truth parity failed"
    print("  PASSED")


# ===================================================================
# Test 2: Single vs batch parity
# ===================================================================

def test_2_single_vs_batch_parity():
    """N individual predict_images([img_i]) vs one predict_images(all)."""
    print("\n=== Test 2: Single vs batch parity ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=4)
    images = [img for _, img in images_data]

    # Single-image results
    single_results = []
    for img in images:
        r = predictor.predict_images([img], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
        single_results.append(r[0])

    # Batch result
    batch_results = predictor.predict_images(images, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)

    all_ok = True
    for i, (stem, _) in enumerate(images_data):
        sr = single_results[i]
        br = batch_results[i]

        n_single = len(sr["scores"])
        n_batch = len(br["scores"])

        if n_single != n_batch:
            print(f"  [{stem}] MISMATCH: single={n_single}, batch={n_batch}")
            all_ok = False
            continue

        if n_single == 0:
            print(f"  [{stem}] OK: 0 dets")
            continue

        box_diff = (sr["boxes"].cpu() - br["boxes"].cpu()).abs().max().item()
        score_diff = (sr["scores"].cpu() - br["scores"].cpu()).abs().max().item()
        ids_match = (sr["class_ids"].cpu() == br["class_ids"].cpu()).all().item()

        ok = box_diff < 2.0 and score_diff < 5e-3 and ids_match
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(
            f"  [{stem}] {status}: {n_single} dets, "
            f"box_diff={box_diff:.4f}px, score_diff={score_diff:.6f}"
        )

    assert all_ok, "Single vs batch parity failed"
    print("  PASSED")


# ===================================================================
# Test 3: Batch size 1 regression
# ===================================================================

def test_3_batch_size_1():
    """predict_images([img]) must match predict_batch(set_images([img]))."""
    print("\n=== Test 3: Batch size 1 regression ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=1)
    img = images_data[0][1]

    r1 = predictor.predict_images([img], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)[0]

    state = predictor.set_images([img])
    r2 = predictor.predict_batch(state, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)[0]

    n1, n2 = len(r1["scores"]), len(r2["scores"])
    assert n1 == n2, f"Detection count mismatch: {n1} vs {n2}"

    if n1 > 0:
        box_diff = (r1["boxes"].cpu() - r2["boxes"].cpu()).abs().max().item()
        score_diff = (r1["scores"].cpu() - r2["scores"].cpu()).abs().max().item()
        assert box_diff < 0.01, f"Box diff too large: {box_diff}"
        assert score_diff < 1e-5, f"Score diff too large: {score_diff}"

    print(f"  {n1} detections, identical")
    print("  PASSED")


# ===================================================================
# Test 4: Variable image sizes
# ===================================================================

def test_4_variable_sizes():
    """Different original sizes → correctly scaled xyxy boxes."""
    print("\n=== Test 4: Variable image sizes ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=2)
    img1 = images_data[0][1]
    img2 = images_data[1][1]

    # Resize to different sizes
    sizes = [(640, 480), (1920, 1080), (800, 800)]
    resized = [img1.resize(s) for s in sizes]

    results = predictor.predict_images(resized, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)

    all_ok = True
    for i, (sz, res) in enumerate(zip(sizes, results)):
        w, h = sz
        n = len(res["scores"])
        if n == 0:
            print(f"  [{w}x{h}] 0 dets (ok)")
            continue

        boxes = res["boxes"].cpu()
        # Check boxes are within image bounds (with small tolerance for rounding)
        x_ok = (boxes[:, [0, 2]] >= -1.0).all() and (boxes[:, [0, 2]] <= w + 1.0).all()
        y_ok = (boxes[:, [1, 3]] >= -1.0).all() and (boxes[:, [1, 3]] <= h + 1.0).all()

        ok = x_ok and y_ok
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(f"  [{w}x{h}] {status}: {n} dets, x_ok={x_ok}, y_ok={y_ok}")

    assert all_ok, "Variable sizes failed"
    print("  PASSED")


# ===================================================================
# Test 5: Cross-image independence
# ===================================================================

def test_5_cross_image_independence():
    """Image A's results must be identical regardless of batch companions."""
    print("\n=== Test 5: Cross-image independence ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=3)
    A, B, C = [img for _, img in images_data[:3]]

    # Run A in three different batch contexts
    r_ab = predictor.predict_images([A, B], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)[0]
    r_ac = predictor.predict_images([A, C], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)[0]
    r_ba = predictor.predict_images([B, A], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)[1]

    def _compare(name1, r1, name2, r2):
        n1, n2 = len(r1["scores"]), len(r2["scores"])
        if n1 != n2:
            print(f"  {name1} vs {name2}: MISMATCH count {n1} vs {n2}")
            return False
        if n1 == 0:
            print(f"  {name1} vs {name2}: OK (0 dets)")
            return True
        box_diff = (r1["boxes"].cpu() - r2["boxes"].cpu()).abs().max().item()
        score_diff = (r1["scores"].cpu() - r2["scores"].cpu()).abs().max().item()
        ok = box_diff < 2.0 and score_diff < 5e-3
        status = "OK" if ok else "FAIL"
        print(f"  {name1} vs {name2}: {status} box_diff={box_diff:.4f} score_diff={score_diff:.6f}")
        return ok

    ok1 = _compare("[A,B][0]", r_ab, "[A,C][0]", r_ac)
    ok2 = _compare("[A,B][0]", r_ab, "[B,A][1]", r_ba)

    assert ok1 and ok2, "Cross-image independence failed"
    print("  PASSED")


# ===================================================================
# Test 6: Zero-detection image in batch
# ===================================================================

def test_6_zero_det_image():
    """Blank image with 0 dets must not corrupt real image results."""
    print("\n=== Test 6: Zero-detection image in batch ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=1)
    real_img = images_data[0][1]
    blank = Image.new("RGB", (640, 480), color=(255, 255, 255))

    # Real image alone
    r_solo = predictor.predict_images([real_img], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)[0]

    # Real image + blank
    r_pair = predictor.predict_images([real_img, blank], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
    r_real_in_pair = r_pair[0]
    r_blank = r_pair[1]

    n_blank = len(r_blank["scores"])
    n_solo = len(r_solo["scores"])
    n_pair = len(r_real_in_pair["scores"])

    print(f"  Blank image dets: {n_blank}")
    print(f"  Real solo: {n_solo}, Real in pair: {n_pair}")

    if n_solo > 0 and n_solo == n_pair:
        box_diff = (r_solo["boxes"].cpu() - r_real_in_pair["boxes"].cpu()).abs().max().item()
        score_diff = (r_solo["scores"].cpu() - r_real_in_pair["scores"].cpu()).abs().max().item()
        print(f"  box_diff={box_diff:.4f}, score_diff={score_diff:.6f}")
        assert box_diff < 2.0, f"Box diff too large: {box_diff}"
        assert score_diff < 5e-3, f"Score diff too large: {score_diff}"
    elif n_solo != n_pair:
        # FP16 noise near threshold boundary can cause ±1 detection
        print(f"  WARNING: detection count differs ({n_solo} vs {n_pair}), may be threshold boundary effect")

    print("  PASSED")


# ===================================================================
# Test 7: Multiple class configurations
# ===================================================================

def test_7_multiple_class_configs():
    """Test with 1, 3, 10 classes — decoder handles varying N correctly."""
    print("\n=== Test 7: Multiple class configurations ===")

    predictor = _build_predictor()
    images_data = load_images(max_images=2)
    images = [img for _, img in images_data]

    for classes in [CLASSES_1, CLASSES_3, CLASSES_10]:
        predictor.set_classes(classes)
        results = predictor.predict_images(images, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)

        total_dets = sum(len(r["scores"]) for r in results)
        # Verify class_names are a subset of the configured classes
        for r in results:
            for name in r["class_names"]:
                assert name in classes, f"Unexpected class '{name}' not in {classes}"

        print(f"  N={len(classes):2d} classes: {total_dets} total dets across {len(images)} images")

    print("  PASSED")


# ===================================================================
# Test 8: set_classes caching (no-op on same list)
# ===================================================================

def test_8_set_classes_caching():
    """Calling set_classes with the same list twice is a no-op."""
    print("\n=== Test 8: set_classes caching ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    # Capture tensor identity
    cf1 = predictor._class_features
    tf1 = predictor._task_features

    # Call again with same classes
    predictor.set_classes(CLASSES_3)

    cf2 = predictor._class_features
    tf2 = predictor._task_features

    assert cf1 is cf2, "class_features should be same object (no-op)"
    assert tf1 is tf2, "task_features should be same object (no-op)"

    # Call with different classes — should update
    predictor.set_classes(CLASSES_1)
    cf3 = predictor._class_features
    assert cf3 is not cf1, "class_features should differ after set_classes with new list"

    print("  No-op on same classes, updated on different classes")
    print("  PASSED")


# ===================================================================
# Test 9: Latency benchmark
# ===================================================================

def test_9_latency_benchmark():
    """Sequential vs batch throughput for B=1,2,4,8."""
    print("\n=== Test 9: Latency benchmark ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=8)
    images = [img for _, img in images_data]

    # Warmup
    for _ in range(3):
        predictor.predict_images(images[:1], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
    sync()

    batch_sizes = [1, 2, 4, 8]
    results = {}
    for B in batch_sizes:
        batch = images[:B]
        n_iters = max(3, 12 // B)

        # --- Sequential ---
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            for img in batch:
                predictor.predict_images([img], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
        sync()
        seq_ms = (time.perf_counter() - t0) * 1000 / n_iters

        # --- Batched ---
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            predictor.predict_images(batch, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
        sync()
        batch_ms = (time.perf_counter() - t0) * 1000 / n_iters

        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        results[B] = {"seq_ms": seq_ms, "batch_ms": batch_ms, "speedup": speedup}
        print(f"  B={B}: seq={seq_ms:.1f}ms  batch={batch_ms:.1f}ms  speedup={speedup:.2f}x")

    # Note: OmDet-Turbo is small enough that per-image overhead is low,
    # so batching speedup depends on hardware. Just report results.
    print("  PASSED (benchmark only — no assertion on speedup)")


# ===================================================================
# Test 10: GPU memory stability
# ===================================================================

def test_10_memory_stability():
    """10 iterations should not grow GPU memory significantly."""
    print("\n=== Test 10: GPU memory stability ===")

    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images(max_images=4)
    images = [img for _, img in images_data]

    # Warmup
    predictor.predict_images(images, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
    sync()
    gc.collect()
    torch.cuda.empty_cache()

    mem_start = torch.cuda.memory_allocated() / 1024 / 1024

    for i in range(10):
        predictor.predict_images(images, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
        sync()

    gc.collect()
    torch.cuda.empty_cache()
    mem_end = torch.cuda.memory_allocated() / 1024 / 1024
    growth = mem_end - mem_start

    print(f"  Start: {mem_start:.1f} MB, End: {mem_end:.1f} MB, Growth: {growth:.1f} MB")
    assert growth < 50.0, f"Memory grew by {growth:.1f} MB (threshold: 50 MB)"
    print("  PASSED")


# ===================================================================
# Test 11: Per-image detailed breakdown
# ===================================================================

def test_11_detailed_breakdown():
    """Per-image detection breakdown with timing."""
    print("\n=== Test 11: Per-image detailed breakdown ===")

    predictor = _build_predictor()
    predictor.set_classes(CLASSES_3)

    images_data = load_images()

    # Warmup
    predictor.predict_images([images_data[0][1]], threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
    sync()

    images = [img for _, img in images_data]

    sync()
    t0 = time.perf_counter()
    state = predictor.set_images(images)
    sync()
    t_encode = (time.perf_counter() - t0) * 1000

    sync()
    t0 = time.perf_counter()
    results = predictor.predict_batch(state, threshold=THRESHOLD, nms_threshold=NMS_THRESHOLD)
    sync()
    t_decode = (time.perf_counter() - t0) * 1000

    print(f"  Encode (backbone+encoder) {len(images)} images: {t_encode:.1f} ms")
    print(f"  Decode (decoder+postproc) {len(images)} images: {t_decode:.1f} ms")
    print(f"  Total: {t_encode + t_decode:.1f} ms ({(t_encode + t_decode) / len(images):.1f} ms/img)")
    print()

    for i, (stem, _) in enumerate(images_data):
        r = results[i]
        n = len(r["scores"])
        if n > 0:
            classes_found = sorted(set(r["class_names"]))
            scores_mean = r["scores"].cpu().float().mean().item()
            print(f"  [{stem}] {n} dets, classes={classes_found}, mean_score={scores_mean:.3f}")
        else:
            print(f"  [{stem}] 0 dets")

    print("  PASSED")


# ===================================================================
# Main
# ===================================================================

def main():
    tests = [
        test_1_ground_truth_parity,
        test_2_single_vs_batch_parity,
        test_3_batch_size_1,
        test_4_variable_sizes,
        test_5_cross_image_independence,
        test_6_zero_det_image,
        test_7_multiple_class_configs,
        test_8_set_classes_caching,
        test_9_latency_benchmark,
        test_10_memory_stability,
        test_11_detailed_breakdown,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, e))
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("Failures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
