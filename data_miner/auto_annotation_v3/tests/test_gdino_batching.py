"""Tests for GDINOBatchPredictor — batched N-prompt GroundingDINO inference.

Validates:
  1. Batched vs sequential parity (same boxes/scores for identical inputs)
  2. Single-prompt regression (N=1 works correctly)
  3. Variable prompt counts (1, 3, 8 prompts)
  4. Cross-image independence (image A's results unchanged by neighbors)
  5. Zero-detection (blank image doesn't crash or corrupt batch)
  6. Multi-image batch parity (predict_images vs per-image predict)
  7. Variable image sizes (different aspect ratios in multi-image batch)
  8. Latency benchmark (sequential vs batched throughput)
  9. Memory stability (no leaks over repeated runs)

Run from repo root:
    CUDA_VISIBLE_DEVICES=0 python -m data_miner.auto_annotation_v3.tests.test_gdino_batching
"""

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path("/media/data_2/vlm/code/data_miner")
SAMPLES = REPO / "output" / "sample" / "fl_pj_sample"

CLASSES_3 = ["person", "forklift", "pallet jack"]
CLASSES_1 = ["person"]
CLASSES_8 = [
    "person", "forklift", "pallet jack", "car",
    "truck", "bicycle", "dog", "cat",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Tolerance for floating-point comparison (batched vs sequential may differ
# slightly due to padding-induced numerical differences in softmax/layernorm)
SCORE_ATOL = 0.02
BOX_ATOL = 2.0  # pixels


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_images(max_images: int | None = None) -> list[tuple[str, Image.Image]]:
    imgs = sorted(SAMPLES.glob("*.jpg"))
    if max_images:
        imgs = imgs[:max_images]
    return [(p.stem, Image.open(p).convert("RGB")) for p in imgs]


_PREDICTOR = None

def _build_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from data_miner.auto_annotation_v3.gdino_batch import GDINOBatchPredictor
        _PREDICTOR = GDINOBatchPredictor(device=DEVICE)
    return _PREDICTOR


def _compare_results(
    seq_results: list[dict],
    bat_results: list[dict],
    label: str,
    score_atol: float = SCORE_ATOL,
    box_atol: float = BOX_ATOL,
) -> bool:
    """Compare sequential vs batched results, allowing float tolerance.

    Returns True if results match within tolerance.

    STRICT: detection count mismatches always FAIL (no silent tolerance).
    Prompt identity is checked — result[i] must belong to the same prompt in both lists.
    """
    ok = True
    if len(seq_results) != len(bat_results):
        print(f"  FAIL [{label}]: result count mismatch: seq={len(seq_results)} bat={len(bat_results)}")
        return False

    for i, (sr, br) in enumerate(zip(seq_results, bat_results)):
        # Verify prompt identity — result at index i must be for the same prompt
        s_prompt = sr.get("prompt", f"prompt_{i}")
        b_prompt = br.get("prompt", f"prompt_{i}")
        if s_prompt != b_prompt:
            print(f"  FAIL [{label}]: prompt mismatch at idx {i}: seq='{s_prompt}' bat='{b_prompt}'")
            ok = False
            continue

        prompt = s_prompt
        s_boxes = sr["boxes"]
        b_boxes = br["boxes"]
        s_scores = sr["scores"]
        b_scores = br["scores"]

        s_n = len(s_scores) if not torch.is_tensor(s_scores) else s_scores.shape[0]
        b_n = len(b_scores) if not torch.is_tensor(b_scores) else b_scores.shape[0]

        # STRICT: any detection count mismatch is a failure
        if s_n != b_n:
            print(f"  FAIL [{label}/{prompt}]: detection count mismatch: seq={s_n} bat={b_n}")
            ok = False
            continue

        if s_n == 0:
            continue

        # Compare scores
        s_sc = s_scores if torch.is_tensor(s_scores) else torch.tensor(s_scores)
        b_sc = b_scores if torch.is_tensor(b_scores) else torch.tensor(b_scores)
        score_diff = (s_sc.float() - b_sc.float()).abs().max().item()

        # Compare boxes
        s_bx = s_boxes if torch.is_tensor(s_boxes) else torch.tensor(s_boxes)
        b_bx = b_boxes if torch.is_tensor(b_boxes) else torch.tensor(b_boxes)
        box_diff = (s_bx.float() - b_bx.float()).abs().max().item()

        if score_diff > score_atol:
            print(f"  FAIL [{label}/{prompt}]: score diff={score_diff:.4f} > atol={score_atol}")
            ok = False
        if box_diff > box_atol:
            print(f"  FAIL [{label}/{prompt}]: box diff={box_diff:.2f}px > atol={box_atol}")
            ok = False

    return ok


# ===================================================================
# Test 1: Batched vs Sequential parity
# ===================================================================

def test_1_batched_vs_sequential_parity():
    """Batched predict() must produce same results as predict_sequential()."""
    print("\n=== Test 1: Batched vs Sequential parity ===")
    predictor = _build_predictor()

    images_data = load_images(max_images=4)
    passed = 0
    total = 0

    for name, image in images_data:
        seq_results = predictor.predict_sequential(
            image, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )
        bat_results = predictor.predict(
            image, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )

        total += 1
        label = f"img={name}"
        if _compare_results(seq_results, bat_results, label):
            passed += 1
            print(f"  OK [{label}]: {sum(len(r['scores']) if not torch.is_tensor(r['scores']) else r['scores'].shape[0] for r in seq_results)} detections match")
        else:
            # Print details for debugging
            for i, (sr, br) in enumerate(zip(seq_results, bat_results)):
                s_n = len(sr["scores"]) if not torch.is_tensor(sr["scores"]) else sr["scores"].shape[0]
                b_n = len(br["scores"]) if not torch.is_tensor(br["scores"]) else br["scores"].shape[0]
                print(f"    prompt={sr['prompt']}: seq={s_n} bat={b_n} detections")

    print(f"\n  Result: {passed}/{total} images passed")
    return passed == total


# ===================================================================
# Test 2: Single prompt regression (N=1)
# ===================================================================

def test_2_single_prompt():
    """N=1 should work identically to sequential with one prompt."""
    print("\n=== Test 2: Single prompt (N=1) ===")
    predictor = _build_predictor()

    images_data = load_images(max_images=3)
    passed = 0
    total = 0

    for name, image in images_data:
        seq_results = predictor.predict_sequential(
            image, CLASSES_1, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )
        bat_results = predictor.predict(
            image, CLASSES_1, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )

        total += 1
        if _compare_results(seq_results, bat_results, f"img={name}"):
            s_n = len(seq_results[0]["scores"]) if not torch.is_tensor(seq_results[0]["scores"]) else seq_results[0]["scores"].shape[0]
            print(f"  OK [img={name}]: {s_n} detections match")
            passed += 1

    print(f"\n  Result: {passed}/{total} images passed")
    return passed == total


# ===================================================================
# Test 3: Variable prompt counts
# ===================================================================

def test_3_variable_prompt_counts():
    """Test with 1, 3, and 8 prompts."""
    print("\n=== Test 3: Variable prompt counts (1, 3, 8) ===")
    predictor = _build_predictor()

    image = load_images(max_images=1)[0][1]
    passed = 0

    for prompt_list in [CLASSES_1, CLASSES_3, CLASSES_8]:
        n = len(prompt_list)
        seq_results = predictor.predict_sequential(
            image, prompt_list, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )
        bat_results = predictor.predict(
            image, prompt_list, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )

        if _compare_results(seq_results, bat_results, f"N={n}"):
            total_det = sum(
                len(r["scores"]) if not torch.is_tensor(r["scores"]) else r["scores"].shape[0]
                for r in bat_results
            )
            print(f"  OK [N={n}]: {total_det} total detections match")
            passed += 1

    print(f"\n  Result: {passed}/3 prompt configs passed")
    return passed == 3


# ===================================================================
# Test 4: Cross-image independence
# ===================================================================

def test_4_cross_image_independence():
    """predict_images() must return identical results for image A regardless
    of which other images accompany it in the list.

    Since predict_images processes each image independently (a simple loop
    over predict), this test verifies the loop preserves per-image state
    correctly — e.g. no residual state in the model between calls, no
    torch cache corruption, no prompt/image mismatch.
    """
    print("\n=== Test 4: Cross-image independence ===")
    predictor = _build_predictor()

    images_data = load_images(max_images=4)
    if len(images_data) < 3:
        print("  SKIP: need >= 3 images")
        return True

    img_a = images_data[0][1]
    img_b = images_data[1][1]
    img_c = images_data[2][1]

    # A alone
    results_a_solo = predictor.predict(
        img_a, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    # A with B (multi-image list)
    results_ab = predictor.predict_images(
        [img_a, img_b], CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    # A with C (multi-image list)
    results_ac = predictor.predict_images(
        [img_a, img_c], CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    ok1 = _compare_results(results_a_solo, results_ab[0], "A_solo vs A_in_AB")
    ok2 = _compare_results(results_a_solo, results_ac[0], "A_solo vs A_in_AC")
    ok3 = _compare_results(results_ab[0], results_ac[0], "A_in_AB vs A_in_AC")

    passed = ok1 and ok2 and ok3
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# ===================================================================
# Test 5: Zero-detection / blank image
# ===================================================================

def test_5_blank_image():
    """Blank image should produce few/zero detections without crashing."""
    print("\n=== Test 5: Blank image (zero detections) ===")
    predictor = _build_predictor()

    blank = Image.new("RGB", (640, 480), color=(128, 128, 128))
    real_image = load_images(max_images=1)[0][1]

    # Blank alone — batched predict
    results_blank = predictor.predict(
        blank, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )
    total_blank = sum(
        len(r["scores"]) if not torch.is_tensor(r["scores"]) else r["scores"].shape[0]
        for r in results_blank
    )
    print(f"  Blank (batched predict): {total_blank} detections")

    # Blank — sequential baseline
    results_blank_seq = predictor.predict_sequential(
        blank, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )
    ok1 = _compare_results(results_blank_seq, results_blank, "blank_seq vs blank_bat")

    # Real + blank in predict_images (independent processing)
    results_mixed = predictor.predict_images(
        [real_image, blank], CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    # Real image results must match solo predict
    results_real_solo = predictor.predict(
        real_image, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )
    ok2 = _compare_results(results_real_solo, results_mixed[0], "real_solo vs real_in_list")

    total_blank_in_list = sum(
        len(r["scores"]) if not torch.is_tensor(r["scores"]) else r["scores"].shape[0]
        for r in results_mixed[1]
    )
    print(f"  Blank in list: {total_blank_in_list} detections")

    ok = ok1 and ok2
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 6: Multi-image batch parity
# ===================================================================

def test_6_multi_image_parity():
    """predict_images([A,B,C]) must match individual predict() calls exactly."""
    print("\n=== Test 6: predict_images vs individual predict ===")
    predictor = _build_predictor()

    images_data = load_images(max_images=4)
    images = [img for _, img in images_data]

    # Individual predictions
    individual = []
    for img in images:
        r = predictor.predict(
            img, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )
        individual.append(r)

    # predict_images (processes each image independently with N-prompt batching)
    batch_results = predictor.predict_images(
        images, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    passed = 0
    for i, (name, _) in enumerate(images_data):
        # Exact match expected since predict_images calls predict() per image
        if _compare_results(individual[i], batch_results[i], f"img={name}", score_atol=0.0, box_atol=0.0):
            total_det = sum(
                len(r["scores"]) if not torch.is_tensor(r["scores"]) else r["scores"].shape[0]
                for r in individual[i]
            )
            print(f"  OK [img={name}]: {total_det} detections match (exact)")
            passed += 1

    print(f"\n  Result: {passed}/{len(images)} images passed")
    return passed == len(images)


# ===================================================================
# Test 7: Variable image sizes
# ===================================================================

def test_7_variable_image_sizes():
    """Different aspect ratio images in predict_images."""
    print("\n=== Test 7: Variable image sizes ===")
    predictor = _build_predictor()

    # Create images of different sizes
    real_image = load_images(max_images=1)[0][1]
    landscape = real_image.resize((800, 600))
    portrait = real_image.resize((400, 700))
    square = real_image.resize((500, 500))

    images = [landscape, portrait, square]
    sizes = [(img.width, img.height) for img in images]
    print(f"  Image sizes: {sizes}")

    # Individual
    individual = []
    for img in images:
        r = predictor.predict(
            img, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
        )
        individual.append(r)

    # predict_images (per-image processing, exact match expected)
    batch_results = predictor.predict_images(
        images, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    passed = 0
    labels = ["landscape", "portrait", "square"]
    for i, label in enumerate(labels):
        if _compare_results(individual[i], batch_results[i], label, score_atol=0.0, box_atol=0.0):
            total_det = sum(
                len(r["scores"]) if not torch.is_tensor(r["scores"]) else r["scores"].shape[0]
                for r in individual[i]
            )
            print(f"  OK [{label} {sizes[i]}]: {total_det} detections match (exact)")
            passed += 1

    print(f"\n  Result: {passed}/3 sizes passed")
    return passed == 3


# ===================================================================
# Test 8: Latency benchmark
# ===================================================================

def test_8_latency_benchmark():
    """Compare throughput: sequential (N passes) vs batched (1 pass) per image."""
    print("\n=== Test 8: Latency benchmark ===")
    predictor = _build_predictor()

    images_data = load_images(max_images=4)
    images = [img for _, img in images_data]
    warmup = 2
    trials = 5

    # Test with increasing prompt counts to show scaling
    for prompts, label in [(CLASSES_3, "3 prompts"), (CLASSES_8, "8 prompts")]:
        N = len(prompts)
        B = len(images)

        print(f"\n  --- {label} ({B} images) ---")
        print(f"  Sequential: {B} images × {N} passes = {B * N} forward passes")
        print(f"  Batched:    {B} images × 1 pass (batch={N}) = {B} forward passes")

        # Warmup
        for _ in range(warmup):
            predictor.predict(images[0], prompts, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD)
            sync()

        # Sequential
        sync()
        t0 = time.perf_counter()
        for _ in range(trials):
            for img in images:
                predictor.predict_sequential(img, prompts, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD)
            sync()
        seq_time = (time.perf_counter() - t0) / trials
        seq_per_image = seq_time / B

        # Batched
        sync()
        t0 = time.perf_counter()
        for _ in range(trials):
            for img in images:
                predictor.predict(img, prompts, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD)
            sync()
        bat_time = (time.perf_counter() - t0) / trials
        bat_per_image = bat_time / B

        speedup = seq_per_image / bat_per_image
        print(f"  Sequential:  {seq_time*1000:7.1f} ms total, {seq_per_image*1000:7.1f} ms/image")
        print(f"  Batched:     {bat_time*1000:7.1f} ms total, {bat_per_image*1000:7.1f} ms/image")
        print(f"  Speedup:     {speedup:.2f}x")

        # Assert batched is actually faster (minimum 1.1x — anything less
        # suggests the batching is broken or there's serious overhead).
        if speedup < 1.1:
            print(f"  FAIL [{label}]: speedup {speedup:.2f}x < 1.1x threshold")
            return False

    return True


# ===================================================================
# Test 9: Memory stability
# ===================================================================

def test_9_memory_stability():
    """Run 10 iterations and check GPU memory doesn't grow significantly."""
    print("\n=== Test 9: Memory stability ===")
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA")
        return True

    predictor = _build_predictor()
    image = load_images(max_images=1)[0][1]

    # Warmup
    for _ in range(3):
        predictor.predict(image, CLASSES_3, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD)
    sync()
    gc.collect()
    torch.cuda.empty_cache()

    mem_start = torch.cuda.memory_allocated()

    for i in range(10):
        predictor.predict(image, CLASSES_8, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD)
        sync()

    gc.collect()
    torch.cuda.empty_cache()
    mem_end = torch.cuda.memory_allocated()
    growth = (mem_end - mem_start) / 1024 / 1024

    print(f"  Start: {mem_start / 1024 / 1024:.1f} MB")
    print(f"  End:   {mem_end / 1024 / 1024:.1f} MB")
    print(f"  Growth: {growth:.1f} MB")

    ok = growth < 50  # allow up to 50 MB growth
    print(f"\n  Result: {'PASS' if ok else 'FAIL'} (growth={growth:.1f} MB)")
    return ok


# ===================================================================
# Test 10a: Prompt-to-slot mapping validation
# ===================================================================

def test_10a_prompt_slot_mapping():
    """Verify that results[i] actually detects prompt[i], not some other prompt.

    Uses semantically distinct prompts and checks that the decoded token labels
    from the model output actually match the prompt text. This catches bugs
    where prompts could be silently permuted or assigned to wrong slots.
    """
    print("\n=== Test 10a: Prompt-to-slot mapping ===")
    predictor = _build_predictor()

    # Use distinct prompts with unique tokens
    image = load_images(max_images=1)[0][1]
    prompts = ["person", "forklift", "pallet jack"]

    results = predictor.predict(
        image, prompts, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    ok = True
    for i, result in enumerate(results):
        # 1. Verify the prompt field echoes the input prompt at position i
        if result["prompt"] != prompts[i]:
            print(f"  FAIL: result[{i}].prompt='{result['prompt']}' != prompts[{i}]='{prompts[i]}'")
            ok = False
            continue

        # 2. Verify the decoded labels contain tokens from the corresponding prompt
        #    (the model's text_labels are decoded from input_ids at slot i)
        labels = result["labels"]
        n = len(labels)
        if n == 0:
            print(f"  skip [{prompts[i]}]: no detections to check")
            continue

        # The decoded label should contain words from the prompt
        prompt_words = set(prompts[i].lower().split())
        mismatches = []
        for label in labels:
            label_str = str(label).lower()
            # At least one prompt word should appear in the decoded label
            if not any(word in label_str for word in prompt_words):
                mismatches.append(label)

        if mismatches:
            print(f"  FAIL [{prompts[i]}]: decoded labels don't contain prompt words: {mismatches[:3]}")
            ok = False
        else:
            print(f"  OK [{prompts[i]}]: all {n} labels map correctly to prompt")

    # 3. Cross-check: shuffle prompts and verify batched result order follows shuffled order
    shuffled = list(reversed(prompts))
    shuffled_results = predictor.predict(
        image, shuffled, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    for i, result in enumerate(shuffled_results):
        if result["prompt"] != shuffled[i]:
            print(f"  FAIL (shuffle): result[{i}].prompt='{result['prompt']}' != shuffled[{i}]='{shuffled[i]}'")
            ok = False

    if ok:
        print("  OK: prompt order preserved under shuffle")

    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 10b: Tokenization padding (long & short prompts mixed)
# ===================================================================

def test_10b_tokenization_padding():
    """Verify correctness when prompts have very different tokenized lengths.

    The batched tokenizer pads all N prompts to the longest in the batch,
    while sequential tokenizer pads each independently. This test ensures
    the batched version still matches sequential results for prompts of
    very different lengths (which force max-length padding behavior).
    """
    print("\n=== Test 10b: Tokenization padding with mixed-length prompts ===")
    predictor = _build_predictor()

    image = load_images(max_images=1)[0][1]

    # Mix short and long prompts to force different padding lengths
    prompts = [
        "car",  # 1 word
        "person",  # 1 word
        "a yellow industrial forklift with two prongs used in warehouses",  # 11 words
        "pallet jack",  # 2 words
        "a person wearing a safety vest and a hard hat on a work site",  # 15 words
    ]

    seq_results = predictor.predict_sequential(
        image, prompts, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )
    bat_results = predictor.predict(
        image, prompts, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD
    )

    ok = _compare_results(seq_results, bat_results, "mixed-length")
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 10: Empty prompts
# ===================================================================

def test_10_empty_prompts():
    """Empty prompts list should return empty results without crashing."""
    print("\n=== Test 10: Empty prompts ===")
    predictor = _build_predictor()
    image = load_images(max_images=1)[0][1]

    results = predictor.predict(image, [], threshold=THRESHOLD)
    ok = results == []
    print(f"  predict([], ...): {results}")

    results_multi = predictor.predict_images([image], [], threshold=THRESHOLD)
    ok2 = results_multi == [[]]
    print(f"  predict_images([img], []): {results_multi}")

    print(f"\n  Result: {'PASS' if ok and ok2 else 'FAIL'}")
    return ok and ok2


# ===================================================================
# Main
# ===================================================================

def main():
    tests = [
        ("1. Batched vs Sequential parity", test_1_batched_vs_sequential_parity),
        ("2. Single prompt (N=1)", test_2_single_prompt),
        ("3. Variable prompt counts", test_3_variable_prompt_counts),
        ("4. Cross-image independence", test_4_cross_image_independence),
        ("5. Blank image", test_5_blank_image),
        ("6. Multi-image batch parity", test_6_multi_image_parity),
        ("7. Variable image sizes", test_7_variable_image_sizes),
        ("8. Latency benchmark", test_8_latency_benchmark),
        ("9. Memory stability", test_9_memory_stability),
        ("10a. Prompt-to-slot mapping", test_10a_prompt_slot_mapping),
        ("10b. Tokenization padding", test_10b_tokenization_padding),
        ("10. Empty prompts", test_10_empty_prompts),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        indicator = "✓" if status == "PASS" else "✗"
        print(f"  {indicator} {name}: {status}")

    failures = sum(1 for _, s in results if s != "PASS")
    print(f"\n  {len(results) - failures}/{len(results)} passed")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
