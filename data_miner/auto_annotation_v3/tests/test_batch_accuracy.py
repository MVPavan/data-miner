"""Test batching accuracy: LitServe batched (concurrent) vs LitServe sequential.

Sends 4 images concurrently to each server (triggering LitServe batching),
then runs the same images sequentially through LitServe. Compares per-image
bbox IoU and score to verify batching does not alter results.

Usage:
    # Servers must be running first:
    #   python data_miner/auto_annotation_v3/servers/launch_all.py

    # Run as a module
    python -m data_miner.auto_annotation_v3.tests.test_batch_accuracy

    # Or directly as a script
    python data_miner/auto_annotation_v3/tests/test_batch_accuracy.py
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Allow running either as a module or as a script from the repo root
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from data_miner.auto_annotation_v3.tests.compare_litserve import (
        _parse_server_response,
        bbox_iou,
    )
else:
    from .compare_litserve import _parse_server_response, bbox_iou


SAMPLE_DIR = Path("/media/data_2/vlm/code/data_miner/output/sample/fl_pj_sample")
IMAGES = sorted(SAMPLE_DIR.glob("*.jpg"))[:4]


def litserve_batched(name: str, port: int, payload_fn) -> tuple[list, list, float]:
    """Send 4 concurrent requests to the server, collect results in image order."""

    def call_one(idx: int, img_path: Path):
        t0 = time.time()
        r = requests.post(
            f"http://127.0.0.1:{port}/predict",
            json=payload_fn(str(Path(img_path).resolve())),
            timeout=120,
        )
        r.raise_for_status()
        return idx, _parse_server_response(r.json(), f"{name}_ls"), time.time() - t0

    t0_total = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(IMAGES)) as ex:
        futures = [ex.submit(call_one, i, img) for i, img in enumerate(IMAGES)]
        results: list = [None] * len(IMAGES)
        latencies: list = [None] * len(IMAGES)
        for f in concurrent.futures.as_completed(futures):
            idx, dets, lat = f.result()
            results[idx] = dets
            latencies[idx] = lat
    wall = time.time() - t0_total
    return results, latencies, wall


def litserve_sequential(name: str, port: int, payload_fn) -> tuple[list, float]:
    """Send 4 requests one after another (no concurrency — no batching)."""
    results = []
    t0_total = time.time()
    for img in IMAGES:
        r = requests.post(
            f"http://127.0.0.1:{port}/predict",
            json=payload_fn(str(Path(img).resolve())),
            timeout=120,
        )
        r.raise_for_status()
        results.append(_parse_server_response(r.json(), f"{name}_ls"))
    wall = time.time() - t0_total
    return results, wall


def compare_results(
    name: str,
    batched_results: list,
    sequential_results: list,
    iou_threshold: float = 0.5,
) -> bool:
    """Compare batched vs sequential for each image."""
    print(f"\n  Per-image comparison (batched vs sequential):")
    all_match = True
    for i, (b, s) in enumerate(zip(batched_results, sequential_results)):
        if len(b) != len(s):
            print(
                f"    [{i+1}] {IMAGES[i].name}: count mismatch  "
                f"batched={len(b)}  sequential={len(s)}"
            )
            all_match = False
            continue

        if len(b) == 0:
            print(f"    [{i+1}] {IMAGES[i].name}: both empty ✓")
            continue

        matched = 0
        total_iou = 0.0
        max_score_diff = 0.0
        used: set[int] = set()
        for bd in b:
            best_iou, best_j = 0.0, -1
            for j, sd in enumerate(s):
                if j in used:
                    continue
                iou = bbox_iou(bd["bbox"], sd["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                used.add(best_j)
                matched += 1
                total_iou += best_iou
                max_score_diff = max(
                    max_score_diff, abs(bd["score"] - s[best_j]["score"])
                )

        status = "✓" if matched == len(b) else "✗"
        mean_iou = total_iou / matched if matched else 0
        print(
            f"    [{i+1}] {IMAGES[i].name}: {matched}/{len(b)} matched  "
            f"mIoU={mean_iou:.3f}  max|Δscore|={max_score_diff:.4f} {status}"
        )
        if matched < len(b):
            all_match = False
    return all_match


def run_test(name: str, port: int, payload_fn) -> bool:
    print(f"\n{'=' * 72}")
    print(f"  {name.upper()}")
    print(f"{'=' * 72}")

    print(f"\n  → Sending {len(IMAGES)} concurrent requests (triggers batching)...")
    batched, latencies, wall_batched = litserve_batched(name, port, payload_fn)
    print(f"    Wall time: {wall_batched:.2f}s  (mean latency: {np.mean(latencies):.2f}s)")

    print(f"\n  → Sending {len(IMAGES)} sequential requests (no batching)...")
    sequential, wall_seq = litserve_sequential(name, port, payload_fn)
    print(f"    Wall time: {wall_seq:.2f}s")

    print(f"\n  → Speedup from batching: {wall_seq / wall_batched:.2f}x")

    all_match = compare_results(name, batched, sequential)
    verdict = "✅ BATCHED == SEQUENTIAL" if all_match else "❌ BATCHING CHANGES RESULTS"
    print(f"\n  Verdict: {verdict}")
    return all_match


def main() -> int:
    if not IMAGES:
        print(f"ERROR: No images found in {SAMPLE_DIR}")
        return 1

    print(f"Using {len(IMAGES)} images:")
    for i, img in enumerate(IMAGES):
        print(f"  {i+1}. {img.name}")

    results = {
        "GDINO": run_test(
            "GDINO", 3001, lambda p: {"image_path": p, "text_prompt": "person ."}
        ),
        "Falcon": run_test(
            "Falcon", 3002, lambda p: {"image_path": p, "text_prompt": "person"}
        ),
        "SAM3": run_test(
            "SAM3",
            3003,
            lambda p: {"mode": "proposal", "image_path": p, "text_prompt": "person"},
        ),
        "OWLv2": run_test(
            "OWLv2",
            3004,
            lambda p: {"image_path": p, "text_queries": ["a photo of a person"]},
        ),
    }

    print(f"\n\n{'#' * 72}")
    print(f"# OVERALL: BATCHED-vs-SEQUENTIAL ACCURACY")
    print(f"{'#' * 72}")
    for name, ok in results.items():
        print(f"  {name:<10} {'✅ EQUIVALENT' if ok else '❌ DIVERGED'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
