"""Full proposal-phase parity matrix: HF sam3 vs DART M3 vs DART M4.

Three variants served simultaneously on different ports; for each scenario we
POST the same payload to each and compare pairwise IoU:

  Scenarios
  ---------
    single-class   : prompts = ["person"] — per-class attention ceiling.
    multi-class    : prompts = ["person", "forklift", "pallet jack"].
    batch          : same as multi-class, but all images sent CONCURRENTLY
                     (asyncio.gather) to stress LitServe batching.

  Variants
  --------
    hf             : serve_sam3.py (HF transformers, per-class loop).
    dart_m4        : serve_sam3_dart.py with detection_only=True (box-NMS).
    dart_m3        : serve_sam3_dart.py with detection_only=False (mask-NMS).

Expected ports (override with --hf/--m4/--m3):
    HF on 3003, DART M4 on 3013, DART M3 on 3014.

Prereqs (from repo root):
    CUDA_VISIBLE_DEVICES=1 python -m data_miner.auto_annotation_v3.servers.serve_sam3 \
        --port 3003 --device 0 --max-batch-size 8 --batch-timeout 0.05 &
    CUDA_VISIBLE_DEVICES=0 python -m data_miner.auto_annotation_v3.servers.serve_sam3_dart \
        --port 3013 --device 0 --detection-only true  &
    CUDA_VISIBLE_DEVICES=4 python -m data_miner.auto_annotation_v3.servers.serve_sam3_dart \
        --port 3014 --device 0 --detection-only false &

Run:
    python -m data_miner.auto_annotation_v3.tests.test_sam3_full_matrix
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

import aiohttp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_miner.auto_annotation_v3.contracts import DetectorRequest, DetectorResponse

SAMPLES = Path("/media/data_2/vlm/code/data_miner/output/sample/fl_pj_sample")
RESULTS = Path(__file__).with_name("test_sam3_full_matrix_results.json")

CLASSES_MULTI = ["person", "forklift", "pallet jack"]
CLASS_SINGLE = ["person"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pair_by_label(a_boxes, a_labels, b_boxes, b_labels, iou_thr=0.5):
    used = [False] * len(b_boxes)
    pairs = []
    for ab, al in zip(a_boxes, a_labels):
        best, best_j = 0.0, -1
        for j, (bb, bl) in enumerate(zip(b_boxes, b_labels)):
            if used[j] or al != bl:
                continue
            v = iou(ab, bb)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > iou_thr:
            used[best_j] = True
            pairs.append(best)
    return pairs, len(a_boxes) - len(pairs), sum(1 for u in used if not u)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

async def call(session, port, image_path, prompts, threshold=0.5):
    req = DetectorRequest(image_path=str(image_path), prompts=prompts, threshold=threshold)
    t0 = time.time()
    async with session.post(
        f"http://localhost:{port}/predict", json=req.model_dump()
    ) as r:
        r.raise_for_status()
        data = await r.json()
    return DetectorResponse.model_validate(data), time.time() - t0


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

async def run_sequential(session, ports, images, prompts, label):
    """Query each server sequentially per image, collect responses + latency."""
    per_variant: dict[str, list] = {v: [] for v in ports}
    latencies: dict[str, list[float]] = {v: [] for v in ports}
    for img in images:
        for v, port in ports.items():
            resp, t = await call(session, port, img, prompts)
            per_variant[v].append(resp)
            latencies[v].append(t)
    return per_variant, latencies


async def run_concurrent(session, ports, images, prompts):
    """Fire all image requests for each variant in parallel — stresses batching."""
    per_variant: dict[str, list] = {}
    latencies: dict[str, list[float]] = {}
    for v, port in ports.items():
        t0 = time.time()
        tasks = [call(session, port, img, prompts) for img in images]
        rows = await asyncio.gather(*tasks)
        total = time.time() - t0
        per_variant[v] = [r for r, _ in rows]
        latencies[v] = [t for _, t in rows]
        latencies[v + "_wall_s"] = total   # total wall time for the whole batch
    return per_variant, latencies


# ---------------------------------------------------------------------------
# Parity computation
# ---------------------------------------------------------------------------

def compute_parity(per_variant: dict[str, list[DetectorResponse]]):
    """Return {(a,b): [iou,…]} pairwise for all variant pairs.
    Each list entry is one matched-detection IoU across all images."""
    variants = list(per_variant.keys())
    pairs: dict[tuple[str, str], list[float]] = {}
    counts_a: dict[tuple[str, str], int] = {}
    counts_b: dict[tuple[str, str], int] = {}
    unmatched: dict[tuple[str, str], tuple[int, int]] = {}
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            all_ious, u_a, u_b, ca, cb = [], 0, 0, 0, 0
            for ra, rb in zip(per_variant[a], per_variant[b]):
                p, ua, ub = pair_by_label(
                    list(ra.boxes), list(ra.labels),
                    list(rb.boxes), list(rb.labels),
                )
                all_ious.extend(p)
                u_a += ua; u_b += ub
                ca += len(ra.boxes); cb += len(rb.boxes)
            pairs[(a, b)] = all_ious
            counts_a[(a, b)] = ca
            counts_b[(a, b)] = cb
            unmatched[(a, b)] = (u_a, u_b)
    return pairs, counts_a, counts_b, unmatched


def _stats(arr):
    if not arr:
        return {"n": 0, "mean": 0.0, "min": 0.0, "pct_0.9": 0.0}
    return {
        "n": len(arr),
        "mean": round(st.mean(arr), 4),
        "min": round(min(arr), 4),
        "pct_0.9": round(sum(1 for x in arr if x > 0.9) / len(arr) * 100, 1),
    }


def report_scenario(name, per_variant, latencies, wall_batch=None):
    print("\n" + "=" * 78)
    print(f"SCENARIO: {name}")
    print("=" * 78)
    pairs, ca, cb, u = compute_parity(per_variant)

    # Per-variant detection totals + per-image latency
    print(f"{'variant':<10} {'total_dets':>12} {'mean_lat_s':>12} {'p95_lat_s':>12}")
    for v in per_variant:
        total = sum(len(r.boxes) for r in per_variant[v])
        lats = latencies[v] if isinstance(latencies[v], list) else []
        mean_l = round(st.mean(lats), 3) if lats else 0.0
        p95 = round(sorted(lats)[int(0.95 * (len(lats) - 1))], 3) if lats else 0.0
        print(f"{v:<10} {total:>12d} {mean_l:>12.3f} {p95:>12.3f}")
    if wall_batch:
        print("\nwall_s (batch-total):")
        for v, w in wall_batch.items():
            print(f"  {v}: {w:.3f}")

    print(f"\n{'pair':<18} {'iou_n':>6} {'mean':>6} {'min':>6} {'>0.9':>6} "
          f"{'a_tot':>6} {'b_tot':>6} {'u_a':>5} {'u_b':>5}")
    out = {}
    for (a, b), ious in pairs.items():
        s = _stats(ious)
        ua, ub = u[(a, b)]
        print(f"{a}->{b:<14} {s['n']:>6d} {s['mean']:>6.3f} {s['min']:>6.3f} "
              f"{s['pct_0.9']:>5.1f}% {ca[(a,b)]:>6d} {cb[(a,b)]:>6d} "
              f"{ua:>5d} {ub:>5d}")
        out[f"{a}_vs_{b}"] = {**s, "a_total": ca[(a, b)], "b_total": cb[(a, b)],
                              "unmatched_a": ua, "unmatched_b": ub}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-port", type=int, default=3003)
    ap.add_argument("--m4-port", type=int, default=3013)
    ap.add_argument("--m3-port", type=int, default=3014)
    ap.add_argument("--n-images", type=int, default=8)
    args = ap.parse_args()

    ports = {"hf": args.hf_port, "m4": args.m4_port, "m3": args.m3_port}
    images = sorted(SAMPLES.glob("*.jpg"))[: args.n_images]
    print(f"Ports: {ports}")
    print(f"Images: {len(images)}")

    results: dict = {"ports": ports, "n_images": len(images), "scenarios": {}}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
        # Warmup: one round per variant so text-embedding / first-call costs
        # don't skew scenario latencies.
        print("\nWarming up all variants …")
        for port in ports.values():
            await call(s, port, images[0], CLASSES_MULTI)

        # --- Scenario 1: single-class sequential ---
        pv, lat = await run_sequential(s, ports, images, CLASS_SINGLE, "single")
        results["scenarios"]["single_class_sequential"] = report_scenario(
            f"single-class ({CLASS_SINGLE[0]}), sequential, {len(images)} imgs",
            pv, lat,
        )

        # --- Scenario 2: multi-class sequential ---
        pv, lat = await run_sequential(s, ports, images, CLASSES_MULTI, "multi")
        results["scenarios"]["multi_class_sequential"] = report_scenario(
            f"multi-class {CLASSES_MULTI}, sequential, {len(images)} imgs",
            pv, lat,
        )

        # --- Scenario 3: multi-class concurrent batch ---
        pv, lat = await run_concurrent(s, ports, images, CLASSES_MULTI)
        wall = {v: lat[v + "_wall_s"] for v in ports}
        per_img_lat = {v: lat[v] for v in ports}
        results["scenarios"]["multi_class_concurrent"] = report_scenario(
            f"multi-class {CLASSES_MULTI}, concurrent batch, {len(images)} imgs",
            pv, per_img_lat, wall_batch=wall,
        )

    # Top-line verdict
    print("\n" + "=" * 78)
    print("TOP-LINE SUMMARY")
    print("=" * 78)
    for scn_name, scn_data in results["scenarios"].items():
        print(f"\n{scn_name}:")
        for pair, stats in scn_data.items():
            print(f"  {pair:<22} mean_iou={stats['mean']:.3f}  "
                  f"min={stats['min']:.3f}  n={stats['n']}  "
                  f"dets(a/b)={stats['a_total']}/{stats['b_total']}  "
                  f"unmatched(a/b)={stats['unmatched_a']}/{stats['unmatched_b']}")

    with RESULTS.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS}")


if __name__ == "__main__":
    asyncio.run(amain())
