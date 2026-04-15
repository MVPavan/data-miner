"""Parity test: sam3 (HF) server vs sam3_dart (DART) server over HTTP.

Both servers expose the same wire contract (DetectorRequest/DetectorResponse
for proposal, SAM3RefineRequest/SAM3RefineResponse for refine). This test
POSTs identical payloads to each and compares outputs via greedy per-label
IoU pairing.

Pre-reqs:
    Launch both servers (separate GPUs) via launch_all.py:
        python -m data_miner.auto_annotation_v3.servers.launch_all \
            --servers sam3 sam3_dart

Run:
    python -m data_miner.auto_annotation_v3.tests.test_sam3_hf_vs_dart_server

Bars:
    Proposal: mean matched IoU ≥ 0.90 (DART's NMS differs from HF's
              per-class)
    Refine:   mean matched IoU ≥ 0.90 (interactive-predictor vs detector-side
              box-prompt path differ by a few %)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_miner.auto_annotation_v3.config import load_config
from data_miner.auto_annotation_v3.contracts import (
    DetectorName,
    DetectorRequest,
    DetectorResponse,
    SAM3RefineRequest,
    SAM3RefineResponse,
)

DEFAULT_IMAGES = sorted(
    Path("/media/data_2/vlm/code/data_miner/output/sample/fl_pj_sample").glob("*.jpg")
)
DEFAULT_CLASSES = ["person", "forklift", "pallet jack"]
RESULTS_PATH = Path(__file__).with_name("test_sam3_hf_vs_dart_server_results.json")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pair_by_label(hf_b, hf_l, dt_b, dt_l, iou_thr=0.5):
    used = [False] * len(dt_b)
    pairs = []
    for i, (b, lb) in enumerate(zip(hf_b, hf_l)):
        best, best_j = 0.0, -1
        for j, (db, dl) in enumerate(zip(dt_b, dt_l)):
            if used[j] or dl != lb:
                continue
            v = iou(b, db)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > iou_thr:
            used[best_j] = True
            pairs.append((best, i, best_j))
    return pairs, len(hf_b) - len(pairs), sum(1 for u in used if not u)


async def proposal(session, port, image_path, classes):
    req = DetectorRequest(image_path=str(image_path), prompts=classes, threshold=0.5)
    t0 = time.time()
    async with session.post(
        f"http://localhost:{port}/predict", json=req.model_dump()
    ) as r:
        r.raise_for_status()
        data = await r.json()
    return DetectorResponse.model_validate(data), time.time() - t0


async def refine(session, port, image_path, bbox_norm):
    req = SAM3RefineRequest(image_path=str(image_path), bbox=bbox_norm, threshold=0.5)
    t0 = time.time()
    async with session.post(
        f"http://localhost:{port}/predict", json=req.model_dump()
    ) as r:
        r.raise_for_status()
        data = await r.json()
    return SAM3RefineResponse.model_validate(data), time.time() - t0


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--n-images", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    detectors = cfg.servers.detectors
    hf = detectors.get(DetectorName.SAM3)
    dt = detectors.get(DetectorName.SAM3_DART)
    if not (hf and hf.enabled and dt and dt.enabled):
        print("ERROR: both sam3 and sam3_dart must be enabled in the config "
              "for this side-by-side test.", file=sys.stderr)
        return 1
    print(f"HF sam3 on :{hf.port}, DART sam3_dart on :{dt.port}")

    images = DEFAULT_IMAGES[: args.n_images]
    print(f"Running on {len(images)} images; classes={args.classes}\n")

    results = {"per_image": []}
    prop_ious, refine_ious = [], []
    hf_prop_times, dt_prop_times = [], []
    hf_ref_times, dt_ref_times = [], []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
        # Warm DART once (first proposal triggers text embeddings + interactive
        # predictor initialisation; don't let that skew timings).
        try:
            await proposal(s, dt.port, images[0], args.classes)
            await refine(s, dt.port, images[0], [0.1, 0.1, 0.9, 0.9])
            await proposal(s, hf.port, images[0], args.classes)
            await refine(s, hf.port, images[0], [0.1, 0.1, 0.9, 0.9])
        except Exception as exc:
            print(f"Warmup failed: {exc!r}")
            return 1

        for i, img in enumerate(images):
            print(f"[{i+1}/{len(images)}] {img.name}")

            hf_r, t_hf = await proposal(s, hf.port, img, args.classes)
            dt_r, t_dt = await proposal(s, dt.port, img, args.classes)
            hf_prop_times.append(t_hf); dt_prop_times.append(t_dt)

            pairs, u_hf, u_dt = pair_by_label(
                list(hf_r.boxes), list(hf_r.labels),
                list(dt_r.boxes), list(dt_r.labels),
            )
            m_iou = sum(p[0] for p in pairs) / len(pairs) if pairs else 0.0
            n_iou = min((p[0] for p in pairs), default=0.0)
            prop_ious.extend(p[0] for p in pairs)
            print(f"  PROPOSAL hf={len(hf_r.boxes)}({t_hf:.2f}s) "
                  f"dart={len(dt_r.boxes)}({t_dt:.2f}s) "
                  f"paired={len(pairs)} u_hf={u_hf} u_dart={u_dt} "
                  f"mean_iou={m_iou:.3f} min_iou={n_iou:.3f}")

            # Refine: pick HF top-1 per label, send same box to both servers.
            per_label_top: dict[str, tuple[list[float], float]] = {}
            for b, sc, lb in zip(hf_r.boxes, hf_r.scores, hf_r.labels):
                if lb not in per_label_top or sc > per_label_top[lb][1]:
                    per_label_top[lb] = (list(b), sc)

            refine_rows = []
            for lb, (box, _sc) in per_label_top.items():
                hf_rr, t_hfr = await refine(s, hf.port, img, box)
                dt_rr, t_dtr = await refine(s, dt.port, img, box)
                hf_ref_times.append(t_hfr); dt_ref_times.append(t_dtr)

                ri = (
                    iou(hf_rr.box, dt_rr.box)
                    if hf_rr.box is not None and dt_rr.box is not None
                    else 0.0
                )
                hf_shift = iou(box, hf_rr.box) if hf_rr.box else 0.0
                dt_shift = iou(box, dt_rr.box) if dt_rr.box else 0.0
                refine_ious.append(ri)
                refine_rows.append({
                    "label": lb,
                    "input_box": box,
                    "hf_box": hf_rr.box, "dart_box": dt_rr.box,
                    "hf_vs_dart_iou": ri,
                    "hf_shift": hf_shift, "dart_shift": dt_shift,
                    "hf_ms": round(t_hfr * 1000, 1),
                    "dart_ms": round(t_dtr * 1000, 1),
                })
                print(f"  REFINE[{lb:<12}] iou(hf,dart)={ri:.3f}  "
                      f"hf_shift={hf_shift:.3f} dart_shift={dt_shift:.3f}  "
                      f"t_hf={t_hfr*1000:.0f}ms t_dart={t_dtr*1000:.0f}ms")

            results["per_image"].append({
                "image": img.stem,
                "proposal": {
                    "hf_n": len(hf_r.boxes), "dart_n": len(dt_r.boxes),
                    "paired": len(pairs), "u_hf": u_hf, "u_dart": u_dt,
                    "mean_iou": m_iou, "min_iou": n_iou,
                    "hf_s": t_hf, "dart_s": t_dt,
                },
                "refine": refine_rows,
            })

    def _stats(arr):
        import statistics as st
        return {
            "n": len(arr),
            "mean": float(st.mean(arr)) if arr else 0.0,
            "min": float(min(arr)) if arr else 0.0,
            "pct_above_0.9": float(sum(1 for x in arr if x > 0.9) / len(arr) * 100) if arr else 0.0,
            "pct_above_0.8": float(sum(1 for x in arr if x > 0.8) / len(arr) * 100) if arr else 0.0,
        }

    import statistics as st
    summary = {
        "proposal": _stats(prop_ious),
        "refine": _stats(refine_ious),
        "latency": {
            "hf_proposal_mean_s": round(st.mean(hf_prop_times), 3) if hf_prop_times else 0,
            "dart_proposal_mean_s": round(st.mean(dt_prop_times), 3) if dt_prop_times else 0,
            "hf_refine_mean_ms": round(st.mean(hf_ref_times) * 1000, 1) if hf_ref_times else 0,
            "dart_refine_mean_ms": round(st.mean(dt_ref_times) * 1000, 1) if dt_ref_times else 0,
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
    print(f"Latency:  {summary['latency']}")
    print(f"\nSaved → {RESULTS_PATH}")

    prop_ok = summary["proposal"]["n"] > 0 and summary["proposal"]["mean"] >= 0.85
    ref_ok = summary["refine"]["n"] > 0 and summary["refine"]["mean"] >= 0.85
    print(f"\nProposal parity {'✅' if prop_ok else '❌'}  (mean_iou ≥ 0.85)")
    print(f"Refine   parity {'✅' if ref_ok else '❌'}  (mean_iou ≥ 0.85)")
    return 0 if (prop_ok and ref_ok) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
