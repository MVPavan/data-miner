"""Accuracy test: aav3 litserve servers vs raw data_miner/models/ helpers.

For each enabled detector in configs/default.yaml:
  * run Helper.detect() directly in-process on a free GPU
    (nms_threshold=None, matched threshold and dtype / image dims so both
    paths produce the same post-processed bbox set)
  * POST a uniform DetectorRequest to the aav3 litserve server
  * pair boxes by IoU ≥ 0.5, report counts / min-IoU / max score delta

Pre-reqs:
    # servers must be running
    python -m data_miner.auto_annotation_v3.servers.launch_all

Run:
    python -m data_miner.auto_annotation_v3.tests.test_raw_vs_litserve
    python -m data_miner.auto_annotation_v3.tests.test_raw_vs_litserve \
        --direct-gpu 4 --classes person forklift "pallet jack"

Equivalence bar: all boxes paired, min IoU ≥ 0.98, |Δscore| < 0.01.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _pin_direct_gpu() -> None:
    """Pin raw-model direct inference to a free GPU via CUDA_VISIBLE_DEVICES.

    Must run BEFORE any torch/transformers import. We peek at --direct-gpu
    without invoking argparse yet so that the env var is set first.
    """
    gpu = "4"
    for i, arg in enumerate(sys.argv):
        if arg == "--direct-gpu" and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
            break
        if arg.startswith("--direct-gpu="):
            gpu = arg.split("=", 1)[1]
            break
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)


_pin_direct_gpu()

import asyncio  # noqa: E402

import aiohttp  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_miner.auto_annotation_v3.config import load_config  # noqa: E402
from data_miner.auto_annotation_v3.contracts import (  # noqa: E402
    DetectorName,
    DetectorRequest,
    DetectorResponse,
)


DEFAULT_IMAGE = (
    "/media/data_2/vlm/code/data_miner/output/sample/fl_pj_sample/"
    "-7vYGr_2DVI_006150.jpg"
)
DEFAULT_CLASSES = ["person", "forklift", "pallet jack"]

# Thresholds matching each server's _to_response default when req.threshold is None.
SERVER_THRESHOLDS: dict[DetectorName, float] = {
    DetectorName.GROUNDING_DINO: 0.25,
    DetectorName.FALCON:         0.0,
    DetectorName.SAM3:           0.5,
    DetectorName.OWLVIT2:        0.1,
}

# Equivalence thresholds
MIN_IOU_REQUIRED = 0.98
MAX_SCORE_DELTA  = 0.01


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _direct_to_xyxy(rows: list) -> tuple[list[list[float]], list[float]]:
    """Helper output [class_id, x, y, w, h, conf, (label)] → xyxy, scores."""
    boxes, scores = [], []
    for r in rows:
        x, y, w, h, c = r[1], r[2], r[3], r[4], r[5]
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        scores.append(float(c))
    return boxes, scores


def _pair(d_boxes, d_scores, s_boxes, s_scores):
    """Greedy IoU pairing at threshold 0.5. Returns (pairs, unmatched_direct, unmatched_server)."""
    used = [False] * len(s_boxes)
    pairs = []
    for ab, asc in zip(d_boxes, d_scores):
        best_j, best = -1, 0.0
        for j, bb in enumerate(s_boxes):
            if used[j]:
                continue
            v = _iou(ab, bb)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > 0.5:
            used[best_j] = True
            pairs.append((best, abs(asc - s_scores[best_j])))
    return pairs, len(d_boxes) - len(pairs), sum(1 for u in used if not u)


def _run_direct(name: DetectorName, image_path: str, classes: list[str]) -> list:
    """Load each Helper with params matched to its server and run detect()."""
    threshold = SERVER_THRESHOLDS[name]

    if name is DetectorName.GROUNDING_DINO:
        # Server loops per-prompt, so Helper must too for apples-to-apples.
        from data_miner.models.grounding_dino import GroundingDINOHelper
        h = GroundingDINOHelper(device="cuda:0")
        out = []
        for cls in classes:
            out.extend(h.detect(
                image_path, threshold=threshold, detection_classes=[cls],
                output_format="normalized", text_threshold=0.2,
                nms_threshold=None,
            ))
        return out

    if name is DetectorName.FALCON:
        # Server uses max_dimension=512 (matching pre-refactor).
        from data_miner.models.falcon_perception import FalconPerceptionHelper
        h = FalconPerceptionHelper(
            device="cuda:0", max_length=4096,
            min_dimension=256, max_dimension=512,
        )
        return h.detect(image_path, threshold=threshold, detection_classes=classes,
                        output_format="normalized", nms_threshold=None)

    if name is DetectorName.SAM3:
        # SAM3 server dtype (fp16 per serve_sam3.py) must match Helper exactly.
        # Helper hardcodes fp16 in load_model() which already matches, so no override needed.
        from data_miner.models.sam import SAMHelper
        h = SAMHelper(device="cuda:0")
        return h.detect(image_path, threshold=threshold, detection_classes=classes,
                        output_format="normalized", nms_threshold=None)

    if name is DetectorName.OWLVIT2:
        from data_miner.models.owlvit import OWLViTHelper
        h = OWLViTHelper(device="cuda:0")
        return h.detect(image_path, threshold=threshold, detection_classes=classes,
                        output_format="normalized", nms_threshold=None)

    raise ValueError(f"No direct Helper mapping for {name}")


async def _call_server(session, port: int, image_path: str,
                       classes: list[str], threshold: float) -> DetectorResponse:
    req = DetectorRequest(image_path=image_path, prompts=classes, threshold=threshold)
    async with session.post(
        f"http://localhost:{port}/predict", json=req.model_dump()
    ) as r:
        r.raise_for_status()
        return DetectorResponse.model_validate(await r.json())


async def compare_one(name: DetectorName, port: int,
                      image_path: str, classes: list[str]) -> bool:
    print(f"\n=== {name.value} ===")

    t0 = time.time()
    direct = _run_direct(name, image_path, classes)
    d_elapsed = time.time() - t0
    d_boxes, d_scores = _direct_to_xyxy(direct)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
        t0 = time.time()
        resp = await _call_server(
            s, port, image_path, classes, SERVER_THRESHOLDS[name]
        )
        s_elapsed = time.time() - t0

    pairs, u_d, u_s = _pair(d_boxes, d_scores, list(resp.boxes), list(resp.scores))
    mean_iou = sum(p[0] for p in pairs) / len(pairs) if pairs else 0.0
    min_iou  = min((p[0] for p in pairs), default=0.0)
    max_sd   = max((p[1] for p in pairs), default=0.0)

    print(f"  direct: n={len(d_boxes)}  ({d_elapsed:.2f}s)")
    print(f"  server: n={len(resp.boxes)}  ({s_elapsed:.2f}s)")
    print(f"  paired={len(pairs)}  unmatched(direct/server)={u_d}/{u_s}")
    print(f"  mean_iou={mean_iou:.4f}  min_iou={min_iou:.4f}  max_score_diff={max_sd:.5f}")

    ok = (
        len(d_boxes) == len(resp.boxes) == len(pairs)
        and min_iou >= MIN_IOU_REQUIRED
        and max_sd < MAX_SCORE_DELTA
    )
    print(f"  {'✅ EQUIVALENT' if ok else '❌ DIVERGED'}")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--direct-gpu", type=str, default="4",
                    help="GPU index for direct Helper inference "
                         "(default 4 — servers are on 0/1 per default.yaml)")
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    enabled = cfg.servers.enabled_detectors()
    if not enabled:
        print("No enabled detectors in config", file=sys.stderr)
        return 1

    results: dict[str, bool] = {}
    for name, dcfg in enabled.items():
        results[name.value] = await compare_one(
            name, dcfg.port, args.image, args.classes
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:<20} {'✅ EQUIVALENT' if ok else '❌ DIVERGED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
