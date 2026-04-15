"""Multi-class proposal test: all-classes-at-once vs per-class looping.

Motivation: during e2e on ``fl_pj_sample`` we saw SAM3 and OWLv2 returning
nothing, and Falcon conflating forklifts as "person". Hypothesis: multi-class
prompts (dotted phrase for GDINO/Falcon/SAM3, multi-entry list for OWLv2)
dilute per-class attention. This test quantifies the gap.

For each (image × class × model), hit the server twice:

  1. ``joint``   — pass all target classes in one call (what detect.py does today).
  2. ``per_class`` — pass only that class's prompt; per-class is the control.

Compares per-model counts per class so we can tell which models degrade under
joint prompts and need per-class looping in the pipeline.

Usage::

    # Servers must already be running
    python -m data_miner.auto_annotation_v3.tests.test_multiclass_proposal
    python -m data_miner.auto_annotation_v3.tests.test_multiclass_proposal \
        --classes person forklift palletjack trolley --threshold 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SAMPLE_DIR = Path("/media/data_2/vlm/code/data_miner/output/sample/fl_pj_sample")

DEFAULT_CLASSES = ["person", "forklift", "palletjack"]

# Detector-specific prompt text for each canonical class.
PROMPTS: dict[str, str] = {
    "person": "person",
    "forklift": "forklift",
    "palletjack": "pallet jack",
    "trolley": "trolley",
    "car": "car",
    "truck": "truck",
}

def _load_enabled_from_config() -> dict[str, int]:
    """Read enabled detectors + ports from configs/default.yaml."""
    from data_miner.auto_annotation_v3.config import load_config
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    return {n.value: c.port for n, c in cfg.servers.enabled_detectors().items()}


PORTS = _load_enabled_from_config()


# ---------------------------------------------------------------------------
# Payload builders — mirror detect.py._build_payload
# ---------------------------------------------------------------------------

def build_payload(model: str, image_path: str, classes: list[str]) -> dict:
    """Uniform DetectorRequest wire — server handles per-prompt iteration
    and any model-specific prompt formatting internally."""
    return {"image_path": image_path, "prompts": [PROMPTS[c] for c in classes]}


def _post(model: str, payload: dict, timeout: float = 60) -> dict:
    port = PORTS[model]
    r = requests.post(f"http://127.0.0.1:{port}/predict", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Label → canonical class mapping
# ---------------------------------------------------------------------------

def canonicalise(model: str, raw_label: str) -> str:
    """Normalise server-echoed label back to canonical class name.

    Servers echo the prompt they were given (uniform wire), so we reverse
    the PROMPTS map. Special-case 'pallet jack' → 'palletjack'.
    """
    s = str(raw_label).lower().strip().strip(".")
    if s == "pallet jack":
        return "palletjack"
    return s


def count_per_class(
    resp: dict,
    model: str,
    target_classes: list[str],
) -> dict[str, list[float]]:
    """Return {class_name: [score, …]} over the response for classes in target_classes."""
    boxes = resp.get("boxes") or []
    scores = resp.get("scores") or []
    labels = resp.get("labels") or []
    targets = set(target_classes)
    out: dict[str, list[float]] = {c: [] for c in target_classes}
    for box, score, lab in zip(boxes, scores, labels):
        canon = canonicalise(model, lab)
        if canon in targets:
            out[canon].append(float(score))
    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    images: list[Path],
    classes: list[str],
    models: list[str],
) -> dict[str, Any]:
    # Aggregate: [model][mode][class] = {"count": int, "scores": [..], "images_with_det": int}
    agg: dict[str, dict[str, dict[str, dict]]] = {
        m: {"joint": {c: {"count": 0, "scores": [], "images_with_det": 0}
                      for c in classes},
            "per_class": {c: {"count": 0, "scores": [], "images_with_det": 0}
                          for c in classes}}
        for m in models
    }

    for img_path in images:
        img_abs = str(img_path.resolve())
        print(f"\n=== {img_path.name} ===")

        for model in models:
            # --- joint: one call with all classes ---
            try:
                t0 = time.time()
                resp = _post(model, build_payload(model, img_abs, classes))
                lat = time.time() - t0
                joint_counts = count_per_class(resp, model, classes)
            except Exception as exc:
                print(f"  {model:16s} joint      FAILED: {exc}")
                joint_counts = {c: [] for c in classes}
                lat = 0.0
            for c in classes:
                scores = joint_counts[c]
                agg[model]["joint"][c]["count"] += len(scores)
                agg[model]["joint"][c]["scores"].extend(scores)
                if scores:
                    agg[model]["joint"][c]["images_with_det"] += 1
            print(
                f"  {model:16s} joint     "
                + "  ".join(f"{c}={len(joint_counts[c])}" for c in classes)
                + f"  ({lat:.2f}s)"
            )

            # --- per_class: one call per class ---
            per_class_counts: dict[str, list[float]] = {}
            per_class_lat = 0.0
            for c in classes:
                try:
                    t0 = time.time()
                    resp = _post(model, build_payload(model, img_abs, [c]))
                    per_class_lat += time.time() - t0
                    per_class_counts[c] = count_per_class(resp, model, [c])[c]
                except Exception as exc:
                    print(f"  {model:16s} per_class[{c}] FAILED: {exc}")
                    per_class_counts[c] = []
            for c in classes:
                scores = per_class_counts[c]
                agg[model]["per_class"][c]["count"] += len(scores)
                agg[model]["per_class"][c]["scores"].extend(scores)
                if scores:
                    agg[model]["per_class"][c]["images_with_det"] += 1
            print(
                f"  {model:16s} per_class "
                + "  ".join(f"{c}={len(per_class_counts[c])}" for c in classes)
                + f"  ({per_class_lat:.2f}s)"
            )

    return agg


def summarise(agg: dict, classes: list[str], n_images: int) -> None:
    print("\n" + "=" * 86)
    print(f"SUMMARY across {n_images} image(s)")
    print("=" * 86)
    header = f"{'model':16s} {'mode':10s} " + "".join(
        f"{c + '_imgs/count':>22s}" for c in classes
    ) + f"{'mean_score':>14s}"
    print(header)
    print("-" * len(header))
    for model, by_mode in agg.items():
        for mode in ("joint", "per_class"):
            row = f"{model:16s} {mode:10s} "
            total_scores: list[float] = []
            for c in classes:
                e = by_mode[mode][c]
                row += f"{e['images_with_det']:>3d}/{e['count']:<18d}"
                total_scores.extend(e["scores"])
            mean = (sum(total_scores) / len(total_scores)) if total_scores else 0.0
            row += f"{mean:>14.3f}"
            print(row)
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, default=SAMPLE_DIR)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument(
        "--models", nargs="+",
        default=sorted(PORTS.keys()),
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap to first N images (0=all)")
    args = ap.parse_args()

    images = sorted(args.image_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {args.image_dir}", file=sys.stderr)
        return 1

    # Require prompts for each requested class.
    for c in args.classes:
        if c not in PROMPTS:
            print(f"Unknown class '{c}' — add to PROMPTS map", file=sys.stderr)
            return 1

    # Health check.
    for m in args.models:
        try:
            r = requests.get(f"http://127.0.0.1:{PORTS[m]}/health", timeout=2)
            if r.status_code != 200:
                print(f"{m} unhealthy (status {r.status_code})", file=sys.stderr)
                return 1
        except requests.ConnectionError:
            print(f"{m} not reachable on port {PORTS[m]}", file=sys.stderr)
            return 1

    agg = run(images, args.classes, args.models)
    summarise(agg, args.classes, len(images))
    return 0


if __name__ == "__main__":
    sys.exit(main())
