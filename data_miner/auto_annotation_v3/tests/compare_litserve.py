"""Compare LitServe server outputs vs aa_v2 direct inference.

Validates that each LitServe server produces equivalent results to the
aa_v2 proposal stage running models directly in-process.

Usage:
    # Single image (servers must be running)
    python -m data_miner.auto_annotation_v3.compare_litserve \
        --image output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg

    # Whole directory
    python -m data_miner.auto_annotation_v3.compare_litserve \
        --image-dir output/sample/fl_pj_sample

    # Specific servers only
    python -m data_miner.auto_annotation_v3.compare_litserve \
        --image output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg \
        --servers grounding_dino falcon

    # Skip direct inference (compare against cached v2 results)
    python -m data_miner.auto_annotation_v3.compare_litserve \
        --image output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg \
        --litserve-only

Servers should be launched first:
    python -m data_miner.auto_annotation_v3.servers.launch_all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# LitServe HTTP clients (v3 servers)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_PORTS = {
    "grounding_dino": 3001,
    "falcon": 3002,
    "sam3": 3003,
    "owlvit2": 3004,
}

DETECTION_CLASSES = ["person"]


def _litserve_health(port: int) -> bool:
    """Check if a LitServe server is reachable."""
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def litserve_gdino(image_path: str, port: int = 3001) -> list[dict]:
    text_prompt = " . ".join(DETECTION_CLASSES) + " ."
    r = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json={"image_path": str(Path(image_path).resolve()), "text_prompt": text_prompt},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return _parse_server_response(data, "grounding_dino_litserve")


def litserve_falcon(image_path: str, port: int = 3002) -> list[dict]:
    text_prompt = " . ".join(DETECTION_CLASSES)
    r = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json={"image_path": str(Path(image_path).resolve()), "text_prompt": text_prompt},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return _parse_server_response(data, "falcon_litserve")


def litserve_sam3(image_path: str, port: int = 3003) -> list[dict]:
    text_prompt = " . ".join(DETECTION_CLASSES)
    r = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json={
            "mode": "proposal",
            "image_path": str(Path(image_path).resolve()),
            "text_prompt": text_prompt,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return _parse_server_response(data, "sam3_litserve")


def litserve_owlvit2(image_path: str, port: int = 3004) -> list[dict]:
    queries = [f"a photo of a {cls}" for cls in DETECTION_CLASSES]
    r = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json={
            "image_path": str(Path(image_path).resolve()),
            "text_queries": queries,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return _parse_server_response(data, "owlvit2_litserve")


def _parse_server_response(data: dict, source: str) -> list[dict]:
    """Normalize server response into a flat list of dicts."""
    boxes = data.get("boxes", [])
    scores = data.get("scores", [])
    labels = data.get("labels", [])
    results = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        results.append({
            "bbox": box,
            "score": float(score),
            "label": str(label),
            "source": source,
        })
    return results


LITSERVE_RUNNERS = {
    "grounding_dino": litserve_gdino,
    "falcon": litserve_falcon,
    "sam3": litserve_sam3,
    "owlvit2": litserve_owlvit2,
}


# ──────────────────────────────────────────────────────────────────────
# aa_v2 direct inference (in-process, GPU)
# ──────────────────────────────────────────────────────────────────────


def direct_falcon(image_path: str, device: str = "cuda:0") -> list[dict]:
    from data_miner.auto_annotation_v2.config import ClassPackConfig, DetectionModelConfig
    from data_miner.auto_annotation_v2.stages.proposal import _get_model, _run_falcon

    results = []
    for cls_name in DETECTION_CLASSES:
        class_pack = ClassPackConfig(
            name=cls_name, synonyms=[], negatives=[], prompt_variants=[cls_name]
        )
        cfg = DetectionModelConfig(
            kind="falcon",
            model_id="tiiuae/Falcon-Perception",
            device=device,
            params={
                "task": "segmentation",
                "dtype": "bfloat16",
                "min_dimension": 256,
                "max_dimension": 512,
                "max_length": 4096,
                "max_new_tokens": 2048,
                "seed": 42,
            },
        )
        loaded = _get_model("falcon", cfg)
        pil_image = Image.open(image_path).convert("RGB")
        candidates = _run_falcon(loaded, pil_image, class_pack, cls_name, cfg.params, "falcon")
        for c in candidates:
            results.append({
                "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
                "score": c.score,
                "label": c.label,
                "source": "falcon_direct",
            })
    return results


def direct_grounding_dino(image_path: str, device: str = "cuda:0") -> list[dict]:
    from data_miner.auto_annotation_v2.config import ClassPackConfig, DetectionModelConfig
    from data_miner.auto_annotation_v2.stages.proposal import _get_model, _run_grounding_dino

    results = []
    for cls_name in DETECTION_CLASSES:
        class_pack = ClassPackConfig(
            name=cls_name, synonyms=[], negatives=[], prompt_variants=[cls_name]
        )
        cfg = DetectionModelConfig(
            kind="grounding_dino",
            model_id="IDEA-Research/grounding-dino-base",
            device=device,
            params={"box_threshold": 0.25, "text_threshold": 0.2},
        )
        loaded = _get_model("grounding_dino", cfg)
        pil_image = Image.open(image_path).convert("RGB")
        candidates = _run_grounding_dino(
            loaded, pil_image, class_pack, cls_name, cfg.params, "grounding_dino"
        )
        for c in candidates:
            results.append({
                "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
                "score": c.score,
                "label": c.label,
                "source": "grounding_dino_direct",
            })
    return results


def direct_sam3(image_path: str, device: str = "cuda:1") -> list[dict]:
    from data_miner.auto_annotation_v2.config import ClassPackConfig, DetectionModelConfig
    from data_miner.auto_annotation_v2.stages.proposal import _get_model, _run_sam

    results = []
    for cls_name in DETECTION_CLASSES:
        class_pack = ClassPackConfig(
            name=cls_name, synonyms=[], negatives=[], prompt_variants=[cls_name]
        )
        cfg = DetectionModelConfig(
            kind="sam",
            model_id="facebook/sam3",
            device=device,
            params={"threshold": 0.5},
        )
        loaded = _get_model("sam", cfg)
        pil_image = Image.open(image_path).convert("RGB")
        candidates = _run_sam(loaded, pil_image, class_pack, cls_name, cfg.params, "sam")
        for c in candidates:
            results.append({
                "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
                "score": c.score,
                "label": c.label,
                "source": "sam3_direct",
            })
    return results


def direct_owlvit2(image_path: str, device: str = "cuda:0") -> list[dict]:
    from data_miner.auto_annotation_v2.config import ClassPackConfig, DetectionModelConfig
    from data_miner.auto_annotation_v2.stages.proposal import _get_model, _run_owlvit

    results = []
    for cls_name in DETECTION_CLASSES:
        class_pack = ClassPackConfig(
            name=cls_name, synonyms=[], negatives=[], prompt_variants=[cls_name]
        )
        cfg = DetectionModelConfig(
            kind="owlvit",
            model_id="google/owlv2-base-patch16-ensemble",
            device=device,
            params={"threshold": 0.1},
        )
        loaded = _get_model("owlvit", cfg)
        pil_image = Image.open(image_path).convert("RGB")
        candidates = _run_owlvit(loaded, pil_image, class_pack, cls_name, cfg.params, "owlvit")
        for c in candidates:
            results.append({
                "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
                "score": c.score,
                "label": c.label,
                "source": "owlvit2_direct",
            })
    return results


DIRECT_RUNNERS = {
    "grounding_dino": direct_grounding_dino,
    "falcon": direct_falcon,
    "sam3": direct_sam3,
    "owlvit2": direct_owlvit2,
}


# ──────────────────────────────────────────────────────────────────────
# Comparison logic
# ──────────────────────────────────────────────────────────────────────


def bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    litserve_dets: list[dict],
    direct_dets: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Match LitServe detections against direct inference detections.

    Returns:
        dict with keys: matched, litserve_only, direct_only, stats
    """
    used_direct = set()
    matched = []

    for ls_det in litserve_dets:
        best_iou = 0.0
        best_idx = -1
        for j, d_det in enumerate(direct_dets):
            if j in used_direct:
                continue
            iou = bbox_iou(ls_det["bbox"], d_det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx >= 0:
            used_direct.add(best_idx)
            matched.append({
                "litserve": ls_det,
                "direct": direct_dets[best_idx],
                "iou": best_iou,
                "score_diff": abs(ls_det["score"] - direct_dets[best_idx]["score"]),
                "label_match": ls_det["label"].lower().strip()
                == direct_dets[best_idx]["label"].lower().strip(),
            })

    litserve_only = [d for i, d in enumerate(litserve_dets) if not any(
        m["litserve"] is d for m in matched
    )]
    direct_only = [d for j, d in enumerate(direct_dets) if j not in used_direct]

    ious = [m["iou"] for m in matched]
    score_diffs = [m["score_diff"] for m in matched]
    label_matches = sum(1 for m in matched if m["label_match"])

    stats = {
        "litserve_count": len(litserve_dets),
        "direct_count": len(direct_dets),
        "matched": len(matched),
        "litserve_only": len(litserve_only),
        "direct_only": len(direct_only),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "mean_score_diff": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "max_score_diff": float(np.max(score_diffs)) if score_diffs else 0.0,
        "label_match_rate": label_matches / len(matched) if matched else 1.0,
    }

    return {
        "matched": matched,
        "litserve_only": litserve_only,
        "direct_only": direct_only,
        "stats": stats,
    }


# ──────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────


def print_detections(title: str, dets: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    if not dets:
        print("  (no detections)")
        return
    for i, d in enumerate(dets, 1):
        bbox = d["bbox"]
        print(
            f"  {i:2d}. bbox=[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"
            f"  score={d['score']:.4f}  label={d['label']}"
        )
    print(f"  Total: {len(dets)}")


def print_comparison(model_name: str, result: dict) -> None:
    stats = result["stats"]
    print(f"\n{'─' * 72}")
    print(f"  {model_name}: LitServe vs Direct Inference Comparison")
    print(f"{'─' * 72}")
    print(f"  LitServe detections:   {stats['litserve_count']}")
    print(f"  Direct detections:     {stats['direct_count']}")
    print(f"  Matched (IoU≥0.5):     {stats['matched']}")
    print(f"  LitServe-only:         {stats['litserve_only']}")
    print(f"  Direct-only:           {stats['direct_only']}")
    if stats["matched"] > 0:
        print(f"  Mean IoU (matched):    {stats['mean_iou']:.4f}")
        print(f"  Min IoU (matched):     {stats['min_iou']:.4f}")
        print(f"  Mean |Δscore|:         {stats['mean_score_diff']:.4f}")
        print(f"  Max |Δscore|:          {stats['max_score_diff']:.4f}")
        print(f"  Label match rate:      {stats['label_match_rate']:.1%}")

    # Verdict
    equivalent = (
        stats["litserve_only"] == 0
        and stats["direct_only"] == 0
        and stats["label_match_rate"] == 1.0
        and stats["max_score_diff"] < 0.05
    )
    close = (
        stats["litserve_only"] <= 1
        and stats["direct_only"] <= 1
        and stats["label_match_rate"] >= 0.9
    )

    if equivalent:
        verdict = "EQUIVALENT"
    elif close:
        verdict = "CLOSE (minor differences — likely batching / numerical)"
    else:
        verdict = "DIVERGED"
    print(f"  Verdict:               {verdict}")

    # Show unmatched detections
    if result["litserve_only"]:
        print(f"\n  LitServe-only detections:")
        for d in result["litserve_only"]:
            print(f"    bbox={[f'{v:.4f}' for v in d['bbox']]}  score={d['score']:.4f}  label={d['label']}")
    if result["direct_only"]:
        print(f"\n  Direct-only detections:")
        for d in result["direct_only"]:
            print(f"    bbox={[f'{v:.4f}' for v in d['bbox']]}  score={d['score']:.4f}  label={d['label']}")


def print_summary(all_results: dict[str, dict]) -> None:
    print(f"\n{'#' * 72}")
    print(f"# OVERALL SUMMARY")
    print(f"{'#' * 72}")
    print(f"\n  {'Model':<20} {'LS':>4} {'Direct':>7} {'Match':>6} {'LS-only':>8} {'D-only':>7} {'mIoU':>6} {'Verdict':<12}")
    print(f"  {'─' * 78}")
    for model, result in all_results.items():
        s = result["stats"]
        eq = (
            s["litserve_only"] == 0
            and s["direct_only"] == 0
            and s["label_match_rate"] == 1.0
            and s["max_score_diff"] < 0.05
        )
        close = s["litserve_only"] <= 1 and s["direct_only"] <= 1 and s["label_match_rate"] >= 0.9
        verdict = "EQUIV" if eq else ("CLOSE" if close else "DIVERGED")
        print(
            f"  {model:<20} {s['litserve_count']:>4} {s['direct_count']:>7}"
            f" {s['matched']:>6} {s['litserve_only']:>8} {s['direct_only']:>7}"
            f" {s['mean_iou']:>6.3f} {verdict:<12}"
        )


# ──────────────────────────────────────────────────────────────────────
# Multi-image batch comparison
# ──────────────────────────────────────────────────────────────────────


def compare_image(
    image_path: str,
    servers: list[str],
    ports: dict[str, int],
    litserve_only: bool = False,
    device: str = "cuda:0",
) -> dict[str, dict]:
    """Compare all specified servers for a single image."""
    print(f"\n{'█' * 72}")
    print(f"  Image: {image_path}")
    img = Image.open(image_path)
    print(f"  Size:  {img.size}")
    print(f"{'█' * 72}")

    all_results = {}

    for model_name in servers:
        print(f"\n{'#' * 72}")
        print(f"# {model_name.upper()}")
        print(f"{'#' * 72}")

        # LitServe
        port = ports.get(model_name, DEFAULT_PORTS.get(model_name))
        if not _litserve_health(port):
            print(f"  WARNING: {model_name} server not reachable on port {port}, skipping")
            continue

        t0 = time.time()
        ls_dets = LITSERVE_RUNNERS[model_name](image_path, port)
        ls_time = time.time() - t0
        print_detections(f"{model_name} — LitServe (port {port}, {ls_time:.2f}s)", ls_dets)

        if litserve_only:
            all_results[model_name] = {
                "matched": [], "litserve_only": ls_dets, "direct_only": [],
                "stats": {
                    "litserve_count": len(ls_dets), "direct_count": 0,
                    "matched": 0, "litserve_only": len(ls_dets), "direct_only": 0,
                    "mean_iou": 0.0, "min_iou": 0.0,
                    "mean_score_diff": 0.0, "max_score_diff": 0.0,
                    "label_match_rate": 1.0,
                },
            }
            continue

        # Direct inference
        t0 = time.time()
        d_dets = DIRECT_RUNNERS[model_name](image_path, device)
        d_time = time.time() - t0
        print_detections(f"{model_name} — Direct aa_v2 ({d_time:.2f}s)", d_dets)

        torch.cuda.empty_cache()

        # Compare
        result = match_detections(ls_dets, d_dets)
        all_results[model_name] = result
        print_comparison(model_name, result)

    return all_results


def compare_directory(
    image_dir: str,
    servers: list[str],
    ports: dict[str, int],
    litserve_only: bool = False,
    device: str = "cuda:0",
    max_images: int = 0,
) -> dict:
    """Compare all images in a directory. Returns aggregate stats."""
    img_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in extensions)

    if max_images > 0:
        images = images[:max_images]

    print(f"\nFound {len(images)} images in {image_dir}")

    # Aggregate stats per model
    agg: dict[str, list[dict]] = {m: [] for m in servers}

    for img_path in images:
        results = compare_image(str(img_path), servers, ports, litserve_only, device)
        for model, result in results.items():
            agg[model].append(result["stats"])

    # Print aggregate
    print(f"\n{'█' * 72}")
    print(f"  AGGREGATE RESULTS ({len(images)} images)")
    print(f"{'█' * 72}")

    aggregate_results = {}
    for model, stats_list in agg.items():
        if not stats_list:
            continue
        total_ls = sum(s["litserve_count"] for s in stats_list)
        total_d = sum(s["direct_count"] for s in stats_list)
        total_matched = sum(s["matched"] for s in stats_list)
        total_ls_only = sum(s["litserve_only"] for s in stats_list)
        total_d_only = sum(s["direct_only"] for s in stats_list)
        ious = [s["mean_iou"] for s in stats_list if s["matched"] > 0]
        mean_iou = float(np.mean(ious)) if ious else 0.0

        aggregate_results[model] = {
            "stats": {
                "litserve_count": total_ls,
                "direct_count": total_d,
                "matched": total_matched,
                "litserve_only": total_ls_only,
                "direct_only": total_d_only,
                "mean_iou": mean_iou,
                "min_iou": float(np.min(ious)) if ious else 0.0,
                "mean_score_diff": float(np.mean([s["mean_score_diff"] for s in stats_list])),
                "max_score_diff": float(np.max([s["max_score_diff"] for s in stats_list])),
                "label_match_rate": float(np.mean([s["label_match_rate"] for s in stats_list])),
            },
        }

    print_summary(aggregate_results)
    return aggregate_results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare LitServe server outputs vs aa_v2 direct inference",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single image")
    input_group.add_argument("--image-dir", type=str, help="Directory of images")

    parser.add_argument(
        "--servers",
        nargs="+",
        default=["grounding_dino", "falcon", "sam3", "owlvit2"],
        choices=["grounding_dino", "falcon", "sam3", "owlvit2"],
        help="Which servers to test (default: all)",
    )
    parser.add_argument(
        "--litserve-only",
        action="store_true",
        help="Only run LitServe, skip direct inference comparison",
    )
    parser.add_argument("--device", default="cuda:0", help="GPU for direct inference")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to process (0 = all)",
    )
    parser.add_argument(
        "--port",
        nargs=2,
        action="append",
        metavar=("SERVER", "PORT"),
        help="Override port: --port grounding_dino 3001",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save comparison results to JSON file",
    )

    args = parser.parse_args()

    # Build port map
    ports = dict(DEFAULT_PORTS)
    if args.port:
        for server, port in args.port:
            ports[server] = int(port)

    # Check server availability
    print("Checking server availability...")
    for server in args.servers:
        port = ports.get(server, DEFAULT_PORTS[server])
        ok = _litserve_health(port)
        status = "OK" if ok else "UNREACHABLE"
        print(f"  {server:<20} :{port}  {status}")

    if args.image:
        results = compare_image(
            args.image, args.servers, ports, args.litserve_only, args.device
        )
        if not args.litserve_only:
            print_summary(results)
    else:
        results = compare_directory(
            args.image_dir, args.servers, ports, args.litserve_only,
            args.device, args.max_images,
        )

    if args.save_results and results:
        # Serialize — strip non-serializable items
        serializable = {}
        for model, data in results.items():
            serializable[model] = data.get("stats", data)
        Path(args.save_results).write_text(json.dumps(serializable, indent=2))
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
