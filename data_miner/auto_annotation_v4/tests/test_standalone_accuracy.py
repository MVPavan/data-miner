"""Integration test: standalone YOLO output vs pipeline stored proposals.

Picks a sample of image_ids from an existing ``pipeline.db``, symlinks them
into a staging dir, runs the standalone runner twice (all classes + a
filtered subset), and compares the written ``.txt`` files against the
pipeline's stored ``sam3_dart`` candidates using IoU-based greedy matching.

Not a pytest — requires a populated pipeline.db and the image files on disk.
Invoke directly:

    .venv/bin/python -m data_miner.auto_annotation_v4.tests.test_standalone_accuracy \\
        --job-dir output/auto_annotation_v4/laion_good1 \\
        --image-root /mnt/data/deepak/DATASET/Laion/Good1/good1 \\
        [--samples 100] [--confidence 0.5] [--iou 0.5] \\
        [--classes forklift,palletjack]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(0.0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def pick_samples(
    conn: sqlite3.Connection, n: int, image_root: Path, model_name: str,
) -> list[tuple[str, str]]:
    """Return [(image_id, path)] — mix of rare-class + rich-detection + random."""
    rare = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT image_id FROM proposals "
            "WHERE model=? AND (data LIKE '%\"forklift\"%' OR data LIKE '%palletjack%') "
            "ORDER BY image_id LIMIT ?", (model_name, max(20, n // 4)),
        )
    ]
    with_dets = [
        r[0] for r in conn.execute(
            "SELECT image_id FROM proposals "
            "WHERE model=? AND data LIKE '%\"candidates\":[{%' "
            "ORDER BY image_id LIMIT ?", (model_name, n),
        )
    ]
    random_ids = [
        r[0] for r in conn.execute(
            "SELECT image_id FROM proposals WHERE model=? "
            "ORDER BY image_id LIMIT ?", (model_name, n * 10),
        )
    ]
    random.shuffle(random_ids)

    picked: list[str] = []
    seen: set[str] = set()
    for bag in (rare, with_dets, random_ids):
        for iid in bag:
            if iid not in seen:
                seen.add(iid)
                picked.append(iid)
            if len(picked) >= n:
                break
        if len(picked) >= n:
            break

    id_to_path: dict[str, str] = {}
    for stem in picked:
        for ext in IMG_EXTS:
            p = image_root / f"{stem}{ext}"
            if p.exists():
                id_to_path[stem] = str(p)
                break
    return [(s, id_to_path[s]) for s in picked if s in id_to_path]


def stage(stage_dir: Path, samples: list[tuple[str, str]]) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for image_id, path in samples:
        (stage_dir / f"{image_id}{Path(path).suffix}").symlink_to(path)


# ---------------------------------------------------------------------------
# Runner subprocess
# ---------------------------------------------------------------------------


def run_standalone(
    *, repo: Path, python: str, tag: str, stage_dir: Path, job_dir: Path,
    out_dir: Path, confidence: float, extra_args: list[str],
) -> None:
    print(f"\n[run] {tag}: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [
        python, "-u", "-m", "data_miner.auto_annotation_v4.models.run_sam3_dart_standalone",
        "--image-dir", str(stage_dir),
        "--job-dir", str(job_dir),
        "--out-dir", str(out_dir),
        "--device", "cuda:0",
        "--batch-size", "8",
        "--num-workers", "6",
        "--confidence", str(confidence),
        "--log-every", "10",
        *extra_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    r = subprocess.run(cmd, cwd=str(repo), env=env,
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print("STDERR tail:")
        print(r.stderr[-2000:])
        raise SystemExit(f"{tag} failed rc={r.returncode}")
    tail = r.stdout.strip().splitlines()[-5:]
    print("  " + "\n  ".join(tail))


# ---------------------------------------------------------------------------
# Reading sides
# ---------------------------------------------------------------------------


def load_pipeline_proposals(
    db: Path, model_name: str,
) -> dict[str, list[dict]]:
    """Read pipeline ``proposals`` rows → {image_id: [candidate, ...]}."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, list[dict]] = {}
    for image_id, data in conn.execute(
        "SELECT image_id, data FROM proposals WHERE model=?", (model_name,),
    ):
        try:
            payload = json.loads(data)
        except Exception:
            continue
        out[image_id] = list(payload.get("candidates") or [])
    conn.close()
    return out


def load_yolo_dir(
    yolo_dir: Path, id_to_class: dict[int, str],
) -> dict[str, list[dict]]:
    """Read YOLO .txt files → {image_id: [{class_name, bbox, score}, ...]}.

    Expects ``class_id cx cy w h [score]`` per line; trailing newline OK;
    empty files produce an empty list (consistent with the runner).
    """
    out: dict[str, list[dict]] = {}
    for txt in yolo_dir.glob("*.txt"):
        image_id = txt.stem
        cands: list[dict] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cls_name = id_to_class.get(cid)
            if cls_name is None:
                continue
            cx, cy, w, h = (float(x) for x in parts[1:5])
            score = float(parts[5]) if len(parts) >= 6 else 1.0
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            cands.append({
                "class_name": cls_name,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "score": score,
            })
        out[image_id] = cands
    return out


def load_id_to_class(job_dir: Path) -> dict[int, str]:
    cfg_path = job_dir / "config.yaml"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    reg = data.get("class_registry") or {}
    return {
        int(cls["id"]): name
        for name, cls in reg.items()
        if isinstance(cls, dict) and "id" in cls
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def greedy_match(
    a: list[dict], b: list[dict], iou_threshold: float,
) -> tuple[int, list[float]]:
    matched = 0
    diffs: list[float] = []
    used_b: set[int] = set()
    for ca in a:
        best_j, best_iou = -1, 0.0
        for j, cb in enumerate(b):
            if j in used_b or ca.get("class_name") != cb.get("class_name"):
                continue
            ov = iou(ca["bbox"], cb["bbox"])
            if ov > best_iou:
                best_j, best_iou = j, ov
        if best_j >= 0 and best_iou >= iou_threshold:
            used_b.add(best_j)
            matched += 1
            diffs.append(
                float(ca.get("score", 0.0)) - float(b[best_j].get("score", 0.0))
            )
    return matched, diffs


def compare(
    name: str, pipe: dict[str, list[dict]], mine: dict[str, list[dict]],
    iou_threshold: float, restrict_classes: set[str] | None = None,
) -> dict:
    if restrict_classes is not None:
        pipe = {
            k: [c for c in v if c.get("class_name") in restrict_classes]
            for k, v in pipe.items()
        }
    shared = sorted(set(pipe) & set(mine))
    pipe_only = sorted(set(pipe) - set(mine))
    mine_only = sorted(set(mine) - set(pipe))
    print(f"\n===== compare: {name} =====")
    print(f"shared images: {len(shared)}  pipe-only: {len(pipe_only)}  mine-only: {len(mine_only)}")

    total_pipe = total_mine = total_matched = 0
    class_pipe: Counter = Counter()
    class_mine: Counter = Counter()
    all_diffs: list[float] = []
    perfect = 0

    for iid in shared:
        p, m = pipe[iid], mine[iid]
        matched, diffs = greedy_match(p, m, iou_threshold)
        total_pipe += len(p)
        total_mine += len(m)
        total_matched += matched
        all_diffs.extend(diffs)
        for c in p: class_pipe[c["class_name"]] += 1
        for c in m: class_mine[c["class_name"]] += 1
        if matched == len(p) == len(m):
            perfect += 1

    def pct(num: int, den: int) -> str:
        return f"{100.0 * num / max(1, den):.1f}%"

    print(f"pipeline candidates (filtered): {total_pipe}")
    print(f"standalone candidates:          {total_mine}")
    print(f"matched (IoU>={iou_threshold}, same class): {total_matched}")
    print(f"  recall (matched/pipe):   {pct(total_matched, total_pipe)}")
    print(f"  precision (matched/mine):{pct(total_matched, total_mine)}")
    print(f"  images perfect:          {perfect}/{len(shared)} ({pct(perfect, len(shared))})")
    if all_diffs:
        all_diffs.sort()
        med = all_diffs[len(all_diffs) // 2]
        mean = sum(all_diffs) / len(all_diffs)
        print(f"  score Δ (mine - pipe): mean={mean:+.4f} median={med:+.4f} n={len(all_diffs)}")

    print("\n  class distribution:")
    for cls in sorted(set(class_pipe) | set(class_mine)):
        print(f"    {cls:<20} pipe={class_pipe[cls]:<5} mine={class_mine[cls]:<5}")

    return {
        "shared": len(shared), "pipe": total_pipe, "mine": total_mine,
        "matched": total_matched, "perfect": perfect,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", type=Path,
                    default=repo / "output/auto_annotation_v4/laion_good1")
    ap.add_argument("--image-root", type=Path,
                    default=Path("/mnt/data/deepak/DATASET/Laion/Good1/good1"))
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--confidence", type=float, default=0.5,
                    help="Match the pipeline's SAM3DartModel default (0.5)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classes", default="forklift,palletjack",
                    help="Comma-separated class subset for the filtered run")
    ap.add_argument("--model-name", default="sam3_dart")
    ap.add_argument("--python", default=str(repo / ".venv/bin/python"))
    ap.add_argument("--stage-dir", type=Path, default=Path("/tmp/accuracy_sample"))
    ap.add_argument("--full-out", type=Path,
                    default=Path("/tmp/accuracy_full_yolo"))
    ap.add_argument("--cls-out", type=Path,
                    default=Path("/tmp/accuracy_2class_yolo"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-full", action="store_true",
                    help="Only run the class-filtered comparison")
    ap.add_argument("--skip-filter", action="store_true",
                    help="Only run the all-classes comparison")
    args = ap.parse_args()

    random.seed(args.seed)

    pipeline_db = args.job_dir / "pipeline.db"
    if not pipeline_db.exists():
        raise SystemExit(f"pipeline.db not found: {pipeline_db}")
    print(f"pipeline.db = {pipeline_db}")

    # --- Pick + stage sample images ---
    conn = sqlite3.connect(f"file:{pipeline_db}?mode=ro", uri=True)
    samples = pick_samples(conn, args.samples, args.image_root, args.model_name)
    conn.close()
    print(f"[sample] picked {len(samples)}/{args.samples}")
    if not samples:
        raise SystemExit("no samples found — check pipeline.db and --image-root")

    stage(args.stage_dir, samples)
    print(f"[stage] {len(samples)} symlinks in {args.stage_dir}")

    # --- Run standalone ---
    if not args.skip_full:
        run_standalone(
            repo=repo, python=args.python, tag="FULL (all classes)",
            stage_dir=args.stage_dir, job_dir=args.job_dir,
            out_dir=args.full_out, confidence=args.confidence, extra_args=[],
        )
    if not args.skip_filter:
        run_standalone(
            repo=repo, python=args.python,
            tag=f"FILTERED ({args.classes})",
            stage_dir=args.stage_dir, job_dir=args.job_dir,
            out_dir=args.cls_out, confidence=args.confidence,
            extra_args=["--classes", args.classes],
        )

    # --- Load both sides ---
    id_to_class = load_id_to_class(args.job_dir)
    pipe = load_pipeline_proposals(pipeline_db, args.model_name)
    sample_ids = {iid for iid, _ in samples}
    pipe = {k: v for k, v in pipe.items() if k in sample_ids}

    rc = 0
    if not args.skip_full:
        full = load_yolo_dir(args.full_out, id_to_class)
        r = compare("ALL-CLASSES standalone vs pipeline", pipe, full, args.iou)
        if r["matched"] != r["pipe"] or r["matched"] != r["mine"]:
            print("\n[FAIL] ALL-CLASSES standalone diverges from pipeline")
            rc = 1
    if not args.skip_filter:
        cls2 = load_yolo_dir(args.cls_out, id_to_class)
        subset = {c.strip() for c in args.classes.split(",") if c.strip()}
        r = compare(
            f"FILTERED standalone vs pipeline (restricted to {sorted(subset)})",
            pipe, cls2, args.iou, restrict_classes=subset,
        )
        if r["matched"] != r["pipe"] or r["matched"] != r["mine"]:
            print("\n[FAIL] FILTERED standalone diverges from pipeline")
            rc = 1

    if rc == 0:
        print("\n[OK] standalone YOLO output matches pipeline proposals on the sampled set")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
