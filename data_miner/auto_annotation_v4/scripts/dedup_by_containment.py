"""One-shot: remove same-class bboxes that are nearly fully contained in
another bbox of the same class and that IoU-only dedup missed.

Why this script exists
----------------------
Within-class dedup (``cluster_and_collapse``) uses IoU >= threshold (default
0.7) to decide whether two same-class bboxes are the same physical object.
But a small bbox fully inside a larger bbox of the same class can have a
very low IoU (e.g. a face-region head nested in a wider head-with-hair
detection gives IoU ~0.45 and containment ~0.99). IoU-based NMS keeps both.

This script reads an existing job's pipeline.db read-only, finds same-class
pairs where ``containment = intersection / min(area_a, area_b) >= threshold``
(default 0.8), drops the LOWER-scoring member of each pair, and writes
cleaned YOLO labels to ``<job_dir>/<out_subdir>/``.

Revert = ``rm -rf <job_dir>/<out_subdir>/``. Production labels/ and
pipeline.db are never touched.

Example:
    python -m data_miner.auto_annotation_v4.scripts.dedup_by_containment \\
        output/auto_annotation_v4/fl_pj_frames_dedup_v1_cls_0_85 \\
        --containment-min 0.8 \\
        --out-subdir labels_deduped
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_dir", type=Path)
    p.add_argument(
        "--containment-min", type=float, default=0.8,
        help="Drop the lower-scoring bbox of a same-class pair when "
             "intersection/min(area_a, area_b) >= this threshold. 0.0 disables.",
    )
    p.add_argument(
        "--out-subdir", default="labels_deduped",
        help="Subdirectory under <job_dir> to write cleaned labels into.",
    )
    p.add_argument(
        "--skip-overlap-exempt", action="store_true",
        help="Skip classes like person/head where nested detections can be "
             "legitimate (disabled by default — head-in-head is the main fix).",
    )
    return p.parse_args()


def _iou_containment(a: dict[str, float], b: dict[str, float]) -> tuple[float, float]:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (a["x2"] - a["x1"])) * max(0.0, (a["y2"] - a["y1"]))
    area_b = max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"]))
    if inter <= 0 or area_a == 0 or area_b == 0:
        return 0.0, 0.0
    iou = inter / (area_a + area_b - inter)
    contain = inter / min(area_a, area_b)
    return iou, contain


def _bbox_to_yolo_line(class_id: int, bb: dict[str, float], score: float | None) -> str:
    cx = (bb["x1"] + bb["x2"]) / 2
    cy = (bb["y1"] + bb["y2"]) / 2
    w = bb["x2"] - bb["x1"]
    h = bb["y2"] - bb["y1"]
    base = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    return f"{base} {score:.6f}" if score is not None else base


def _online_backup(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    s = sqlite3.connect(src)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        s.close(); d.close()


def main() -> int:
    args = _parse_args()
    job_dir = args.job_dir.resolve()
    db_path = job_dir / "pipeline.db"
    out_dir = job_dir / args.out_subdir
    out_labels = out_dir / "labels"
    shadow_db = out_dir / "pipeline.db"

    if not db_path.exists():
        print(f"ERROR: pipeline.db not found at {db_path}"); return 2
    out_labels.mkdir(parents=True, exist_ok=True)

    # 1. Seed a shadow pipeline.db — viewer will read stages.finalize from
    #    this copy. Production DB stays untouched.
    print(f"online-backup {db_path} -> {shadow_db}")
    _online_backup(db_path, shadow_db)

    # Mirror classes.txt and config.yaml so the viewer has context files.
    for aux in ("classes.txt", "config.yaml"):
        src = job_dir / aux
        if src.exists():
            (out_dir / aux).write_bytes(src.read_bytes())

    # Classes to skip even if containment triggers (overlap_exempt ≈ head,
    # person, backpack, handbag). Only skipped when --skip-overlap-exempt.
    exempt = {"backpack", "handbag"} if args.skip_overlap_exempt else set()
    # NB: head and person ARE legitimately the target classes we want to
    # clean up (per the bug report), so they're NOT in the default exempt set.

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # Separate writable connection on the shadow DB for stages.finalize patches.
    shadow_con = sqlite3.connect(shadow_db)
    dropped_by_class: Counter[str] = Counter()
    kept_by_class: Counter[str] = Counter()
    images_changed = 0
    audit_rows = []

    for image_id, data in con.execute(
        "SELECT image_id, data FROM stages WHERE stage='finalize'"
    ):
        d = json.loads(data)
        anns = d.get("final_annotations") or []
        # Group by class
        by_class: dict[str, list[dict[str, Any]]] = {}
        for a in anns:
            by_class.setdefault(a["class_name"], []).append(a)

        to_drop: set[str] = set()
        for cn, items in by_class.items():
            if len(items) < 2 or cn in exempt:
                continue
            # Sort by score desc, then mark lower-scoring members of any
            # pair (k, j) where containment(k,j) >= threshold as dropped.
            # Stable: ensures when a==b==c (chain), we drop the lower ones.
            order = sorted(range(len(items)), key=lambda i: -float(items[i].get("confidence") or items[i].get("score") or 0.0))
            for a_idx in order:
                if items[a_idx]["candidate_id"] in to_drop:
                    continue
                for b_idx in order:
                    if b_idx == a_idx:
                        continue
                    b_cid = items[b_idx]["candidate_id"]
                    if b_cid in to_drop:
                        continue
                    _, contain = _iou_containment(items[a_idx]["bbox"], items[b_idx]["bbox"])
                    if contain >= args.containment_min:
                        # Keep a_idx (higher score), drop b_idx.
                        to_drop.add(b_cid)
                        audit_rows.append({
                            "image_id": image_id,
                            "kept_id": items[a_idx]["candidate_id"],
                            "dropped_id": b_cid,
                            "class": cn,
                            "containment": round(contain, 4),
                            "kept_score": items[a_idx].get("confidence"),
                            "dropped_score": items[b_idx].get("confidence"),
                        })

        # Write cleaned YOLO labels for this image
        kept_rows = []
        for a in anns:
            if a["candidate_id"] in to_drop:
                dropped_by_class[a["class_name"]] += 1
                continue
            kept_by_class[a["class_name"]] += 1
            kept_rows.append(_bbox_to_yolo_line(
                a["class_id"], a["bbox"],
                score=a.get("confidence"),
            ))
        (out_labels / f"{image_id}.txt").write_text(
            "\n".join(kept_rows) + ("\n" if kept_rows else "")
        )

        # Patch shadow pipeline.db's stages.finalize row so the viewer sees
        # the deduped annotations (no change to production DB).
        if to_drop:
            images_changed += 1
            patched_anns = [a for a in anns if a["candidate_id"] not in to_drop]
            # Also append a dropped entry per removed candidate so the viewer
            # surfaces the dedup reason rather than the bbox just vanishing.
            patched_dropped = list(d.get("dropped") or [])
            for cid in to_drop:
                patched_dropped.append({
                    "candidate_id": cid,
                    "class_name": next(
                        (a["class_name"] for a in anns if a["candidate_id"] == cid),
                        "?",
                    ),
                    "reason": "same_class_containment",
                    "context": "post_finalize",
                    "detail": f"dedup_by_containment containment>={args.containment_min}",
                })
            patched = dict(d)
            patched["final_annotations"] = patched_anns
            patched["dropped"] = patched_dropped
            patched.setdefault("filter_stats", {})["same_class_containment_dropped"] = len(to_drop)
            shadow_con.execute(
                "UPDATE stages SET data=? WHERE image_id=? AND stage='finalize'",
                (json.dumps(patched), image_id),
            )

    shadow_con.commit()
    shadow_con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shadow_con.close()

    # Audit trail
    with (out_dir / "audit.jsonl").open("w") as f:
        for r in audit_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "job_dir": str(job_dir),
        "containment_min": args.containment_min,
        "exempt_classes": sorted(exempt),
        "images_changed": images_changed,
        "total_boxes_dropped": sum(dropped_by_class.values()),
        "dropped_by_class": dict(dropped_by_class.most_common()),
        "kept_by_class": dict(kept_by_class.most_common()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"done.")
    print(f"images changed:         {images_changed}")
    print(f"total boxes dropped:    {sum(dropped_by_class.values())}")
    print(f"dropped by class (top 10):")
    for cn, n in dropped_by_class.most_common(10):
        print(f"  {cn:20s} {n}")
    print(f"output: {out_dir}")
    print()
    print(f"View the deduped result:")
    print(f"  python -m data_miner.auto_annotation_v4.viewer "
          f"--job-dir {out_dir} --port 8995")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
