"""Export YOLO-format .txt labels from detect-stage proposals.

Streams `proposals` rows (no bulk fetch) for images whose detect stage has
completed, converts each candidate bbox to YOLO format (class_id cx cy w h,
normalized to [0,1]), and writes one .txt per image into an output folder.

Usage:
    python export_yolo_from_proposals.py <JOB_DIR> [--model MODEL]
                                         [--out-dir DIR]
                                         [--min-score FLOAT]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def load_class_id_map(job_dir: Path) -> dict[str, int]:
    """Load {class_name: class_id} from config.yaml's class_registry.

    Falls back to positional indexing of classes.txt if the config is absent
    or unparseable.
    """
    cfg_path = job_dir / "config.yaml"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict):
            reg = data.get("class_registry") or {}
            if isinstance(reg, dict):
                out = {
                    name: int(cls["id"])
                    for name, cls in reg.items()
                    if isinstance(cls, dict) and "id" in cls
                }
                if out:
                    return out

    classes_txt = job_dir / "classes.txt"
    if classes_txt.exists():
        names = [
            ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        return {n: i for i, n in enumerate(names)}

    return {}


def bbox_to_yolo(bbox: dict) -> tuple[float, float, float, float] | None:
    """Convert {x1,y1,x2,y2} (normalized) to (cx,cy,w,h), clamped to [0,1]."""
    try:
        x1 = max(0.0, min(1.0, float(bbox["x1"])))
        y1 = max(0.0, min(1.0, float(bbox["y1"])))
        x2 = max(0.0, min(1.0, float(bbox["x2"])))
        y2 = max(0.0, min(1.0, float(bbox["y2"])))
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path, help="Pipeline job directory")
    ap.add_argument(
        "--model",
        default="sam3_dart",
        help="Which proposals.model to export (default: sam3_dart)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: <job_dir>/labels_detect)",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop candidates below this score (default: 0.0 = keep all)",
    )
    args = ap.parse_args()

    job_dir: Path = args.job_dir.resolve()
    db_path = job_dir / "pipeline.db"
    out_dir: Path = (args.out_dir or (job_dir / "labels_detect")).resolve()

    if not db_path.exists():
        print(f"[error] pipeline.db not found at {db_path}", file=sys.stderr)
        return 2

    class_id = load_class_id_map(job_dir)
    if not class_id:
        print(
            f"[error] no class mapping (config.yaml or classes.txt) in {job_dir}",
            file=sys.stderr,
        )
        return 3
    print(f"[info] loaded {len(class_id)} classes")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] writing to {out_dir}")

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA query_only = TRUE")
    conn.execute("PRAGMA cache_size = -65536")

    total_rows = conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE model = ?", (args.model,)
    ).fetchone()[0]
    print(f"[info] {total_rows} proposals rows for model={args.model!r}")

    n_images = 0
    n_empty = 0
    n_detections = 0
    n_skipped_class = 0
    n_skipped_bbox = 0
    n_skipped_score = 0
    unknown_classes: dict[str, int] = {}

    cursor = conn.execute(
        "SELECT image_id, data FROM proposals WHERE model = ? ORDER BY image_id",
        (args.model,),
    )

    for row in cursor:
        image_id, raw = row
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = payload.get("candidates") or []

        lines: list[str] = []
        for c in candidates:
            name = c.get("class_name") or c.get("label")
            if name not in class_id:
                n_skipped_class += 1
                unknown_classes[name or "<none>"] = unknown_classes.get(name or "<none>", 0) + 1
                continue
            score = c.get("score")
            if score is not None and float(score) < args.min_score:
                n_skipped_score += 1
                continue
            yolo = bbox_to_yolo(c.get("bbox") or {})
            if yolo is None:
                n_skipped_bbox += 1
                continue
            cx, cy, w, h = yolo
            lines.append(f"{class_id[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Write even when empty (dataset invariant: one .txt per image).
        (out_dir / f"{image_id}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )

        n_images += 1
        n_detections += len(lines)
        if not lines:
            n_empty += 1

        if n_images % 10000 == 0:
            print(
                f"[progress] {n_images}/{total_rows}  "
                f"detections={n_detections}  empty={n_empty}",
                flush=True,
            )

    conn.close()

    print()
    print("=== done ===")
    print(f"images written:     {n_images}")
    print(f"  with detections:  {n_images - n_empty}")
    print(f"  empty .txt:       {n_empty}")
    print(f"total detections:   {n_detections}")
    print(f"skipped bbox bad:   {n_skipped_bbox}")
    print(f"skipped score:      {n_skipped_score}  (threshold={args.min_score})")
    print(f"skipped unknown cls:{n_skipped_class}")
    if unknown_classes:
        top = sorted(unknown_classes.items(), key=lambda x: -x[1])[:10]
        print("  top unknown:      " + ", ".join(f"{n}={c}" for n, c in top))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
