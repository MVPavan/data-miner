"""Build a shadow pipeline.db under <job_dir>/<rereview_subdir>/ so the
production viewer can visualise the re-review results without any changes
to production state.

What it does:
  1. Online-copy production pipeline.db -> rereview/pipeline.db.
  2. For every image in rereview/verdicts.jsonl, rewrite its
     stages.finalize row so:
       - non-target annotations are preserved verbatim from production,
       - target (re-reviewed) annotations are replaced with the re-review
         outcomes (accepted [possibly relabeled] or review or dropped).
  3. Copy classes.txt + config.yaml (if present) so the viewer can render.

Revert = rm -rf <rereview_subdir>/pipeline.db (or the whole subdir).

Run:
  python -m data_miner.auto_annotation_v4.scripts.prepare_rereview_viewer \\
      output/auto_annotation_v4/loco_unannotated_full_sam_filtered_detect \\
      --rereview-subdir rereview

Then:
  python -m data_miner.auto_annotation_v4.viewer \\
      --job-dir output/.../loco_.../rereview --port 8994
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_dir", type=Path)
    p.add_argument("--rereview-subdir", default="rereview")
    p.add_argument("--classes", default="forklift,palletjack",
                   help="Target classes in the re-review (for bookkeeping only).")
    return p.parse_args()


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
    rr_dir = job_dir / args.rereview_subdir
    verdicts_path = rr_dir / "verdicts.jsonl"
    prod_db = job_dir / "pipeline.db"
    shadow_db = rr_dir / "pipeline.db"
    target_classes = {c.strip() for c in args.classes.split(",") if c.strip()}

    if not prod_db.exists():
        print(f"ERROR: production pipeline.db missing: {prod_db}", file=sys.stderr); return 2
    if not verdicts_path.exists():
        print(f"ERROR: verdicts.jsonl missing: {verdicts_path}", file=sys.stderr); return 2
    rr_dir.mkdir(parents=True, exist_ok=True)

    # 1. Online backup (handles WAL) — fresh copy every run, so re-prepping is idempotent.
    print(f"online-backup {prod_db} -> {shadow_db}")
    _online_backup(prod_db, shadow_db)

    # 2. Index verdicts by image_id.
    verdicts_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in verdicts_path.open():
        r = json.loads(line)
        verdicts_by_image[r["image_id"]].append(r)
    print(f"verdicts: {sum(len(v) for v in verdicts_by_image.values())} "
          f"rows across {len(verdicts_by_image)} images")

    # Build class_name -> class_id from production job (read classes.txt if
    # present, otherwise scrape from existing finalize rows).
    classes_txt = job_dir / "classes.txt"
    if classes_txt.exists():
        shutil.copyfile(classes_txt, rr_dir / "classes.txt")
    config_yaml = job_dir / "config.yaml"
    if config_yaml.exists():
        shutil.copyfile(config_yaml, rr_dir / "config.yaml")

    # 3. Patch stages.finalize for each image with verdicts.
    #    For each such image, load the existing finalize JSON, split into
    #    target/non-target, replace targets with re-review outcomes.
    con = sqlite3.connect(shadow_db)
    con.row_factory = sqlite3.Row

    # Build {candidate_id -> class_id} from the original finalize so we can
    # map rereview's class_name back to the canonical numeric id used in
    # FinalAnnotation. Only read target candidates per image on demand.

    patched = 0
    skipped = 0
    for image_id, verdicts in verdicts_by_image.items():
        row = con.execute(
            "SELECT data FROM stages WHERE image_id=? AND stage='finalize'",
            (image_id,),
        ).fetchone()
        if row is None:
            skipped += 1
            continue
        fin = json.loads(row["data"])
        orig_anns = fin.get("final_annotations") or []
        # Map candidate_id -> class_id from the ORIGINAL finalize (so relabels
        # keep a stable id when the class name stays target-set).
        cid_to_class_id: dict[str, int] = {
            a["candidate_id"]: a["class_id"] for a in orig_anns
        }
        # Non-target rows: preserved verbatim. Target rows: replaced by verdicts.
        kept: list[dict[str, Any]] = [
            a for a in orig_anns if a.get("class_name") not in target_classes
        ]

        # Build a class_name -> class_id lookup using ALL images' finalize rows
        # once (lazy cache, module-level). Simplest: read from this image if the
        # class is already here; otherwise fall back to any row.
        def _class_id_for(cn: str) -> int | None:
            for a in orig_anns:
                if a["class_name"] == cn:
                    return a["class_id"]
            return None

        new_review_items: list[dict[str, Any]] = list(fin.get("review_items") or [])
        new_dropped: list[dict[str, Any]] = list(fin.get("dropped") or [])

        for v in verdicts:
            cid = v["candidate_id"]
            outcome = v["outcome"]
            orig_class = v["orig_class"]
            relabel_to = v.get("relabel_to")
            # The candidate MUST be in orig_anns (we only wrote rereview for accepted targets).
            src = next((a for a in orig_anns if a["candidate_id"] == cid), None)
            if src is None:
                # Was filtered or routed away pre-finalize — nothing to patch.
                continue
            if outcome == "accepted":
                new_class = relabel_to or orig_class
                new_id = _class_id_for(new_class)
                if new_id is None:
                    # Relabeled to a class we don't have in any finalize row
                    # (rare — e.g. a class that was never accepted anywhere).
                    # Fall back to the original id so the box still renders.
                    new_id = src["class_id"]
                patched_ann = dict(src)
                patched_ann["class_name"] = new_class
                patched_ann["class_id"] = new_id
                trace = list(patched_ann.get("trace") or [])
                trace.append(f"rereview:accepted conf={v['class_confidence']} "
                             f"bbox={v['bbox_score']}" +
                             (f" relabel={orig_class}->{new_class}" if relabel_to else ""))
                patched_ann["trace"] = trace
                kept.append(patched_ann)
            elif outcome == "review":
                new_review_items.append({
                    "candidate_id": cid,
                    "class_name": v.get("detected_class") or orig_class,
                    "orig_class": orig_class,
                    "bbox": src["bbox"],
                    "class_confidence": v["class_confidence"],
                    "bbox_score": v["bbox_score"],
                    "reasoning": v.get("reasoning"),
                    "source": "rereview",
                })
            elif outcome == "rejected":
                new_dropped.append({
                    "candidate_id": cid,
                    "class_name": orig_class,
                    "reason": "rereview_rejected",
                    "detail": v.get("reasoning"),
                    "context": "post_review",
                })
            # outcome == "skip": leave original in place, add to kept verbatim
            elif outcome == "skip":
                kept.append(dict(src))

        fin["final_annotations"] = kept
        fin["review_items"] = new_review_items
        fin["dropped"] = new_dropped
        fin.setdefault("filter_stats", {})["rereview"] = len(verdicts)

        con.execute(
            "UPDATE stages SET data=? WHERE image_id=? AND stage='finalize'",
            (json.dumps(fin), image_id),
        )
        patched += 1

    con.commit()
    # Free the WAL so the viewer sees a flat db.
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()

    print(f"patched finalize rows: {patched}")
    print(f"skipped (no finalize row): {skipped}")
    print()
    print(f"Now run:  python -m data_miner.auto_annotation_v4.viewer "
          f"--job-dir {rr_dir} --port 8994")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
