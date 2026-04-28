"""Promote <job_dir>/<rereview_subdir>/labels/ over <job_dir>/labels/ with backups.

Only run this AFTER inspecting rereview/ and confirming it's better than the
production labels. This is a one-way mutation of production state — the
script refuses to overwrite without first taking two snapshots:

    1. pipeline.db            -> backups/pipeline-pre-promote-<ts>.db  (SQLite online backup)
    2. labels/                -> backups/labels-pre-promote-<ts>/       (cp -a)

Revert recipe (printed at the end of a successful promote):

    rm -rf labels/
    mv backups/labels-pre-promote-<ts>/ labels/
    # (optional) restore pipeline.db from backups/pipeline-pre-promote-<ts>.db
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_dir", type=Path)
    p.add_argument("--rereview-subdir", default="rereview")
    p.add_argument(
        "--include-review", action="store_true",
        help="Also copy rereview/review/ into the job's review/ dir (appends).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _online_backup(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(src)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        s.close(); d.close()


def main() -> int:
    args = _parse_args()
    job_dir = args.job_dir.resolve()
    src_labels = job_dir / args.rereview_subdir / "labels"
    src_review = job_dir / args.rereview_subdir / "review"
    dst_labels = job_dir / "labels"
    dst_review = job_dir / "review"
    db_path = job_dir / "pipeline.db"
    backups = job_dir / "backups"

    if not src_labels.exists():
        print(f"ERROR: {src_labels} not found", file=sys.stderr); return 2
    if not dst_labels.exists():
        print(f"ERROR: {dst_labels} not found", file=sys.stderr); return 2
    if not db_path.exists():
        print(f"ERROR: {db_path} not found", file=sys.stderr); return 2

    ts = time.strftime("%Y%m%d-%H%M%S")
    db_bk = backups / f"pipeline-pre-promote-{ts}.db"
    labels_bk = backups / f"labels-pre-promote-{ts}"

    n_src = len(list(src_labels.glob("*.txt")))
    n_dst = len(list(dst_labels.glob("*.txt")))
    print(f"source labels:      {src_labels}  ({n_src} files)")
    print(f"production labels:  {dst_labels}  ({n_dst} files)")
    print(f"pipeline.db backup: {db_bk}")
    print(f"labels backup:      {labels_bk}")
    print(f"include review:     {args.include_review}")

    if args.dry_run:
        print("(dry-run — no changes written)")
        return 0

    # --- 1. snapshot pipeline.db via SQLite online backup ---
    print("backing up pipeline.db …")
    _online_backup(db_path, db_bk)

    # --- 2. snapshot labels/ via cp -a ---
    print("copying labels/ …")
    shutil.copytree(dst_labels, labels_bk)

    # --- 3. overwrite labels/<image>.txt from rereview/labels/ ---
    # Files only in rereview/ (not in prod) are also copied — covers images
    # where the original was empty and re-review rescued some.
    print("promoting rereview/labels/* over labels/ …")
    n_overwritten = 0
    for src in src_labels.glob("*.txt"):
        shutil.copy2(src, dst_labels / src.name)
        n_overwritten += 1
    print(f"overwrote {n_overwritten} label files.")

    # --- 4. optionally copy review items ---
    if args.include_review and src_review.exists():
        dst_review.mkdir(exist_ok=True)
        n_rev = 0
        for src in src_review.glob("*.txt"):
            # Append-rather-than-overwrite: existing review items from earlier
            # passes are preserved below the new lines.
            existing = ""
            tgt = dst_review / src.name
            if tgt.exists():
                existing = tgt.read_text()
                if existing and not existing.endswith("\n"):
                    existing += "\n"
            tgt.write_text(existing + src.read_text())
            n_rev += 1
        print(f"merged {n_rev} review files into {dst_review}/")

    print()
    print("revert recipe:")
    print(f"    rm -rf {dst_labels}")
    print(f"    mv {labels_bk} {dst_labels}")
    print(f"    # (optional) cp {db_bk} {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
