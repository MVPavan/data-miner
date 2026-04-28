"""Rewrite a YOLO labels_standalone tree, dropping bboxes whose class was
marked ``fail`` in the ``label_validation`` sidecar table.

Classes not present in the verdict table (i.e. not in any
``classes_to_check`` run) are passed through untouched.

Run::

    python -m data_miner.label_validation.apply_verdicts \\
        --lance-uri /mnt/data/data_miner_lance/laion_good3 \\
        --verdict-table label_validation \\
        --labels-in output/auto_annotation_v4/laion_good3/labels_standalone \\
        --labels-out output/auto_annotation_v4/laion_good3/labels_standalone_validated \\
        --classes-yaml output/auto_annotation_v4/laion_good3/data.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from .validate_labels import load_id_to_name

log = logging.getLogger("apply_verdicts")


def load_failed_pairs(lance_uri: Path, table_name: str) -> dict[str, set[str]]:
    """Return ``{image_id: {class_name, ...}}`` for every verdict == 'fail'."""
    import lancedb
    db = lancedb.connect(str(lance_uri))
    if table_name not in db.table_names():
        raise FileNotFoundError(f"verdict table missing: {lance_uri}/{table_name}")
    tbl = db.open_table(table_name)
    failed: dict[str, set[str]] = defaultdict(set)
    ds = tbl.to_lance()
    reader = ds.scanner(columns=["image_id", "class_name", "verdict"]).to_reader()
    for rb in reader:
        ids = rb.column("image_id").to_pylist()
        names = rb.column("class_name").to_pylist()
        verdicts = rb.column("verdict").to_pylist()
        for i, n, v in zip(ids, names, verdicts):
            if v == "fail":
                failed[i].add(n)
    return failed


def rewrite_labels(
    labels_in: Path,
    labels_out: Path,
    failed: dict[str, set[str]],
    id_to_name: dict[int, str],
    drop_empty: bool,
    limit: int = 0,
) -> dict:
    labels_out.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_copied = 0
    n_changed = 0
    n_lines_kept = 0
    n_lines_dropped = 0
    n_empty_after = 0

    for entry in os.scandir(labels_in):
        if not entry.name.endswith(".txt"):
            continue
        n_files += 1
        stem = entry.name[:-4]
        failed_classes = failed.get(stem)

        # Fast-path: no failure recorded for this image — plain copy.
        if not failed_classes:
            # Symlink would be lighter, but copying is safer for downstream
            # tools that resolve paths. We read+write to preserve line
            # endings as whatever the input uses.
            with open(entry.path, "rb") as src:
                data = src.read()
            with open(labels_out / entry.name, "wb") as dst:
                dst.write(data)
            n_copied += 1
            n_lines_kept += data.count(b"\n")
            if limit and n_files >= limit:
                break
            continue

        kept_lines: list[str] = []
        dropped = 0
        with open(entry.path) as f:
            for raw in f:
                stripped = raw.strip()
                if not stripped:
                    continue
                head = stripped.split(" ", 1)[0]
                try:
                    cid = int(head)
                except ValueError:
                    kept_lines.append(raw.rstrip("\n"))
                    continue
                name = id_to_name.get(cid)
                if name in failed_classes:
                    dropped += 1
                    continue
                kept_lines.append(raw.rstrip("\n"))

        n_lines_kept += len(kept_lines)
        n_lines_dropped += dropped
        n_changed += 1

        if not kept_lines:
            n_empty_after += 1
            if drop_empty:
                if limit and n_files >= limit:
                    break
                continue

        out_path = labels_out / entry.name
        with open(out_path, "w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")
            # else: empty file — caller opted out of drop_empty
        if limit and n_files >= limit:
            break

    return {
        "files_total": n_files,
        "files_untouched_copy": n_copied,
        "files_edited": n_changed,
        "files_empty_after_filter": n_empty_after,
        "lines_kept": n_lines_kept,
        "lines_dropped": n_lines_dropped,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-uri", type=str,
                    default="/mnt/data/data_miner_lance/laion_good3")
    ap.add_argument("--verdict-table", type=str, default="label_validation")
    ap.add_argument("--labels-in", type=str,
                    default="output/auto_annotation_v4/laion_good3/labels_standalone")
    ap.add_argument("--labels-out", type=str,
                    default="output/auto_annotation_v4/laion_good3/labels_standalone_validated")
    ap.add_argument("--classes-yaml", type=str,
                    default="output/auto_annotation_v4/laion_good3/data.yaml")
    ap.add_argument("--drop-empty", action="store_true",
                    help="Skip writing label files with zero surviving bboxes "
                         "(otherwise write an empty file).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N label files (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    id_to_name = load_id_to_name(Path(args.classes_yaml))
    failed = load_failed_pairs(Path(args.lance_uri), args.verdict_table)
    log.info("verdict table: %d images with >=1 failed class", len(failed))

    t0 = time.perf_counter()
    stats = rewrite_labels(
        labels_in=Path(args.labels_in),
        labels_out=Path(args.labels_out),
        failed=failed,
        id_to_name=id_to_name,
        drop_empty=args.drop_empty,
        limit=args.limit,
    )
    dt = time.perf_counter() - t0
    log.info("done in %.1fs: %s", dt, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
