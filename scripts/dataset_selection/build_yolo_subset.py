"""
Build a YOLO dataset from an aav4 job's final labels, restricted to the
images selected by select_diverse_subset.py.

Reads:
    - manifest.json     (from select_diverse_subset.py — selected stems)
    - stem_to_path.json (image stem → source image path)
    - aav4 job dir:
          labels/<stem>.txt
          classes.txt

Writes (Ultralytics YOLO layout):
    <out_dir>/
        images/<stem>.<ext>     (copy or symlink of source image)
        labels/<stem>.txt       (copy of aav4 label)
        classes.txt             (copied from aav4 job)
        dataset.yaml            (Ultralytics-friendly: train/val both point at images/)
        missing.json            (selected stems that had no aav4 label)

Usage:
    python -m scripts.dataset_selection.build_yolo_subset \
        --manifest output/dataset_selection/datatang_diverse_1000/manifest.json \
        --aav4-job-dir output/auto_annotation_v4/datatang_val_detect \
        --out-dir output/dataset_selection/datatang_diverse_1000/yolo \
        --link
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="manifest.json from select_diverse_subset.py")
    ap.add_argument("--aav4-job-dir", type=Path, required=True,
                    help="aav4 job dir containing labels/ and classes.txt")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write the YOLO subset")
    ap.add_argument("--link", action="store_true",
                    help="Symlink images instead of copying (saves disk)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Don't error out when some stems have no aav4 label")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    stem_to_path: dict[str, str] = json.loads(
        (args.manifest.parent / "stem_to_path.json").read_text()
    )
    selected: list[str] = manifest["selected"]

    aav4_labels_dir = args.aav4_job_dir / "labels"
    aav4_classes = args.aav4_job_dir / "classes.txt"
    if not aav4_labels_dir.is_dir():
        raise SystemExit(f"missing labels dir: {aav4_labels_dir}")
    if not aav4_classes.is_file():
        raise SystemExit(f"missing classes.txt: {aav4_classes}")

    out_dir = args.out_dir.resolve()
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_label: list[str] = []
    missing_image: list[str] = []

    for stem in selected:
        src_label = aav4_labels_dir / f"{stem}.txt"
        src_image = stem_to_path.get(stem)

        if not src_label.exists():
            missing_label.append(stem)
            continue
        if src_image is None or not Path(src_image).exists():
            missing_image.append(stem)
            continue

        src_image_path = Path(src_image)
        dst_image = out_images / src_image_path.name
        dst_label = out_labels / f"{stem}.txt"

        # idempotent: clear stale symlink/file before re-creating
        if dst_image.is_symlink() or dst_image.exists():
            dst_image.unlink()
        if args.link:
            dst_image.symlink_to(src_image_path)
        else:
            shutil.copy2(src_image_path, dst_image)
        shutil.copy2(src_label, dst_label)
        written += 1

    shutil.copy2(aav4_classes, out_dir / "classes.txt")

    classes = [c for c in aav4_classes.read_text().splitlines() if c.strip()]
    yaml_text = (
        f"# YOLO subset built from {args.aav4_job_dir}\n"
        f"# selected via {args.manifest}\n"
        f"path: {out_dir}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(classes)}\n"
        f"names: [{', '.join(repr(c) for c in classes)}]\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text)

    (out_dir / "missing.json").write_text(json.dumps({
        "missing_label": missing_label,
        "missing_image": missing_image,
    }, indent=2))

    print(f"Wrote {written} image+label pairs to {out_dir}")
    print(f"  classes      : {len(classes)}")
    print(f"  missing label: {len(missing_label)}")
    print(f"  missing image: {len(missing_image)}")
    if (missing_label or missing_image) and not args.allow_missing:
        if missing_label:
            print(f"\nFirst 5 missing labels: {missing_label[:5]}")
        if missing_image:
            print(f"First 5 missing images: {missing_image[:5]}")
        raise SystemExit(
            "Some selected stems were missing — pass --allow-missing to ignore"
        )


if __name__ == "__main__":
    main()
