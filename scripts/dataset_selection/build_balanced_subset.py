"""
Build an N-image YOLO subset from select_diverse_subset.py's selected list,
biased toward rare videos.

Strategy:
    1. Group `manifest.selected` by clip (same clip-derivation rules as
       viewer_subset.py — auto-derived, or canonical via --clips-file).
    2. Sort clips ascending by count (rarest first; tie-break by clip name).
    3. Add whole clips in that order until adding the next clip would exceed
       --target. Then take just enough leading frames from the next clip to
       land exactly on --target (preserving FPS order within the clip — i.e.
       the order frames appear in `manifest.selected`).
    4. Materialize a YOLO dataset (images/ + labels/ + classes.txt +
       dataset.yaml) using aav4 final labels.

Output:
    <out_dir>/
        sub_manifest.json   stems chosen + per-clip counts + selection trace
        images/             symlink (or copy) of source images
        labels/             aav4 YOLO labels for those stems
        classes.txt
        dataset.yaml

Usage:
    python -m scripts.dataset_selection.build_balanced_subset \
        --manifest output/dataset_selection/datatang_diverse_1000/manifest.json \
        --aav4-job-dir output/auto_annotation_v4/datatang_val_detect \
        --out-dir output/dataset_selection/datatang_balanced_250 \
        --target 250 \
        --link
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from scripts.dataset_selection.viewer_subset import derive_clip_map_with_overrides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="manifest.json from select_diverse_subset.py")
    ap.add_argument("--aav4-job-dir", type=Path, required=True,
                    help="aav4 job dir containing labels/ and classes.txt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--clips-file", type=Path, default=None,
                    help="Optional canonical video-name list (longest-prefix match)")
    ap.add_argument("--link", action="store_true",
                    help="Symlink images instead of copying")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    stem_to_path: dict[str, str] = json.loads(
        (args.manifest.parent / "stem_to_path.json").read_text()
    )
    selected: list[str] = manifest["selected"]

    # Clip mapping over the WHOLE stem set so we use the same heuristic as the
    # viewer (canonical names depend on dataset-wide frequency).
    all_stems = manifest["selected"] + manifest["dedup_drops"] + manifest["fps_drops"]
    canonical = None
    if args.clips_file is not None:
        canonical = [
            line.strip()
            for line in args.clips_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    stem_to_clip = derive_clip_map_with_overrides(all_stems, canonical)

    # Group selected stems by clip; preserve original order (FPS-determined).
    by_clip: dict[str, list[str]] = defaultdict(list)
    for stem in selected:
        by_clip[stem_to_clip[stem]].append(stem)

    # Sort clips by ascending count, ties by name.
    clips_sorted = sorted(by_clip.keys(), key=lambda c: (len(by_clip[c]), c))

    chosen: list[str] = []
    trace: list[dict] = []
    for clip in clips_sorted:
        stems_here = by_clip[clip]
        remaining = args.target - len(chosen)
        if remaining <= 0:
            break
        if len(stems_here) <= remaining:
            chosen.extend(stems_here)
            trace.append({"clip": clip, "available": len(stems_here),
                          "taken": len(stems_here), "partial": False})
        else:
            chosen.extend(stems_here[:remaining])
            trace.append({"clip": clip, "available": len(stems_here),
                          "taken": remaining, "partial": True})
            break

    if len(chosen) != args.target:
        print(f"WARNING: only assembled {len(chosen)} stems (target {args.target}) — "
              f"selected pool has too few frames")

    # Materialize the YOLO dir.
    out_dir = args.out_dir.resolve()
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    aav4_labels_dir = args.aav4_job_dir / "labels"
    aav4_classes = args.aav4_job_dir / "classes.txt"
    if not aav4_labels_dir.is_dir():
        raise SystemExit(f"missing labels dir: {aav4_labels_dir}")
    if not aav4_classes.is_file():
        raise SystemExit(f"missing classes.txt: {aav4_classes}")

    written = 0
    missing_label, missing_image = [], []
    for stem in chosen:
        src_label = aav4_labels_dir / f"{stem}.txt"
        src_image = stem_to_path.get(stem)
        if not src_label.exists():
            missing_label.append(stem); continue
        if src_image is None or not Path(src_image).exists():
            missing_image.append(stem); continue
        src_image_path = Path(src_image)
        dst_image = out_images / src_image_path.name
        dst_label = out_labels / f"{stem}.txt"
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
        f"# rare-video bias, target {args.target}, "
        f"selected via {args.manifest}\n"
        f"path: {out_dir}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(classes)}\n"
        f"names: [{', '.join(repr(c) for c in classes)}]\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text)

    sub_manifest = {
        "source_manifest": str(args.manifest),
        "target": args.target,
        "actual": len(chosen),
        "clips_used": len(trace),
        "selection_trace": trace,
        "stems": chosen,
        "missing_label": missing_label,
        "missing_image": missing_image,
    }
    (out_dir / "sub_manifest.json").write_text(json.dumps(sub_manifest, indent=2))

    print(f"\nWrote {written} image+label pairs to {out_dir}")
    print(f"  target        : {args.target}")
    print(f"  chosen        : {len(chosen)}")
    print(f"  clips used    : {len(trace)} (smallest first)")
    print(f"  partial last  : {trace[-1]['partial'] if trace else False}")
    print(f"  missing label : {len(missing_label)}")
    print(f"  missing image : {len(missing_image)}")
    print(f"\nSelection trace (clip → taken/available):")
    for t in trace:
        flag = " *partial*" if t["partial"] else ""
        print(f"  {t['taken']:4d}/{t['available']:<4d}  {t['clip']}{flag}")


if __name__ == "__main__":
    main()
