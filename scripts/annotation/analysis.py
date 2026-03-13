"""
YOLO Label Analysis for CVAT-exported datasets.

Usage:
    python scripts/annotation/analysis.py --dataset cvat_output/doors_v1
    python scripts/annotation/analysis.py --dataset cvat_output/doors_v1 --split train
    python scripts/annotation/analysis.py --dataset cvat_output/delivery_pov_v1 --yaml dataset.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data_yaml(dataset_dir: Path, yaml_name: str = "data.yaml") -> dict:
    """Load the YOLO data.yaml / dataset.yaml from the dataset directory."""
    candidates = [
        dataset_dir / yaml_name,
        dataset_dir / "dataset.yaml",
        dataset_dir / "data.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No YAML config found in {dataset_dir}")


def parse_class_names(cfg: dict) -> dict[int, str]:
    """Return {class_id: name} from data.yaml 'names' field."""
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def load_labels(labels_dir: Path) -> pd.DataFrame:
    """
    Parse all YOLO label .txt files in a directory tree.

    Returns a DataFrame with columns:
        file, class_id, cx, cy, w, h
    Empty label files produce no rows (they appear in the image-level stats as
    files with 0 annotations).
    """
    records = []
    for txt_path in sorted(labels_dir.rglob("*.txt")):
        content = txt_path.read_text().strip()
        if not content:
            # Track empty files so we can report them
            records.append(
                {
                    "file": txt_path.stem,
                    "class_id": np.nan,
                    "cx": np.nan,
                    "cy": np.nan,
                    "w": np.nan,
                    "h": np.nan,
                }
            )
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            records.append(
                {
                    "file": txt_path.stem,
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns to the annotation DataFrame."""
    ann = df.dropna(subset=["class_id"]).copy()
    ann["class_id"] = ann["class_id"].astype(int)
    ann["area"] = ann["w"] * ann["h"]
    ann["aspect_ratio"] = ann["w"] / ann["h"].replace(0, np.nan)
    return ann


def class_distribution(ann: pd.DataFrame, class_names: dict) -> pd.DataFrame:
    counts = ann["class_id"].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    dist = pd.DataFrame(
        {
            "class_id": counts.index,
            "class_name": [class_names.get(i, str(i)) for i in counts.index],
            "count": counts.values,
            "pct": pct.values,
        }
    ).reset_index(drop=True)
    return dist


def per_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image annotation counts (including empty-label images with count=0).
    """
    all_files = df["file"].unique()

    ann = df.dropna(subset=["class_id"])
    counts = ann.groupby("file").size().rename("n_annotations")

    img_df = pd.DataFrame(index=all_files)
    img_df.index.name = "file"
    img_df = img_df.join(counts, how="left").fillna(0).astype(int).reset_index()
    return img_df


def bbox_size_stats(ann: pd.DataFrame, class_names: dict) -> pd.DataFrame:
    """Width, height, area, aspect-ratio statistics per class."""
    rows = []
    for cid, grp in ann.groupby("class_id"):
        rows.append(
            {
                "class_id": cid,
                "class_name": class_names.get(cid, str(cid)),
                "n": len(grp),
                "w_mean": grp["w"].mean(),
                "w_std": grp["w"].std(),
                "w_min": grp["w"].min(),
                "w_max": grp["w"].max(),
                "h_mean": grp["h"].mean(),
                "h_std": grp["h"].std(),
                "h_min": grp["h"].min(),
                "h_max": grp["h"].max(),
                "area_mean": grp["area"].mean(),
                "area_std": grp["area"].std(),
                "area_min": grp["area"].min(),
                "area_max": grp["area"].max(),
                "ar_mean": grp["aspect_ratio"].mean(),
                "ar_median": grp["aspect_ratio"].median(),
            }
        )
    return pd.DataFrame(rows).round(4)


def area_buckets(ann: pd.DataFrame, class_names: dict) -> pd.DataFrame:
    """
    Break bounding-box areas into small/medium/large buckets
    (thresholds in normalised area: <1%, 1-9%, >=9%).
    """
    bins = [0, 0.01, 0.09, 1.01]
    labels_b = ["small (<1%)", "medium (1-9%)", "large (≥9%)"]
    ann = ann.copy()
    ann["size_bucket"] = pd.cut(ann["area"], bins=bins, labels=labels_b, right=False)
    pivot = (
        ann.groupby(["class_id", "size_bucket"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = [f"{i} – {class_names.get(i, str(i))}" for i in pivot.index]
    pivot.index.name = "class"
    return pivot


def co_occurrence(ann: pd.DataFrame, class_names: dict) -> pd.DataFrame:
    """How often do classes appear together in the same image?"""
    image_classes = ann.groupby("file")["class_id"].apply(
        lambda x: sorted(x.unique().tolist())
    )
    all_ids = sorted(ann["class_id"].unique())
    matrix = pd.DataFrame(0, index=all_ids, columns=all_ids)
    for classes in image_classes:
        for i in range(len(classes)):
            for j in range(len(classes)):
                matrix.loc[classes[i], classes[j]] += 1
    matrix.index = [f"{i}:{class_names.get(i,str(i))}" for i in matrix.index]
    matrix.columns = [f"{i}:{class_names.get(i,str(i))}" for i in matrix.columns]
    matrix.index.name = "class \\ class"
    return matrix


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

REMOVE_CLASSES = {0, 5}  # bad, lift
HEIGHT_TOLERANCE = 0.15  # inner bbox height within 15% of outer → nested door


def _bbox_contains(outer: dict, inner: dict) -> bool:
    """Check if outer bbox fully contains inner bbox (normalised YOLO coords)."""
    ox1 = outer["cx"] - outer["w"] / 2
    oy1 = outer["cy"] - outer["h"] / 2
    ox2 = outer["cx"] + outer["w"] / 2
    oy2 = outer["cy"] + outer["h"] / 2
    ix1 = inner["cx"] - inner["w"] / 2
    iy1 = inner["cy"] - inner["h"] / 2
    ix2 = inner["cx"] + inner["w"] / 2
    iy2 = inner["cy"] + inner["h"] / 2
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def remove_nested_doors(
    bboxes: list[dict], height_tol: float = HEIGHT_TOLERANCE
) -> list[dict]:
    """
    Remove inner bboxes that are contained within a larger bbox and have
    nearly the same height (single door inside a double-door annotation).
    """
    if len(bboxes) <= 1:
        return bboxes
    # Sort by area descending so we check large boxes first
    bboxes = sorted(bboxes, key=lambda b: b["w"] * b["h"], reverse=True)
    keep = [True] * len(bboxes)
    for i in range(len(bboxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue
            if _bbox_contains(bboxes[i], bboxes[j]):
                h_ratio = bboxes[j]["h"] / bboxes[i]["h"] if bboxes[i]["h"] > 0 else 0
                if abs(1.0 - h_ratio) <= height_tol:
                    keep[j] = False
    return [b for b, k in zip(bboxes, keep) if k]


def filter_labels(
    labels_dir: Path,
    output_dir: Path,
    remove_classes: set[int] = REMOVE_CLASSES,
    height_tol: float = HEIGHT_TOLERANCE,
) -> dict:
    """
    Filter YOLO labels:
      1. Remove annotations with class_id in remove_classes (bad=0, lift=5).
      2. Remap all remaining classes to 0 (single 'door' class).
      3. Remove nested bboxes with similar height (single door in double door).
      4. Write cleaned labels to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "total_files": 0,
        "removed_class_annotations": 0,
        "removed_nested_annotations": 0,
        "kept_annotations": 0,
        "empty_after_filter": 0,
    }

    for txt_path in sorted(labels_dir.rglob("*.txt")):
        stats["total_files"] += 1
        content = txt_path.read_text().strip()
        if not content:
            continue

        # Parse and filter classes
        bboxes = []
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid in remove_classes:
                stats["removed_class_annotations"] += 1
                continue
            bboxes.append(
                {
                    "class_id": 0,  # remap everything to single class
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                }
            )

        # Remove nested doors
        n_before = len(bboxes)
        bboxes = remove_nested_doors(bboxes, height_tol)
        stats["removed_nested_annotations"] += n_before - len(bboxes)

        if not bboxes:
            stats["empty_after_filter"] += 1
            continue

        stats["kept_annotations"] += len(bboxes)

        # Write filtered labels
        out_path = output_dir / txt_path.name
        lines = [
            f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in bboxes
        ]
        out_path.write_text("\n".join(lines) + "\n")

    # Write a new data.yaml for the filtered dataset
    # output_dir is <dataset_root>/labels_filtered/labels/<split>
    # so data.yaml goes two levels up at <dataset_root>/labels_filtered/
    new_yaml = output_dir.parents[1] / "data.yaml"
    yaml_data = {"names": {0: "door"}, "path": ".", "train": "train.txt"}
    with open(new_yaml, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return stats


def run_filter(
    dataset_dir: Path,
    split: str = "train",
    output_name: str = "labels_filtered",
    height_tol: float = HEIGHT_TOLERANCE,
) -> None:
    """Run the label filtering pipeline and print summary."""
    labels_dir = dataset_dir / "labels" / split
    if not labels_dir.exists():
        labels_dir = dataset_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Write to a proper YOLO layout: <output_name>/labels/<split>/
    output_dir = dataset_dir / output_name / "labels" / split
    filtered_dataset_dir = dataset_dir / output_name
    print("\nFiltering labels")
    print(f"  Source       : {labels_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  Remove classes: {REMOVE_CLASSES} (bad, lift)")
    print(f"  Height tol   : {height_tol:.0%}")

    stats = filter_labels(labels_dir, output_dir, height_tol=height_tol)

    section("FILTER RESULTS")
    print(f"  Total files processed       : {stats['total_files']}")
    print(f"  Removed (bad/lift class)    : {stats['removed_class_annotations']}")
    print(f"  Removed (nested doors)      : {stats['removed_nested_annotations']}")
    print(f"  Kept annotations            : {stats['kept_annotations']}")
    print(f"  Empty after filter (skipped): {stats['empty_after_filter']}")
    print(f"\n  Filtered labels written to : {output_dir.resolve()}")
    print(f"  Run analysis with          : --dataset {filtered_dataset_dir}")
    print(f"  e.g.: python scripts/annotation/analysis.py --dataset {filtered_dataset_dir}")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

SECTION_WIDTH = 70


def section(title: str) -> None:
    print(f"\n{'='*SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'='*SECTION_WIDTH}")


def run_analysis(
    dataset_dir: Path, split: str = "train", yaml_name: str | None = None
) -> None:
    # --- Load config ---
    cfg = load_data_yaml(dataset_dir, yaml_name or "data.yaml")
    class_names = parse_class_names(cfg)

    labels_dir = dataset_dir / "labels" / split
    if not labels_dir.exists():
        # Flat layout: labels/ directly
        labels_dir = dataset_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    print(f"\nDataset  : {dataset_dir.resolve()}")
    print(f"Split    : {split}")
    print(f"Labels   : {labels_dir}")
    print(f"Classes  : {class_names}")

    # --- Parse ---
    df = load_labels(labels_dir)
    ann = derive_metrics(df)

    n_files = df["file"].nunique()
    n_empty = df[df["class_id"].isna()]["file"].nunique()
    n_annotations = len(ann)

    # ------------------------------------------------------------------ #
    section("DATASET OVERVIEW")
    print(f"  Total label files   : {n_files}")
    print(f"  Empty label files   : {n_empty}")
    print(f"  Annotated files     : {n_files - n_empty}")
    print(f"  Total annotations   : {n_annotations}")
    print(f"  Avg ann / image     : {n_annotations / max(n_files - n_empty, 1):.2f}")

    # ------------------------------------------------------------------ #
    section("CLASS DISTRIBUTION")
    dist = class_distribution(ann, class_names)
    print(dist.to_string(index=False))

    # ------------------------------------------------------------------ #
    section("PER-IMAGE ANNOTATION COUNTS")
    img_stats = per_image_stats(df)
    cnt = img_stats["n_annotations"]
    print(f"  Mean   : {cnt.mean():.2f}")
    print(f"  Median : {cnt.median():.1f}")
    print(f"  Std    : {cnt.std():.2f}")
    print(f"  Min    : {cnt.min()}")
    print(f"  Max    : {cnt.max()}")
    summary_counts = cnt.value_counts().sort_index().reset_index()
    summary_counts.columns = ["n_annotations", "n_images"]
    print("\n  Annotation count distribution:")
    print(summary_counts.to_string(index=False))

    # ------------------------------------------------------------------ #
    section("BOUNDING BOX SIZE STATS (normalised coords)")
    bbox = bbox_size_stats(ann, class_names)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(bbox.to_string(index=False))

    # ------------------------------------------------------------------ #
    section("BOUNDING BOX AREA BUCKETS (per class)")
    buckets = area_buckets(ann, class_names)
    print(buckets.to_string())

    # ------------------------------------------------------------------ #
    section("CLASS CO-OCCURRENCE MATRIX (images)")
    cooc = co_occurrence(ann, class_names)
    print(cooc.to_string())

    print(f"\n{'='*SECTION_WIDTH}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="YOLO label analysis")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("cvat_output/doors_v1"),
        help="Path to dataset root (containing data.yaml and labels/)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Sub-folder inside labels/ to analyse (default: train)",
    )
    parser.add_argument(
        "--yaml",
        default=None,
        help="Name of the YAML config file (default: auto-detect data.yaml / dataset.yaml)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter labels: remove bad/lift, remove nested doors, write to labels_filtered/",
    )
    parser.add_argument(
        "--output-name",
        default="labels_filtered",
        help="Output folder name for filtered labels (default: labels_filtered)",
    )
    parser.add_argument(
        "--height-tol",
        type=float,
        default=HEIGHT_TOLERANCE,
        help=f"Height tolerance for nested door removal (default: {HEIGHT_TOLERANCE})",
    )
    args = parser.parse_args()

    if args.filter:
        run_filter(
            args.dataset,
            split=args.split,
            output_name=args.output_name,
            height_tol=args.height_tol,
        )
    else:
        run_analysis(args.dataset, split=args.split, yaml_name=args.yaml)


if __name__ == "__main__":
    main()
