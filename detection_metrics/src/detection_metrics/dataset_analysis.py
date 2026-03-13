"""
YOLO dataset analysis — class distribution, bbox stats, area buckets, co-occurrence.

Usage as CLI:
    detection-metrics analyze-dataset --dataset /path/to/dataset
    detection-metrics analyze-dataset --dataset /path/to/dataset --split train

Usage as library:
    from detection_metrics.dataset_analysis import run_analysis
    run_analysis(Path("/path/to/dataset"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate


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
    Empty label files produce no rows (they appear in per-image stats as
    files with 0 annotations).
    """
    records = []
    for txt_path in sorted(labels_dir.rglob("*.txt")):
        content = txt_path.read_text().strip()
        if not content:
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
    return pd.DataFrame(
        {
            "class_id": counts.index,
            "class_name": [class_names.get(i, str(i)) for i in counts.index],
            "count": counts.values,
            "pct": pct.values,
        }
    ).reset_index(drop=True)


def per_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-image annotation counts (including empty-label images with count=0)."""
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
    """Break bounding-box areas into small/medium/large buckets."""
    bins = [0, 0.01, 0.09, 1.01]
    labels_b = ["small (<1%)", "medium (1-9%)", "large (>=9%)"]
    ann = ann.copy()
    ann["size_bucket"] = pd.cut(ann["area"], bins=bins, labels=labels_b, right=False)
    pivot = (
        ann.groupby(["class_id", "size_bucket"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = [f"{i} - {class_names.get(i, str(i))}" for i in pivot.index]
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
    matrix.index = [f"{i}:{class_names.get(i, str(i))}" for i in matrix.index]
    matrix.columns = [f"{i}:{class_names.get(i, str(i))}" for i in matrix.columns]
    matrix.index.name = "class \\ class"
    return matrix


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

TABLE_FMT = "simple_grid"


def _tab(df: pd.DataFrame, **kw) -> str:
    return tabulate(df, headers="keys", tablefmt=TABLE_FMT, showindex=False, **kw)


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(1, 66 - len(title))}")


def _discover_labels(dataset_dir: Path, split: str | None) -> list[tuple[str, Path]]:
    """Auto-detect label directories."""
    SPLIT_NAMES = ("train", "valid", "val", "test")

    split_dirs = [
        (s, dataset_dir / s / "labels")
        for s in SPLIT_NAMES
        if (dataset_dir / s / "labels").is_dir()
    ]
    if split_dirs:
        if split:
            match = [(s, d) for s, d in split_dirs if s == split]
            if not match:
                raise FileNotFoundError(
                    f"Split '{split}' not found. Available: {[s for s, _ in split_dirs]}"
                )
            return match
        return split_dirs

    if split and (dataset_dir / "labels" / split).is_dir():
        return [(split, dataset_dir / "labels" / split)]

    if (dataset_dir / "labels").is_dir():
        return [(split or "all", dataset_dir / "labels")]

    raise FileNotFoundError(f"No labels directory found under {dataset_dir}")


def _ann_count_row(name: str, df: pd.DataFrame, max_bin: int = 10) -> dict:
    cnt = per_image_stats(df)["n_annotations"]
    vc = cnt.value_counts()
    row: dict = {"split": name}
    for i in range(1, max_bin + 1):
        row[str(i)] = int(vc.get(i, 0))
    row[f">{max_bin}"] = int(cnt[cnt > max_bin].count())
    return row


def _split_summary(name: str, df: pd.DataFrame, ann: pd.DataFrame) -> dict:
    n_files = df["file"].nunique()
    n_empty = df[df["class_id"].isna()]["file"].nunique()
    cnt = per_image_stats(df)["n_annotations"]
    return {
        "split": name,
        "files": n_files,
        "empty": n_empty,
        "annotations": len(ann),
        "avg/img": f"{len(ann) / max(n_files - n_empty, 1):.2f}",
        "med/img": f"{cnt.median():.0f}",
        "max/img": f"{cnt.max()}",
    }


def run_analysis(
    dataset_dir: Path, split: str | None = None, yaml_name: str | None = None
) -> None:
    """
    Run dataset analysis and print formatted tables.

    Args:
        dataset_dir: Root of the YOLO dataset.
        split:       Analyse only this split (default: auto-detect all).
        yaml_name:   Name override for data.yaml.
    """
    cfg = load_data_yaml(dataset_dir, yaml_name or "data.yaml")
    class_names = parse_class_names(cfg)
    splits = _discover_labels(dataset_dir, split)

    print(f"\nDataset : {dataset_dir.resolve()}")
    print(f"Classes : {class_names}")

    summaries: list[dict] = []
    ann_count_rows: list[dict] = []
    split_data: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

    for name, labels_dir in splits:
        df = load_labels(labels_dir)
        ann = derive_metrics(df)
        summaries.append(_split_summary(name, df, ann))
        ann_count_rows.append(_ann_count_row(name, df))
        split_data.append((name, df, ann))

    if len(split_data) > 1:
        combined_df = pd.concat([d for _, d, _ in split_data], ignore_index=True)
        combined_ann = derive_metrics(combined_df)
        summaries.append(_split_summary("TOTAL", combined_df, combined_ann))
        ann_count_rows.append(_ann_count_row("TOTAL", combined_df))
        split_data.append(("TOTAL", combined_df, combined_ann))
    else:
        combined_ann = split_data[0][2]

    section("OVERVIEW")
    print(_tab(pd.DataFrame(summaries)))

    section("ANNOTATIONS PER IMAGE")
    print(_tab(pd.DataFrame(ann_count_rows)))

    section("CLASS DISTRIBUTION")
    print(_tab(class_distribution(combined_ann, class_names)))

    section("BBOX STATS (normalised)")
    bbox_rows = []
    for name, _df, ann in split_data:
        for _, r in bbox_size_stats(ann, class_names).iterrows():
            bbox_rows.append({"split": name, **r.to_dict()})
    print(_tab(pd.DataFrame(bbox_rows), floatfmt=".4f"))

    section("AREA BUCKETS")
    bucket_rows = []
    for name, _df, ann in split_data:
        ab = area_buckets(ann, class_names).reset_index()
        ab.insert(0, "split", name)
        bucket_rows.append(ab)
    print(_tab(pd.concat(bucket_rows, ignore_index=True)))

    if combined_ann["class_id"].nunique() > 1:
        section("CO-OCCURRENCE")
        cooc = co_occurrence(combined_ann, class_names)
        print(_tab(cooc.reset_index()))

    print()
