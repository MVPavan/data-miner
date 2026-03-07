"""
Dataset format converter for object detection annotations.

Converts between 5 standard formats:
  coco      — Standard COCO (annotations/instances_*.json + images/{split}/)
  darknet   — Original YOLO/Darknet (flat images/ + labels/ + train.txt)
  roboflow  — Roboflow COCO export ({split}/_annotations.coco.json)
  yolo_v5a  — Ultralytics YOLO split-first ({split}/images/ + {split}/labels/)
  yolo_v5b  — Ultralytics YOLO modality-first (images/{split}/ + labels/{split}/)

Uses an intermediate representation so any source → any target works.
Images are symlinked by default (use copy_images=True to copy).
"""
from __future__ import annotations

import json
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from detection_metrics.logging import logger, create_progress

# Re-use the canonical extension set from utils
from detection_metrics.utils import IMGS_EXTNS

# ─────────────────────────────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────────────────────────────

IMG_EXTS: frozenset[str] = frozenset(IMGS_EXTNS)


class DatasetFormat(StrEnum):
    """Supported annotation formats."""

    COCO = "coco"
    DARKNET = "darknet"
    ROBOFLOW = "roboflow"
    YOLO_V5A = "yolo_v5a"
    YOLO_V5B = "yolo_v5b"


class Split(StrEnum):
    """Standard dataset splits (canonical names)."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


# Standard COCO uses train2017/val2017/test2017 naming
_COCO_SPLIT_MAP: dict[Split, str] = {
    Split.TRAIN: "train2017",
    Split.VALID: "val2017",
    Split.TEST: "test2017",
}

# ─────────────────────────────────────────────────────────────────────
# Pydantic models — intermediate representation
# ─────────────────────────────────────────────────────────────────────


class Annotation(BaseModel):
    """Single bounding box — 0-indexed class_id, absolute pixel xyxy."""

    model_config = ConfigDict(frozen=True)

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("class_id")
    @classmethod
    def _class_id_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"class_id must be >= 0, got {v}")
        return v


class ImageEntry(BaseModel):
    """One image and its annotations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_name: str  # e.g. "000001.jpg"
    abs_path: Path  # resolved path to the actual image file
    width: int
    height: int
    annotations: list[Annotation] = []


class DatasetBundle(BaseModel):
    """Format-agnostic dataset: categories + per-split image lists."""

    categories: list[str] = []
    splits: dict[Split, list[ImageEntry]] = {}

    def summary(self) -> str:
        lines = [f"Categories ({len(self.categories)}): {self.categories}"]
        for split, imgs in self.splits.items():
            n_ann = sum(len(img.annotations) for img in imgs)
            lines.append(f"  {split}: {len(imgs)} images, {n_ann} annotations")
        return "\n".join(lines)


class ConvertConfig(BaseModel):
    """Validated configuration for a dataset conversion operation."""

    source: Path
    target: Path
    source_format: DatasetFormat | None = None
    target_format: DatasetFormat
    splits: list[Split] = [Split.TRAIN, Split.VALID, Split.TEST]
    copy_images: bool = False

    @field_validator("source", "target")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v.resolve()

    @model_validator(mode="after")
    def _source_ne_target(self) -> ConvertConfig:
        if self.source == self.target:
            raise ValueError("Source and target directories must be different")
        return self


# ─────────────────────────────────────────────────────────────────────
# Image / file helpers
# ─────────────────────────────────────────────────────────────────────


def list_images(directory: Path) -> list[Path]:
    """List image files in a directory, sorted."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image (reads header only)."""
    with Image.open(path) as img:
        return img.size


def find_split_dir(base: Path, split: Split) -> Path | None:
    """Find directory for a split, trying val/valid aliases."""
    candidates = [split.value]
    if split == Split.VALID:
        candidates.append("val")
    for name in candidates:
        p = base / name
        if p.is_dir():
            return p
    return None


def link_images(
    images: list[ImageEntry], target_dir: Path, *, copy: bool = False
) -> None:
    """Symlink or copy images into *target_dir*."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        dst = target_dir / img.file_name
        if dst.exists() or dst.is_symlink():
            continue
        if not img.abs_path.exists():
            logger.warning(f"Source image missing: {img.abs_path}")
            continue
        if copy:
            shutil.copy2(img.abs_path, dst)
        else:
            dst.symlink_to(img.abs_path.resolve())


# ─────────────────────────────────────────────────────────────────────
# YOLO helpers (shared by yolo_v5a, yolo_v5b, darknet)
# ─────────────────────────────────────────────────────────────────────


def parse_yolo_labels(
    label_path: Path, img_w: int, img_h: int
) -> list[Annotation]:
    """Parse a YOLO .txt label file → list of Annotation (absolute xyxy)."""
    if not label_path.exists():
        return []
    anns: list[Annotation] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = (float(p) for p in parts[1:5])
        anns.append(
            Annotation(
                class_id=cid,
                x1=(cx - bw / 2) * img_w,
                y1=(cy - bh / 2) * img_h,
                x2=(cx + bw / 2) * img_w,
                y2=(cy + bh / 2) * img_h,
            )
        )
    return anns


def write_yolo_label(
    label_path: Path, annotations: list[Annotation], img_w: int, img_h: int
) -> None:
    """Write annotations as a YOLO .txt label file (normalized cxcywh)."""
    lines: list[str] = []
    for a in annotations:
        cx = ((a.x1 + a.x2) / 2) / img_w
        cy = ((a.y1 + a.y2) / 2) / img_h
        bw = (a.x2 - a.x1) / img_w
        bh = (a.y2 - a.y1) / img_h
        lines.append(f"{a.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_yolo_split(
    images_dir: Path, labels_dir: Path, *, desc: str = ""
) -> list[ImageEntry]:
    """Read one YOLO split (images dir + labels dir) → list of ImageEntry."""
    if not images_dir.is_dir():
        return []
    img_files = list_images(images_dir)
    entries: list[ImageEntry] = []
    with create_progress() as progress:
        task = progress.add_task(
            desc or str(images_dir), total=len(img_files)
        )
        for img_path in img_files:
            w, h = image_size(img_path)
            label_path = labels_dir / f"{img_path.stem}.txt"
            anns = parse_yolo_labels(label_path, w, h)
            entries.append(
                ImageEntry(
                    file_name=img_path.name,
                    abs_path=img_path.resolve(),
                    width=w,
                    height=h,
                    annotations=anns,
                )
            )
            progress.update(task, advance=1)
    return entries


def write_yolo_split(
    entries: list[ImageEntry],
    images_dir: Path,
    labels_dir: Path,
    *,
    copy_images: bool = False,
) -> None:
    """Write images + YOLO labels for one split."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    link_images(entries, images_dir, copy=copy_images)
    for img in entries:
        write_yolo_label(
            labels_dir / f"{Path(img.file_name).stem}.txt",
            img.annotations,
            img.width,
            img.height,
        )


def read_data_yaml_names(source: Path) -> list[str]:
    """Read category names from data.yaml (handles both list and dict styles)."""
    yaml_path = source / "data.yaml"
    if not yaml_path.exists():
        # Also try dataset.yaml
        yaml_path = source / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {source}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return list(names)


def write_data_yaml(
    target: Path, categories: list[str], split_paths: dict[Split, str]
) -> None:
    """Write a data.yaml file (Ultralytics convention: 'val' key)."""
    data: dict = {}
    for split, path in split_paths.items():
        key = "val" if split == Split.VALID else split.value
        data[key] = path
    data["nc"] = len(categories)
    data["names"] = categories
    with open(target / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────
# COCO JSON helpers (shared by coco, roboflow)
# ─────────────────────────────────────────────────────────────────────


def build_coco_json(categories: list[str], images: list[ImageEntry]) -> dict:
    """Build a COCO JSON dict from intermediate data (1-based category IDs)."""
    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": n} for i, n in enumerate(categories)
        ],
    }
    ann_id = 1
    for img_id, img in enumerate(images, start=1):
        coco["images"].append(
            {
                "id": img_id,
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
            }
        )
        for a in img.annotations:
            w, h = a.x2 - a.x1, a.y2 - a.y1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": a.class_id + 1,  # 0-based → 1-based
                    "bbox": [
                        round(a.x1, 2),
                        round(a.y1, 2),
                        round(w, 2),
                        round(h, 2),
                    ],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco


def parse_coco_json(
    json_path: Path, images_dir: Path
) -> tuple[list[str], list[ImageEntry]]:
    """Parse a COCO JSON annotation file → (categories, images)."""
    with open(json_path) as f:
        data = json.load(f)

    # Categories → 0-based sequential (handles non-sequential COCO IDs)
    raw_cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    id_remap = {cat["id"]: idx for idx, cat in enumerate(raw_cats)}
    categories = [cat["name"] for cat in raw_cats]

    # Images
    img_map: dict[int, ImageEntry] = {}
    for info in data.get("images", []):
        fname = Path(info["file_name"]).name
        abs_path = images_dir / fname
        img_map[info["id"]] = ImageEntry(
            file_name=fname,
            abs_path=abs_path.resolve() if abs_path.exists() else abs_path,
            width=info["width"],
            height=info["height"],
        )

    # Annotations
    for ann in data.get("annotations", []):
        entry = img_map.get(ann["image_id"])
        if entry is None:
            continue
        cid = id_remap.get(ann["category_id"])
        if cid is None:
            continue
        x, y, w, h = ann["bbox"]
        entry.annotations.append(
            Annotation(class_id=cid, x1=x, y1=y, x2=x + w, y2=y + h)
        )

    return categories, list(img_map.values())


# ─────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────


def detect_format(source: Path) -> DatasetFormat:
    """Detect the annotation format of a dataset directory.

    Returns the detected ``DatasetFormat`` enum value.
    Raises ``ValueError`` if no format could be identified.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Not a directory: {source}")

    # 1. Standard COCO: annotations/ dir with instances_*.json
    ann_dir = source / "annotations"
    if ann_dir.is_dir() and any(ann_dir.glob("instances_*.json")):
        return DatasetFormat.COCO

    # 2. Roboflow COCO: _annotations.coco.json inside split dirs
    for name in ("train", "valid", "val", "test"):
        if (source / name / "_annotations.coco.json").exists():
            return DatasetFormat.ROBOFLOW

    # 3. YOLO with data.yaml → distinguish v5a from v5b
    if (source / "data.yaml").exists():
        for name in ("train", "valid", "val", "test"):
            if (source / name / "images").is_dir():
                return DatasetFormat.YOLO_V5A
            if (source / "images" / name).is_dir():
                return DatasetFormat.YOLO_V5B
        # data.yaml exists but no standard sub-dirs — default to v5a
        return DatasetFormat.YOLO_V5A

    # 4. Darknet: obj.data or path-list txt files
    if (source / "obj.data").exists() or (source / "train.txt").exists():
        return DatasetFormat.DARKNET

    all_formats = ", ".join(f.value for f in DatasetFormat)
    raise ValueError(
        f"Cannot auto-detect format for {source}.\n"
        f"Supported formats: {all_formats}"
    )


# ─────────────────────────────────────────────────────────────────────
# Readers  (source → DatasetBundle)
# ─────────────────────────────────────────────────────────────────────


def _read_coco(source: Path, splits: list[Split]) -> DatasetBundle:
    """Read Standard COCO format (annotations/ + images/)."""
    ann_dir = source / "annotations"
    images_root = source / "images"
    categories: list[str] | None = None
    bundle = DatasetBundle()

    for split in splits:
        coco_name = _COCO_SPLIT_MAP.get(split, split.value)

        json_path = None
        for name in (coco_name, split.value):
            p = ann_dir / f"instances_{name}.json"
            if p.exists():
                json_path = p
                break
        if json_path is None:
            continue

        img_dir = None
        for name in (coco_name, split.value):
            p = images_root / name
            if p.is_dir():
                img_dir = p
                break
        if img_dir is None:
            logger.warning(
                f"Images dir not found for split '{split.value}', skipping"
            )
            continue

        cats, entries = parse_coco_json(json_path, img_dir)
        if categories is None:
            categories = cats
        bundle.splits[split] = entries
        logger.info(
            f"Read {split.value}: {len(entries)} images from {json_path.name}"
        )

    bundle.categories = categories or []
    return bundle


def _read_darknet(source: Path, splits: list[Split]) -> DatasetBundle:
    """Read Standard YOLO/Darknet format (flat images/ + labels/ + path lists)."""
    names_file = source / "obj.names"
    categories = (
        [
            l.strip()
            for l in names_file.read_text().strip().splitlines()
            if l.strip()
        ]
        if names_file.exists()
        else []
    )

    labels_dir = source / "labels"
    bundle = DatasetBundle(categories=categories)

    for split in splits:
        path_file = source / f"{split.value}.txt"
        if not path_file.exists():
            continue

        lines = [
            l.strip()
            for l in path_file.read_text().strip().splitlines()
            if l.strip()
        ]
        entries: list[ImageEntry] = []
        with create_progress() as progress:
            task = progress.add_task(
                f"Reading {split.value}", total=len(lines)
            )
            for line in lines:
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = (source / img_path).resolve()
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    progress.update(task, advance=1)
                    continue
                w, h = image_size(img_path)
                label_path = labels_dir / f"{img_path.stem}.txt"
                anns = parse_yolo_labels(label_path, w, h)
                entries.append(
                    ImageEntry(
                        file_name=img_path.name,
                        abs_path=img_path.resolve(),
                        width=w,
                        height=h,
                        annotations=anns,
                    )
                )
                progress.update(task, advance=1)
        bundle.splits[split] = entries
        logger.info(f"Read {split.value}: {len(entries)} images")

    # Infer categories from annotations if obj.names was missing
    if not bundle.categories:
        max_cls = max(
            (
                a.class_id
                for imgs in bundle.splits.values()
                for img in imgs
                for a in img.annotations
            ),
            default=-1,
        )
        if max_cls >= 0:
            bundle.categories = [f"class_{i}" for i in range(max_cls + 1)]

    return bundle


def _read_roboflow(source: Path, splits: list[Split]) -> DatasetBundle:
    """Read Roboflow COCO export ({split}/_annotations.coco.json)."""
    categories: list[str] | None = None
    bundle = DatasetBundle()

    for split in splits:
        split_dir = find_split_dir(source, split)
        if split_dir is None:
            continue
        json_path = split_dir / "_annotations.coco.json"
        if not json_path.exists():
            continue

        img_dir = split_dir / "images"
        if not img_dir.is_dir():
            img_dir = split_dir

        cats, entries = parse_coco_json(json_path, img_dir)
        if categories is None:
            categories = cats
        bundle.splits[split] = entries
        logger.info(
            f"Read {split.value}: {len(entries)} images from {json_path.name}"
        )

    bundle.categories = categories or []
    return bundle


def _read_yolo_v5a(source: Path, splits: list[Split]) -> DatasetBundle:
    """Read Ultralytics YOLO v5a — split-first layout."""
    categories = read_data_yaml_names(source)
    bundle = DatasetBundle(categories=categories)

    for split in splits:
        split_dir = find_split_dir(source, split)
        if split_dir is None:
            continue
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.is_dir():
            continue
        entries = read_yolo_split(
            images_dir, labels_dir, desc=f"Reading {split.value}"
        )
        bundle.splits[split] = entries
        n_ann = sum(len(e.annotations) for e in entries)
        logger.info(
            f"Read {split.value}: {len(entries)} images, {n_ann} annotations"
        )

    return bundle


def _read_yolo_v5b(source: Path, splits: list[Split]) -> DatasetBundle:
    """Read Ultralytics YOLO v5b — modality-first layout."""
    categories = read_data_yaml_names(source)
    bundle = DatasetBundle(categories=categories)

    images_root = source / "images"
    labels_root = source / "labels"

    for split in splits:
        img_dir = find_split_dir(images_root, split)
        if img_dir is None:
            continue
        lbl_dir = labels_root / img_dir.name
        entries = read_yolo_split(
            img_dir, lbl_dir, desc=f"Reading {split.value}"
        )
        bundle.splits[split] = entries
        n_ann = sum(len(e.annotations) for e in entries)
        logger.info(
            f"Read {split.value}: {len(entries)} images, {n_ann} annotations"
        )

    return bundle


# ─────────────────────────────────────────────────────────────────────
# Writers  (DatasetBundle → target format)
# ─────────────────────────────────────────────────────────────────────


def _write_coco(
    bundle: DatasetBundle, target: Path, *, copy_images: bool = False
) -> None:
    """Write Standard COCO format (annotations/ + images/{split}/)."""
    ann_dir = target / "annotations"
    images_root = target / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split, entries in bundle.splits.items():
        coco_name = _COCO_SPLIT_MAP.get(split, split.value)
        coco = build_coco_json(bundle.categories, entries)
        (ann_dir / f"instances_{coco_name}.json").write_text(
            json.dumps(coco, indent=2)
        )
        link_images(entries, images_root / coco_name, copy=copy_images)
        logger.info(
            f"Wrote {coco_name}: {len(entries)} images, "
            f"{len(coco['annotations'])} annotations"
        )


def _write_darknet(
    bundle: DatasetBundle, target: Path, *, copy_images: bool = False
) -> None:
    """Write Standard YOLO/Darknet format (flat images/ + labels/ + path lists)."""
    target.mkdir(parents=True, exist_ok=True)
    images_dir = target / "images"
    labels_dir = target / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    (target / "obj.names").write_text("\n".join(bundle.categories) + "\n")

    all_images: list[ImageEntry] = []
    for split, entries in bundle.splits.items():
        paths: list[str] = []
        for img in entries:
            all_images.append(img)
            write_yolo_label(
                labels_dir / f"{Path(img.file_name).stem}.txt",
                img.annotations,
                img.width,
                img.height,
            )
            paths.append(f"images/{img.file_name}")
        (target / f"{split.value}.txt").write_text("\n".join(paths) + "\n")
        logger.info(f"Wrote {split.value}.txt ({len(entries)} images)")

    link_images(all_images, images_dir, copy=copy_images)

    obj_lines = [f"classes = {len(bundle.categories)}"]
    for split in bundle.splits:
        obj_lines.append(f"{split.value}  = {split.value}.txt")
    obj_lines += ["names  = obj.names", "backup = backup/"]
    (target / "obj.data").write_text("\n".join(obj_lines) + "\n")
    logger.info(
        f"Wrote obj.names ({len(bundle.categories)} classes), obj.data"
    )


def _write_roboflow(
    bundle: DatasetBundle, target: Path, *, copy_images: bool = False
) -> None:
    """Write Roboflow COCO export."""
    for split, entries in bundle.splits.items():
        split_dir = target / split.value
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        coco = build_coco_json(bundle.categories, entries)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco, indent=2)
        )
        link_images(entries, img_dir, copy=copy_images)
        logger.info(
            f"Wrote {split.value}/: {len(entries)} images, "
            f"{len(coco['annotations'])} annotations"
        )


def _write_yolo_v5a(
    bundle: DatasetBundle, target: Path, *, copy_images: bool = False
) -> None:
    """Write Ultralytics YOLO v5a — split-first layout."""
    target.mkdir(parents=True, exist_ok=True)
    split_paths: dict[Split, str] = {}

    for split, entries in bundle.splits.items():
        write_yolo_split(
            entries,
            target / split.value / "images",
            target / split.value / "labels",
            copy_images=copy_images,
        )
        split_paths[split] = f"{split.value}/images"
        n_ann = sum(len(e.annotations) for e in entries)
        logger.info(
            f"Wrote {split.value}/: {len(entries)} images, {n_ann} annotations"
        )

    write_data_yaml(target, bundle.categories, split_paths)
    logger.info(f"Wrote data.yaml ({len(bundle.categories)} classes)")


def _write_yolo_v5b(
    bundle: DatasetBundle, target: Path, *, copy_images: bool = False
) -> None:
    """Write Ultralytics YOLO v5b — modality-first layout."""
    target.mkdir(parents=True, exist_ok=True)
    split_paths: dict[Split, str] = {}

    for split, entries in bundle.splits.items():
        write_yolo_split(
            entries,
            target / "images" / split.value,
            target / "labels" / split.value,
            copy_images=copy_images,
        )
        split_paths[split] = f"images/{split.value}"
        n_ann = sum(len(e.annotations) for e in entries)
        logger.info(
            f"Wrote images/{split.value}/ + labels/{split.value}/: "
            f"{len(entries)} images, {n_ann} annotations"
        )

    write_data_yaml(target, bundle.categories, split_paths)
    logger.info(f"Wrote data.yaml ({len(bundle.categories)} classes)")


# ─────────────────────────────────────────────────────────────────────
# Reader / Writer registries
# ─────────────────────────────────────────────────────────────────────

ReaderFn = Callable[[Path, list[Split]], DatasetBundle]
WriterFn = Callable[[DatasetBundle, Path], None]  # copy_images via keyword

READERS: dict[DatasetFormat, ReaderFn] = {
    DatasetFormat.COCO: _read_coco,
    DatasetFormat.DARKNET: _read_darknet,
    DatasetFormat.ROBOFLOW: _read_roboflow,
    DatasetFormat.YOLO_V5A: _read_yolo_v5a,
    DatasetFormat.YOLO_V5B: _read_yolo_v5b,
}

WRITERS: dict[DatasetFormat, Callable] = {
    DatasetFormat.COCO: _write_coco,
    DatasetFormat.DARKNET: _write_darknet,
    DatasetFormat.ROBOFLOW: _write_roboflow,
    DatasetFormat.YOLO_V5A: _write_yolo_v5a,
    DatasetFormat.YOLO_V5B: _write_yolo_v5b,
}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def convert(cfg: ConvertConfig) -> DatasetBundle:
    """Convert a detection dataset between annotation formats.

    Accepts a validated ``ConvertConfig`` and returns the intermediate
    ``DatasetBundle`` so callers can inspect the result.
    """
    if not cfg.source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {cfg.source}")

    # Auto-detect source format
    src_fmt = cfg.source_format
    if src_fmt is None:
        src_fmt = detect_format(cfg.source)
        logger.info(f"Auto-detected source format: {src_fmt.value}")

    logger.info(f"Converting: {src_fmt.value} → {cfg.target_format.value}")
    logger.info(f"  Source: {cfg.source}")
    logger.info(f"  Target: {cfg.target}")
    logger.info(f"  Splits: {[s.value for s in cfg.splits]}")
    logger.info(f"  Images: {'copy' if cfg.copy_images else 'symlink'}")

    # Read
    logger.info(f"Reading {src_fmt.value} dataset...")
    bundle = READERS[src_fmt](cfg.source, cfg.splits)
    logger.info(f"\n{bundle.summary()}")

    if not bundle.splits:
        logger.warning("No data found. Check source path and split names.")
        return bundle

    # Write
    logger.info(f"Writing {cfg.target_format.value} dataset...")
    WRITERS[cfg.target_format](bundle, cfg.target, copy_images=cfg.copy_images)
    logger.info(f"Done! Output → {cfg.target}")
    return bundle
