"""
Remote visualisation via FiftyOne — view images, GT, and predictions in-browser.

Usage as library:
    from detection_metrics.remote_view import DatasetConfig, visualize_remote_view_images
    cfg = DatasetConfig(
        name="my_dataset",
        images_dir="/data/images",
        coco_gt_json_path="/data/gt.json",
    )
    visualize_remote_view_images(cfg)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, model_validator


# Lazy-import fiftyone so the rest of the package works without it.
def _get_fo():
    try:
        import fiftyone as fo
        return fo
    except ImportError:
        raise ImportError(
            "fiftyone is required for remote_view. "
            "Install it with: pip install fiftyone"
        )


FO_DATASET_MAP_KEYS = {"yolo", "coco", "images"}


class DatasetConfig(BaseModel):
    """Configuration for loading a dataset in FiftyOne."""

    name: str
    images_dir: str
    dataset_type: str = "coco"  # 'yolo', 'coco', 'images'
    yolo_gt_folder: Optional[str] = None
    prediction_fields: List[str] = ["predictions"]
    yolo_pred_folders: Optional[List[str]] = None
    coco_gt_json_path: Optional[str] = None
    coco_pred_json_path: Optional[str] = None
    overwrite: bool = False
    port: int = 7770

    @model_validator(mode="after")
    def validate_config(self):
        if self.dataset_type not in FO_DATASET_MAP_KEYS:
            raise ValueError(
                f"Unsupported dataset_type: {self.dataset_type}. "
                f"Supported: {sorted(FO_DATASET_MAP_KEYS)}"
            )
        if self.dataset_type == "yolo":
            assert (
                self.yolo_gt_folder is not None
            ), "yolo_gt_folder must be provided for yolo dataset_type"
        elif self.dataset_type == "coco":
            assert (
                self.coco_gt_json_path is not None
            ), "coco_gt_json_path must be provided for coco dataset_type"

        if self.yolo_pred_folders is not None:
            assert len(self.prediction_fields) == len(
                self.yolo_pred_folders
            ), "Length of prediction_fields must match yolo_pred_folders"
        return self


def load_dataset(dataset_config: DatasetConfig):
    """Load a dataset into FiftyOne."""
    fo = _get_fo()

    FO_TYPE_MAP = {
        "yolo": fo.types.YOLOv4Dataset,
        "coco": fo.types.COCODetectionDataset,
        "images": fo.types.ImageDirectory,
    }

    if fo.dataset_exists(dataset_config.name):
        if dataset_config.overwrite:
            fo.delete_dataset(dataset_config.name)
        else:
            print(
                f"Dataset '{dataset_config.name}' already exists. Loading existing."
            )
            return fo.load_dataset(dataset_config.name)

    dt = dataset_config.dataset_type
    if dt == "coco":
        dataset = fo.Dataset.from_dir(
            dataset_type=FO_TYPE_MAP[dt],
            data_path=dataset_config.images_dir,
            labels_path=dataset_config.coco_gt_json_path,
            label_field="ground_truth",
            name=dataset_config.name,
        )
    elif dt == "yolo":
        dataset = fo.Dataset.from_dir(
            dataset_type=FO_TYPE_MAP[dt],
            data_path=dataset_config.images_dir,
            labels_path=dataset_config.yolo_gt_folder,
            label_field="ground_truth",
            name=dataset_config.name,
        )
    elif dt == "images":
        dataset = fo.Dataset.from_dir(
            dataset_type=FO_TYPE_MAP[dt],
            data_path=dataset_config.images_dir,
            name=dataset_config.name,
        )
    else:
        raise ValueError(f"Unsupported type: {dt}")

    if dataset_config.yolo_pred_folders:
        import fiftyone.utils.yolo as fouyolo

        for pred_folder, label_field in zip(
            dataset_config.yolo_pred_folders,
            dataset_config.prediction_fields,
        ):
            fouyolo.add_yolo_labels(
                dataset, labels_path=pred_folder, label_field=label_field
            )

    return dataset


def visualize_remote_view_images(dataset_config: DatasetConfig):
    """Launch FiftyOne app for the given dataset configuration."""
    fo = _get_fo()
    dataset = load_dataset(dataset_config)
    print(
        f"Launching FiftyOne app for '{dataset_config.name}' "
        f"on port {dataset_config.port}..."
    )
    session = fo.launch_app(
        dataset, address="0.0.0.0", port=dataset_config.port
    )
    while True:
        time.sleep(10)
