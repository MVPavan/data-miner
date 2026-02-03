# python script to visualize remote view images and annotations using voxel51

import time
from typing import List

import fiftyone as fo
import fiftyone.utils.yolo as fouyolo
from pydantic import BaseModel, model_validator

FO_DATASET_MAP = {
    "yolo": fo.types.YOLOv4Dataset,
    "coco": fo.types.COCODetectionDataset,
    "images": fo.types.ImageDirectory,
}


class DatasetConfig(BaseModel):
    name: str
    images_dir: str
    dataset_type: str = "coco"  # 'yolo', 'coco', 'images'
    yolo_gt_folder: str | None = None
    prediction_fields: List[str] = ["predictions"]
    yolo_pred_folders: List[str] | None = None
    coco_gt_json_path: str | None = None
    coco_pred_json_path: str | None = None
    overwrite: bool = False
    port: int = 7770

    @model_validator(mode="after")
    def validate(self):
        if self.dataset_type not in FO_DATASET_MAP:
            raise ValueError(
                f"Unsupported dataset_type: {self.dataset_type}. Supported types are: {list(FO_DATASET_MAP.keys())}"
            )
        if self.dataset_type == "yolo":
            assert self.yolo_gt_folder is not None, (
                "yolo_gt_folder must be provided for yolo dataset_type"
            )
        elif self.dataset_type == "coco":
            assert self.coco_gt_json_path is not None, (
                "coco_gt_json_path must be provided for coco dataset_type"
            )

        if self.yolo_pred_folders is not None:
            assert len(self.prediction_fields) == len(self.yolo_pred_folders), (
                "Length of prediction_field must match length of yolo_pred_folder"
            )
        return self


def load_dataset(dataset_config: DatasetConfig) -> fo.Dataset:
    """
    Load a dataset into FiftyOne using the provided configuration.

    dataset_config: Configuration for the dataset to be loaded.
    """
    if fo.dataset_exists(dataset_config.name):
        if dataset_config.overwrite:
            fo.delete_dataset(dataset_config.name)
        else:
            print(
                f"Dataset '{dataset_config.name}' already exists. Loading existing dataset."
            )
            return fo.load_dataset(dataset_config.name)

    if not fo.dataset_exists(dataset_config.name):
        if dataset_config.dataset_type == "coco":
            dataset = fo.Dataset.from_dir(
                dataset_type=FO_DATASET_MAP[dataset_config.dataset_type],
                data_path=dataset_config.images_dir,
                labels_path=dataset_config.coco_gt_json_path,
                label_field="ground_truth",
                name=dataset_config.name,
            )
        elif dataset_config.dataset_type == "yolo":
            dataset = fo.Dataset.from_dir(
                dataset_type=FO_DATASET_MAP[dataset_config.dataset_type],
                data_path=dataset_config.images_dir,
                labels_path=dataset_config.yolo_gt_folder,
                label_field="ground_truth",
                name=dataset_config.name,
            )
        elif dataset_config.dataset_type == "images":
            dataset = fo.Dataset.from_dir(
                dataset_type=FO_DATASET_MAP[dataset_config.dataset_type],
                dataset_dir=dataset_config.images_dir,
                name=dataset_config.name,
            )

    if dataset_config.yolo_pred_folders is not None:
        for pred_folder, label_field in zip(
            dataset_config.yolo_pred_folders, dataset_config.prediction_fields
        ):
            fouyolo.add_yolo_labels(
                dataset,
                labels_path=pred_folder,
                label_field=label_field,
                classes=None,  # {0: 'door'}
            )

    return dataset


def visualize_remote_view_images(dataset_config: DatasetConfig, wait=True):
    dataset = load_dataset(dataset_config)
    # Launch the FiftyOne app to visualize the dataset
    print(
        f"Launching FiftyOne app for dataset '{dataset_config.name}' on port {dataset_config.port}..."
    )
    session = fo.launch_app(dataset, address="0.0.0.0", port=dataset_config.port)
    # session.wait()  # Keep the app running until closed by the user
    # run in infinite loop
    if wait:
        while True:
            time.sleep(10)
    return session


def visualize_remote_view_images_multiple(
    dataset_configs: List[DatasetConfig], port: int = 0
):
    dataset = None
    sessions = []
    for dataset_config in dataset_configs:
        if port != 0:
            if dataset is None:
                dataset = load_dataset(dataset_config)
                session = fo.launch_app(dataset, address="0.0.0.0", port=port)
            else:
                session.dataset = load_dataset(dataset_config)
        else:
            sessions.append(visualize_remote_view_images(dataset_config, wait=False))

    # run in infinite loop
    # while True:
    #     time.sleep(10)


def dataset_config_creator(
    dataset_paths: dict[str, str], port_start: int = 7770
) -> List[DatasetConfig]:
    dataset_configs = []
    _port = port_start
    for name, path in dataset_paths.items():
        dataset_configs.append(
            DatasetConfig(
                name=name,
                images_dir=path,
                dataset_type="images",
                overwrite=True,
                port=_port,
            )
        )
        _port += 1
    return dataset_configs


if __name__ == "__main__":
    # dataset_paths = {
    #     # "delivery_dedup": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_dedup",
    #     # "real_estate_dedup": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/real_estate_v1/frames_dedup",
    #     # "direct_doors_dedup": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup",
    #     # "direct_doors_dedup_v2": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_v2",
    #     # "direct_doors_dedup_v2_diff": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_v2_diff",
    #     # "direct_doors_filtered_v2": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_filter_v2",
    #     # "direct_doors_dedup_v3": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_v3",
    #     # "direct_doors_dedup_v3_filtered": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_v3_filtered",
    #     # "direct_doors_filter_v2_dedup": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_filter_v2_dedup",
    #     "delivery_dedup": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_dedup",
    #     "delivery_dedup_v2": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_dedup_v2",
    #     "delivery_dedup_v2_filtered": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_dedup_v2_filter",
    #     "delivery_filtered_v2_dedup": "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup",
    # }

    # dataset_paths = {
    #     "door_crops": "/data/datasets/intel_datasets/doors_croped/doors_selected_3k_crops",
    # }

    # # visualize_remote_view_images(dataset_config=delivery_videos)
    # visualize_remote_view_images_multiple(
    #     dataset_configs=dataset_config_creator(dataset_paths, port_start=7770),
    #     port=0,  # if port not zero, all datasets will be visualized in same port
    #     # ignoring individual dataset ports
    # )

    # delivery_filtered_v2_dedup_md = DatasetConfig(
    #     name="delivery_filtered_v2_dedup",
    #     images_dir="/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup",
    #     dataset_type="images",
    #     yolo_pred_folders=[
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/moondream/frames_filtered_v2_dedup/pred_txt_merged_filtered",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/owlvit/frames_filtered_v2_dedup/pred_txt",
    #         # "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/yolo_world/frames_filtered_v2_dedup/pred_txt",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/sam3/frames_filtered_v2_dedup/pred_txt",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/gdino/frames_filtered_v2_dedup/pred_txt",
    #         # "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/paligemma/frames_filtered_v2_dedup/pred_txt",
    #         # "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/paligemma_v2/frames_filtered_v2_dedup/pred_txt",
    #         # "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/owlvit_ref/frames_filtered_v2_dedup/pred_txt",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/merged_detections/frames_filtered_v2_dedup",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/merged_detections_03iou/frames_filtered_v2_dedup",
    #     ],
    #     prediction_fields=[
    #         "moon",
    #         "owlvit",
    #         # "yolo_world",
    #         "sam3",
    #         "gdino",
    #         # "paligemma",
    #         # "paligemma_v2",
    #         # "owlvit_ref",
    #         "merged_detections",
    #         "merged_detections_03iou",
    #     ],
    #     overwrite=True,
    #     port=7773,
    # )
    # visualize_remote_view_images(dataset_config=delivery_filtered_v2_dedup_md)

    # delivery_filtered_v2_dedup_md = DatasetConfig(
    #     name="delivery_filtered_v2_dedup_v1",
    #     images_dir="/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup",
    #     dataset_type="images",
    #     yolo_pred_folders=[
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/merged_detections_03iou_08conf/frames_filtered_v2_dedup",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/merged_detections_03iou_07conf/frames_filtered_v2_dedup",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/merged_detections_03iou_06conf/frames_filtered_v2_dedup",
    #         "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/merged_detections_03iou_05conf/frames_filtered_v2_dedup",
    #     ],
    #     prediction_fields=[
    #         "merged_detections_03iou_08conf",
    #         "merged_detections_03iou_07conf",
    #         "merged_detections_03iou_06conf",
    #         "merged_detections_03iou_05conf",
    #     ],
    #     overwrite=True,
    #     port=7774,
    # )
    # visualize_remote_view_images(dataset_config=delivery_filtered_v2_dedup_md)

    delivery_filtered_v2_dedup_md = DatasetConfig(
        name="delivery_filtered_v2_dedup_qwen",
        images_dir="/data/pavan/codes/tycoai/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup",
        dataset_type="images",
        yolo_pred_folders=[
            "/data/pavan/codes/tycoai/data_miner/output/projects/delivery_pov_v1/qwen3vl/frames_filtered_v2_dedup/pred_txt",
        ],
        prediction_fields=[
            "qwen3vl",
        ],
        overwrite=True,
        port=7775,
    )
    visualize_remote_view_images(dataset_config=delivery_filtered_v2_dedup_md)

