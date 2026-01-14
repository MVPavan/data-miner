# python script to visualize remote view images and annotations using voxel51

import time
from typing import List

import fiftyone as fo
import fiftyone.utils.yolo as fouyolo
from pydantic import BaseModel, model_validator

FO_DATASET_MAP = {
    'yolo': fo.types.YOLOv4Dataset,
    'coco': fo.types.COCODetectionDataset,
    'images' : fo.types.ImageDirectory,
}

class DatasetConfig(BaseModel):
    name: str
    images_dir: str
    dataset_type: str = 'coco'  # 'yolo', 'coco', 'images'
    yolo_gt_folder: str|None = None
    prediction_fields: List[str] = ["predictions"]
    yolo_pred_folders: List[str]|None = None
    coco_gt_json_path: str|None = None
    coco_pred_json_path: str|None = None
    overwrite: bool = False
    port:int = 7770

    @model_validator(mode='after')
    def validate(self):
        if self.dataset_type not in FO_DATASET_MAP:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}. Supported types are: {list(FO_DATASET_MAP.keys())}")
        if self.dataset_type == 'yolo':
            assert self.yolo_gt_folder is not None, "yolo_gt_folder must be provided for yolo dataset_type"
        elif self.dataset_type == 'coco':
            assert self.coco_gt_json_path is not None, "coco_gt_json_path must be provided for coco dataset_type"
        
        if self.yolo_pred_folders is not None:
            assert len(self.prediction_fields) == len(self.yolo_pred_folders), "Length of prediction_field must match length of yolo_pred_folder"
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
            print(f"Dataset '{dataset_config.name}' already exists. Loading existing dataset.")
            return fo.load_dataset(dataset_config.name)

    if not fo.dataset_exists(dataset_config.name):
        if dataset_config.dataset_type == 'coco':
            dataset = fo.Dataset.from_dir(
                dataset_type=FO_DATASET_MAP[dataset_config.dataset_type],
                data_path=dataset_config.images_dir,
                labels_path=dataset_config.coco_gt_json_path,
                label_field="ground_truth",
                name=dataset_config.name,
            )
        elif dataset_config.dataset_type == 'yolo':
            dataset = fo.Dataset.from_dir(
                dataset_type=FO_DATASET_MAP[dataset_config.dataset_type],
                data_path=dataset_config.images_dir,
                labels_path=dataset_config.yolo_gt_folder,
                label_field="ground_truth",
                name=dataset_config.name,
            )
        elif dataset_config.dataset_type == 'images':
            dataset = fo.Dataset.from_dir(
                dataset_type=FO_DATASET_MAP[dataset_config.dataset_type],
                dataset_dir=dataset_config.images_dir,
                name=dataset_config.name,
            )

    if dataset_config.yolo_pred_folders is not None:
        for pred_folder, label_field in zip(dataset_config.yolo_pred_folders, dataset_config.prediction_fields):
            fouyolo.add_yolo_labels(
                dataset,
                labels_path=pred_folder,
                label_field=label_field,
                classes=None # {0: 'door'}
            )
    
    return dataset


def visualize_remote_view_images(dataset_config: DatasetConfig):
    
    dataset = load_dataset(dataset_config)
    # Launch the FiftyOne app to visualize the dataset
    print(f"Launching FiftyOne app for dataset '{dataset_config.name}' on port {dataset_config.port}...")
    session = fo.launch_app(dataset, address="0.0.0.0", port=dataset_config.port)
    # session.wait()  # Keep the app running until closed by the user
    return session

def visualize_remote_view_images_multiple(dataset_configs: List[DatasetConfig], port:int=0):
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
            sessions.append(visualize_remote_view_images(dataset_config))
    
    # run in infinite loop            
    while True:
        time.sleep(10)


if __name__ == "__main__":
    
    delivery_videos = DatasetConfig(
        name="delivery_videos",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_dedup",
        overwrite=False,
        port=7770
    )

    real_estate_videos = DatasetConfig(
        name="real_estate_videos",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/real_estate_v1/frames_dedup",
        overwrite=False,
        port=7771
    )

    direct_doors_videos = DatasetConfig(
        name="direct_doors_videos",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup",
        overwrite=False,
        port=7772
    )

    direct_doors_videos_v2 = DatasetConfig(
        name="direct_doors_videos_v2",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_filter_v2",
        overwrite=True,
        port=7773
    )

    direct_doors_removed = DatasetConfig(
        name="direct_doors_removed",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_diff",
        overwrite=True,
        port=7774
    )

    direct_doors_videos_v3 = DatasetConfig(
        name="direct_doors_videos_v3",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_filter_v3",
        overwrite=True,
        port=7775
    )

    direct_doors_removed_v3 = DatasetConfig(
        name="direct_doors_removed_v3",
        dataset_type="images",
        images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup_v3_diff",
        overwrite=True,
        port=7776
    )

    # filter_test = DatasetConfig(
    #     name="test",
    #     dataset_type="images",
    #     images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/test/0Fe0mUrqmBg_test",
    #     overwrite=True,
    #     port=7773
    # )

    # filter_test_fv2 = DatasetConfig(
    #     name="test_fv2",
    #     dataset_type="images",
    #     images_dir="/mnt/data_2/pavan/project_helpers/data_miner/output/projects/test/0Fe0mUrqmBg_fv2",
    #     overwrite=True,
    #     port=7774
    # )

    # visualize_remote_view_images(dataset_config=delivery_videos)
    visualize_remote_view_images_multiple(
        dataset_configs=[
            # delivery_videos,
            # real_estate_videos,
            direct_doors_videos,
            # direct_doors_videos_v2,
            # direct_doors_removed,
            # direct_doors_videos_v3,
            # direct_doors_removed_v3
        ],
        port = 0 # if port not zero, all datasets will be visualized in same port 
        # ignoring individual dataset ports
    )
