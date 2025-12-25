"""Unified data loader for COCO JSON and YOLO TXT formats."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from detection_metrics.logging import logger, create_progress
from detection_metrics.configs.config import DatasetConfig, PredictionEntry


# Constants
IMGS_EXTNS = ['.jpg', '.jpeg', '.png']


class DataLoader:
    """
    Handles data ingestion from various formats (COCO, YOLO).
    
    Outputs consistent format: List of [img_id, class_id, conf, x1, y1, x2, y2]
    """
    
    def __init__(self, coco_standard_classids: bool = False):
        """
        Initialize DataLoader.
        
        Args:
            coco_standard_classids: If True, class IDs start from 1 (COCO standard).
                                   If False, class IDs start from 0.
        """
        self.coco_standard_classids = coco_standard_classids
        self.categories: Dict[int, str] = {}
        self.image_file_to_ids: Dict[str, int] = {}
    
    def load_coco(
        self, 
        json_path: Path, 
        is_gt: bool = False, 
        only_labels: bool = False
    ) -> List[List[float]]:
        """
        Parse COCO JSON format.
        
        Args:
            json_path: Path to COCO JSON file.
            is_gt: Whether this is ground truth (contains categories/images).
            only_labels: If True, only load category labels.
        
        Returns:
            List of [img_id, class_id, conf, x1, y1, x2, y2]
        """
        data = []
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        if is_gt:
            if not all(k in coco_data for k in ['categories', 'images', 'annotations']):
                raise ValueError(
                    "COCO GT file must contain 'categories', 'images', and 'annotations' keys"
                )
            
            # Parse Categories
            for cat in coco_data['categories']:
                self.categories[cat['id']] = cat['name']
            
            if only_labels:
                return []

            # Parse Images
            for img in coco_data['images']:
                self.image_file_to_ids[Path(img['file_name']).stem] = img['id']

            logger.info(f"Loading {len(coco_data['annotations'])} GT annotations...")
            with create_progress() as progress:
                task = progress.add_task("Parsing COCO GT", total=len(coco_data['annotations']))
                for ann in coco_data['annotations']:
                    x, y, w, h = ann['bbox']
                    cid = ann['category_id']
                    data.append([ann['image_id'], cid, 1.0, x, y, x + w, y + h])
                    progress.update(task, advance=1)
        
        elif isinstance(coco_data, list):
            # Prediction List format
            if not self.categories:
                raise ValueError("Categories not set - load GT first")
            if not self.image_file_to_ids:
                raise ValueError("Image IDs not set - load GT first")
            
            image_ids = set(self.image_file_to_ids.values())
            
            logger.info(f"Loading {len(coco_data)} predictions...")
            with create_progress() as progress:
                task = progress.add_task("Parsing COCO Preds", total=len(coco_data))
                for ann in coco_data:
                    x, y, w, h = ann['bbox']
                    cid = ann['category_id']
                    
                    if cid not in self.categories:
                        raise ValueError(f"Category ID {cid} not found in GT categories")
                    
                    img_id = ann['image_id']
                    if img_id not in image_ids:
                        raise ValueError(f"Image ID {img_id} not found in GT images")
                    
                    data.append([img_id, cid, ann['score'], x, y, x + w, y + h])
                    progress.update(task, advance=1)
        else:
            raise ValueError("Unsupported COCO format for predictions")
            
        return data

    def load_yolo(
        self, 
        folder_path: Path, 
        is_gt: bool = False, 
        only_labels: bool = False
    ) -> List[List[float]]:
        """
        Parse YOLO TXT files.
        
        Args:
            folder_path: Path to folder containing YOLO labels and data.yaml.
            is_gt: Whether this is ground truth.
            only_labels: If True, only load category labels.
        
        Returns:
            List of [img_id, class_id, conf, x1, y1, x2, y2]
        """
        from PIL import Image

        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        data_yaml_path = folder_path / 'data.yaml'
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {folder_path}")
        
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)

        # Parse Categories
        class_idx_delta = 1 if self.coco_standard_classids else 0
        
        if is_gt:
            if self.categories:
                raise ValueError("Categories already set - GT already loaded")
            self.categories = {
                i + class_idx_delta: name 
                for i, name in enumerate(data_yaml['names'])
            }
            
            if only_labels:
                return []
            
            if self.image_file_to_ids:
                raise ValueError("Image IDs already set - GT already loaded")
            
            # Generate image IDs from filenames
            self.image_file_to_ids = {
                img_id_str: int(hashlib.sha256(img_id_str.encode('utf-8')).hexdigest()[:16], 16) 
                for img_id_str in data_yaml.get('images', [])
            }
        else:
            if not self.categories:
                raise ValueError("Categories not set - load GT first")
            if not self.image_file_to_ids:
                raise ValueError("Image IDs not set - load GT first")
        
        # Build image size map
        images_path = Path(data_yaml['images_path'])
        img_wh_map = {}
        
        for img_path in images_path.glob('*'):
            if img_path.suffix.lower() not in IMGS_EXTNS:
                continue
            img_id_str = img_path.stem
            with Image.open(img_path) as img:
                img_wh_map[img_id_str] = img.size  # (W, H)

        data = []
        txt_files = list(folder_path.glob("*.txt"))
        
        logger.info(f"Loading {len(txt_files)} YOLO {'GT' if is_gt else 'prediction'} files...")
        with create_progress() as progress:
            task = progress.add_task(f"Parsing YOLO {'GT' if is_gt else 'Preds'}", total=len(txt_files))
            
            for file_path in txt_files:
                img_id_str = file_path.stem
                
                # For GT, generate image ID if not already present
                if is_gt and img_id_str not in self.image_file_to_ids:
                    self.image_file_to_ids[img_id_str] = abs(hash(img_id_str)) % (10**9)
                
                img_id = self.image_file_to_ids.get(img_id_str)
                if img_id is None:
                    raise ValueError(f"Image ID not found for {img_id_str}")
                
                wh = img_wh_map.get(img_id_str)
                if wh is None:
                    raise ValueError(f"Image size not found for {img_id_str}")
                W, H = wh
                
                raw = np.loadtxt(file_path, ndmin=2)
                if raw.size == 0:
                    progress.update(task, advance=1)
                    continue
                
                # raw: [class, xc, yc, w, h, (conf)]
                # Convert to [img_id, class, conf, x1, y1, x2, y2]
                cls_ids = raw[:, 0] + class_idx_delta
                xc, yc, w, h = raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
                x1 = (xc - w/2) * W
                y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W
                y2 = (yc + h/2) * H
                
                confs = raw[:, 5] if not is_gt and raw.shape[1] > 5 else np.ones_like(cls_ids)
                
                batch = np.column_stack((
                    np.full_like(cls_ids, img_id),
                    cls_ids,
                    confs,
                    x1, y1, x2, y2
                ))
                data.extend(batch.tolist())
                progress.update(task, advance=1)
            
        return data
    
    def load_gt(self, gt_path: Path, only_labels: bool = False) -> List[List[float]]:
        """
        Load ground truth from COCO or YOLO format (auto-detected).
        
        Args:
            gt_path: Path to GT file (JSON) or folder (YOLO).
            only_labels: If True, only load category labels.
        
        Returns:
            List of [img_id, class_id, conf, x1, y1, x2, y2]
        """
        gt_path = Path(gt_path)
        
        if gt_path.suffix == ".json":
            logger.info("Loading COCO GT...")
            return self.load_coco(gt_path, is_gt=True, only_labels=only_labels)
        elif gt_path.is_dir():
            logger.info("Loading YOLO GT...")
            return self.load_yolo(gt_path, is_gt=True, only_labels=only_labels)
        else:
            raise ValueError(f"Unsupported GT format: {gt_path}")
    
    def load_predictions(
        self, 
        predictions: List[PredictionEntry]
    ) -> Dict[str, List[List[float]]]:
        """
        Load predictions from multiple files.
        
        Args:
            predictions: List of PredictionEntry with path and name.
        
        Returns:
            Dict mapping prediction name to list of detections.
        """
        pred_data_dict = {}
        
        for pred_entry in predictions:
            pred_path = Path(pred_entry.path)
            pred_name = pred_entry.name
            
            if pred_path.suffix == ".json":
                logger.info(f"Loading COCO predictions: {pred_name}")
                pred_data = self.load_coco(pred_path, is_gt=False)
            elif pred_path.is_dir():
                logger.info(f"Loading YOLO predictions: {pred_name}")
                pred_data = self.load_yolo(pred_path, is_gt=False)
            else:
                raise ValueError(f"Unsupported prediction format: {pred_path}")
            
            pred_data_dict[pred_name] = pred_data
        
        return pred_data_dict

    def load_dataset(
        self, 
        dataset_config: DatasetConfig,
        skip_models: Optional[List[str]] = None
    ) -> Tuple[List[List[float]], Dict[str, List[List[float]]]]:
        """
        Load complete dataset (GT + predictions).
        
        Args:
            dataset_config: Dataset configuration.
            skip_models: List of model names to skip loading.
        
        Returns:
            Tuple of (gt_data, predictions_dict)
        """
        skip_models = skip_models or []
        
        if not dataset_config.gt_path:
            raise ValueError("GT path not specified in config")
        if not dataset_config.predictions:
            raise ValueError("No predictions specified in config")
        
        # Filter predictions
        predictions = [
            p for p in dataset_config.predictions 
            if p.name not in skip_models
        ]
        
        if not predictions:
            # Only load labels if all models are skipped
            self.load_gt(dataset_config.gt_path, only_labels=True)
            return [], {}
        
        gt_data = self.load_gt(dataset_config.gt_path)
        pred_data_dict = self.load_predictions(predictions)
        
        return gt_data, pred_data_dict
