# Detection utilities for visualization and post-processing
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def visualize_detections(
    images_folder: Union[str, Path],
    annotations_folder: Union[str, Path],
    output_folder: Union[str, Path],
    class_names: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    line_width: int = 3,
    show_labels: bool = True,
    desc: str = "Visualizing Detections",
):
    """
    Visualize detections by overlaying bounding boxes on images.
    
    Expects YOLO format annotations: class x_center y_center width height (normalized)
    
    Args:
        images_folder: Folder containing input images
        annotations_folder: Folder containing YOLO format .txt annotation files
        output_folder: Folder to save visualized images
        class_names: Optional list of class names for labels
        colors: Optional list of colors for classes (default: red, green, blue, ...)
        line_width: Width of bounding box lines
        show_labels: Whether to show class labels
        desc: Description for progress bar
    """
    images_folder = Path(images_folder)
    annotations_folder = Path(annotations_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if colors is None:
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    
    for ann_file in tqdm(
        list(annotations_folder.glob("*.txt")),
        desc=desc,
    ):
        # Try multiple image extensions
        image_file = None
        for ext in IMG_EXTENSIONS:
            candidate = images_folder / f"{ann_file.stem}{ext}"
            if candidate.exists():
                image_file = candidate
                break
            candidate = images_folder / f"{ann_file.stem}{ext.upper()}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file is None:
            print(f"Image file for {ann_file.stem} not found. Skipping.")
            continue
        
        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size
        
        # Load annotations
        try:
            anns = np.loadtxt(ann_file)
            if len(anns) == 0:
                continue
            if len(anns.shape) == 1:
                anns = anns[np.newaxis, :]
        except Exception:
            continue
        
        for ann in anns:
            # YOLO format: class x_center y_center width height (normalized)
            cls_id = int(ann[0])
            x_center, y_center, width, height = ann[1:5]
            
            # Convert to pixel coordinates
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h
            
            color = colors[cls_id % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            if show_labels and class_names and cls_id < len(class_names):
                label = class_names[cls_id]
                draw.text((x1, y1 - 15), label, fill=color)
        
        output_path = output_folder / image_file.name
        image.save(output_path)


def visualize_masks(
    images_folder: Union[str, Path],
    masks_folder: Union[str, Path],
    output_folder: Union[str, Path],
    alpha: float = 0.5,
    draw_bboxes: bool = True,
):
    """
    Visualize segmentation masks overlaid on images.
    
    Args:
        images_folder: Folder containing input images
        masks_folder: Folder containing .npz mask files
        output_folder: Folder to save visualized images
        alpha: Transparency for mask overlay
        draw_bboxes: Whether to draw bounding boxes around masks
    """
    from .sam import mask_to_bbox
    
    images_folder = Path(images_folder)
    masks_folder = Path(masks_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate random colors for masks
    np.random.seed(42)
    
    for mask_file in tqdm(
        list(masks_folder.glob("*.npz")),
        desc="Visualizing Masks",
    ):
        # Find corresponding image
        image_file = None
        for ext in IMG_EXTENSIONS:
            candidate = images_folder / f"{mask_file.stem}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file is None:
            continue
        
        image = np.array(Image.open(image_file).convert("RGB"))
        data = np.load(mask_file)
        masks = data["masks"]
        
        # Create colored overlay
        overlay = image.copy()
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, 3)
            overlay[mask > 0] = (
                overlay[mask > 0] * (1 - alpha) + color * alpha
            ).astype(np.uint8)
        
        # Draw bounding boxes
        pil_image = Image.fromarray(overlay)
        if draw_bboxes:
            draw = ImageDraw.Draw(pil_image)
            for mask in masks:
                x1, y1, x2, y2 = mask_to_bbox(mask)
                draw.rectangle([x1, y1, x2, y2], outline="white", width=2)
        
        output_path = output_folder / image_file.name
        pil_image.save(output_path)


def load_yolo_annotations(ann_file: Union[str, Path]) -> list[dict]:
    """
    Load YOLO format annotations from a file.
    
    Args:
        ann_file: Path to annotation file
        
    Returns:
        List of dicts with 'class_id', 'x_center', 'y_center', 'width', 'height'
    """
    ann_file = Path(ann_file)
    if not ann_file.exists():
        return []
    
    try:
        anns = np.loadtxt(ann_file)
        if len(anns) == 0:
            return []
        if len(anns.shape) == 1:
            anns = anns[np.newaxis, :]
    except Exception:
        return []
    
    return [
        {
            "class_id": int(ann[0]),
            "x_center": ann[1],
            "y_center": ann[2],
            "width": ann[3],
            "height": ann[4],
        }
        for ann in anns
    ]


def save_yolo_annotations(
    detections: list,
    output_path: Union[str, Path],
):
    """
    Save detections in YOLO format.
    
    Args:
        detections: List of [class_id, x, y, w, h, ...] (normalized coords)
        output_path: Path to save the annotation file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for det in detections:
            cls_id, x_min, y_min, width, height = det[:5]
            # Convert to YOLO format: class x_center y_center width height
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_detection_summary(
    output_dir: Union[str, Path],
    model_name: str,
    detection_classes: list[str],
    threshold: float,
    total_images: int,
    images_with_detections: int,
    total_detections: int,
    detection_counts: dict,
    avg_latency: Optional[float] = None,
):
    """Save detection summary as JSON."""
    output_dir = Path(output_dir)
    summary = {
        "model": model_name,
        "detection_classes": detection_classes,
        "threshold": threshold,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "detection_counts": detection_counts,
    }
    if avg_latency is not None:
        summary["avg_latency_s"] = avg_latency
    
    summary_path = output_dir / "detection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")
