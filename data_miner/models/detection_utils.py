# Detection utilities for visualization and post-processing
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


# =============================================================================
# NMS Utilities
# =============================================================================


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Non-Maximum Suppression for bounding boxes.

    Args:
        boxes: Array of shape (N, 4) with xyxy format (x1, y1, x2, y2)
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Array of indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # Try torchvision first (faster, GPU-enabled)
    try:
        import torch
        from torchvision.ops import nms as tv_nms

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        return tv_nms(boxes_t, scores_t, iou_threshold).numpy()
    except ImportError:
        pass

    # Numpy fallback
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=int)


def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    NMS applied separately per class.

    Args:
        boxes: Array of shape (N, 4) with xyxy format
        scores: Array of shape (N,)
        class_ids: Array of shape (N,) with class indices
        iou_threshold: IoU threshold for suppression

    Returns:
        Array of indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    keep = []
    for cls in np.unique(class_ids):
        mask = class_ids == cls
        indices = np.where(mask)[0]
        cls_keep = nms(boxes[mask], scores[mask], iou_threshold)
        keep.extend(indices[cls_keep])
    return np.array(sorted(keep), dtype=int)


# =============================================================================
# Detection Post-Processing Utilities
# =============================================================================


def create_class_ids(
    n: int,
    original_ids: Optional[np.ndarray] = None,
    merge_classes: bool = True,
) -> np.ndarray:
    """
    Create class IDs array for text-based detection.

    For text-based detectors (Grounding DINO, SAM3, OWLv2), multiple text prompts
    often refer to the same semantic class (e.g., "door", "glass door", "entrance door").

    Args:
        n: Number of detections
        original_ids: Original class indices from detection (e.g., which text prompt matched)
        merge_classes: If True, all class IDs are 0 (single class).
                      If False, preserve original_ids or use sequential indices.

    Returns:
        Array of shape (n,) with class IDs
    """
    if merge_classes:
        return np.zeros(n, dtype=int)
    if original_ids is not None:
        return np.asarray(original_ids, dtype=int)
    return np.arange(n, dtype=int)  # Fallback: sequential indices


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    nms_threshold: Optional[float] = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply NMS to detections.

    Args:
        boxes: (N, 4) xyxy format
        scores: (N,)
        class_ids: (N,)
        nms_threshold: IoU threshold, None to skip NMS

    Returns:
        Filtered (boxes, scores, class_ids)
    """
    if len(boxes) == 0 or nms_threshold is None:
        return boxes, scores, class_ids

    keep = nms_per_class(boxes, scores, class_ids, nms_threshold)
    return boxes[keep], scores[keep], class_ids[keep]


def format_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    image_size: tuple[int, int],
    output_format: str = "normalized",
) -> Union[list, tuple]:
    """
    Format detections for output.

    Args:
        boxes: (N, 4) xyxy pixel format
        scores: (N,)
        class_ids: (N,) 0-indexed
        image_size: (width, height)
        output_format: "normalized" or "pixel"

    Returns:
        If "normalized": List of [class_id, x, y, w, h, conf] with 0-1 coords
        If "pixel": Tuple of (bboxes, confidences, class_ids) with 1-indexed class_ids
    """
    if len(boxes) == 0:
        return [] if output_format == "normalized" else 0

    w, h = image_size

    if output_format == "pixel":
        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        return bboxes, scores.tolist(), (class_ids + 1).tolist()  # 1-indexed
    else:
        results = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            results.append(
                [
                    int(cls_id),
                    x1 / w,
                    y1 / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h,
                    float(score),
                ]
            )
        return results


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
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "magenta",
        ]

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
            f.write(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


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


# =============================================================================
# Helper Functions for Generating Query Image Crops
# =============================================================================


def crop_from_yolo(
    image: Union[str, Path, Image.Image],
    label_path: Union[str, Path],
    class_id: Optional[int] = None,
    padding: float = 0.0,
) -> list[Image.Image]:
    """
    Generate object crops from YOLO format annotations.

    YOLO format: class_id x_center y_center width height (normalized 0-1)

    Args:
        image: Image path or PIL Image
        label_path: Path to YOLO .txt label file
        class_id: Optional filter for specific class (None = all classes)
        padding: Extra padding around crops as fraction of box size (default: 0.0)

    Returns:
        List of cropped PIL Images
    """
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    img_width, img_height = pil_image.size
    crops = []

    label_path = Path(label_path)
    if not label_path.exists():
        return crops

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if class_id is not None and cls_id != class_id:
                continue

            # Parse YOLO format (normalized center x, y, w, h)
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            # Convert to corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Add padding
            if padding > 0:
                pad_w = width * padding
                pad_h = height * padding
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(img_width, x2 + pad_w)
                y2 = min(img_height, y2 + pad_h)

            crop = pil_image.crop((int(x1), int(y1), int(x2), int(y2)))
            crops.append(crop)

    return crops


def crop_from_coco(
    image: Union[str, Path, Image.Image],
    annotation: dict,
    padding: float = 0.0,
) -> Image.Image:
    """
    Generate object crop from a COCO annotation dict.

    COCO bbox format: [x_min, y_min, width, height] (pixel coordinates)

    Args:
        image: Image path or PIL Image
        annotation: COCO annotation dict with 'bbox' key
        padding: Extra padding around crop as fraction of box size (default: 0.0)

    Returns:
        Cropped PIL Image
    """
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    img_width, img_height = pil_image.size

    # COCO format: [x_min, y_min, width, height]
    bbox = annotation["bbox"]
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h

    # Add padding
    if padding > 0:
        pad_w = w * padding
        pad_h = h * padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(img_width, x2 + pad_w)
        y2 = min(img_height, y2 + pad_h)

    return pil_image.crop((int(x1), int(y1), int(x2), int(y2)))


def crops_from_coco_json(
    images_dir: Union[str, Path],
    coco_json_path: Union[str, Path],
    category_id: Optional[int] = None,
    max_crops: Optional[int] = None,
    padding: float = 0.0,
) -> list[Image.Image]:
    """
    Generate object crops from a COCO format JSON annotations file.

    Args:
        images_dir: Directory containing images
        coco_json_path: Path to COCO annotations JSON file
        category_id: Optional filter for specific category (None = all)
        max_crops: Maximum number of crops to return (None = all)
        padding: Extra padding around crops as fraction of box size

    Returns:
        List of cropped PIL Images
    """
    images_dir = Path(images_dir)
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Build image id -> filename mapping
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    crops = []
    for ann in coco_data["annotations"]:
        if category_id is not None and ann.get("category_id") != category_id:
            continue

        if "bbox" not in ann:
            continue

        image_id = ann["image_id"]
        filename = id_to_filename.get(image_id)
        if not filename:
            continue

        image_path = images_dir / filename
        if not image_path.exists():
            continue

        try:
            crop = crop_from_coco(image_path, ann, padding=padding)
            crops.append(crop)
        except Exception:
            continue

        if max_crops and len(crops) >= max_crops:
            break

    return crops


def crops_from_yolo_folder(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    class_id: Optional[int] = None,
    max_crops: Optional[int] = None,
    padding: float = 0.0,
) -> list[Image.Image]:
    """
    Generate object crops from a folder of YOLO format labels.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO .txt label files
        class_id: Optional filter for specific class (None = all)
        max_crops: Maximum number of crops to return (None = all)
        padding: Extra padding around crops as fraction of box size

    Returns:
        List of cropped PIL Images
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    crops = []

    for label_path in labels_dir.glob("*.txt"):
        # Find corresponding image
        stem = label_path.stem
        image_path = None
        for ext in IMG_EXTENSIONS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            continue

        try:
            file_crops = crop_from_yolo(
                image_path, label_path, class_id=class_id, padding=padding
            )
            crops.extend(file_crops)
        except Exception:
            continue

        if max_crops and len(crops) >= max_crops:
            crops = crops[:max_crops]
            break

    return crops
