import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

load_dotenv(".env")
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login

    login(os.getenv("HF_TOKEN"))

from data_miner.models.detection_utils import (
    apply_nms,
    create_class_ids,
    format_detections,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _center_size_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
) -> tuple[float, float, float, float]:
    """Convert normalized center-size coordinates to normalized xyxy."""
    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2
    return (
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    )


def _mask_rle_to_xyxy(mask_rle: dict) -> Optional[tuple[float, float, float, float]]:
    """Decode a Falcon mask RLE and return normalized xyxy bounds."""
    if not isinstance(mask_rle, dict):
        return None

    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not isinstance(size, list) or len(size) != 2 or counts is None:
        return None

    try:
        encoded_mask = {
            "size": size,
            "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
        }
        decoded_mask = mask_utils.decode(encoded_mask)
    except Exception:
        return None

    if decoded_mask is None or decoded_mask.size == 0:
        return None

    foreground = np.argwhere(decoded_mask.astype(bool))
    if foreground.size == 0:
        return None

    y_coords = foreground[:, 0]
    x_coords = foreground[:, 1]
    image_height, image_width = decoded_mask.shape[:2]

    x1 = x_coords.min() / image_width
    y1 = y_coords.min() / image_height
    x2 = (x_coords.max() + 1) / image_width
    y2 = (y_coords.max() + 1) / image_height

    return (
        max(0.0, min(1.0, float(x1))),
        max(0.0, min(1.0, float(y1))),
        max(0.0, min(1.0, float(x2))),
        max(0.0, min(1.0, float(y2))),
    )


def _bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU for two normalized xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


class FalconPerceptionHelper:
    """Helper for Falcon Perception open-vocabulary auto-annotation."""

    DEFAULT_MODEL_ID = "tiiuae/Falcon-Perception"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["object"])
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL_ID)
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]

    def load_model(self):
        """Load Falcon Perception model."""
        if self.model_loaded:
            return

        print(f"Loading Falcon Perception: {self.model_id}")

        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

        self.model_loaded = True
        print(f"Falcon Perception loaded on {self.device}")

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Falcon Perception model unloaded")

    def get_model(self):
        return self.model

    def _parse_prediction(self, prediction: dict, label: str) -> Optional[dict[str, Any]]:
        """Parse one Falcon prediction and retain both geometry sources."""
        mask_bbox = _mask_rle_to_xyxy(prediction.get("mask_rle"))

        coord_bbox = None
        xy = prediction.get("xy")
        hw = prediction.get("hw")
        if isinstance(xy, dict) and isinstance(hw, dict):
            try:
                coord_bbox = _center_size_to_xyxy(
                    float(xy["x"]),
                    float(xy["y"]),
                    float(hw["w"]),
                    float(hw["h"]),
                )
            except (KeyError, TypeError, ValueError):
                coord_bbox = None

        primary_bbox = mask_bbox or coord_bbox
        if primary_bbox is None:
            return None

        raw_confidence = prediction.get("confidence", prediction.get("score", 1.0))
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 1.0

        bbox_iou = None
        review_required = False
        if mask_bbox is not None and coord_bbox is not None:
            bbox_iou = _bbox_iou(mask_bbox, coord_bbox)
            review_required = bbox_iou < 0.95

        return {
            "primary_bbox": primary_bbox,
            "mask_bbox": mask_bbox,
            "coord_bbox": coord_bbox,
            "confidence": confidence,
            "label": str(prediction.get("label") or label),
            "bbox_iou": bbox_iou,
            "review_required": review_required,
        }

    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.3,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
        nms_threshold: Optional[float] = 0.5,
        include_metadata: bool = False,
    ):
        """Run Falcon Perception detection on an image."""
        if not self.model_loaded:
            self.load_model()

        t0 = time.perf_counter()
        classes = detection_classes or self.detection_class

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            img_path = Path(image)
        else:
            pil_image = image.convert("RGB")
            img_path = None

        image_width, image_height = pil_image.size

        all_boxes = []
        all_scores = []
        all_labels = []
        all_metadata = []

        for text_prompt in classes:
            outputs = self.model.generate(pil_image, text_prompt)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
                outputs = outputs[0]

            for prediction in outputs or []:
                if not isinstance(prediction, dict):
                    continue
                parsed = self._parse_prediction(prediction, text_prompt)
                if parsed is None:
                    continue
                bbox = parsed["primary_bbox"]
                confidence = parsed["confidence"]
                label = parsed["label"]
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = bbox
                all_boxes.append(
                    np.array(
                        [
                            x1 * image_width,
                            y1 * image_height,
                            x2 * image_width,
                            y2 * image_height,
                        ]
                    )
                )
                all_scores.append(confidence)
                all_labels.append(label)
                all_metadata.append(parsed)

        self.latencies.append(time.perf_counter() - t0)

        if len(all_boxes) == 0:
            if img_path:
                pass
            return [] if output_format == "normalized" else 0

        boxes_np = np.stack(all_boxes)
        scores_np = np.array(all_scores)
        class_ids_np = create_class_ids(
            len(all_boxes),
            original_ids=np.arange(len(all_boxes)),
            merge_classes=False,
        )

        boxes_np, scores_np, class_ids_np = apply_nms(
            boxes_np, scores_np, class_ids_np, nms_threshold
        )

        detections = format_detections(
            boxes_np,
            scores_np,
            class_ids_np,
            (image_width, image_height),
            output_format,
        )

        if include_metadata:
            detailed_detections = []
            for detection in detections:
                label_index = int(detection[0])
                metadata = all_metadata[label_index] if label_index < len(all_metadata) else None

                if metadata is None:
                    continue

                detailed_detections.append(
                    {
                        "label": metadata["label"],
                        "confidence": float(detection[5]),
                        "primary_bbox": [float(value) for value in metadata["primary_bbox"]],
                        "mask_bbox": (
                            [float(value) for value in metadata["mask_bbox"]]
                            if metadata["mask_bbox"] is not None
                            else None
                        ),
                        "coord_bbox": (
                            [float(value) for value in metadata["coord_bbox"]]
                            if metadata["coord_bbox"] is not None
                            else None
                        ),
                        "bbox_iou": metadata["bbox_iou"],
                        "review_required": metadata["review_required"],
                        "yolo_bbox": [float(value) for value in detection[1:5]],
                    }
                )

            return detailed_detections

        if output_format == "normalized":
            for detection in detections:
                label_index = int(detection[0])
                detection[0] = 0
                detection.append(
                    all_labels[label_index] if label_index < len(all_labels) else classes[0]
                )

        return detections

    def infer_image(self, img_path, threshold=0.3, detection_classes=None):
        """Alias for detect() with pixel output format."""
        return self.detect(
            img_path, threshold, detection_classes, output_format="pixel"
        )

    def detect_objects(self, image, detection_classes, threshold=0.3):
        """Alias for detect() with normalized output format."""
        return self.detect(
            image, threshold, detection_classes, output_format="normalized"
        )

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.3,
    ):
        """Process all images in a folder and save YOLO-style txt output."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        predictions_dir = output_dir / "pred_txt"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_loaded:
            self.load_model()

        image_files = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Processing {len(image_files)} images")

        total_detections = 0
        images_with_detections = 0

        for image_file in tqdm(image_files, desc="Processing"):
            try:
                detections = self.detect(image_file, threshold=threshold)
                if len(detections) == 0:
                    continue

                images_with_detections += 1
                total_detections += len(detections)

                txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(txt_path, "w") as file_handle:
                    for det in detections:
                        cls_id, x_min, y_min, w, h = det[:5]
                        score = det[5]
                        x_center = x_min + w / 2
                        y_center = y_min + h / 2
                        file_handle.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {score:.4f}\n"
                        )
            except Exception as exc:
                print(f"Error processing {image_file}: {exc}")

        print(
            f"\nDone! {total_detections} detections in {images_with_detections}/{len(image_files)} images"
        )

        with open(output_dir / "summary.json", "w") as file_handle:
            json.dump(
                {
                    "model": self.model_id,
                    "classes": self.detection_class,
                    "total_images": len(image_files),
                    "images_with_detections": images_with_detections,
                    "total_detections": total_detections,
                },
                file_handle,
                indent=2,
            )