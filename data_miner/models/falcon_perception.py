import json
import os
import argparse
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

    def __init__(
        self,
        detection_class: Union[str, list[str]] = "object",
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: Optional[str] = None,
        compile: bool = False,
        task: str = "segmentation",
        seed: int = 42,
        max_length: int = 4096,
        min_dimension: int = 256,
        max_dimension: int = 1024,
    ):
        self.model_loaded = False
        self.detection_class = detection_class
        self.model_id = model_id
        self.device = device
        self.dtype = dtype or ("bfloat16" if torch.cuda.is_available() else "float32")
        self.compile = compile
        self.task = task
        self.seed = seed
        self.max_length = max_length
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.model = None
        self.tokenizer = None
        self.model_args = None
        self.engine = None
        self.stop_token_ids = None

        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]

    def _resolved_device(self) -> Optional[str]:
        if self.device in (None, "auto"):
            return None
        return self.device

    def load_model(self):
        """Load Falcon Perception model."""
        if self.model_loaded:
            return

        print(f"Loading Falcon Perception: {self.model_id}")

        from falcon_perception import load_and_prepare_model, setup_torch_config
        from falcon_perception.batch_inference import BatchInferenceEngine

        setup_torch_config()
        self.model, self.tokenizer, self.model_args = load_and_prepare_model(
            hf_model_id=self.model_id,
            device=self._resolved_device(),
            dtype=self.dtype,
            compile=self.compile,
        )
        self.engine = BatchInferenceEngine(self.model, self.tokenizer)
        self.stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.end_of_query_token_id,
        ]

        if self.task == "segmentation" and not self.model_args.do_segmentation:
            self.task = "detection"

        self.model_loaded = True
        print(f"Falcon Perception loaded on {self.model.device}")

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        self.model_args = None
        self.engine = None
        self.stop_token_ids = None
        self.model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Falcon Perception model unloaded")

    def _prepare_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def _finalize_detections(
        self,
        image_size: tuple[int, int],
        all_boxes: list[np.ndarray],
        all_scores: list[float],
        all_labels: list[str],
        all_metadata: list[dict[str, Any]],
        classes: list[str],
        output_format: str,
        nms_threshold: Optional[float],
        include_metadata: bool,
    ):
        image_width, image_height = image_size

        if len(all_boxes) == 0:
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
        return self.detect_batch(
            [image],
            threshold=threshold,
            detection_classes=detection_classes,
            output_format=output_format,
            nms_threshold=nms_threshold,
            include_metadata=include_metadata,
        )[0]

    @torch.no_grad()
    def detect_batch(
        self,
        images: list[Union[str, Path, Image.Image]],
        threshold: float = 0.3,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
        nms_threshold: Optional[float] = 0.5,
        include_metadata: bool = False,
    ):
        """Run Falcon Perception detection on a batch of images."""
        if not self.model_loaded:
            self.load_model()

        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import process_batch_and_generate
        from falcon_perception.visualization_utils import pair_bbox_entries

        classes = detection_classes or self.detection_class

        pil_images = [self._prepare_image(image) for image in images]

        per_image_boxes: list[list[np.ndarray]] = [[] for _ in pil_images]
        per_image_scores: list[list[float]] = [[] for _ in pil_images]
        per_image_labels: list[list[str]] = [[] for _ in pil_images]
        per_image_metadata: list[list[dict[str, Any]]] = [[] for _ in pil_images]

        for text_prompt in classes:
            prompt = build_prompt_for_task(text_prompt, self.task)
            batch_inputs = process_batch_and_generate(
                self.tokenizer,
                [(pil_image, prompt) for pil_image in pil_images],
                max_length=self.max_length,
                min_dimension=self.min_dimension,
                max_dimension=self.max_dimension,
            )
            batch_inputs = {
                key: (value.to(self.model.device) if torch.is_tensor(value) else value)
                for key, value in batch_inputs.items()
            }

            _, aux_out = self.engine.generate(
                **batch_inputs,
                max_new_tokens=2048,
                temperature=0.0,
                stop_token_ids=self.stop_token_ids,
                seed=self.seed,
                task=self.task,
            )

            for image_index, aux in enumerate(aux_out):
                image_width, image_height = pil_images[image_index].size
                paired_bboxes = pair_bbox_entries(aux.bboxes_raw)
                masks_rle = aux.masks_rle if self.task == "segmentation" else []

                for index, bbox_entry in enumerate(paired_bboxes):
                    prediction = {
                        "xy": {
                            "x": bbox_entry.get("x"),
                            "y": bbox_entry.get("y"),
                        },
                        "hw": {
                            "h": bbox_entry.get("h"),
                            "w": bbox_entry.get("w"),
                        },
                        "mask_rle": masks_rle[index] if index < len(masks_rle) else None,
                        "score": 1.0,
                        "label": text_prompt,
                    }
                    parsed = self._parse_prediction(prediction, text_prompt)
                    if parsed is None:
                        continue
                    bbox = parsed["primary_bbox"]
                    confidence = parsed["confidence"]
                    label = parsed["label"]
                    if confidence < threshold:
                        continue

                    x1, y1, x2, y2 = bbox
                    per_image_boxes[image_index].append(
                        np.array(
                            [
                                x1 * image_width,
                                y1 * image_height,
                                x2 * image_width,
                                y2 * image_height,
                            ]
                        )
                    )
                    per_image_scores[image_index].append(confidence)
                    per_image_labels[image_index].append(label)
                    per_image_metadata[image_index].append(parsed)

        return [
            self._finalize_detections(
                pil_images[index].size,
                per_image_boxes[index],
                per_image_scores[index],
                per_image_labels[index],
                per_image_metadata[index],
                classes,
                output_format,
                nms_threshold,
                include_metadata,
            )
            for index in range(len(pil_images))
        ]

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.3,
        batch_size: int = 4,
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

        for start_index in tqdm(range(0, len(image_files), batch_size), desc="Processing"):
            batch_files = image_files[start_index:start_index + batch_size]
            try:
                batch_detections = self.detect_batch(batch_files, threshold=threshold)
                for image_file, detections in zip(batch_files, batch_detections):
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
                print(f"Error processing batch starting at {batch_files[0]}: {exc}")

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


def main():
    parser = argparse.ArgumentParser(description="Batch process images with Falcon Perception.")
    parser.add_argument("input_dir", type=Path, help="Folder containing images to process")
    parser.add_argument("output_dir", type=Path, help="Folder where YOLO predictions will be written")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="One or more text queries/classes to run against every image",
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of images per inference batch")
    parser.add_argument(
        "--model-id",
        default=FalconPerceptionHelper.DEFAULT_MODEL_ID,
        help="Falcon model id to use",
    )
    parser.add_argument(
        "--task",
        choices=["segmentation", "detection"],
        default="segmentation",
        help="Falcon task mode",
    )
    parser.add_argument("--device", default="auto", help="Device to run on, e.g. auto/cuda/cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Torch dtype passed to the upstream Falcon loader",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable upstream model compilation",
    )
    args = parser.parse_args()

    print(f"Running Falcon Perception:{args}")
    # return
    helper = FalconPerceptionHelper(
        detection_class=args.classes,
        model_id=args.model_id,
        task=args.task,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
    )
    try:
        helper.process_folder(
            args.input_dir,
            args.output_dir,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
    finally:
        helper.unload_model()


if __name__ == "__main__":
    main()