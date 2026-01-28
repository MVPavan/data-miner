# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv(".env")
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login

    login(os.getenv("HF_TOKEN"))

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


# =============================================================================
# Mask-to-Bounding-Box Utilities
# =============================================================================


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Convert binary mask to (x_min, y_min, x_max, y_max) bounding box."""
    if mask.sum() == 0:
        return (0, 0, 0, 0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def mask_to_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Convert binary mask to (x, y, width, height) format."""
    x_min, y_min, x_max, y_max = mask_to_bbox(mask)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def masks_to_bboxes(masks: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Convert multiple masks to bounding boxes."""
    return [mask_to_bbox(mask) for mask in masks]


# =============================================================================
# SAM3 Helper Class
# =============================================================================


class SAMHelper:
    """
    SAM 3 Helper for text-promptable concept segmentation.

    SAM 3 performs Promptable Concept Segmentation (PCS):
    - Detects and segments based on text prompts (e.g., "yellow school bus")
    - Can recognize 4 million+ visual concepts
    - Returns instance masks with bounding boxes

    Uses: facebook/sam3 via HuggingFace Transformers
    """

    DEFAULT_MODEL = "facebook/sam3"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["object"])
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL)
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.processor = None

        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]

    def load_model(self):
        """Load SAM3 model and processor."""
        if self.model_loaded:
            return

        print(f"Loading SAM3: {self.model_id}")

        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained(self.model_id)
        self.model = Sam3Model.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        self.model_loaded = True
        print(f"SAM3 loaded on {self.device}")

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("SAM3 model unloaded")

    def get_model(self):
        return self.model

    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.5,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
    ):
        """
        Run text-promptable segmentation on an image.

        Uses efficient multi-prompt inference: vision embeddings are computed once
        and reused across all text prompts.

        Args:
            image: Path to image or PIL Image
            threshold: Confidence threshold (default: 0.5)
            detection_classes: Text prompts to detect (uses self.detection_class if None)
            output_format: "normalized" (default) or "pixel"

        Returns:
            If output_format="normalized":
                List of [class_id, x, y, w, h, confidence] with 0-1 coords
            If output_format="pixel":
                Tuple of (bboxes, confidences, class_ids) with pixel coords
        """
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

        # Pre-compute vision embeddings ONCE for the image
        img_inputs = self.processor(images=pil_image, return_tensors="pt").to(
            self.device
        )
        vision_embeds = self.model.get_vision_features(
            pixel_values=img_inputs.pixel_values
        )
        original_sizes = img_inputs.get("original_sizes").tolist()

        # Accumulate results across all text prompts
        all_boxes = []
        all_scores = []
        all_class_ids = []

        # Run each text prompt efficiently using cached vision embeddings
        for cls_idx, text_prompt in enumerate(classes):
            text_inputs = self.processor(text=text_prompt, return_tensors="pt").to(
                self.device
            )
            outputs = self.model(vision_embeds=vision_embeds, **text_inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=0.5,
                target_sizes=original_sizes,
            )[0]

            for box, score in zip(results.get("boxes", []), results.get("scores", [])):
                all_boxes.append(box.cpu().numpy() if hasattr(box, "cpu") else box)
                all_scores.append(
                    float(score.cpu()) if hasattr(score, "cpu") else float(score)
                )
                all_class_ids.append(cls_idx)

        self.latencies.append(time.perf_counter() - t0)

        if len(all_boxes) == 0:
            if img_path:
                # print(f"No objects detected in {img_path.name}")
                pass
            if output_format == "pixel":
                return 0
            return []

        if output_format == "pixel":
            bboxes = []
            for box in all_boxes:
                x1, y1, x2, y2 = box
                bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            class_ids = [c + 1 for c in all_class_ids]  # 1-indexed
            return bboxes, all_scores, class_ids
        else:
            detection_results = []
            for box, score, cls_id in zip(all_boxes, all_scores, all_class_ids):
                x1, y1, x2, y2 = box
                detection_results.append(
                    [
                        cls_id,  # 0-indexed
                        x1 / image_width,
                        y1 / image_height,
                        (x2 - x1) / image_width,
                        (y2 - y1) / image_height,
                        score,
                    ]
                )
            return detection_results

    # Aliases for backwards compatibility
    def infer_image(self, img_path, threshold=0.5, detection_classes=None):
        """Alias for detect() with pixel output format."""
        return self.detect(
            img_path, threshold, detection_classes, output_format="pixel"
        )

    def detect_objects(self, image, detection_classes, threshold=0.5):
        """Alias for detect() with normalized output format."""
        return self.detect(
            image, threshold, detection_classes, output_format="normalized"
        )

    @torch.no_grad()
    def segment_with_text(
        self,
        image: Union[str, Path, Image.Image],
        text: str,
        threshold: float = 0.5,
    ) -> dict:
        """
        Segment objects matching a text prompt.

        Args:
            image: Image path or PIL Image
            text: Text description (e.g., "yellow school bus")
            threshold: Confidence threshold

        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        if not self.model_loaded:
            self.load_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        inputs = self.processor(
            images=pil_image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        return {
            "masks": [
                m.cpu().numpy() if hasattr(m, "cpu") else m
                for m in results.get("masks", [])
            ],
            "boxes": [
                b.cpu().numpy() if hasattr(b, "cpu") else b
                for b in results.get("boxes", [])
            ],
            "scores": [
                float(s.cpu()) if hasattr(s, "cpu") else float(s)
                for s in results.get("scores", [])
            ],
            "image_size": pil_image.size,
        }

    @torch.no_grad()
    def segment_with_boxes(
        self,
        image: Union[str, Path, Image.Image],
        input_boxes: list[list[int]],
        box_labels: Optional[list[int]] = None,
    ) -> dict:
        """
        Segment using bounding box prompts.

        Args:
            image: Image path or PIL Image
            input_boxes: List of [x1, y1, x2, y2] boxes
            box_labels: 1 for positive, 0 for negative (default: all positive)

        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        if not self.model_loaded:
            self.load_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        if box_labels is None:
            box_labels = [1] * len(input_boxes)

        inputs = self.processor(
            images=pil_image,
            input_boxes=[input_boxes],
            input_boxes_labels=[box_labels],
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        return {
            "masks": [
                m.cpu().numpy() if hasattr(m, "cpu") else m
                for m in results.get("masks", [])
            ],
            "boxes": [
                b.cpu().numpy() if hasattr(b, "cpu") else b
                for b in results.get("boxes", [])
            ],
            "scores": [
                float(s.cpu()) if hasattr(s, "cpu") else float(s)
                for s in results.get("scores", [])
            ],
        }

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.5,
        save_masks: bool = False,
    ):
        """Process all images in a folder."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        predictions_dir = output_dir / "pred_txt"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        if save_masks:
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

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

                # Save YOLO format
                txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, w, h = det[:5]
                        x_center = x_min + w / 2
                        y_center = y_min + h / 2
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                        )

                if save_masks:
                    result = self.segment_with_text(
                        image_file, self.detection_class[0], threshold
                    )
                    if result["masks"]:
                        np.savez_compressed(
                            masks_dir / f"{image_file.stem}.npz",
                            masks=np.array(result["masks"]),
                        )

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        print(
            f"\nDone! {total_detections} detections in {images_with_detections}/{len(image_files)} images"
        )

        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "model": self.model_id,
                    "classes": self.detection_class,
                    "total_images": len(image_files),
                    "images_with_detections": images_with_detections,
                    "total_detections": total_detections,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    # from .detection_utils import visualize_detections, visualize_masks

    input_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/sam3/frames_filtered_v2_dedup"
    )

    sam = SAMHelper(detection_class=["door", "glass door", "entrance door"])
    sam.process_folder(input_folder, output_folder, threshold=0.5, save_masks=False)

    # # Single image with text prompt
    # result = sam.segment_with_text("/path/to/image.jpg", text="yellow school bus")
    # print(f"Found {len(result['masks'])} objects")

    # # Visualize
    # visualize_detections(input_folder, output_folder / "pred_txt", output_folder / "vis")
