# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import os
import time
from pathlib import Path
from typing import Optional, Union

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

load_dotenv(".env")
if os.getenv("HF_TOKEN"):
    login(os.getenv("HF_TOKEN"))

from data_miner.models.detection_utils import (
    apply_nms,
    create_class_ids,
    format_detections,
    nms,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class OWLViTHelper:
    """
    OWL-ViT v2 (OWLv2) Helper for zero-shot and few-shot object detection.

    Uses Google's OWL-ViT v2 model for open-vocabulary object detection
    based on text prompts or reference images, without requiring task-specific training.

    Features:
    - Zero-shot detection from text descriptions
    - Image-guided (few-shot) detection using reference images
    - Strong performance on rare objects (44.6% mAPrare on LVIS)
    - Self-training for improved rare class detection

    Model variants:
    - google/owlv2-base-patch16-ensemble (default, good balance)
    - google/owlv2-large-patch14-ensemble (higher accuracy)
    """

    DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["door"])
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL_ID)
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: None | Owlv2ForObjectDetection = None
        self.processor: None | Owlv2Processor = None

        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]

    def load_model(self):
        """Load the OWL-ViT v2 model and processor."""
        if self.model_loaded:
            return

        print(f"Loading OWL-ViT v2 model: {self.model_id}")

        self.processor = Owlv2Processor.from_pretrained(self.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        print(f"OWL-ViT v2 loaded on {self.device}")

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
        print("OWL-ViT v2 model unloaded")

    def get_model(self):
        return self.model

    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.1,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
        nms_threshold: Optional[float] = 0.5,
    ):
        """
        Run zero-shot object detection on an image.

        Args:
            image: Path to image or PIL Image
            threshold: Confidence threshold for detections (default: 0.1)
            detection_classes: List of class names to detect (uses self.detection_class if None)
            output_format: Output format - "normalized" (default) or "pixel"
                - "normalized": Returns list of [class_id, x, y, w, h, confidence] with 0-1 coords
                - "pixel": Returns (bboxes, confidences, class_ids) tuple with pixel coords

        Returns:
            If output_format="normalized":
                List of [class_id, x_min, y_min, width, height, confidence] (0-indexed, 0-1 coords)
                Empty list if no detections.
            If output_format="pixel":
                Tuple of (bboxes, confidences, class_ids) where:
                - bboxes: List of [x, y, width, height] in pixel coordinates
                - confidences: List of confidence scores
                - class_ids: List of class indices (1-indexed for compatibility)
                Returns 0 if no objects detected.
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

        # Format text queries for OWLv2
        text_queries = [[f"a photo of a {cls}" for cls in classes]]

        # Process inputs
        inputs = self.processor(
            text=text_queries,
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([[image_height, image_width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]

        self.latencies.append(time.perf_counter() - t0)

        # Extract results
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        if len(boxes) == 0:
            if img_path:
                print(f"No objects detected in {img_path.name}")
            return [] if output_format == "normalized" else 0

        # Create class IDs (all zeros for text-based detection)
        class_ids = create_class_ids(len(boxes))

        # Apply NMS
        boxes, scores, class_ids = apply_nms(boxes, scores, class_ids, nms_threshold)

        return format_detections(
            boxes, scores, class_ids, (image_width, image_height), output_format
        )

    # Aliases for backwards compatibility
    def infer_image(self, img_path, threshold=0.1, detection_classes=None):
        """Alias for detect() with pixel output format."""
        return self.detect(
            img_path, threshold, detection_classes, output_format="pixel"
        )

    def detect_objects(self, image, detection_classes, threshold=0.1):
        """Alias for detect() with normalized output format."""
        return self.detect(
            image, threshold, detection_classes, output_format="normalized"
        )

    def detect_objects_guided(
        self,
        image: Union[str, Path, Image.Image],
        query_images: Union[str, Path, Image.Image, list],
        threshold: float = 0.9,
        nms_threshold: float = 0.3,
        output_format: str = "normalized",
    ):
        """Alias for detect_with_query_images()"""
        return self.detect_with_query_images(
            image, query_images, threshold, nms_threshold, output_format
        )

    @torch.no_grad()
    def detect_with_query_images(
        self,
        image: Union[str, Path, Image.Image],
        query_images: Union[str, Path, Image.Image, list],
        threshold: float = 0.9,
        nms_threshold: float = 0.3,
        output_format: str = "normalized",
    ):
        """
        Run image-guided (few-shot) object detection with multiple query images.

        Detects objects in the target image that are similar to objects shown
        in the query images. Since OWLv2 only supports one query image per
        inference, this method iterates over all query images and aggregates
        the results with NMS deduplication.

        Args:
            image: Target image to search in (path or PIL Image)
            query_images: Reference image(s) showing the object to find.
                Can be a single image or a list of images. Multiple images
                will be processed sequentially and results aggregated.
            threshold: Confidence threshold for detections (default: 0.9)
                Higher than text-based detection because of visual similarity matching.
            nms_threshold: Non-maximum suppression IoU threshold (default: 0.3)
            output_format: Output format - "normalized" (default) or "pixel"
                - "normalized": Returns list of [0, x, y, w, h, confidence] with 0-1 coords
                - "pixel": Returns (bboxes, confidences, class_ids) tuple with pixel coords

        Returns:
            If output_format="normalized":
                List of [class_id, x_min, y_min, width, height, confidence] (0-1 coords)
                Note: class_id is always 0 for image-guided detection.
                Empty list if no detections.
            If output_format="pixel":
                Tuple of (bboxes, confidences, class_ids) where:
                - bboxes: List of [x, y, width, height] in pixel coordinates
                - confidences: List of confidence scores
                - class_ids: List of class indices (always 1 for image-guided)
                Returns 0 if no objects detected.
        """
        if not self.model_loaded:
            self.load_model()

        t0 = time.perf_counter()

        # Load target image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        image_width, image_height = pil_image.size

        # Load query image(s)
        if not isinstance(query_images, list):
            query_images = [query_images]

        pil_query_images = []
        for q_img in query_images:
            if isinstance(q_img, (str, Path)):
                pil_query_images.append(Image.open(q_img).convert("RGB"))
            else:
                pil_query_images.append(q_img.convert("RGB"))

        # Accumulate raw model outputs from all query images
        # OWLv2 only supports one query image per inference call
        all_logits = []
        all_boxes = []

        for query_img in pil_query_images:
            # Process inputs for image-guided detection
            inputs = self.processor(
                images=pil_image,
                query_images=query_img,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run image-guided detection
            outputs = self.model.image_guided_detection(**inputs)

            # Collect raw outputs (shape: [1, num_patches, ...])
            all_logits.append(outputs.logits)
            all_boxes.append(outputs.target_pred_boxes)

        # Concatenate outputs along batch dimension
        combined_logits = torch.cat(all_logits, dim=0)  # [N, num_patches, 1]
        combined_boxes = torch.cat(all_boxes, dim=0)  # [N, num_patches, 4]

        # Create a combined output object for post-processing
        from transformers.models.owlv2.modeling_owlv2 import (
            Owlv2ImageGuidedObjectDetectionOutput,
        )

        combined_outputs = Owlv2ImageGuidedObjectDetectionOutput(
            logits=combined_logits,
            target_pred_boxes=combined_boxes,
            # Other fields not needed for post_process_image_guided_detection
            image_embeds=None,
            query_image_embeds=None,
            query_pred_boxes=None,
            class_embeds=None,
            text_model_output=None,
            vision_model_output=None,
        )

        # Target sizes repeated for each query image (all same target image)
        target_sizes = torch.tensor(
            [[image_height, image_width]] * len(pil_query_images), device=self.device
        )

        # Apply post-processing with built-in NMS (processes each batch item)
        results_list = self.processor.post_process_image_guided_detection(
            outputs=combined_outputs,
            threshold=threshold,
            nms_threshold=nms_threshold,
            target_sizes=target_sizes,
        )

        # Aggregate results from all query images
        import numpy as np

        all_result_boxes = []
        all_result_scores = []
        for result in results_list:
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            if len(boxes) > 0:
                all_result_boxes.extend(boxes.tolist())
                all_result_scores.extend(scores.tolist())

        self.latencies.append(time.perf_counter() - t0)

        if len(all_result_boxes) == 0:
            if output_format == "pixel":
                return 0
            return []

        # Final NMS across all query results to remove cross-query duplicates
        boxes_array = np.array(all_result_boxes)
        scores_array = np.array(all_result_scores)

        keep_indices = nms(boxes_array, scores_array, nms_threshold)
        boxes_array = boxes_array[keep_indices]
        scores_array = scores_array[keep_indices]

        if output_format == "pixel":
            # Return pixel coordinates (moondream.py compatible)
            bboxes = []
            for box in boxes_array:
                x1, y1, x2, y2 = box
                bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            class_ids = [1] * len(boxes_array)
            confidences = [float(score) for score in scores_array]
            return bboxes, confidences, class_ids
        else:
            # Return normalized coordinates (default)
            detection_results = []
            for box, score in zip(boxes_array, scores_array):
                x1, y1, x2, y2 = box
                detection_results.append(
                    [
                        0,  # class_id always 0 for image-guided detection
                        x1 / image_width,
                        y1 / image_height,
                        (x2 - x1) / image_width,
                        (y2 - y1) / image_height,
                        float(score),
                    ]
                )
            return detection_results

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.1,
    ):
        """
        Process all images in a folder and save detection results.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save detection results
            threshold: Confidence threshold for detections
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        predictions_dir = output_dir / "pred_txt"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")

        if not self.model_loaded:
            self.load_model()

        # Get all image files
        image_files = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        total_detections = 0
        images_with_detections = 0
        detection_counts = {}

        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                detections = self.detect_objects(
                    image_file,
                    detection_classes=self.detection_class,
                    threshold=threshold,
                )

                # reference_images = list(Path("output/samples").glob("*.png"))
                # detections = self.detect_objects_guided(
                #     image_file,
                #     query_images=reference_images,
                #     threshold=threshold,
                # )

                detection_counts[image_file.stem] = len(detections)

                if len(detections) == 0:
                    continue

                images_with_detections += 1
                total_detections += len(detections)

                # Save detections as txt in YOLO format
                detection_txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(detection_txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, width, height, score = det
                        # Convert to YOLO format: class x_center y_center width height
                        x_center = x_min + width / 2
                        y_center = y_min + height / 2
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                        )

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        print("\nProcessing complete!")
        print(f"Total detections: {total_detections}")
        print(f"Images with detections: {images_with_detections}/{len(image_files)}")

        # Save summary
        summary_path = output_dir / "detection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "model": self.model_id,
                    "detection_classes": self.detection_class,
                    "threshold": threshold,
                    "total_images": len(image_files),
                    "images_with_detections": images_with_detections,
                    "total_detections": total_detections,
                    "detection_counts": detection_counts,
                },
                f,
                indent=4,
            )
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    # Example usage
    input_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/owlvit/frames_filtered_v2_dedup"
    )

    ##################################################################################################
    ####################### Inference ################################################################

    owlvit = OWLViTHelper(
        detection_class=["door", "glass door", "entrance door"],
        model_id="google/owlv2-large-patch14-ensemble",
    )

    # Process entire folder
    owlvit.process_folder(input_folder, output_folder, threshold=0.1)

    ##################################################################################################
    ####################### Visualization ############################################################

    # visualize_detections(
    #     images_folder=input_folder,
    #     annotations_folder=output_folder / "pred_txt",
    #     output_folder=output_folder / "visualizations",
    #     class_names=["door", "glass door", "entrance door"],
    # )

    ##################################################################################################
    ####################### Image-Guided (Few-Shot) Detection ########################################

    # Example: Use a reference image to find similar objects
    # import requests
    # target_image = Image.open(requests.get(
    #     "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    # ).raw)
    # query_image = Image.open(requests.get(
    #     "http://images.cocodataset.org/val2017/000000001675.jpg", stream=True
    # ).raw)
    #
    # owlvit = OWLViTHelper()
    # detections = owlvit.detect_with_query_images(
    #     target_image, query_image, threshold=0.6
    # )
    # print(f"Found {len(detections)} similar objects")
