# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from tqdm import tqdm

load_dotenv(".env")
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login
    login(os.getenv("HF_TOKEN"))

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class GroundingDINOHelper:
    """
    Grounding DINO 1.5 Helper for zero-shot object detection.
    
    Grounding DINO 1.5 (May 2024) is the state-of-the-art open-set object detector:
    - **Pro**: 54.3 AP on COCO, 55.7 AP on LVIS-minival (highest accuracy)
    - **Edge**: Optimized for edge devices (36.2 AP at 75.2 FPS with TensorRT)
    
    Features:
    - Zero-shot detection from natural language descriptions
    - Open-vocabulary object detection without retraining
    - Referring expression comprehension (locate specific objects)
    - Trained on 20M+ images
    
    Model variants:
    - IDEA-Research/grounding-dino-base (original, stable)
    - IDEA-Research/grounding-dino-tiny (smaller, faster)
    - IDEA-Research/grounding-dino-1.5-pro (best accuracy, May 2024)
    - IDEA-Research/grounding-dino-1.5-edge (optimized for speed)
    """
    
    DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"
    MODEL_1_5_PRO = "IDEA-Research/grounding-dino-1.5-pro"
    MODEL_1_5_EDGE = "IDEA-Research/grounding-dino-1.5-edge"
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["door"])
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL_ID)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]
    
    def load_model(self):
        """Load the Grounding DINO model and processor."""
        if self.model_loaded:
            return
        
        print(f"Loading Grounding DINO: {self.model_id}")
        
        # Try transformers pipeline first (works for most models)
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
            )
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Standard loading failed ({e}), trying alternative method...")
            # Alternative: Try loading via groundingdino package
            try:
                from groundingdino.util.inference import load_model as load_gdino_model
                from groundingdino.util.inference import predict
                
                # This requires the groundingdino package and manual weight download
                self.model = load_gdino_model(
                    "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                    "weights/groundingdino_swinb_cogcoor.pth",
                )
                self._use_native_gdino = True
            except ImportError:
                raise ImportError(
                    "Failed to load Grounding DINO. Install with:\n"
                    "pip install transformers>=4.35.0\n"
                    "or: pip install groundingdino-py"
                )
        
        self._use_native_gdino = False
        self.model_loaded = True
        print(f"Grounding DINO loaded on {self.device}")
    
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
        print("Grounding DINO model unloaded")
    
    def get_model(self):
        return self.model
    
    def _format_prompt(self, classes: list[str]) -> str:
        """
        Format detection classes as Grounding DINO text prompt.
        
        Grounding DINO expects classes separated by periods.
        """
        return ". ".join(classes) + "."
    
    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.3,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
    ):
        """
        Run zero-shot object detection on an image.
        
        Args:
            image: Path to image or PIL Image
            threshold: Confidence threshold (box_threshold and text_threshold)
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
        
        # Format text prompt
        text_prompt = self._format_prompt(classes)
        
        # Process inputs
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(image_height, image_width)],
        )[0]
        
        self.latencies.append(time.perf_counter() - t0)
        
        # Extract results
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]  # List of strings
        
        if len(boxes) == 0:
            if img_path:
                print(f"No objects detected in {img_path.name}")
            if output_format == "pixel":
                return 0
            return []
        
        # Map labels to class indices
        label_to_idx_1 = {cls.lower(): i + 1 for i, cls in enumerate(classes)}
        label_to_idx_0 = {cls.lower(): i for i, cls in enumerate(classes)}
        
        if output_format == "pixel":
            # Return pixel coordinates (moondream.py compatible)
            bboxes = []
            confidences = []
            class_ids = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(score))
                cls_id = label_to_idx_1.get(label.lower().strip(), 1)
                class_ids.append(cls_id)
            return bboxes, confidences, class_ids
        else:
            # Return normalized coordinates (default)
            detection_results = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                cls_id = label_to_idx_0.get(label.lower().strip(), 0)
                detection_results.append([
                    cls_id,  # 0-indexed
                    x1 / image_width,
                    y1 / image_height,
                    (x2 - x1) / image_width,
                    (y2 - y1) / image_height,
                    float(score),
                ])
            return detection_results
    
    # Aliases for backwards compatibility
    def infer_image(self, img_path, threshold=0.3, detection_classes=None):
        """Alias for detect() with pixel output format."""
        return self.detect(img_path, threshold, detection_classes, output_format="pixel")
    
    def detect_objects(self, image, detection_classes, threshold=0.3):
        """Alias for detect() with normalized output format."""
        return self.detect(image, threshold, detection_classes, output_format="normalized")
    
    def query_image(
        self,
        image: Union[str, Path, Image.Image],
        query: str,
        threshold: float = 0.3,
    ) -> list:
        """
        Detect objects using a natural language query.
        
        This enables referring expression comprehension, e.g.:
        - "the red car on the left"
        - "person wearing a hat"
        - "glass door used for entrance"
        
        Args:
            image: Image path or PIL Image
            query: Natural language description
            threshold: Confidence threshold
            
        Returns:
            List of detections matching the query
        """
        if not self.model_loaded:
            self.load_model()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        image_width, image_height = pil_image.size
        
        inputs = self.processor(
            images=pil_image,
            text=query,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(image_height, image_width)],
        )[0]
        
        detections = []
        for box, score, label in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"],
        ):
            x1, y1, x2, y2 = box
            detections.append({
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "bbox_normalized": [
                    x1 / image_width,
                    y1 / image_height,
                    (x2 - x1) / image_width,
                    (y2 - y1) / image_height,
                ],
                "confidence": float(score),
                "label": label,
            })
        
        return detections
    
    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.3,
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
                
                detection_counts[image_file.stem] = len(detections)
                
                if len(detections) == 0:
                    continue
                
                images_with_detections += 1
                total_detections += len(detections)
                
                # Save detections as txt in YOLO format
                detection_txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(detection_txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, width, height = det[:5]
                        # Convert to YOLO format: class x_center y_center width height
                        x_center = x_min + width / 2
                        y_center = y_min + height / 2
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"\nProcessing complete!")
        print(f"Total detections: {total_detections}")
        print(f"Images with detections: {images_with_detections}/{len(image_files)}")
        
        if len(self.latencies) > 0:
            avg_latency = sum(self.latencies) / len(self.latencies)
            print(f"Average inference time: {avg_latency:.3f}s ({1/avg_latency:.1f} FPS)")
        
        # Save summary
        summary_path = output_dir / "detection_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "model": self.model_id,
                "detection_classes": self.detection_class,
                "threshold": threshold,
                "total_images": len(image_files),
                "images_with_detections": images_with_detections,
                "total_detections": total_detections,
                "avg_latency_s": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
                "detection_counts": detection_counts,
            }, f, indent=4)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    from .detection_utils import visualize_detections
    
    # Example usage
    input_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/grounding_dino/frames_filtered_v2_dedup"
    )
    
    ##################################################################################################
    ####################### Inference ################################################################
    
    # Use GroundingDINO 1.5 Pro for best accuracy
    gdino = GroundingDINOHelper(
        detection_class=["door", "glass door", "entrance door"],
        model_id="IDEA-Research/grounding-dino-base",  # or "IDEA-Research/grounding-dino-1.5-pro"
    )
    
    # Process entire folder
    gdino.process_folder(input_folder, output_folder, threshold=0.3)
    
    ##################################################################################################
    ####################### Natural Language Query ###################################################
    
    # # Query with natural language
    # detections = gdino.query_image(
    #     "/path/to/image.jpg",
    #     query="glass door used for human entrance, not windows or vehicle doors",
    #     threshold=0.3,
    # )
    # for det in detections:
    #     print(f"Found: {det['label']} at {det['bbox']} with confidence {det['confidence']:.2f}")
    
    ##################################################################################################
    ####################### Visualization ############################################################
    
    # visualize_detections(
    #     images_folder=input_folder,
    #     annotations_folder=output_folder / "pred_txt",
    #     output_folder=output_folder / "visualizations",
    #     class_names=["door", "glass door", "entrance door"],
    # )
