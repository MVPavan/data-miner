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


class GroundingDINOHelper:
    """
    Grounding DINO Helper for zero-shot object detection.
    
    Uses the free, open-source GroundingDINO models via HuggingFace Transformers.
    
    Available models:
    - IDEA-Research/grounding-dino-base (default, best accuracy ~50AP COCO)
    - IDEA-Research/grounding-dino-tiny (faster, smaller, ~48AP COCO)
    
    Features:
    - Zero-shot detection from text prompts
    - Open-vocabulary detection without retraining
    - High accuracy on COCO, LVIS, and custom objects
    """
    
    DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["object"])
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]
    
    def load_model(self):
        """Load Grounding DINO model and processor."""
        if self.model_loaded:
            return
        
        print(f"Loading Grounding DINO: {self.model_id}")
        
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        
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
    
    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.3,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
        text_threshold: float = 0.25,
    ):
        """
        Detect objects in an image using text prompts.
        
        Args:
            image: Path to image or PIL Image
            threshold: Box confidence threshold (default: 0.3)
            detection_classes: Text labels to detect (uses self.detection_class if None)
            output_format: "normalized" (default) or "pixel"
            text_threshold: Text matching threshold (default: 0.25)
            
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
        
        # Build text prompt: classes as list [["a cat", "a remote control"]]
        text_labels = [classes]
        
        inputs = self.processor(
            images=pil_image,
            text=text_labels,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image_height, image_width)],
        )[0]
        
        self.latencies.append(time.perf_counter() - t0)
        
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", [])
        
        if len(boxes) == 0:
            if img_path:
                print(f"No objects detected in {img_path.name}")
            if output_format == "pixel":
                return 0
            return []
        
        # Map labels to class indices
        class_to_idx = {c.lower(): i for i, c in enumerate(classes)}
        
        if output_format == "pixel":
            bboxes = []
            class_ids = []
            confidences = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(score))
                cls_idx = class_to_idx.get(label.lower(), 0) + 1  # 1-indexed
                class_ids.append(cls_idx)
            return bboxes, confidences, class_ids
        else:
            detection_results = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                cls_idx = class_to_idx.get(label.lower(), 0)  # 0-indexed
                detection_results.append([
                    cls_idx,
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
    
    @torch.no_grad()
    def query_image(
        self,
        image: Union[str, Path, Image.Image],
        query: str,
        threshold: float = 0.3,
    ) -> list[dict]:
        """
        Query image with natural language for referring expression comprehension.
        
        Args:
            image: Path to image or PIL Image
            query: Natural language query (e.g., "the red car on the left")
            threshold: Confidence threshold
            
        Returns:
            List of dicts with 'label', 'confidence', 'bbox' keys
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
            text=[[query]],
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=0.25,
            target_sizes=[(image_height, image_width)],
        )[0]
        
        detections = []
        for box, score, label in zip(
            results.get("boxes", []),
            results.get("scores", []),
            results.get("labels", []),
        ):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "label": label,
                "confidence": float(score),
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            })
        
        return detections
    
    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.3,
    ):
        """Process all images in a folder."""
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
                
                # Save YOLO format
                txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, w, h = det[:5]
                        x_center = x_min + w / 2
                        y_center = y_min + h / 2
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"\nDone! {total_detections} detections in {images_with_detections}/{len(image_files)} images")
        
        if len(self.latencies) > 0:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"Average inference: {avg:.3f}s ({1/avg:.1f} FPS)")
        
        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            json.dump({
                "model": self.model_id,
                "classes": self.detection_class,
                "total_images": len(image_files),
                "images_with_detections": images_with_detections,
                "total_detections": total_detections,
            }, f, indent=2)


if __name__ == "__main__":
    from .detection_utils import visualize_detections
    
    input_folder = Path("/path/to/images")
    output_folder = Path("/path/to/output")
    
    gdino = GroundingDINOHelper(
        detection_class=["door", "glass door", "entrance door"],
        model_id="IDEA-Research/grounding-dino-base",  # or grounding-dino-tiny
    )
    
    # Process folder
    gdino.process_folder(input_folder, output_folder, threshold=0.3)
    
    # # Natural language query
    # results = gdino.query_image("image.jpg", "the main entrance door")
    # for det in results:
    #     print(f"Found: {det['label']} at {det['bbox']}")
    
    # # Visualize
    # visualize_detections(input_folder, output_folder / "pred_txt", output_folder / "vis")
