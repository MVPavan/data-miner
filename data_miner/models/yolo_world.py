# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv(".env")

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class YOLOWorldHelper:
    """
    YOLO-World Helper for real-time open-vocabulary object detection.

    Uses Ultralytics YOLO-World model for fast open-vocabulary object detection.
    Approximately 20x faster than GroundingDINO with competitive accuracy.

    Features:
    - Real-time open-vocabulary detection
    - ~20x faster than GroundingDINO
    - Supports custom class vocabularies at runtime
    - Efficient for video processing

    Model variants:
    - yolov8s-worldv2 (small, fastest)
    - yolov8m-worldv2 (medium, balanced)
    - yolov8l-worldv2 (large)
    - yolov8x-worldv2 (xlarge, highest accuracy, default)
    """

    DEFAULT_MODEL = "yolov8x-worldv2"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["door"])
        self.model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]

    def load_model(self):
        """Load the YOLO-World model."""
        if self.model_loaded:
            return

        print(f"Loading YOLO-World model: {self.model_name}")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )

        # Load the model - Ultralytics will download if needed
        self.model = YOLO(f"{self.model_name}.pt")

        # Set custom classes for open-vocabulary detection
        self.model.set_classes(self.detection_class)

        self.model_loaded = True
        print(f"YOLO-World loaded with classes: {self.detection_class}")

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("YOLO-World model unloaded")

    def set_classes(self, classes: list[str]):
        """
        Update detection classes at runtime.

        Args:
            classes: List of class names to detect
        """
        self.detection_class = classes
        if self.model_loaded and self.model is not None:
            self.model.set_classes(classes)
            print(f"YOLO-World classes updated to: {classes}")

    def get_model(self):
        return self.model

    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        threshold: float = 0.25,
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
    ):
        """
        Run open-vocabulary object detection on an image.

        Args:
            image: Path to image, PIL Image, or numpy array
            threshold: Confidence threshold for detections (default: 0.25)
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

        # Update classes if specified
        if detection_classes and detection_classes != self.detection_class:
            self.set_classes(detection_classes)

        t0 = time.perf_counter()

        # Handle different input types
        if isinstance(image, Image.Image):
            image_input = np.array(image)
            image_height, image_width = image_input.shape[:2]
            img_path = None
        elif isinstance(image, np.ndarray):
            image_input = image
            image_height, image_width = image.shape[:2]
            img_path = None
        else:
            image_input = str(image)
            img_path = Path(image)
            with Image.open(image) as img:
                image_width, image_height = img.size

        # Run inference
        results = self.model.predict(
            source=image_input,
            conf=threshold,
            device=self.device,
            verbose=False,
        )

        self.latencies.append(time.perf_counter() - t0)

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            if img_path:
                # print(f"No objects detected in {img_path.name}")
                pass
            if output_format == "pixel":
                return 0
            return []

        result = results[0]
        boxes = result.boxes

        if output_format == "pixel":
            # Return pixel coordinates (moondream.py compatible)
            bboxes = []
            confidences = []
            class_ids = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(box.conf[0].cpu()))
                class_ids.append(int(box.cls[0].cpu()) + 1)  # 1-indexed
            return bboxes, confidences, class_ids
        else:
            # Return normalized coordinates (default)
            detection_results = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                detection_results.append(
                    [
                        cls_id,  # 0-indexed
                        x1 / image_width,
                        y1 / image_height,
                        (x2 - x1) / image_width,
                        (y2 - y1) / image_height,
                        conf,
                    ]
                )
            return detection_results

    # Aliases for backwards compatibility
    def infer_image(self, img_path, threshold=0.25, detection_classes=None):
        """Alias for detect() with pixel output format."""
        return self.detect(
            img_path, threshold, detection_classes, output_format="pixel"
        )

    def detect_objects(self, image, detection_classes, threshold=0.25):
        """Alias for detect() with normalized output format."""
        return self.detect(
            image, threshold, detection_classes, output_format="normalized"
        )

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.25,
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
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        print("\nProcessing complete!")
        print(f"Total detections: {total_detections}")
        print(f"Images with detections: {images_with_detections}/{len(image_files)}")

        if len(self.latencies) > 0:
            avg_latency = sum(self.latencies) / len(self.latencies)
            print(
                f"Average inference time: {avg_latency:.3f}s ({1 / avg_latency:.1f} FPS)"
            )

        # Save summary
        summary_path = output_dir / "detection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "detection_classes": self.detection_class,
                    "threshold": threshold,
                    "total_images": len(image_files),
                    "images_with_detections": images_with_detections,
                    "total_detections": total_detections,
                    "avg_latency_s": sum(self.latencies) / len(self.latencies)
                    if self.latencies
                    else 0,
                    "detection_counts": detection_counts,
                },
                f,
                indent=4,
            )
        print(f"Summary saved to {summary_path}")

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.25,
        save_video: bool = True,
        save_frames: bool = False,
    ):
        """
        Process a video and optionally save annotated output.

        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            threshold: Confidence threshold
            save_video: Whether to save annotated video
            save_frames: Whether to save individual frames
        """
        if not self.model_loaded:
            self.load_model()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run tracking/detection on video
        results = self.model.track(
            source=str(video_path),
            conf=threshold,
            device=self.device,
            stream=True,
            persist=True,
            verbose=False,
        )

        frame_detections = []
        for frame_idx, result in enumerate(tqdm(results, desc="Processing video")):
            frame_data = {
                "frame": frame_idx,
                "detections": [],
            }

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    frame_data["detections"].append(
                        {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(box.conf[0].cpu()),
                            "class_id": int(box.cls[0].cpu()),
                            "class_name": self.detection_class[int(box.cls[0].cpu())],
                            "track_id": int(box.id[0].cpu())
                            if box.id is not None
                            else None,
                        }
                    )

            frame_detections.append(frame_data)

            if save_frames and result.plot() is not None:
                frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                Image.fromarray(result.plot()[:, :, ::-1]).save(frame_path)

        # Save detection data
        json_path = output_dir / "video_detections.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "video": str(video_path),
                    "model": self.model_name,
                    "detection_classes": self.detection_class,
                    "frames": frame_detections,
                },
                f,
                indent=2,
            )
        print(f"Video detections saved to {json_path}")


if __name__ == "__main__":
    # Example usage
    input_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/yolo_world/frames_filtered_v2_dedup"
    )

    ##################################################################################################
    ####################### Inference ################################################################

    yolo = YOLOWorldHelper(
        detection_class=["door", "glass door", "entrance door"],
        model_name="yolov8x-worldv2",  # Options: yolov8s/m/l/x-worldv2
    )

    # Process entire folder
    yolo.process_folder(input_folder, output_folder, threshold=0.25)

    ##################################################################################################
    ####################### Video Processing (optional) ##############################################

    # yolo.process_video(
    #     video_path="/path/to/video.mp4",
    #     output_dir=output_folder / "video_output",
    #     threshold=0.25,
    #     save_video=True,
    # )

    ##################################################################################################
    ####################### Visualization ############################################################

    # visualize_detections(
    #     images_folder=input_folder,
    #     annotations_folder=output_folder / "pred_txt",
    #     output_folder=output_folder / "visualizations",
    #     class_names=["door", "glass door", "entrance door"],
    # )
