# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Union

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv(".env")
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login

    login(os.getenv("HF_TOKEN"))

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class PaliGemmaHelper:
    """
    PaliGemma 2 Helper for vision-language object detection.

    Uses the "detect {object} ; {object}" prompt format to locate objects
    and returns bounding boxes from <locXXXX> tokens.

    Available models:
    - google/paligemma2-3b-mix-224
    - google/paligemma2-3b-mix-448
    - google/paligemma2-10b-mix-224
    - google/paligemma2-10b-mix-448
    - google/paligemma2-28b-mix-224
    - google/paligemma2-28b-mix-448 (default - best accuracy)
    """

    DEFAULT_MODEL = "google/paligemma2-28b-mix-448"

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
        """Load PaliGemma2 model and processor."""
        if self.model_loaded:
            return

        print(f"Loading PaliGemma2: {self.model_id}")

        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.model_loaded = True
        print("PaliGemma2 loaded")

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
        print("PaliGemma2 model unloaded")

    def get_model(self):
        return self.model

    def _parse_detections(self, output_text: str, image_size: tuple) -> list[dict]:
        """
        Parse detection output from PaliGemma2.

        Format: <loc0123><loc0456><loc0789><loc1000> object_name
        Coordinates are (y_min, x_min, y_max, x_max) normalized to 0-1024.
        """
        detections = []
        img_w, img_h = image_size

        # Pattern: 4 loc tokens followed by object name
        pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^<]+)"
        matches = re.findall(pattern, output_text)

        for match in matches:
            y_min_norm = int(match[0]) / 1024
            x_min_norm = int(match[1]) / 1024
            y_max_norm = int(match[2]) / 1024
            x_max_norm = int(match[3]) / 1024
            label = match[4].strip()

            detections.append(
                {
                    "label": label,
                    "bbox_norm": [x_min_norm, y_min_norm, x_max_norm, y_max_norm],
                    "bbox_pixel": [
                        int(x_min_norm * img_w),
                        int(y_min_norm * img_h),
                        int(x_max_norm * img_w),
                        int(y_max_norm * img_h),
                    ],
                }
            )

        return detections

    @torch.inference_mode()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.0,  # PaliGemma doesn't output scores
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
    ):
        """
        Detect objects in an image using text prompts.

        Args:
            image: Path to image or PIL Image
            threshold: Not used (PaliGemma doesn't output confidence scores)
            detection_classes: Objects to detect (uses self.detection_class if None)
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
        
        # Build detection prompt with image token prefix (required by PaliGemmaProcessor)
        prompt = "<image>detect " + " ; ".join(classes)
        
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(torch.bfloat16).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]

        generation = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        output_text = self.processor.decode(
            generation[0][input_len:], skip_special_tokens=False
        )

        self.latencies.append(time.perf_counter() - t0)

        # Parse detections from output
        raw_detections = self._parse_detections(
            output_text, (image_width, image_height)
        )

        if len(raw_detections) == 0:
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
            for det in raw_detections:
                x1, y1, x2, y2 = det["bbox_pixel"]
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
                cls_idx = class_to_idx.get(det["label"].lower(), 0) + 1  # 1-indexed
                class_ids.append(cls_idx)
            confidences = [1.0] * len(bboxes)  # PaliGemma doesn't output scores
            return bboxes, confidences, class_ids
        else:
            detection_results = []
            for det in raw_detections:
                x_min, y_min, x_max, y_max = det["bbox_norm"]
                cls_idx = class_to_idx.get(det["label"].lower(), 0)  # 0-indexed
                detection_results.append(
                    [
                        cls_idx,
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min,
                        1.0,  # confidence
                    ]
                )
            return detection_results

    # Aliases for backwards compatibility
    def infer_image(self, img_path, threshold=0.0, detection_classes=None):
        """Alias for detect() with pixel output format."""
        return self.detect(
            img_path, threshold, detection_classes, output_format="pixel"
        )

    def detect_objects(self, image, detection_classes, threshold=0.0):
        """Alias for detect() with normalized output format."""
        return self.detect(
            image, threshold, detection_classes, output_format="normalized"
        )

    @torch.inference_mode()
    def describe(
        self,
        image: Union[str, Path, Image.Image],
        mode: str = "caption",
        language: str = "en",
    ) -> str:
        """
        Generate a description of the image.

        Args:
            image: Path to image or PIL Image
            mode: "cap" (short), "caption" (COCO-like), or "describe" (detailed)
            language: Language code (e.g., "en")

        Returns:
            Generated description text
        """
        if not self.model_loaded:
            self.load_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        prompt = f"<image>{mode} {language}"
        
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(torch.bfloat16).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]

        generation = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        return self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )

    @torch.inference_mode()
    def answer(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        language: str = "en",
    ) -> str:
        """
        Answer a question about the image.

        Args:
            image: Path to image or PIL Image
            question: Question to answer
            language: Language code

        Returns:
            Answer text
        """
        if not self.model_loaded:
            self.load_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        prompt = f"<image>answer {language} {question}"
        
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(torch.bfloat16).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]

        generation = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        return self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.0,
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
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
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
    input_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/paligemma/frames_filtered_v2_dedup"
    )

    pali = PaliGemmaHelper(
        detection_class=["door", "glass door", "entrance door"],
        model_id="google/paligemma2-28b-mix-448",
    )

    # Process folder
    pali.process_folder(input_folder, output_folder)

    # # Single image detection
    # detections = pali.detect("image.jpg", detection_classes=["car", "person"])
    # print(detections)

    # # Visual QA
    # answer = pali.answer("image.jpg", "What color is the car?")
    # print(answer)

    # # Captioning
    # caption = pali.describe("image.jpg", mode="describe")
    # print(caption)
