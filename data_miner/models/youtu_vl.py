"""
Youtu-VL-4B detector for open-vocabulary 2D grounding.

Uses Tencent's Youtu-VL-4B-Instruct model for prompt-based object detection.
Outputs bounding boxes in YOLO format via JSON response parsing.

Requires: transformers>=4.56.0,<=4.57.1
    https://github.com/TencentCloudADP/youtu-vl

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m data_miner.models.youtu_vl
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class YoutuVLHelper:
    """
    Youtu-VL detector for open-vocabulary 2D grounding.

    Uses prompt-based detection with JSON output parsing.
    Model: tencent/Youtu-VL-4B-Instruct (~4B params, single GPU).
    """

    DEFAULT_MODEL = "tencent/Youtu-VL-4B-Instruct"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", ["object"])
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL)
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_flash_attn = kwargs.get("use_flash_attn", True)
        self.model = None
        self.processor = None

        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]

    def load_model(self):
        """Load Youtu-VL model and processor."""
        if self.model_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading Youtu-VL: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=True
        )

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device,
            "trust_remote_code": True,
        }
        if self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        ).eval()

        self.model_loaded = True
        print(f"Youtu-VL loaded on {self.device}")

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
        print("Youtu-VL model unloaded")

    def get_model(self):
        return self.model

    def _parse_json_output(self, text: str) -> list[dict]:
        """Parse JSON bounding box output from model response."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            if not isinstance(data, list):
                data = [data]
            return data
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return [result] if isinstance(result, dict) else result
                except json.JSONDecodeError:
                    pass
            return []

    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        detection_classes: Optional[list[str]] = None,
        output_format: str = "normalized",
    ) -> list:
        """
        Detect objects in an image via grounding prompt.

        Args:
            image: Path to image or PIL Image
            detection_classes: List of class names to detect (uses self.detection_class if None)
            output_format: "normalized" (default) returns list of [class_id, x, y, w, h]

        Returns:
            List of [class_id, x_min, y_min, width, height] with normalized 0-1 coords.
            Empty list if no detections.
        """
        if not self.model_loaded:
            self.load_model()

        classes = detection_classes or self.detection_class

        if isinstance(image, (str, Path)):
            img_path_str = str(image)
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
            img_path_str = None

        width, height = pil_image.size

        t0 = time.perf_counter()

        # Build grounding prompt
        classes_str = ", ".join(classes)
        grounding_prompt = (
            f'Locate every instance that belongs to: "{classes_str}". '
            f'Report bbox coordinates in JSON format as a list of objects '
            f'with "bbox_2d" (as [x1, y1, x2, y2] in pixels) and "label" fields.'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path_str or pil_image},
                    {"type": "text", "text": grounding_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            img_input=img_path_str,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        self.latencies.append(time.perf_counter() - t0)

        # Parse JSON output
        bbox_data = self._parse_json_output(output_text)

        # Convert to normalized detection format
        detections = []
        for item in bbox_data:
            bbox = item.get("bbox_2d", item.get("bbox", []))
            label = item.get("label", classes[0] if classes else "object")

            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Clamp to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Normalize to 0-1
                x_min = x1 / width
                y_min = y1 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                # Match label to class ID
                label_lower = label.lower()
                class_id = 0
                for i, cls in enumerate(classes):
                    if cls.lower() in label_lower or label_lower in cls.lower():
                        class_id = i
                        break

                detections.append([class_id, x_min, y_min, w, h])

        return detections

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        """Process all images in a folder and save detections in YOLO format."""
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
        print(f"Detection classes: {self.detection_class}")

        total_detections = 0
        images_with_detections = 0

        skipped = 0
        for image_file in tqdm(image_files, desc="Processing"):
            txt_path = predictions_dir / f"{image_file.stem}.txt"
            if txt_path.exists():
                skipped += 1
                continue

            try:
                detections = self.detect(image_file)

                if len(detections) == 0:
                    continue

                images_with_detections += 1
                total_detections += len(detections)
                with open(txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, w, h = det
                        x_center = x_min + w / 2
                        y_center = y_min + h / 2
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                        )

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        if skipped:
            print(f"\nSkipped {skipped} images (already processed)")
        print(
            f"Done! {total_detections} new detections in "
            f"{images_with_detections}/{len(image_files) - skipped} images"
        )

        if len(self.latencies) > 0:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"Average inference: {avg:.3f}s ({1 / avg:.1f} FPS)")

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
        "/data/datasets/data_miner_datasets/forklift_palletjack_v1/frames_dedup_v1_cls_0.85"
    )
    output_folder = Path(
        "/data/datasets/data_miner_datasets/forklift_palletjack_v1/detections/youtu_vl"
    )

    yvl = YoutuVLHelper(
        detection_class=["forklift", "pallet jack"],
        use_flash_attn=True,
    )
    yvl.process_folder(input_folder, output_folder)
