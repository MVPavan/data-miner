"""
Qwen3-VL Detector for open-vocabulary 2D grounding.

Uses Qwen3-VL-32B-Instruct model for high-precision object detection.
Requires 2 GPUs (~80GB VRAM total) for deployment.

Usage:
    python -m data_miner.models.qwen3_vl

Example:
    CUDA_VISIBLE_DEVICES=0,1 python -m data_miner.models.qwen3_vl

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import re
import time
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class Qwen3VLHelper:
    """
    Qwen3-VL detector for open-vocabulary 2D grounding.

    Uses prompt-based detection with JSON output parsing.
    Model is sharded across 2 GPUs using device_map="auto".
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-32B-Instruct",
        detection_class: Optional[Union[str, list]] = None,
        use_flash_attn: bool = True,
    ):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.latencies = []
        self.use_flash_attn = use_flash_attn

        # Handle detection classes
        if detection_class is None:
            detection_class = ["door"]
        if isinstance(detection_class, str):
            detection_class = [detection_class]
        self.detection_class = detection_class

    def load_model(self):
        """Load Qwen3-VL model across 2 GPUs."""
        if self.model_loaded:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading Qwen3-VL: {self.model_id}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load model with 2-GPU sharding
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",  # Automatically shard across available GPUs
            "trust_remote_code": True,
        }

        if self.use_flash_attn:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            except Exception as e:
                print(f"Flash Attention not available: {e}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **model_kwargs
        )
        self.model.eval()
        self.model_loaded = True

        # Print device mapping
        if hasattr(self.model, "hf_device_map"):
            devices = set(self.model.hf_device_map.values())
            print(f"Model loaded on devices: {devices}")
        print("Qwen3-VL loaded successfully")

    def _parse_json_output(self, text: str) -> list[dict]:
        """Parse JSON bounding box output from model response."""
        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            if not isinstance(data, list):
                data = [data]
            return data
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            # Try to find single JSON object
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return [result] if isinstance(result, dict) else result
                except json.JSONDecodeError:
                    pass
            print(f"Failed to parse JSON from response: {text[:200]}")
            return []

    @torch.no_grad()
    def detect(
        self,
        image_path: Union[str, Path],
        detection_classes: Optional[list[str]] = None,
    ) -> list[tuple]:
        """
        Detect objects in an image.

        Args:
            image_path: Path to the image file
            detection_classes: List of class names to detect (uses self.detection_class if None)

        Returns:
            List of detections as tuples: (class_id, x_min, y_min, width, height)
            Coordinates are normalized (0-1)
        """
        if not self.model_loaded:
            self.load_model()

        if detection_classes is None:
            detection_classes = self.detection_class

        t0 = time.perf_counter()

        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Build grounding prompt
        classes_str = ", ".join(detection_classes)
        grounding_prompt = f'Locate every instance that belongs to: "{classes_str}". Report bbox coordinates in JSON format as a list of objects with "bbox_2d" (as [x1, y1, x2, y2] in pixels) and "label" fields.'

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": grounding_prompt},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

        # Trim input from output
        generated_ids_trimmed = generated_ids[0][inputs["input_ids"].shape[1] :]
        output_text = self.processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.latencies.append(time.perf_counter() - t0)

        # Parse JSON output
        bbox_data = self._parse_json_output(output_text)

        # Convert to detection format
        detections = []
        for item in bbox_data:
            bbox = item.get("bbox_2d", item.get("bbox", []))
            label = item.get(
                "label", detection_classes[0] if detection_classes else "object"
            )

            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Normalize to 0-1
                x_min = x1 / width
                y_min = y1 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                # Find class ID
                label_lower = label.lower()
                class_id = 0
                for i, cls in enumerate(detection_classes):
                    if cls.lower() in label_lower or label_lower in cls.lower():
                        class_id = i
                        break

                detections.append((class_id, x_min, y_min, w, h))

        return detections

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        """
        Process all images in a folder and save detections in YOLO format.

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for predictions
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        predictions_dir = output_dir / "pred_txt"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_loaded:
            self.load_model()

        # Collect image files
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

        for image_file in tqdm(image_files, desc="Processing"):
            try:
                detections = self.detect(image_file)

                if len(detections) == 0:
                    continue

                images_with_detections += 1
                total_detections += len(detections)

                # Save YOLO format: class x_center y_center width height
                txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, w, h = det
                        # Convert from x_min, y_min to x_center, y_center
                        x_center = x_min + w / 2
                        y_center = y_min + h / 2
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                        )

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                import traceback

                traceback.print_exc()

        print(
            f"\nDone! {total_detections} detections in {images_with_detections}/{len(image_files)} images"
        )

        if len(self.latencies) > 0:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"Average inference: {avg:.3f}s ({1 / avg:.1f} FPS)")

        # Save summary
        summary = {
            "model": self.model_id,
            "classes": self.detection_class,
            "total_images": len(image_files),
            "images_with_detections": images_with_detections,
            "total_detections": total_detections,
            "avg_latency_sec": sum(self.latencies) / len(self.latencies)
            if self.latencies
            else 0,
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    # Example usage - modify paths as needed
    input_folder = Path(
        "/data/pavan/codes/tycoai/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/codes/tycoai/data_miner/output/projects/delivery_pov_v1/qwen3vl/frames_filtered_v2_dedup"
    )

    # Initialize detector
    qwen = Qwen3VLHelper(
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        detection_class=["door", "glass door", "entrance door"],
        use_flash_attn=False,
    )

    # Process folder
    qwen.process_folder(input_folder, output_folder)
