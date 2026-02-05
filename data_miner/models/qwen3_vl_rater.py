"""
Qwen3-VL Annotation Quality Rater for bounding box validation.

Uses Qwen3-VL-32B-Instruct model to score existing bbox annotations.
Requires 2 GPUs (~80GB VRAM total) for deployment.

Scoring thresholds:
- < 0.3: Discard (wrong object, severely misaligned)
- 0.3-0.5: Partial door or excessive glass environment included
- 0.5-0.75: Bbox not tight enough
- > 0.75: Keep (good annotation)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m data_miner.models.qwen3_vl_rater
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import re
import time
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROMPT_VERSION = "v2"

# Scoring thresholds
THRESHOLD_DISCARD = 0.3
THRESHOLD_PARTIAL = 0.5
THRESHOLD_LOOSE = 0.75

# Reason codes → category mapping
REASON_TO_CATEGORY = {
    "not_a_door": "discard",
    "partial_door": "partial",
    "loose_bbox": "loose",
    "good": "keep",
}

# Rating prompt - emphasize entrance doors, distinguish from glass elements
RATING_PROMPT = """You are an expert annotation quality reviewer for object detection datasets.

The ONLY red box in this image is the annotation to evaluate. Do not imagine other boxes.

Task: Evaluate if the RED bounding box correctly annotates an ENTRANCE DOOR.

DEFINITIONS:
- ENTRANCE DOOR: A door people walk through to enter/exit. Can be solid, glass, or partially glass. Must have door cues: frame, handle, hinges, threshold, or opening seam.
- NOT ENTRANCE DOOR: cupboard doors, Windows, glass walls, glass partitions, glass facades, display windows, or glass surfaces without door cues.

RETURN exactly this JSON format, nothing else:
{"is_door": true/false, "score": 0.XX, "reason": "CODE"}

REASON CODES (pick exactly one):
- "not_a_door": bbox is on wrong object (window, glass wall, non-door)
- "partial_door": bbox only covers part of the door, or cuts off edges
- "loose_bbox": bbox covers door but has too much padding/background/adjacent glass
- "good": bbox tightly covers the complete entrance door

SCORING:
- 0.0-0.3: not_a_door
- 0.3-0.5: partial_door
- 0.5-0.75: loose_bbox
- 0.75-1.0: good"""


class Qwen3VLAnnotationRater:
    """
    Qwen3-VL annotation quality rater for bounding box validation.

    Uses prompt-based reasoning to score bbox annotations from 0-1.
    Model is sharded across 2 GPUs using device_map="auto".
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-32B-Instruct",
        use_flash_attn: bool = True,
        bbox_color: tuple = (255, 0, 0),  # Red
        bbox_width: int = 3,
    ):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.latencies = []
        self.use_flash_attn = use_flash_attn
        self.bbox_color = bbox_color
        self.bbox_width = bbox_width
        self.rating_prompt = RATING_PROMPT

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
            "device_map": "auto",
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

        if hasattr(self.model, "hf_device_map"):
            devices = set(self.model.hf_device_map.values())
            print(f"Model loaded on devices: {devices}")
        print("Qwen3-VL loaded successfully")

    def _draw_bbox_on_image(
        self,
        image: Image.Image,
        bbox: tuple,
    ) -> tuple[Image.Image, bool]:
        """
        Draw bounding box on image with clamping.

        Args:
            image: PIL Image
            bbox: (x_center, y_center, width, height) in normalized coordinates (0-1)

        Returns:
            (Image with bbox drawn, is_valid)
        """
        img_w, img_h = image.size
        x_center, y_center, w, h = bbox

        # Convert from YOLO format (center, normalized) to pixel coordinates
        x1 = int((x_center - w / 2) * img_w)
        y1 = int((y_center - h / 2) * img_h)
        x2 = int((x_center + w / 2) * img_w)
        y2 = int((y_center + h / 2) * img_h)

        # Swap if inverted
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Clamp to image bounds
        x1, x2 = max(0, x1), min(img_w - 1, x2)
        y1, y2 = max(0, y1), min(img_h - 1, y2)

        # Check for degenerate box
        if x2 <= x1 or y2 <= y1:
            return image, False

        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([x1, y1, x2, y2], outline=self.bbox_color, width=self.bbox_width)
        return img_copy, True

    def _parse_json_output(self, text: str) -> dict:
        """Parse JSON output with robust extraction."""
        # Strip common wrappers
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```json\s*|```\s*", "", text)
        for prefix in ["Here is the JSON:", "Here's the JSON:", "JSON:", "Output:"]:
            text = text.replace(prefix, "")
        text = text.strip()

        # Extract first balanced {...} containing "score"
        def extract_balanced_json(s: str) -> str | None:
            start = s.find("{")
            if start == -1:
                return None
            depth, end = 0, start
            for i, c in enumerate(s[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = s[start:end]
            return candidate if "score" in candidate else None

        json_str = extract_balanced_json(text)
        if json_str:
            try:
                data = json.loads(json_str)
                # Coerce score to float and clamp
                score = float(data.get("score", -1))
                data["score"] = max(0.0, min(1.0, score)) if score >= 0 else -1
                return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: regex extraction
        score_match = re.search(r"\"?score\"?\s*[:=]\s*[\"']?([\d.]+)", text)
        if score_match:
            score = max(0.0, min(1.0, float(score_match.group(1))))
            return {"is_door": None, "score": score, "reason": "fallback_parse"}

        print(f"Parse failed: {text[:200]}")
        return {"is_door": None, "score": -1, "reason": "parse_error"}

    def _load_annotations(self, txt_path: Path) -> list[tuple]:
        """
        Load YOLO format annotations from txt file.

        Returns:
            List of (class_id, x_center, y_center, width, height)
        """
        annotations = []
        if not txt_path.exists():
            return annotations

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))

        return annotations

    def _derive_category(self, score: float, reason: str, is_door: bool | None) -> str:
        """Derive category from reason with consistency checks."""
        # If is_door explicitly false, force discard
        if is_door is False:
            return "discard"

        # Map reason to category
        category = REASON_TO_CATEGORY.get(reason, "error")

        # Consistency check: score vs category mismatch
        if category == "keep" and score < 0.5:
            return "inconsistent"
        if category == "discard" and score > 0.7:
            return "inconsistent"
        if is_door is False and reason == "good":
            return "inconsistent"

        # Fallback for unknown reasons: use score thresholds
        if category == "error" and score >= 0:
            if score < THRESHOLD_DISCARD:
                return "discard"
            elif score < THRESHOLD_PARTIAL:
                return "partial"
            elif score < THRESHOLD_LOOSE:
                return "loose"
            else:
                return "keep"

        return category

    @torch.no_grad()
    def rate_annotation(
        self,
        image: Union[str, Path, Image.Image],
        bbox: tuple,
    ) -> dict:
        """Rate a single bounding box annotation."""
        if not self.model_loaded:
            self.load_model()

        t0 = time.perf_counter()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Draw bbox on image
        annotated_image, bbox_valid = self._draw_bbox_on_image(image, bbox)
        if not bbox_valid:
            return {
                "is_door": None,
                "score": -1,
                "reason": "degenerate_bbox",
                "category": "error",
                "raw_response": "",
            }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": annotated_image},
                    {"type": "text", "text": self.rating_prompt},
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
            **inputs, max_new_tokens=128, do_sample=False
        )
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.latencies.append(time.perf_counter() - t0)

        # Parse and derive category
        result = self._parse_json_output(output_text)
        is_door = result.get("is_door")
        score = result.get("score", -1)
        reason = result.get("reason", "unknown")
        category = self._derive_category(score, reason, is_door)

        return {
            "is_door": is_door,
            "score": score,
            "reason": reason,
            "category": category,
            "raw_response": output_text,
        }

    def rate_image_annotations(
        self,
        image_path: Union[str, Path],
        annotation_path: Optional[Union[str, Path]] = None,
    ) -> tuple[list[dict], int, int]:
        """Rate all annotations for a single image. Returns (results, img_w, img_h)."""
        image_path = Path(image_path)
        annotation_path = (
            Path(annotation_path) if annotation_path else image_path.with_suffix(".txt")
        )

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        annotations = self._load_annotations(annotation_path)

        results = []
        for i, (class_id, x_center, y_center, w, h) in enumerate(annotations):
            rating = self.rate_annotation(image, (x_center, y_center, w, h))
            rating.update(
                {
                    "bbox_index": i,
                    "class_id": class_id,
                    "bbox": (x_center, y_center, w, h),
                }
            )
            results.append(rating)

        return results, img_w, img_h

    def process_folder(
        self,
        image_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        """Process all images and rate their annotations."""
        image_dir, annotation_dir, output_dir = (
            Path(image_dir),
            Path(annotation_dir),
            Path(output_dir),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir = output_dir / "sidecar_json"
        scored_ann_dir = output_dir / "scored_annotations"
        sidecar_dir.mkdir(exist_ok=True)
        scored_ann_dir.mkdir(exist_ok=True)

        if not self.model_loaded:
            self.load_model()

        # Collect image files
        image_files = [
            f
            for ext in IMG_EXTENSIONS
            for f in list(image_dir.glob(f"*{ext}"))
            + list(image_dir.glob(f"*{ext.upper()}"))
        ]
        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"Processing {len(image_files)} images")
        category_counts = {
            "keep": 0,
            "loose": 0,
            "partial": 0,
            "discard": 0,
            "error": 0,
            "inconsistent": 0,
        }
        total_annotations, missing_ann = 0, 0

        for image_file in tqdm(image_files, desc="Rating annotations"):
            ann_path = annotation_dir / f"{image_file.stem}.txt"
            if not ann_path.exists():
                missing_ann += 1
                continue

            try:
                results, img_w, img_h = self.rate_image_annotations(
                    image_file, ann_path
                )
                if not results:
                    continue

                # Count categories
                for r in results:
                    category_counts[r["category"]] = (
                        category_counts.get(r["category"], 0) + 1
                    )
                    total_annotations += 1

                # Save sidecar JSON per image
                sidecar_data = {
                    "image_path": str(image_file),
                    "annotation_path": str(ann_path),
                    "img_w": img_w,
                    "img_h": img_h,
                    "model_id": self.model_id,
                    "prompt_version": PROMPT_VERSION,
                    "annotations": [
                        {
                            "bbox_index": r["bbox_index"],
                            "class_id": r["class_id"],
                            "bbox": list(r["bbox"]),
                            "is_door": r["is_door"],
                            "score": r["score"],
                            "reason": r["reason"],
                            "category": r["category"],
                        }
                        for r in results
                    ],
                }
                with open(sidecar_dir / f"{image_file.stem}.json", "w") as f:
                    json.dump(sidecar_data, f, indent=2)

                # Save scored annotations (all boxes with score appended)
                lines = []
                for r in results:
                    x_c, y_c, w, h = r["bbox"]
                    lines.append(
                        f"{r['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {r['score']:.3f} {r['category']}"
                    )
                with open(scored_ann_dir / f"{image_file.stem}.txt", "w") as f:
                    f.write("\n".join(lines))

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        # Print summary
        self._print_summary(
            category_counts,
            total_annotations,
            missing_ann,
            len(image_files),
            output_dir,
        )

    def _print_summary(
        self,
        category_counts: dict,
        total: int,
        missing: int,
        num_images: int,
        output_dir: Path,
    ):
        """Print and save summary."""
        print(f"\n{'=' * 50}\nANNOTATION RATING SUMMARY\n{'=' * 50}")
        print(f"Total annotations: {total} | Missing ann files: {missing}")
        for cat in ["keep", "loose", "partial", "discard", "inconsistent", "error"]:
            cnt = category_counts.get(cat, 0)
            print(f"  {cat:12}: {cnt:5} ({100 * cnt / max(1, total):5.1f}%)")

        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"\nAvg inference: {avg:.3f}s ({1 / avg:.1f} ratings/sec)")

        summary = {
            "model_id": self.model_id,
            "prompt_version": PROMPT_VERSION,
            "total_images": num_images,
            "missing_annotations": missing,
            "total_annotations": total,
            "category_counts": category_counts,
            "thresholds": {
                "discard": THRESHOLD_DISCARD,
                "partial": THRESHOLD_PARTIAL,
                "loose": THRESHOLD_LOOSE,
            },
            "avg_latency_sec": sum(self.latencies) / len(self.latencies)
            if self.latencies
            else 0,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage - modify paths as needed
    image_folder = Path(
        "/media/data_2/vlm/code/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    annotation_folder = Path(
        "/media/data_2/vlm/code/data_miner/output/projects/delivery_pov_v1/merged_detections/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/media/data_2/vlm/code/data_miner/output/projects/delivery_pov_v1/qwen3vl_rated"
    )

    # Initialize rater
    rater = Qwen3VLAnnotationRater(
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        use_flash_attn=False,
        bbox_color=(255, 0, 0),  # Red bbox
        bbox_width=4,
    )

    # Process folder
    rater.process_folder(
        image_dir=image_folder,
        annotation_dir=annotation_folder,
        output_dir=output_folder,
    )
