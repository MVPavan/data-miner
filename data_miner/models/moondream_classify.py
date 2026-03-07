"""
Moondream3 image classifier for object presence filtering.

Filters images by:
1. Reality check — reject cartoons, sketches, 3D renders, etc.
2. Object presence — is the target object in the image?
3. Visibility — is >50% of the object visible (not too cropped/zoomed)?

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m data_miner.models.moondream_classify
"""

import json
import shutil
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

REALITY_QUERY = (
    "Is this a real photograph taken by a camera? "
    "Answer no if it is a cartoon, sketch, drawing, illustration, 3D rendering, "
    "CGI, animation, clip art, or digitally generated image. Answer yes or no."
)

PRESENCE_QUERY_TEMPLATE = (
    "Is there a {obj} in this image where more than 50% of the {obj} is visible? "
    "It should not be too zoomed in or too cropped. Answer yes or no."
)


class MoondreamClassifier:
    def __init__(
        self,
        object_classes: list[str] = None,
        filter_non_real: bool = True,
    ):
        self.object_classes = object_classes or ["forklift", "pallet jack"]
        self.filter_non_real = filter_non_real
        self.model = None
        self.model_loaded = False
        self.latencies = []

    def load_model(self):
        if self.model_loaded:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream3-preview",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        self.model.compile()
        self.model_loaded = True
        print("Moondream3 loaded")

    def _query(self, encoded_image, question: str) -> bool | None:
        result = self.model.query(image=encoded_image, question=question, reasoning=False)
        answer = result["answer"].strip().lower()
        if "yes" in answer and "no" not in answer:
            return True
        if "no" in answer:
            return False
        return None

    def classify_image(self, image_path: str | Path) -> dict:
        if not self.model_loaded:
            self.load_model()

        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        encoded = self.model.encode_image(image)

        result = {"is_real": None, "objects": {}}

        # Reality check
        if self.filter_non_real:
            result["is_real"] = self._query(encoded, REALITY_QUERY)
            if result["is_real"] is False:
                for obj in self.object_classes:
                    result["objects"][obj] = False
                self.latencies.append(time.perf_counter() - t0)
                return result

        # Object presence + visibility
        for obj in self.object_classes:
            query = PRESENCE_QUERY_TEMPLATE.format(obj=obj)
            result["objects"][obj] = self._query(encoded, query)

        self.latencies.append(time.perf_counter() - t0)
        return result

    def process_folder(self, input_dir: str | Path, output_dir: str | Path):
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        if not self.model_loaded:
            self.load_model()

        image_files = [
            f
            for ext in IMG_EXTENSIONS
            for f in list(input_dir.glob(f"*{ext}")) + list(input_dir.glob(f"*{ext.upper()}"))
        ]
        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Processing {len(image_files)} images")
        print(f"Object classes: {self.object_classes}")
        print(f"Filter non-real: {self.filter_non_real}")

        all_results = {}
        counts = {"total": 0, "non_real": 0, "real": 0}
        for obj in self.object_classes:
            counts[f"{obj}_present"] = 0

        for img_file in tqdm(image_files, desc="Classifying"):
            try:
                classification = self.classify_image(img_file)
                all_results[img_file.stem] = classification
                counts["total"] += 1

                if classification["is_real"] is False:
                    counts["non_real"] += 1
                else:
                    counts["real"] += 1

                for obj in self.object_classes:
                    if classification["objects"].get(obj):
                        counts[f"{obj}_present"] += 1
            except Exception as e:
                print(f"Error: {img_file.name}: {e}")

        # Summary
        total = counts["total"]
        print(f"\n{'=' * 50}\nSUMMARY\n{'=' * 50}")
        print(f"Total: {total}")
        print(f"  Real:     {counts['real']} ({100 * counts['real'] / max(1, total):.1f}%)")
        print(f"  Non-real: {counts['non_real']} ({100 * counts['non_real'] / max(1, total):.1f}%)")
        for obj in self.object_classes:
            cnt = counts[f"{obj}_present"]
            print(f"  {obj} present: {cnt} ({100 * cnt / max(1, total):.1f}%)")
        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"Avg latency: {avg:.3f}s ({1 / avg:.1f} img/s)")

        # Copy passing images
        images_dir = output_dir / "images"
        images_dir_failed = output_dir / "images_failed"
        images_dir.mkdir(exist_ok=True)
        images_dir_failed.mkdir(exist_ok=True)
        copied = 0
        for img_file in image_files:
            res = all_results.get(img_file.stem)
            if res and res["is_real"] is not False and any(res["objects"].values()):
                shutil.copy2(img_file, images_dir / img_file.name)
                copied += 1
            else:
                shutil.copy2(img_file, images_dir_failed / img_file.name)
        print(f"Copied {copied} images to {images_dir}")

        # Save outputs
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        with open(output_dir / "summary.json", "w") as f:
            json.dump({"object_classes": self.object_classes, "counts": counts}, f, indent=2)
        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    input_folder = Path(
        "/swdfs_mnt/swshared/data_miner_output/projects/forklift_palletjack_v1/frames_dedup_v2_mean"
    )
    output_folder = Path(
        "/swdfs_mnt/swshared/data_miner_output/projects/forklift_palletjack_v1/md_filter_mean"
    )

    classifier = MoondreamClassifier(
        object_classes=["forklift", "pallet jack"],
        filter_non_real=True,
    )
    classifier.process_folder(input_folder, output_folder)
