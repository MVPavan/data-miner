"""
Run detection models sequentially on forklift/pallet jack dataset.

Models: grounding_dino, owlvit, sam3, moondream
Classes: forklift, pallet jack

Usage:
    python scripts/run_detect_forklift.py --models grounding_dino owlvit
    python scripts/run_detect_forklift.py --models sam3 moondream --sanity
    python scripts/run_detect_forklift.py                           # runs all 4
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

ALL_MODELS = ["grounding_dino", "owlvit", "sam3"]#, "moondream"]

# Configuration
img_dir = Path(
    "/data/datasets/data_miner_datasets/forklift_palletjack_v1/frames_dedup_v1_cls_0.85"
)
base_output_dir = Path(
    "/data/datasets/data_miner_datasets/forklift_palletjack_v1/detections/pallet_jack/"
)
detection_classes = ["pallet jack"] # ["forklift", "pallet jack"]  # , 
THRESHOLD = 0.001  # Very low to keep all detections

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sanity", action="store_true", help="Run on 5 images only as a sanity check"
)
parser.add_argument(
    "--models",
    nargs="+",
    choices=ALL_MODELS,
    default=ALL_MODELS,
    help=f"Models to run (default: all). Choices: {ALL_MODELS}",
)
args = parser.parse_args()

SANITY = args.sanity
SANITY_N = 5
models_to_run = args.models

if SANITY:
    base_output_dir = base_output_dir.parent / "detections_sanity"
    print(f"*** SANITY MODE: processing only {SANITY_N} images ***\n")

print(f"Models to run: {models_to_run}\n")


def get_image_files(input_dir):
    image_files = []
    for ext in IMG_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(image_files)


# In sanity mode, create a temp dir with symlinks to a small subset
if SANITY:
    import atexit
    import shutil
    import tempfile

    _sanity_dir = Path(tempfile.mkdtemp(prefix="sanity_"))
    atexit.register(lambda: shutil.rmtree(_sanity_dir, ignore_errors=True))
    _all_imgs = get_image_files(img_dir)[:SANITY_N]
    for img in _all_imgs:
        (_sanity_dir / img.name).symlink_to(img)
    img_dir = _sanity_dir
    print(f"Sanity images: {[p.name for p in _all_imgs]}\n")


# ============================================================
# 1. Grounding DINO
# ============================================================
if "grounding_dino" in models_to_run:
    print("=" * 60)
    print(f"[{models_to_run.index('grounding_dino')+1}/{len(models_to_run)}] Grounding DINO")
    print("=" * 60)

    from data_miner.models.grounding_dino import GroundingDINOHelper

    gdino = GroundingDINOHelper(
        detection_class=detection_classes,
        model_id="IDEA-Research/grounding-dino-base",
    )
    gdino.process_folder(img_dir, base_output_dir / "grounding_dino", threshold=THRESHOLD)
    gdino.unload_model()

# ============================================================
# 2. OWL-ViT v2
# ============================================================
if "owlvit" in models_to_run:
    print("\n" + "=" * 60)
    print(f"[{models_to_run.index('owlvit')+1}/{len(models_to_run)}] OWL-ViT v2")
    print("=" * 60)

    from data_miner.models.owlvit import OWLViTHelper

    owlvit = OWLViTHelper(
        detection_class=detection_classes,
        model_id="google/owlv2-large-patch14-ensemble",
    )
    owlvit.process_folder(img_dir, base_output_dir / "owlvit", threshold=THRESHOLD)
    owlvit.unload_model()

# ============================================================
# 3. SAM3
# ============================================================
if "sam3" in models_to_run:
    print("\n" + "=" * 60)
    print(f"[{models_to_run.index('sam3')+1}/{len(models_to_run)}] SAM3")
    print("=" * 60)

    from data_miner.models.sam import SAMHelper

    sam = SAMHelper(detection_class=detection_classes)
    sam.process_folder(img_dir, base_output_dir / "sam3", threshold=THRESHOLD)
    sam.unload_model()

# ============================================================
# 4. MoonDream
# ============================================================
if "moondream" in models_to_run:
    print("\n" + "=" * 60)
    print(f"[{models_to_run.index('moondream')+1}/{len(models_to_run)}] MoonDream")
    print("=" * 60)

    from data_miner.models.moondream import MoonDreamHelper

    moondream = MoonDreamHelper(detection_class=detection_classes)

    output_dir = base_output_dir / "moondream"
    predictions_dir = output_dir / "pred_txt"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    moondream.load_model()

    image_files = get_image_files(img_dir)
    print(f"Processing {len(image_files)} images")

    total_detections = 0
    images_with_detections = 0

    for image_file in tqdm(image_files, desc="Processing"):
        try:
            image = Image.open(image_file)
            encoded_image = moondream.model.encode_image(image)
            # Direct detection, no query filtering
            detections = moondream.detect_object(encoded_image, detection_classes)

            if len(detections) == 0:
                continue

            images_with_detections += 1
            total_detections += len(detections)

            # Save in YOLO format (class x_center y_center width height confidence)
            txt_path = predictions_dir / f"{image_file.stem}.txt"
            with open(txt_path, "w") as f:
                for det in detections:
                    cls_id, x_min, y_min, w, h = det
                    x_center = x_min + w / 2
                    y_center = y_min + h / 2
                    f.write(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} 1.000000\n"
                    )

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(
        f"\nDone! {total_detections} detections in "
        f"{images_with_detections}/{len(image_files)} images"
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "model": "moondream/moondream3-preview",
                "classes": detection_classes,
                "total_images": len(image_files),
                "images_with_detections": images_with_detections,
                "total_detections": total_detections,
            },
            f,
            indent=2,
        )

print("\n" + "=" * 60)
print(f"Complete! Ran: {', '.join(models_to_run)}")
print(f"Results saved to: {base_output_dir}")
print("=" * 60)
