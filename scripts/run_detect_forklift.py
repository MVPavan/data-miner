"""
Run detection models sequentially on forklift/pallet jack dataset.

Models: grounding_dino, owlvit, sam3, moondream, falcon
Classes: forklift, pallet jack

Usage:
    python scripts/run_detect_forklift.py --models grounding_dino owlvit
    python scripts/run_detect_forklift.py --models falcon --sanity
    python scripts/run_detect_forklift.py                           # runs all configured models
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ALL_MODELS = ["grounding_dino", "owlvit", "sam3", "falcon"]  # , "moondream"]
ALL_MODELS = ["falcon"]  # , "moondream"]

# Configuration
img_dir = Path(
    "/data/datasets/data_miner_datasets/forklift_palletjack_v1/frames_dedup_v1_cls_0.85"
)
base_output_dir = Path(
    "/data/datasets/data_miner_datasets/forklift_palletjack_v1/detections/pallet_jack/"
)
detection_classes = ["forklift", "pallet jack"]  # , 
THRESHOLD = 0.001  # Very low to keep all detections
REVIEW_IOU_THRESHOLD = 0.85

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

SANITY = args.sanity #or True
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


def yolo_line(class_id, x_min, y_min, width, height, score):
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return (
        f"{class_id} {x_center:.6f} {y_center:.6f} "
        f"{width:.6f} {height:.6f} {score:.6f}\n"
    )


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

print("processing:", img_dir)
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
# 4. Falcon Perception
# ============================================================
if "falcon" in models_to_run:
    print("\n" + "=" * 60)
    print(f"[{models_to_run.index('falcon')+1}/{len(models_to_run)}] Falcon Perception")
    print("=" * 60)

    from data_miner.models.falcon_perception import FalconPerceptionHelper

    falcon = FalconPerceptionHelper(
        detection_class=detection_classes,
        model_id="tiiuae/falcon-perception",
    )

    output_dir = base_output_dir / "falcon_perception"
    predictions_dir = output_dir / "pred_txt"
    details_dir = output_dir / "details"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(img_dir)
    print(f"Processing {len(image_files)} images")
    print(
        "Falcon prompt strategy: one query per class, then merge results "
        f"for {detection_classes}\n"
    )

    class_name_to_id = {
        class_name.lower(): index for index, class_name in enumerate(detection_classes)
    }

    total_detections = 0
    images_with_detections = 0
    review_images = []
    review_detection_count = 0

    falcon.load_model()

    for image_file in tqdm(image_files, desc="Processing"):
        try:
            merged_detections = []

            for class_name in detection_classes:
                detections = falcon.detect(
                    image_file,
                    threshold=THRESHOLD,
                    detection_classes=[class_name],
                    output_format="normalized",
                    include_metadata=True,
                )
                for det in detections:
                    label = det["label"] if det.get("label") else class_name
                    merged_detections.append((label, det))

            if not merged_detections:
                continue

            images_with_detections += 1
            total_detections += len(merged_detections)

            image_detail = {
                "image": image_file.name,
                "review_required": False,
                "detections": [],
            }

            txt_path = predictions_dir / f"{image_file.stem}.txt"
            with open(txt_path, "w") as file_handle:
                for label, det in merged_detections:
                    normalized_label = label.lower().strip()
                    class_id = class_name_to_id.get(
                        normalized_label,
                        class_name_to_id.get(label.split(",")[0].lower().strip(), 0),
                    )
                    x_min, y_min, width, height = det["yolo_bbox"]
                    score = det["confidence"]
                    file_handle.write(
                        yolo_line(class_id, x_min, y_min, width, height, score)
                    )

                    bbox_iou = det.get("bbox_iou")
                    review_required = bool(det.get("review_required"))
                    if bbox_iou is not None and bbox_iou >= REVIEW_IOU_THRESHOLD:
                        review_required = False

                    if review_required:
                        image_detail["review_required"] = True
                        review_detection_count += 1

                    image_detail["detections"].append(
                        {
                            "class_id": class_id,
                            "label": label,
                            "confidence": score,
                            "mask_bbox": det.get("mask_bbox"),
                            "coord_bbox": det.get("coord_bbox"),
                            "yolo_bbox": det.get("yolo_bbox"),
                            "bbox_iou": bbox_iou,
                            "review_required": review_required,
                            "review_reason": (
                                "mask_bbox_vs_coord_bbox_iou_below_0.85"
                                if review_required and bbox_iou is not None
                                else None
                            ),
                        }
                    )

            with open(details_dir / f"{image_file.stem}.json", "w") as file_handle:
                json.dump(image_detail, file_handle, indent=2)

            if image_detail["review_required"]:
                review_images.append(image_file.name)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(
        f"\nDone! {total_detections} detections in "
        f"{images_with_detections}/{len(image_files)} images"
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "model": "tiiuae/Falcon-Perception",
                "classes": detection_classes,
                "query_mode": "one_query_per_class_then_merge",
                "review_iou_threshold": REVIEW_IOU_THRESHOLD,
                "total_images": len(image_files),
                "images_with_detections": images_with_detections,
                "total_detections": total_detections,
                "review_detection_count": review_detection_count,
                "review_images": review_images,
            },
            f,
            indent=2,
        )

    falcon.unload_model()

# ============================================================
# 5. MoonDream
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
