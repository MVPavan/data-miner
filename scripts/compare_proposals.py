"""Compare model outputs: models/ (standalone) vs auto_annotation_v2 proposal stage.

Usage:
    python scripts/compare_proposals.py --image output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 1) Run each model via models/ helpers (standalone)
# ---------------------------------------------------------------------------


def run_falcon_standalone(image_path: str, device: str = "cuda:0") -> list[dict]:
    from data_miner.models.falcon_perception import FalconPerceptionHelper

    helper = FalconPerceptionHelper(
        detection_class=["person"],
        model_id="tiiuae/Falcon-Perception",
        device=device,
        dtype="bfloat16",
        task="segmentation",
        seed=42,
        max_length=4096,
        min_dimension=256,
        max_dimension=512,  # match aa_v2 config
    )
    helper.load_model()
    results = helper.detect(
        image_path,
        threshold=0.0,  # get everything
        detection_classes=["person"],
        output_format="normalized",
        nms_threshold=0.5,
        include_metadata=True,
    )
    helper.unload_model()
    return results


def run_grounding_dino_standalone(
    image_path: str, device: str = "cuda:0"
) -> list[dict]:
    from data_miner.models.grounding_dino import GroundingDINOHelper

    helper = GroundingDINOHelper(
        detection_class=["person"],
        model_id="IDEA-Research/grounding-dino-base",
        device=device,
    )
    helper.load_model()
    results = helper.detect(
        image_path,
        threshold=0.25,
        detection_classes=["person"],
        output_format="pixel",
        text_threshold=0.2,
        nms_threshold=0.5,
    )
    helper.unload_model()
    return results


def run_sam_standalone(image_path: str, device: str = "cuda:1") -> list[dict]:
    from data_miner.models.sam import SAMHelper

    helper = SAMHelper(
        detection_class=["person"],
        model_id="facebook/sam3",
        device=device,
    )
    helper.load_model()
    results = helper.detect(
        image_path,
        threshold=0.5,
        detection_classes=["person"],
        output_format="pixel",
        nms_threshold=0.5,
    )
    helper.unload_model()
    return results


# ---------------------------------------------------------------------------
# 2) Run each model via aa_v2 proposal stage internals
# ---------------------------------------------------------------------------


def run_falcon_aav2(image_path: str, device: str = "cuda:0") -> list[dict]:
    from data_miner.auto_annotation_v2.config import (
        ClassPackConfig,
        DetectionModelConfig,
    )
    from data_miner.auto_annotation_v2.stages.proposal import _get_model, _run_falcon

    class_pack = ClassPackConfig(
        name="person", synonyms=[], negatives=[], prompt_variants=["person"]
    )
    cfg = DetectionModelConfig(
        kind="falcon",
        model_id="tiiuae/Falcon-Perception",
        device=device,
        params={
            "task": "segmentation",
            "dtype": "bfloat16",
            "min_dimension": 256,
            "max_dimension": 512,
            "max_length": 4096,
            "max_new_tokens": 2048,
            "seed": 42,
        },
    )

    loaded = _get_model("falcon", cfg)
    pil_image = Image.open(image_path).convert("RGB")

    candidates = _run_falcon(
        loaded, pil_image, class_pack, "person", cfg.params, "falcon"
    )
    return [
        {
            "candidate_id": c.candidate_id,
            "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
            "score": c.score,
            "label": c.label,
        }
        for c in candidates
    ]


def run_grounding_dino_aav2(image_path: str, device: str = "cuda:0") -> list[dict]:
    from data_miner.auto_annotation_v2.config import (
        ClassPackConfig,
        DetectionModelConfig,
    )
    from data_miner.auto_annotation_v2.stages.proposal import (
        _get_model,
        _run_grounding_dino,
    )

    class_pack = ClassPackConfig(
        name="person", synonyms=[], negatives=[], prompt_variants=["person"]
    )
    cfg = DetectionModelConfig(
        kind="grounding_dino",
        model_id="IDEA-Research/grounding-dino-base",
        device=device,
        params={"box_threshold": 0.25, "text_threshold": 0.2},
    )

    loaded = _get_model("grounding_dino", cfg)
    pil_image = Image.open(image_path).convert("RGB")

    candidates = _run_grounding_dino(
        loaded, pil_image, class_pack, "person", cfg.params, "grounding_dino"
    )
    return [
        {
            "candidate_id": c.candidate_id,
            "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
            "score": c.score,
            "label": c.label,
        }
        for c in candidates
    ]


def run_sam_aav2(image_path: str, device: str = "cuda:1") -> list[dict]:
    from data_miner.auto_annotation_v2.config import (
        ClassPackConfig,
        DetectionModelConfig,
    )
    from data_miner.auto_annotation_v2.stages.proposal import _get_model, _run_sam

    class_pack = ClassPackConfig(
        name="person", synonyms=[], negatives=[], prompt_variants=["person"]
    )
    cfg = DetectionModelConfig(
        kind="sam",
        model_id="facebook/sam3",
        device=device,
        params={"threshold": 0.5},
    )

    loaded = _get_model("sam", cfg)
    pil_image = Image.open(image_path).convert("RGB")

    candidates = _run_sam(loaded, pil_image, class_pack, "person", cfg.params, "sam")
    return [
        {
            "candidate_id": c.candidate_id,
            "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2],
            "score": c.score,
            "label": c.label,
        }
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def print_results(title: str, results):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    if not results:
        print("  (no detections)")
        return
    if isinstance(results, list) and len(results) > 0:
        if isinstance(results[0], dict):
            for i, r in enumerate(results, 1):
                bbox = r.get("bbox") or r.get("primary_bbox") or r.get("yolo_bbox")
                score = r.get("score") or r.get("confidence", "?")
                label = r.get("label", "?")
                review = r.get("review_required", "")
                review_str = " [REVIEW]" if review else ""
                print(
                    f"  {i:2d}. bbox={[f'{v:.4f}' for v in bbox]}  score={score:.4f}  label={label}{review_str}"
                )
        elif isinstance(results[0], (list, tuple)):
            for i, r in enumerate(results, 1):
                print(f"  {i:2d}. {[f'{v:.4f}' for v in r]}")
        else:
            print(f"  Raw: {results}")
    else:
        print(f"  Result type: {type(results)}, value: {results}")
    print(f"  Total: {len(results) if isinstance(results, list) else results}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg",
    )
    args = parser.parse_args()

    image_path = args.image
    print(f"Image: {image_path}")
    img = Image.open(image_path)
    print(f"Size: {img.size}")

    # --- Falcon ---
    print("\n" + "#" * 70)
    print("# FALCON PERCEPTION")
    print("#" * 70)

    print("\n--- Running Falcon via models/ (standalone) ---")
    falcon_standalone = run_falcon_standalone(image_path)
    print_results("Falcon Standalone (models/)", falcon_standalone)

    print("\n--- Running Falcon via aa_v2 proposal ---")
    falcon_aav2 = run_falcon_aav2(image_path)
    print_results("Falcon aa_v2 (proposal.py)", falcon_aav2)

    # Free GPU memory
    torch.cuda.empty_cache()

    # --- GroundingDINO ---
    print("\n" + "#" * 70)
    print("# GROUNDING DINO")
    print("#" * 70)

    print("\n--- Running GroundingDINO via models/ (standalone) ---")
    gdino_standalone = run_grounding_dino_standalone(image_path)
    print_results("GroundingDINO Standalone (models/)", gdino_standalone)

    print("\n--- Running GroundingDINO via aa_v2 proposal ---")
    gdino_aav2 = run_grounding_dino_aav2(image_path)
    print_results("GroundingDINO aa_v2 (proposal.py)", gdino_aav2)

    torch.cuda.empty_cache()

    # --- SAM3 ---
    print("\n" + "#" * 70)
    print("# SAM3")
    print("#" * 70)

    print("\n--- Running SAM3 via models/ (standalone) ---")
    sam_standalone = run_sam_standalone(image_path)
    print_results("SAM3 Standalone (models/)", sam_standalone)

    print("\n--- Running SAM3 via aa_v2 proposal ---")
    sam_aav2 = run_sam_aav2(image_path)
    print_results("SAM3 aa_v2 (proposal.py)", sam_aav2)

    torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    print(
        f"  Falcon:  standalone={len(falcon_standalone) if isinstance(falcon_standalone, list) else falcon_standalone}  aa_v2={len(falcon_aav2)}"
    )
    print(
        f"  GDino:   standalone={len(gdino_standalone) if isinstance(gdino_standalone, list) else gdino_standalone}  aa_v2={len(gdino_aav2)}"
    )
    print(
        f"  SAM3:    standalone={len(sam_standalone) if isinstance(sam_standalone, list) else sam_standalone}  aa_v2={len(sam_aav2)}"
    )


if __name__ == "__main__":
    main()
