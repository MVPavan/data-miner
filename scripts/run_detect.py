from pathlib import Path

from data_miner.config import DetectionConfig, DetectorType
from data_miner.modules.detector import ObjectDetector

input_dir = Path(
    "/mnt/data_2/pavan/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
)

detect_conf = DetectionConfig(
    output_dir=input_dir.parent / "detections",
    device="cuda:7",
    detector=DetectorType.GROUNDING_DINO,
    model_ids={
        "grounding_dino": "IDEA-Research/grounding-dino-base",
        "owlv2": "google/owlv2-base-patch16-ensemble",
    },
    confidence_threshold=0.3,
    batch_size=16,
    save_visualizations=True,
)

# Detection prompts - what objects to detect
detection_prompt = (
    "glass door . entrance door . french door . patio door . sliding door"
)

_detector = ObjectDetector(detect_conf, device_map=detect_conf.device)

# Glob recursively for jpg and png files
frame_paths = list(Path(input_dir).rglob("*.jpg")) + list(
    Path(input_dir).rglob("*.png")
)

result = _detector.detect_batch(
    image_paths=frame_paths,
    prompt=detection_prompt,
    show_progress=True,
)

print("\nDetection Results:")
print(f"  Total frames: {result.total_frames}")
print(f"  Frames with detections: {result.frames_with_detections}")
print(f"  Total detections: {result.total_detections}")
print(f"  Detection rate: {(result.frames_with_detections / result.total_frames):.2%}")
print(f"\nAnnotations saved to: {detect_conf.output_dir / 'annotations.json'}")
if detect_conf.save_visualizations:
    print(f"Visualizations saved to: {detect_conf.output_dir / 'visualizations'}")
print("Completed")
