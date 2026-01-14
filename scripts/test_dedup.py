from pathlib import Path

from data_miner.config import DINO_MODELS, DeduplicationConfig
from data_miner.modules.deduplicator import Deduplicator

input_dir = Path("/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup")
dedup_conf = DeduplicationConfig(
    output_dir=input_dir.parent / "frames_dedup_v2",
    dino_model_id=DINO_MODELS["dinov3-giant"],
    device="cuda:7",
    batch_size=8,
    threshold=0.85,
    k_neighbors=1000,
)

_dedup = Deduplicator(dedup_conf, device_map=dedup_conf.device)
frame_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
result = _dedup.deduplicate(
    frame_paths=frame_paths
)
print(f"Unique frames: {result.unique_frames}")
print(f"Total frames: {result.total_frames}")
print(f"Dedup rate: {(result.duplicates_removed/result.total_frames):.2%}")

