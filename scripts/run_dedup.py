from pathlib import Path

from data_miner.config import DINO_MODELS, DeduplicationConfig, DinoEmbeddingStage
from data_miner.modules.deduplicator import Deduplicator



def main(thr = 0.95):
    input_dir = Path("/data/datasets/data_miner_datasets/forklift_palletjack_v1/frames_filtered")
    # input_dir = Path("/swdfs_mnt/swshared/data_miner_output/projects/forklift_palletjack_v1/frames_filtered")
    # input_dir = Path("/data/datasets/data_miner_datasets/doors_jci")
    dedup_conf = DeduplicationConfig(
        output_dir=input_dir.parent / f"frames_dedup_v1_cls_{thr}",
        dino_model_id=DINO_MODELS["dinov3-giant"],
        device="cuda:1",
        batch_size=8,
        threshold=thr,
        k_neighbors=1000,
        dino_embedding_stage=DinoEmbeddingStage.HIDDEN_CLS,
        cache_embeddings=True,
        ignore_cache=False,
    )

    _dedup = Deduplicator(dedup_conf, device_map=dedup_conf.device)

    frame_paths = list(Path(input_dir).rglob("*.jpg")) + list(Path(input_dir).rglob("*.png"))

    result = _dedup.deduplicate(frame_paths=frame_paths, input_dir=input_dir)
    print(f"Unique frames: {result.unique_frames}")
    print(f"Total frames: {result.total_frames}")
    print(f"Dedup rate: {(result.duplicates_removed/result.total_frames):.2%}")


if __name__ == "__main__":
    thrs = [0.8, 0.85, 0.9, 0.92, 0.95]
    for thr in thrs:
        main(thr)