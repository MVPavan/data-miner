import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path

from data_miner.models.youtu_vl import YoutuVLHelper

img_dir = Path(
    "/data/datasets/data_miner_datasets/forklift_palletjack_v1/frames_dedup_v1_cls_0.85"
)
output_dir = Path(
    "/data/datasets/data_miner_datasets/forklift_palletjack_v1/detections/youtu_vl"
)

yvl = YoutuVLHelper(
    detection_class=["forklift", "pallet jack"],
    use_flash_attn=False,
)
yvl.process_folder(img_dir, output_dir)

# CUDA_VISIBLE_DEVICES=1 uv run --with 'transformers>=4.56.0,<=4.57.1'  --with git+https://github.com/lucasb-eyer/pydensecrf.git scripts/run_detect_forklift_youtu_vl.py