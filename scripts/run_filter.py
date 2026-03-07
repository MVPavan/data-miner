from pathlib import Path
from omegaconf import OmegaConf

from data_miner.config import SIGLIP2_MODELS, FilterConfig, get_filter_config
from data_miner.modules.frame_filter import FrameFilter


input_dir = Path("/swdfs_mnt/swshared/data_miner_output/projects/forklift_palletjack_v1/frames_filtered")
output_dir = Path("/swdfs_mnt/swshared/data_miner_output/projects/forklift_palletjack_v1/")

filter_conf = get_filter_config(
    config_path="run_configs/multi_config_v1/forklift_palletjack.yaml"
    )
filter_conf.output_dir = output_dir
# glob recursively for jpg and png files
frame_paths = list(Path(input_dir).rglob("*.jpg")) + list(Path(input_dir).rglob("*.png"))

_filter = FrameFilter(config=filter_conf, device_map=filter_conf.device)

result = _filter.filter_frames(
    frame_paths=frame_paths,
    video_id="frames_filtered_pooler_v2",
)   
print(f"\n {result.passed_frames}/{result.total_frames} frames passed")
print("Completed")

