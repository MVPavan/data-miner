from pathlib import Path
from omegaconf import OmegaConf

from data_miner.config import SIGLIP2_MODELS, FilterConfig, get_filter_config
from data_miner.modules.frame_filter import FrameFilter


input_dir = Path("/swdfs_mnt/swshared/data_miner_output/frames_raw/O-D9FiUmzNc")

filter_conf = get_filter_config(
    config_path="run_configs/multi_config_v1/forklift_palletjack.yaml"
    )
filter_conf.output_dir = Path(f"output/temp/{input_dir.name}")
# glob recursively for jpg and png files
frame_paths = list(Path(input_dir).rglob("*.jpg")) + list(Path(input_dir).rglob("*.png"))

_filter = FrameFilter(config=filter_conf, device_map=filter_conf.device)

result = _filter.filter_frames(
    frame_paths=frame_paths,
    video_id="frames_filtered_v2",
)
print(f"\n {result.passed_frames}/{result.total_frames} frames passed")
print("Completed")

