"""Video Miner v3 - Utilities package."""

try:
    from .device import resolve_device, get_model_device, clear_gpu_cache
    _device_available = True
except ImportError:
    _device_available = False

from .io import save_json, load_json, ensure_dir, get_video_id
from .validators import validate_youtube_url, validate_image_path

__all__ = [
    "resolve_device",
    "get_model_device",
    "clear_gpu_cache",
    "save_json",
    "load_json",
    "ensure_dir",
    "get_video_id",
    "validate_youtube_url",
    "validate_image_path",
]

