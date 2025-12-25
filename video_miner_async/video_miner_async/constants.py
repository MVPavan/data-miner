"""
Constants and Model Registry

Single source of truth for all model IDs, default parameters, and constants.
Update this file to change model versions across the entire codebase.
"""

# =============================================================================
# SigLIP 2 Models (Filtering)
# =============================================================================
SIGLIP2_MODELS = {
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",   # ~400M params, ~2GB
    "siglip2-giant": "google/siglip2-giant-opt-patch16-384",  # ~1B params, ~4GB
}

SIGLIP2_DEFAULT = "siglip2-so400m"


# =============================================================================
# DINOv3 / DINOv2 Models (Deduplication)
# =============================================================================
DINO_MODELS = {
    # DINOv3 - ViT variants (LVD 1.7B images pretrain)
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",      # 21.6M
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",       # 85.7M
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",      # 304M
    "dinov3-huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",   # 0.8B
    "dinov3-giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",     # 7B
    # DINOv2 fallback
    "dinov2-base": "facebook/dinov2-base",                           # 86M
    "dinov2-large": "facebook/dinov2-large",                         # 300M
}

DINO_DEFAULT = "dinov2-base"  # Default fallback (DINOv3 tried first if enabled)


# =============================================================================
# Detector Models (Object Detection)
# =============================================================================
DETECTOR_MODELS = {
    "dino-x": "IDEA-Research/grounding-dino-base",           # Placeholder for DINO-X
    "moondream3": "moondream/moondream3-preview",             # Moondream (v3 when available)
    "florence2": "microsoft/Florence-2-large",               # Multi-task
    "grounding-dino": "IDEA-Research/grounding-dino-base",   # Stable
}

DETECTOR_DEFAULT = "moondream3"


# =============================================================================
# Default Thresholds and Parameters
# =============================================================================
DEFAULT_FILTER_THRESHOLD = 0.25
DEFAULT_DEDUP_THRESHOLD = 0.90
DEFAULT_DETECTION_THRESHOLD = 0.3

DEFAULT_BATCH_SIZE = 16
DEFAULT_DEDUP_BATCH_SIZE = 32
DEFAULT_DETECTION_BATCH_SIZE = 8

DEFAULT_FRAME_INTERVAL = 30
DEFAULT_MAX_FRAMES_PER_VIDEO = 1000

DEFAULT_MAX_CONCURRENT_DOWNLOADS = 3
DEFAULT_DOWNLOAD_TIMEOUT = 300

DEFAULT_IMAGE_QUALITY = 95


# =============================================================================
# Search Defaults
# =============================================================================
DEFAULT_SEARCH_MAX_RESULTS = 50
SEARCH_MAX_LIMIT = 500
SEARCH_TIMEOUT_SECONDS = 120
YOUTUBE_BASE_URL = "https://www.youtube.com/watch?v="
YOUTUBE_DOMAINS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
})


# =============================================================================
# Registry Defaults
# =============================================================================
DEFAULT_REGISTRY_FILE = "video_registry.yaml"

