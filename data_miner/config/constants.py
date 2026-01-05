"""
Constants and Enums - Single source of truth for all constant values.
"""

from enum import Enum


class VideoStatus(str, Enum):
    """Central video processing status (download/extract stages)."""
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    EXTRACTING = "EXTRACTING"
    EXTRACTED = "EXTRACTED"
    FAILED = "FAILED"


class ProjectVideoStatus(str, Enum):
    """Per-video status within a project (filter stage only)."""
    PENDING = "PENDING"
    FILTERING = "FILTERING"
    FILTERED = "FILTERED"
    FILTERED_EMPTY = "FILTERED_EMPTY"  # No frames matched filter prompts
    FAILED = "FAILED"


class ProjectStatus(str, Enum):
    """Project-level processing stage."""
    POPULATING = "POPULATING"       # Videos being added/downloaded/extracted
    FILTERING = "FILTERING"         # Filter workers processing
    DEDUP_READY = "DEDUP_READY"     # Ready for cross-dedup
    DEDUPING = "DEDUPING"           # Cross-dedup in progress
    DETECT_READY = "DETECT_READY"   # Ready for detection
    DETECTING = "DETECTING"         # Detection in progress
    COMPLETE = "COMPLETE"           # Done
    FAILED = "FAILED"               # Failed


# Terminal statuses for DRY status checking
FILTER_TERMINAL_STATUSES = frozenset({
    ProjectVideoStatus.FILTERED,
    ProjectVideoStatus.FILTERED_EMPTY,
    ProjectVideoStatus.FAILED,
})

VIDEO_TERMINAL_STATUSES = frozenset({
    VideoStatus.EXTRACTED,
    VideoStatus.FAILED,
})

class SourceType(str, Enum):
    """How the video was added to the project."""
    SEARCH = "SEARCH"
    MANUAL = "MANUAL"
    FILE = "FILE"


class SamplingStrategy(str, Enum):
    """Frame sampling strategy for extraction."""
    INTERVAL = "interval"
    TIME_BASED = "time"
    KEYFRAME = "keyframe"


class FilterModel(str, Enum):
    """Model for frame filtering."""
    SIGLIP2_SO400M = "siglip2-so400m"
    SIGLIP2_GIANT = "siglip2-giant"


class DetectorType(str, Enum):
    """Object detector type."""
    OWLV2 = "owlv2"
    GROUNDING_DINO = "grounding_dino"


class DedupModelType(str, Enum):
    """Model type for deduplication embeddings."""
    DINO = "dino"
    SIGLIP = "siglip"


class StageName(str, Enum):
    """Pipeline stage names."""
    DOWNLOAD = "download"
    EXTRACT = "extract"
    FILTER = "filter"
    DEDUP = "dedup"
    DETECT = "detect"


# =============================================================================
# Status Transitions (for central video stages)
# =============================================================================
VIDEO_INPUT_STATUS: dict[StageName, VideoStatus] = {
    StageName.DOWNLOAD: VideoStatus.PENDING,
    StageName.EXTRACT: VideoStatus.DOWNLOADED,
}

VIDEO_IN_PROGRESS_STATUS: dict[StageName, VideoStatus] = {
    StageName.DOWNLOAD: VideoStatus.DOWNLOADING,
    StageName.EXTRACT: VideoStatus.EXTRACTING,
}

VIDEO_OUTPUT_STATUS: dict[StageName, VideoStatus] = {
    StageName.DOWNLOAD: VideoStatus.DOWNLOADED,
    StageName.EXTRACT: VideoStatus.EXTRACTED,
}

# Status Transitions (for project-video filter stage only)
# Note: Dedup and Detect are now project-level, not per-video
PROJECT_VIDEO_INPUT_STATUS: dict[StageName, ProjectVideoStatus] = {
    StageName.FILTER: ProjectVideoStatus.PENDING,
}

PROJECT_VIDEO_IN_PROGRESS_STATUS: dict[StageName, ProjectVideoStatus] = {
    StageName.FILTER: ProjectVideoStatus.FILTERING,
}

PROJECT_VIDEO_OUTPUT_STATUS: dict[StageName, ProjectVideoStatus] = {
    StageName.FILTER: ProjectVideoStatus.FILTERED,
}


# =============================================================================
# SigLIP 2 Models (Filtering)
# =============================================================================
SIGLIP2_MODELS = {
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    "siglip2-giant": "google/siglip2-giant-opt-patch16-384",
}
SIGLIP2_DEFAULT = "siglip2-so400m"


# =============================================================================
# DINO Models (Deduplication)
# =============================================================================
DINO_MODELS = {
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",      # 21.6M
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",       # 85.7M
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",      # 304M
    "dinov3-huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",   # 0.8B
    "dinov3-giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",     # 7B
}
DINO_DEFAULT = "dinov3-base"


# =============================================================================
# YouTube Domains
# =============================================================================
YOUTUBE_DOMAINS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
})
YOUTUBE_BASE_URL = "https://www.youtube.com/watch?v="

