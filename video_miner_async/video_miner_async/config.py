"""
Video Miner v3 - Configuration Management

Pydantic-based configuration models with validation and defaults.
Supports YAML config files and CLI overrides.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .constants import (
    SIGLIP2_MODELS,
    SIGLIP2_DEFAULT,
    DINO_MODELS,
    DINO_DEFAULT,
    DETECTOR_MODELS,
    DETECTOR_DEFAULT,
    DEFAULT_FILTER_THRESHOLD,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEDUP_BATCH_SIZE,
    DEFAULT_DETECTION_BATCH_SIZE,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_MAX_FRAMES_PER_VIDEO,
    DEFAULT_MAX_CONCURRENT_DOWNLOADS,
    DEFAULT_DOWNLOAD_TIMEOUT,
    DEFAULT_IMAGE_QUALITY,
)


class DetectorType(str, Enum):
    """Available detection model backends."""
    DINO_X = "dino-x"
    MOONDREAM3 = "moondream3"
    FLORENCE2 = "florence2"
    GROUNDING_DINO = "grounding-dino"


class SamplingStrategy(str, Enum):
    """Frame sampling strategies."""
    INTERVAL = "interval"      # Every N frames
    TIME_BASED = "time"        # Every N seconds
    KEYFRAME = "keyframe"      # I-frames only (scene changes)


class FilterModel(str, Enum):
    """Available SigLIP 2 filter model variants."""
    SIGLIP2_SO400M = "siglip2-so400m"  # ~400M params, 384px, ~2GB VRAM
    SIGLIP2_GIANT = "siglip2-giant"    # ~1B params, 384px, ~4GB VRAM


class StageName(str, Enum):
    """Pipeline stage names for type-safe stage selection."""
    DOWNLOAD = "download"
    EXTRACTION = "extraction"
    FILTER = "filter"
    DEDUPLICATION = "deduplication"
    DETECTION = "detection"


# Ordered list for stage validation (must be contiguous)
STAGE_ORDER = list(StageName)


class DownloadConfig(BaseModel):
    """Configuration for video downloading."""
    
    force: bool = Field(default=False, description="Force rerun, ignore registry status")
    output_dir: Path = Field(default=Path("./downloads/videos"))
    format: str = Field(default="bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best")
    max_resolution: Optional[int] = Field(default=1080, description="Max video height in pixels")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Concurrent downloads")
    timeout: int = Field(default=300, ge=60, description="Download timeout in seconds")
    
    # Rate limiting options (to avoid YouTube blocks)
    sleep_interval: int = Field(default=0, ge=0, le=120, description="Min seconds to sleep between downloads (0=disabled)")
    max_sleep_interval: int = Field(default=0, ge=0, le=180, description="Max seconds to sleep (random between min-max)")
    sleep_requests: float = Field(default=0, ge=0, le=10, description="Seconds to sleep between metadata requests")
    
    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class ExtractionConfig(BaseModel):
    """Configuration for frame extraction."""
    
    force: bool = Field(default=False, description="Force rerun, ignore registry status")
    max_workers: int = Field(default=4, ge=1, le=16, description="Max concurrent video extractions")
    input_dir: Optional[Path] = Field(default=None, description="Input video directory (for standalone runs)")
    output_dir: Path = Field(default=Path("./downloads/frames_raw"))
    strategy: SamplingStrategy = Field(default=SamplingStrategy.INTERVAL)
    interval_frames: int = Field(default=30, ge=1, description="Extract every N frames")
    interval_seconds: float = Field(default=1.0, gt=0, description="Extract every N seconds")
    max_frames_per_video: Optional[int] = Field(default=1000, description="Cap frames per video")
    image_format: str = Field(default="jpg", pattern="^(jpg|png|webp)$")
    quality: int = Field(default=95, ge=1, le=100, description="JPEG/WebP quality")
    
    @field_validator("output_dir", "input_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class FilterConfig(BaseModel):
    """Configuration for SigLIP2-based frame filtering."""
    
    force: bool = Field(default=False, description="Force rerun, ignore registry status")
    input_dir: Optional[Path] = Field(default=None, description="Input frames directory (for standalone runs)")
    output_dir: Path = Field(default=Path("./downloads/frames_filtered"))
    model: FilterModel = Field(default=FilterModel.SIGLIP2_SO400M, description="SigLIP2 model variant")
    threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Similarity threshold")
    batch_size: int = Field(default=16, ge=1, le=64, description="Inference batch size")
    
    @property
    def model_id(self) -> str:
        """Get the HuggingFace model ID for the selected model."""
        return SIGLIP2_MODELS[self.model.value]
    
    @field_validator("output_dir", "input_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class DeduplicationConfig(BaseModel):
    """Configuration for deduplication (DINOv3 or SigLIP2)."""
    
    force: bool = Field(default=False, description="Force rerun, ignore registry status")
    input_dir: Optional[Path] = Field(default=None, description="Input frames directory (for standalone runs)")
    output_dir: Path = Field(default=Path("./downloads/frames_deduplicated"))
    threshold: float = Field(default=0.90, ge=0.0, le=1.0, description="Similarity threshold for duplicates")
    batch_size: int = Field(default=32, ge=1, le=128, description="Embedding batch size")
    k_neighbors: int = Field(default=50, ge=10, description="FAISS k-nearest neighbors to check")
    
    # Model selection
    use_siglip: bool = Field(
        default=False, 
        description="Use SigLIP2 for dedup (memory-efficient, reuses filter model)"
    )

    dino_model_id: str = Field(default=None, description="DINO model ID (auto-selected if None)")
    
    @property
    def model_type(self) -> str:
        """Get the model type being used."""
        return "siglip" if self.use_siglip else "dino"
    
    @field_validator("output_dir", "input_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class DetectionConfig(BaseModel):
    """Configuration for open-set object detection."""
    
    force: bool = Field(default=False, description="Force rerun, ignore registry status")
    input_dir: Optional[Path] = Field(default=None, description="Input frames directory (for standalone runs)")
    output_dir: Path = Field(default=Path("./downloads/detections"))
    detector: DetectorType = Field(default=DetectorType.MOONDREAM3)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    batch_size: int = Field(default=8, ge=1, le=32)
    save_visualizations: bool = Field(default=True)
    
    # Model-specific settings
    model_ids: dict = Field(default_factory=lambda: {
        "dino-x": "IDEA-Research/grounding-dino-base",  # DINO-X when available
        "moondream3": "vikhyatk/moondream2",  # Will use moondream3 when available
        "florence2": "microsoft/Florence-2-large",
        "grounding-dino": "IDEA-Research/grounding-dino-base",
    })
    
    @field_validator("output_dir", "input_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class PipelineConfig(BaseModel):
    """Main pipeline configuration combining all stages."""
    
    # Input
    urls: list[str] = Field(default_factory=list, description="YouTube URLs to process")
    classes: list[str] = Field(default_factory=list, description="Target classes/captions for filtering")
    
    # Stage configs
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    
    # Pipeline control
    stages: list[StageName] = Field(
        default=[
            StageName.DOWNLOAD,
            StageName.EXTRACTION,
            StageName.FILTER,
            StageName.DEDUPLICATION,
            StageName.DETECTION,
        ],
        description="Stages to run (must be contiguous)"
    )
    resume: bool = Field(default=True, description="Resume from checkpoints")
    device_map: str = Field(default="auto", description="Device: 'auto', 'cuda', 'cuda:0', 'cpu'")
    use_fp16: bool = Field(default=True, description="Use fp16 for memory efficiency")
    
    # Output
    output_dir: Path = Field(default=Path("./output"))
    
    @field_validator("stages", mode="before")
    @classmethod
    def validate_stages(cls, v):
        """Validate stages are valid names and contiguous in order."""
        if not v:
            raise ValueError("At least one stage must be specified")
        
        # Convert strings to StageName enum if needed
        stages = []
        for s in v:
            if isinstance(s, str):
                try:
                    stages.append(StageName(s))
                except ValueError:
                    valid = [stage.value for stage in StageName]
                    raise ValueError(f"Invalid stage '{s}'. Valid stages: {valid}")
            elif isinstance(s, StageName):
                stages.append(s)
            else:
                raise ValueError(f"Stage must be string or StageName, got {type(s)}")
        
        # Check stages are contiguous in STAGE_ORDER
        indices = [STAGE_ORDER.index(s) for s in stages]
        expected = list(range(min(indices), max(indices) + 1))
        
        if sorted(indices) != expected:
            raise ValueError(
                f"Stages must be contiguous. Got {[s.value for s in stages]}, "
                f"but indices {sorted(indices)} should be {expected}"
            )
        
        # Check order matches STAGE_ORDER
        if indices != sorted(indices):
            raise ValueError(
                f"Stages must be in order: {[STAGE_ORDER[i].value for i in expected]}"
            )
        
        return stages
    
    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        if v is None:
            return v
        return Path(v) if isinstance(v, str) else v
    
    def get_urls(self) -> list[str]:
        """Get all URLs (processed and gathered externally)."""
        return self.urls

