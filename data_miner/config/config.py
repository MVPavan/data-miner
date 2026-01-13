"""
Configuration models - Pydantic models for YAML config.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .constants import SamplingStrategy, DetectorType, DedupModelType


class DownloadConfig(BaseModel):
    """Download stage config."""
    output_dir: Path = Field(default=Path("./output/videos"))
    
    # Video format settings
    format: str = Field(default="bestvideo[height<=1080]+bestaudio/best[height<=1080]")
    max_resolution: int = Field(default=1080)
    timeout: int = Field(default=300)
    
    # Rate limiting (to avoid YouTube blocks)
    sleep_interval: float = Field(default=0)
    max_sleep_interval: float = Field(default=0)
    sleep_requests: float = Field(default=0)
    
    # Hashtag blocklist (path to txt file with patterns, one per line)
    blocked_hashtag_patterns: Optional[Path] = Field(default=None)


class ExtractionConfig(BaseModel):
    """Extraction stage config."""
    output_dir: Path = Field(default=Path("./output/frames_raw"))
    strategy: SamplingStrategy = Field(default=SamplingStrategy.INTERVAL)
    interval_frames: int = Field(default=30)
    interval_seconds: float = Field(default=1.0)
    max_frames_per_video: Optional[int] = Field(default=1000)
    image_format: str = Field(default="jpg")
    quality: int = Field(default=95)


class FilterConfig(BaseModel):
    """Filter stage config."""
    output_dir: Path = Field(default=Path("./output/frames_filtered"))
    device: str = Field(default="auto")  # auto, cuda, cuda:0, cpu
    model_id: str = Field(default="siglip2-so400m")
    batch_size: int = Field(default=32)
    
    # Positive prompts - frames must match at least one above threshold
    positive_prompts: list[str] = Field(default_factory=list)
    threshold: float = Field(default=0.25, description="Min score for positive prompts")
    
    # Negative prompts - optional, for filtering out false positives
    negative_prompts: list[str] = Field(default_factory=list)
    negative_threshold: float = Field(default=0.25, description="Max score for negative prompts")
    margin_threshold: float = Field(default=0.05, description="Positive must beat negative by this margin")

    # zoom prompts - optional, for filtering out close-up shots
    zoom_prompts: list[str] = Field(default_factory=list)
    zoom_threshold: float = Field(default=0.3, description="Zoom threshold should be less than this")
    zoom_margin_threshold: float = Field(default=0.05, description="Positive must beat zoom by this margin")

class DeduplicationConfig(BaseModel):
    """Deduplication stage config."""
    output_dir: Path = Field(default=Path("./output/frames_dedup"))
    device: str = Field(default="auto")  # auto, cuda, cuda:0, cpu
    threshold: float = Field(default=0.90)
    batch_size: int = Field(default=64)
    
    # Model selection
    model_type: DedupModelType = Field(default=DedupModelType.DINO)
    dino_model_id: str = Field(default="dinov3-base")
    
    # FAISS settings
    k_neighbors: int = Field(default=50)


class DetectionConfig(BaseModel):
    """Detection stage config."""
    output_dir: Path = Field(default=Path("./output/detections"))
    device: str = Field(default="auto")  # auto, cuda, cuda:0, cpu
    
    # Detection model
    detector: DetectorType = Field(default=DetectorType.GROUNDING_DINO)
    model_ids: dict = Field(default_factory=lambda: {
        "grounding_dino": "IDEA-Research/grounding-dino-base",
        "owlv2": "google/owlv2-base-patch16-ensemble",
    })
    
    # Detection settings
    confidence_threshold: float = Field(default=0.3)
    threshold: float = Field(default=0.3)  # Alias
    batch_size: int = Field(default=16)
    save_visualizations: bool = Field(default=True)


class DatabaseConfig(BaseModel):
    """Database connection config."""
    url: str = Field(default="postgresql://postgres:postgres@localhost:5432/data_miner")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    loki_url: str = Field(default="http://localhost:3100/loki/api/v1/push")
    log_dir: str = Field(default="/var/log/data_miner")


class SupervisorConfig(BaseModel):
    """Supervisor worker counts."""
    download_workers: int = Field(default=3)
    extract_workers: int = Field(default=2)
    filter_workers: int = Field(default=1)
    dedup_workers: int = Field(default=1)
    detect_workers: int = Field(default=1)


class MonitorConfig(BaseModel):
    """Monitor worker settings."""
    poll_interval: int = Field(default=10, description="Seconds between monitor checks")
    stale_threshold_minutes: int = Field(default=2, description="Reset locks older than this")
    long_running_threshold_minutes: int = Field(default=30, description="Warn about locks older than this")
    cleanup_extracted_videos: bool = Field(default=False, description="Delete source videos after extraction")


class BackupConfig(BaseModel):
    """Backup worker settings."""
    enabled: bool = Field(default=False, description="Enable backup worker")
    remote_dest: str = Field(description="Destination path (local: /path or remote: user@host:/path)")
    delete_after_backup: bool = Field(default=False, description="Delete local frames after verified backup")
    poll_interval: int = Field(default=300, description="Seconds between backup checks")
    verification_timeout: int = Field(default=1800, description="Seconds for rsync verification")
    
    @property
    def is_remote(self) -> bool:
        """Check if destination is remote (SSH) or local."""
        return "@" in self.remote_dest
    
    @model_validator(mode='after')
    def validate_backup_requirements(self) -> 'BackupConfig':
        """Validate rsync and SSH when backup is enabled."""
        if self.enabled:
            from ..utils.ssh_helper import check_rsync_installed, test_ssh_connection
            
            if self.remote_dest == "":
                raise ValueError("Backup 'remote_dest' must be specified when backup is enabled.")

            # Check rsync is installed
            if not check_rsync_installed():
                raise ValueError(
                    "rsync is not installed.\n"
                    "  Install: sudo apt-get install rsync"
                )
            
            # Check SSH for remote destinations
            if self.is_remote:
                if not test_ssh_connection(self.remote_dest):
                    raise ValueError(
                        f"SSH connection failed to '{self.remote_dest}'\n"
                        "  1. Set up passwordless SSH: ssh-keygen && ssh-copy-id user@host\n"
                        "  2. Or run: ./data_miner/utils/setup_ssh_keys.sh\n"
                        "  3. Verify: ssh user@host 'echo ok'"
                    )
        return self

