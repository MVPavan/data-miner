"""
Video Registry Module

Pydantic-based video registry for tracking video URLs, IDs, and processing status.
Prevents duplicate processing and maintains a master list of all videos.
"""

import logging
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Status Enums
# =============================================================================

class VideoStatus(str, Enum):
    """Overall video processing status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    FILTERING = "filtering"
    FILTERED = "filtered"
    DEDUPLICATING = "deduplicating"
    DEDUPLICATED = "deduplicated"
    DETECTING = "detecting"
    DETECTED = "detected"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Stage Status Models (Pydantic)
# =============================================================================

class DownloadStage(BaseModel):
    """Download stage status."""
    completed: bool = False
    path: Optional[str] = None
    size_mb: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class ExtractionStage(BaseModel):
    """Frame extraction stage status."""
    completed: bool = False
    total_frames: int = 0
    output_dir: Optional[str] = None
    error: Optional[str] = None


class FilterStage(BaseModel):
    """Filtering stage status."""
    completed: bool = False
    input_frames: int = 0
    passed_frames: int = 0
    output_dir: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def pass_rate(self) -> float:
        return self.passed_frames / max(self.input_frames, 1)


class DeduplicationStage(BaseModel):
    """Deduplication stage status."""
    completed: bool = False
    input_frames: int = 0
    unique_frames: int = 0
    duplicates_removed: int = 0
    output_dir: Optional[str] = None
    error: Optional[str] = None


class DetectionStage(BaseModel):
    """Detection stage status."""
    completed: bool = False
    frames_processed: int = 0
    total_detections: int = 0
    output_dir: Optional[str] = None
    error: Optional[str] = None


class PipelineStages(BaseModel):
    """All pipeline stages for a video."""
    download: DownloadStage = Field(default_factory=DownloadStage)
    extraction: ExtractionStage = Field(default_factory=ExtractionStage)
    filter: FilterStage = Field(default_factory=FilterStage)
    deduplication: DeduplicationStage = Field(default_factory=DeduplicationStage)
    detection: DetectionStage = Field(default_factory=DetectionStage)


# =============================================================================
# Video Entry Model
# =============================================================================

class VideoEntry(BaseModel):
    """A single video entry in the registry."""
    video_id: str
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None
    duration_seconds: Optional[int] = None
    source_keyword: Optional[str] = None
    added_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: VideoStatus = VideoStatus.PENDING
    stages: PipelineStages = Field(default_factory=PipelineStages)
    notes: Optional[str] = None
    
    def is_processed(self) -> bool:
        """Check if video has completed all stages."""
        return self.status == VideoStatus.COMPLETE
    
    def get_summary(self) -> dict:
        """Get a summary of processing status."""
        return {
            "video_id": self.video_id,
            "status": self.status.value,
            "downloaded": self.stages.download.completed,
            "extracted": self.stages.extraction.completed,
            "filtered": self.stages.filter.completed,
            "deduplicated": self.stages.deduplication.completed,
            "detected": self.stages.detection.completed,
        }


# =============================================================================
# Registry Metadata
# =============================================================================

class RegistryMetadata(BaseModel):
    """Metadata for the registry."""
    created: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_videos: int = 0
    keywords_searched: list[str] = Field(default_factory=list)
    version: str = "1.0"


# =============================================================================
# Video Registry
# =============================================================================

class VideoRegistry(BaseModel):
    """
    Master video registry for tracking all videos.
    
    Example:
        >>> registry = VideoRegistry.load("video_registry.yaml")
        >>> registry.add_video("dQw4w9WgXcQ", "https://youtube.com/...", keyword="glass door")
        >>> registry.save()
    """
    metadata: RegistryMetadata = Field(default_factory=RegistryMetadata)
    videos: dict[str, VideoEntry] = Field(default_factory=dict)
    
    _file_path: Optional[Path] = None
    _lock: threading.Lock = None  # Thread lock for safe concurrent updates
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, '_lock', threading.Lock())
    
    # -------------------------------------------------------------------------
    # Video Management
    # -------------------------------------------------------------------------
    
    def add_video(
        self,
        video_id: str,
        url: str,
        title: Optional[str] = None,
        channel: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> bool:
        """
        Add a video to the registry.
        
        Returns True if added, False if already exists.
        """
        if video_id in self.videos:
            logger.debug(f"Video {video_id} already in registry, skipping")
            return False
        
        self.videos[video_id] = VideoEntry(
            video_id=video_id,
            url=url,
            title=title,
            channel=channel,
            source_keyword=keyword,
        )
        self.metadata.total_videos = len(self.videos)
        self.metadata.updated = datetime.now().isoformat()
        
        if keyword and keyword not in self.metadata.keywords_searched:
            self.metadata.keywords_searched.append(keyword)
        
        logger.info(f"Added video to registry: {video_id}")
        return True
    
    def get_video(self, video_id: str) -> Optional[VideoEntry]:
        """Get a video entry by ID."""
        return self.videos.get(video_id)
    
    def has_video(self, video_id: str) -> bool:
        """Check if video exists in registry."""
        return video_id in self.videos
    
    def remove_video(self, video_id: str) -> bool:
        """Remove a video from the registry."""
        if video_id in self.videos:
            del self.videos[video_id]
            self.metadata.total_videos = len(self.videos)
            self.metadata.updated = datetime.now().isoformat()
            return True
        return False
    
    # -------------------------------------------------------------------------
    # Status Updates (Thread-Safe)
    # -------------------------------------------------------------------------
    
    def update_status(self, video_id: str, status: VideoStatus) -> None:
        """Update the overall status of a video (thread-safe)."""
        with self._lock:
            if video_id in self.videos:
                self.videos[video_id].status = status
                self.metadata.updated = datetime.now().isoformat()
    
    def update_download_stage(
        self,
        video_id: str,
        completed: bool = True,
        path: Optional[str] = None,
        size_mb: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update download stage status (thread-safe)."""
        with self._lock:
            if video_id in self.videos:
                stage = self.videos[video_id].stages.download
                stage.completed = completed
                stage.path = path
                stage.size_mb = size_mb
                stage.error = error
                self.metadata.updated = datetime.now().isoformat()
    
    def update_extraction_stage(
        self,
        video_id: str,
        completed: bool = True,
        total_frames: int = 0,
        output_dir: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update extraction stage status (thread-safe)."""
        with self._lock:
            if video_id in self.videos:
                stage = self.videos[video_id].stages.extraction
                stage.completed = completed
                stage.total_frames = total_frames
                stage.output_dir = output_dir
                stage.error = error
                self.metadata.updated = datetime.now().isoformat()
    
    def update_filter_stage(
        self,
        video_id: str,
        completed: bool = True,
        input_frames: int = 0,
        passed_frames: int = 0,
        output_dir: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update filter stage status (thread-safe)."""
        with self._lock:
            if video_id in self.videos:
                stage = self.videos[video_id].stages.filter
                stage.completed = completed
                stage.input_frames = input_frames
                stage.passed_frames = passed_frames
                stage.output_dir = output_dir
                stage.error = error
                self.metadata.updated = datetime.now().isoformat()
    
    def update_deduplication_stage(
        self,
        video_id: str,
        completed: bool = True,
        input_frames: int = 0,
        unique_frames: int = 0,
        duplicates_removed: int = 0,
        output_dir: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update deduplication stage status (thread-safe)."""
        with self._lock:
            if video_id in self.videos:
                stage = self.videos[video_id].stages.deduplication
                stage.completed = completed
                stage.input_frames = input_frames
                stage.unique_frames = unique_frames
                stage.duplicates_removed = duplicates_removed
                stage.output_dir = output_dir
                stage.error = error
                self.metadata.updated = datetime.now().isoformat()
    
    def update_detection_stage(
        self,
        video_id: str,
        completed: bool = True,
        frames_processed: int = 0,
        total_detections: int = 0,
        output_dir: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update detection stage status (thread-safe)."""
        with self._lock:
            if video_id in self.videos:
                stage = self.videos[video_id].stages.detection
                stage.completed = completed
                stage.frames_processed = frames_processed
                stage.total_detections = total_detections
                stage.output_dir = output_dir
                stage.error = error
                self.metadata.updated = datetime.now().isoformat()
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def get_by_status(self, status: VideoStatus) -> list[VideoEntry]:
        """Get all videos with a specific status."""
        return [v for v in self.videos.values() if v.status == status]
    
    def get_pending(self) -> list[VideoEntry]:
        """Get all pending videos."""
        return self.get_by_status(VideoStatus.PENDING)
    
    def get_completed(self) -> list[VideoEntry]:
        """Get all completed videos."""
        return self.get_by_status(VideoStatus.COMPLETE)
    
    def get_failed(self) -> list[VideoEntry]:
        """Get all failed videos."""
        return self.get_by_status(VideoStatus.FAILED)
    
    def get_urls_by_status(self, status: VideoStatus|str) -> list[str]:
        """Get URLs of videos with a specific status."""
        return [v.url for v in self.get_by_status(status)]
    
    def get_statistics(self) -> dict:
        """Get registry statistics."""
        status_counts = {}
        for status in VideoStatus:
            count = len(self.get_by_status(status))
            if count > 0:
                status_counts[status.value] = count
        
        return {
            "total_videos": len(self.videos),
            "keywords": self.metadata.keywords_searched,
            "by_status": status_counts,
            "created": self.metadata.created,
            "updated": self.metadata.updated,
        }
    
    def get_stage_summary(self) -> dict:
        """Get pipeline stage summary with frame counts."""
        downloaded = 0
        extracted = 0
        filtered = 0
        deduplicated = 0
        detected = 0
        failed = 0
        
        total_frames_extracted = 0
        total_frames_filtered = 0
        total_frames_unique = 0
        total_detections = 0
        
        for video in self.videos.values():
            stages = video.stages
            
            if stages.download.completed:
                downloaded += 1
            if stages.extraction.completed:
                extracted += 1
                total_frames_extracted += stages.extraction.total_frames
            if stages.filter.completed:
                filtered += 1
                total_frames_filtered += stages.filter.passed_frames
            if stages.deduplication.completed:
                deduplicated += 1
                total_frames_unique += stages.deduplication.unique_frames
            if stages.detection.completed:
                detected += 1
                total_detections += stages.detection.total_detections
            
            if video.status == VideoStatus.FAILED:
                failed += 1
        
        return {
            "total_videos": len(self.videos),
            "downloaded": downloaded,
            "extracted": extracted,
            "filtered": filtered,
            "deduplicated": deduplicated,
            "detected": detected,
            "failed": failed,
            "frames": {
                "extracted": total_frames_extracted,
                "after_filter": total_frames_filtered,
                "unique": total_frames_unique,
            },
            "detections": total_detections,
        }
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save(self, file_path: Optional[Path] = None) -> None:
        """Save registry to YAML file (thread-safe, atomic write)."""
        path = file_path or self._file_path
        if not path:
            raise ValueError("No file path specified for saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe save with lock
        with self._lock:
            # Convert to dict for YAML serialization
            data = self.model_dump(mode="json")
            
            # Atomic write: write to temp file, then rename
            # This prevents data loss if process is killed during save
            temp_path = path.with_suffix(".tmp")
            try:
                with open(temp_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
                # Atomic rename (on POSIX systems)
                temp_path.replace(path)
            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise
        
        logger.debug(f"Registry saved to {path}")
    
    @classmethod
    def load(cls, file_path: Path) -> "VideoRegistry":
        """Load registry from YAML file with recovery support."""
        path = Path(file_path)
        temp_path = path.with_suffix(".tmp")
        
        # Check if main file exists
        if not path.exists():
            # Check for .tmp backup from interrupted save
            if temp_path.exists() and temp_path.stat().st_size > 0:
                logger.warning(f"Main registry missing, recovering from {temp_path}")
                temp_path.replace(path)
            else:
                logger.info(f"Registry file not found, creating new: {path}")
                registry = cls()
                registry._file_path = path
                return registry
        
        # Try to load main file
        try:
            with open(path) as f:
                content = f.read()
            
            # Check for empty/corrupted file
            if not content.strip():
                logger.warning(f"Registry file is empty: {path}")
                # Try to recover from .tmp backup
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    logger.warning(f"Attempting recovery from {temp_path}")
                    with open(temp_path) as f:
                        content = f.read()
                    if content.strip():
                        # Restore from backup
                        temp_path.replace(path)
                        logger.info("Successfully recovered registry from backup")
                    else:
                        logger.error("Backup file is also empty!")
                        registry = cls()
                        registry._file_path = path
                        return registry
                else:
                    logger.warning("No backup available, creating new registry")
                    registry = cls()
                    registry._file_path = path
                    return registry
            
            data = yaml.safe_load(content) or {}
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            registry = cls()
            registry._file_path = path
            return registry
        
        registry = cls.model_validate(data)
        registry._file_path = path
        logger.info(f"Registry loaded: {len(registry.videos)} videos from {path}")
        
        # Cleanup .tmp file if it exists (from successful previous save)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        
        return registry
    
    @classmethod
    def load_or_create(cls, file_path: Path) -> "VideoRegistry":
        """Load existing registry or create new one."""
        return cls.load(file_path)

