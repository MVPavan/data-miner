"""
SQLModel database models.

Architecture:
- Video: Central processing (download/extract), shared across projects
- ProjectVideo: Per-project processing (filter/dedup/detect)
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from ..config import VideoStatus, ProjectVideoStatus, ProjectStatus, SourceType


class Project(SQLModel, table=True):
    """Project table."""
    __tablename__ = "projects"
    
    project_id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True)
    output_dir: Optional[str] = None
    project_stage: ProjectStatus = Field(default=ProjectStatus.POPULATING)

    # Video counts (updated by monitor)
    total_videos: int = Field(default=0)
    videos_pending: int = Field(default=0)
    videos_downloaded: int = Field(default=0)
    videos_extracted: int = Field(default=0)
    videos_failed: int = Field(default=0)

    # Frame counts (updated by monitor)
    extracted_frames: int = Field(default=0)
    filtered_frames: int = Field(default=0)
    unique_frames: int = Field(default=0)
    
    # Output directories (set by workers)
    dedup_dir: Optional[str] = None
    detect_dir: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Video(SQLModel, table=True):
    """
    Central video table - shared across projects.
    
    Tracks download and extraction stages. Videos can belong to multiple
    projects via the ProjectVideo table.
    """
    __tablename__ = "videos"
    
    video_id: str = Field(max_length=20, primary_key=True)
    url: Optional[str] = None
    
    # Source info (where this video came from)
    source_type: SourceType = Field(default=SourceType.MANUAL)
    source_info: Optional[str] = None  # search query, filename, etc.
    
    # Central stage outputs
    video_path: Optional[str] = None
    frames_dir: Optional[str] = None
    frame_count: Optional[int] = None
    title: Optional[str] = None
    
    # Central status (download/extract only)
    status: VideoStatus = Field(default=VideoStatus.PENDING)
    error: Optional[str] = None
    
    # Concurrency control with heartbeat
    locked_by: Optional[str] = Field(default=None, max_length=50)
    locked_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    
    # Backup tracking
    backed_up: bool = Field(default=False)
    backed_up_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectVideo(SQLModel, table=True):
    """
    Per-project video processing state.
    
    Links videos to projects and tracks project-specific stages
    (filter, dedup, detect). One entry per (project, video) pair.
    """
    __tablename__ = "project_videos"
    __table_args__ = (UniqueConstraint("project_id", "video_id"),)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.project_id")
    video_id: str = Field(foreign_key="videos.video_id", max_length=20)
    
    # Project-specific output directories
    filtered_dir: Optional[str] = None
    
    # Project-specific metrics
    passed_frames: Optional[int] = None
    
    # Project status (filter/dedup/detect stages)
    status: ProjectVideoStatus = Field(default=ProjectVideoStatus.PENDING)
    error: Optional[str] = None
    
    # Concurrency control with heartbeat
    locked_by: Optional[str] = Field(default=None, max_length=50)
    locked_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

