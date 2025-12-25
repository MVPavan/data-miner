"""
Async Pipeline Messages

Pydantic models for inter-stage communication.
"""

from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


class StageMessage(BaseModel):
    """Message passed between pipeline stages."""
    
    video_id: str = Field(..., description="YouTube video ID")
    input_path: Path = Field(..., description="Input folder/file from previous stage")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Stage-specific metadata")
    timestamp: float = Field(default_factory=time.time, description="Message creation time")
    
    class Config:
        arbitrary_types_allowed = True


class StageResult(BaseModel):
    """Result of processing a single video through a stage."""
    
    video_id: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
