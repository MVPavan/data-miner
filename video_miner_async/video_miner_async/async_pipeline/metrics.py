"""
Async Pipeline Metrics

Pydantic models for pipeline monitoring and statistics.
"""

from pydantic import BaseModel, Field
from typing import Optional


class StageMetrics(BaseModel):
    """Metrics for a single pipeline stage."""
    
    name: str
    processed_count: int = 0
    failed_count: int = 0
    total_time_seconds: float = 0.0
    
    @property
    def avg_time_seconds(self) -> float:
        if self.processed_count == 0:
            return 0.0
        return self.total_time_seconds / self.processed_count


class PipelineMetrics(BaseModel):
    """Overall pipeline metrics."""
    
    total_videos: int = 0
    completed_videos: int = 0
    failed_videos: int = 0
    
    stage_metrics: dict[str, StageMetrics] = Field(default_factory=dict)
    queue_depths: dict[str, int] = Field(default_factory=dict)
    errors: list[tuple[str, str]] = Field(default_factory=list)
    
    def add_stage(self, name: str) -> None:
        """Add a stage to track."""
        if name not in self.stage_metrics:
            self.stage_metrics[name] = StageMetrics(name=name)
    
    def record_success(self, stage_name: str, duration: float) -> None:
        """Record successful processing."""
        if stage_name in self.stage_metrics:
            self.stage_metrics[stage_name].processed_count += 1
            self.stage_metrics[stage_name].total_time_seconds += duration
    
    def record_failure(self, stage_name: str, video_id: str, error: str) -> None:
        """Record processing failure."""
        if stage_name in self.stage_metrics:
            self.stage_metrics[stage_name].failed_count += 1
        self.errors.append((video_id, error))
        self.failed_videos += 1
    
    def summary(self) -> dict:
        """Get summary of all metrics."""
        return {
            "total_videos": self.total_videos,
            "completed": self.completed_videos,
            "failed": self.failed_videos,
            "stages": {
                name: {
                    "processed": m.processed_count,
                    "failed": m.failed_count,
                    "avg_time": f"{m.avg_time_seconds:.2f}s",
                }
                for name, m in self.stage_metrics.items()
            },
            "errors": self.errors[-10:],  # Last 10 errors
        }
