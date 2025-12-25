"""
Async Pipeline Module

Stage-level parallel processing for video mining at scale.

Example:
    >>> from video_miner_async.async_pipeline import AsyncPipelineOrchestrator, AsyncPipelineConfig
    >>> 
    >>> orchestrator = AsyncPipelineOrchestrator(
    ...     pipeline_config=config,
    ...     async_config=AsyncPipelineConfig(download_workers=3),
    ... )
    >>> result = await orchestrator.run(urls)
"""

from .messages import StageMessage, StageResult
from .metrics import PipelineMetrics, StageMetrics
from .base_worker import BaseStageWorker
from .orchestrator import AsyncPipelineOrchestrator, AsyncPipelineConfig, AsyncPipelineResult
from .workers import DownloadWorker, ExtractWorker, FilterWorker, DedupCollector

__all__ = [
    # Messages
    "StageMessage",
    "StageResult",
    # Metrics
    "PipelineMetrics",
    "StageMetrics",
    # Base
    "BaseStageWorker",
    # Orchestrator
    "AsyncPipelineOrchestrator",
    "AsyncPipelineConfig",
    "AsyncPipelineResult",
    # Workers
    "DownloadWorker",
    "ExtractWorker",
    "FilterWorker",
    "DedupCollector",
]
