"""
Dedup Collector

Collects filtered frames from all videos for cross-video deduplication.
This is a special worker that accumulates results rather than processing individually.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from ..base_worker import BaseStageWorker
from ..messages import StageMessage
from ...config import DeduplicationConfig
from ...modules.deduplicator import Deduplicator, DeduplicationResult

logger = logging.getLogger(__name__)


class DedupCollector(BaseStageWorker):
    """
    Collector for cross-video deduplication.
    
    Unlike other workers, this one:
    1. Accumulates all filtered frame folders
    2. Runs dedup once after all videos are processed
    
    Input: StageMessage with filtered frames folder path
    Output: Accumulates internally, run deduplicate() at the end
    """
    
    def __init__(
        self,
        config: DeduplicationConfig,
        device_map: str = "auto",
        use_fp16: bool = True,
        **kwargs,
    ):
        """
        Initialize dedup collector.
        
        Args:
            config: Deduplication configuration
            device_map: Device for model
            use_fp16: Use fp16 for memory efficiency
            **kwargs: Arguments passed to BaseStageWorker
        """
        super().__init__(**kwargs)
        self.config = config
        self.device_map = device_map
        self.use_fp16 = use_fp16
        
        # Accumulated frame groups
        self.frame_groups: dict[str, list[Path]] = {}
        self._deduplicator: Optional[Deduplicator] = None
    
    @property
    def deduplicator(self) -> Deduplicator:
        """Lazy-load deduplicator."""
        if self._deduplicator is None:
            self._deduplicator = Deduplicator(
                config=self.config,
                device_map=self.device_map,
                use_fp16=self.use_fp16,
            )
        return self._deduplicator
    
    def process(self, msg: StageMessage) -> Optional[StageMessage]:
        """
        Collect frames from a video for later deduplication.
        
        Args:
            msg: Message with filtered frames folder path
            
        Returns:
            Always returns None (accumulates internally)
        """
        frames_dir = Path(msg.input_path)
        video_id = msg.video_id
        
        # Get all frame paths
        frame_paths = (
            sorted(frames_dir.glob("*.jpg")) + 
            sorted(frames_dir.glob("*.png"))
        )
        
        if frame_paths:
            self.frame_groups[video_id] = frame_paths
            logger.info(
                f"[{self.name}] Collected {len(frame_paths)} frames from {video_id}. "
                f"Total videos: {len(self.frame_groups)}"
            )
        
        # Don't pass to output queue - accumulate
        return None
    
    def run_deduplication(self) -> DeduplicationResult:
        """
        Run cross-video deduplication on all collected frames.
        
        Call this after all videos have been processed.
        
        Returns:
            DeduplicationResult with unique frames
        """
        if not self.frame_groups:
            logger.warning(f"[{self.name}] No frames to deduplicate")
            return DeduplicationResult(
                total_frames=0,
                unique_frames=0,
                duplicates_removed=0,
            )
        
        total_frames = sum(len(paths) for paths in self.frame_groups.values())
        logger.info(
            f"[{self.name}] Running cross-video dedup on {total_frames} frames "
            f"from {len(self.frame_groups)} videos"
        )
        
        result = self.deduplicator.deduplicate_cross_video(
            frame_groups=self.frame_groups,
            show_progress=True,
        )
        
        logger.info(
            f"[{self.name}] Dedup complete: {result.unique_frames}/{result.total_frames} unique "
            f"({result.duplicates_removed} removed)"
        )
        
        return result
    
    def cleanup(self) -> None:
        """Unload model to free GPU memory."""
        if self._deduplicator is not None:
            self._deduplicator.unload_model()
            self._deduplicator = None
    
    def get_collected_count(self) -> tuple[int, int]:
        """Get (num_videos, num_frames) collected."""
        num_videos = len(self.frame_groups)
        num_frames = sum(len(paths) for paths in self.frame_groups.values())
        return num_videos, num_frames
