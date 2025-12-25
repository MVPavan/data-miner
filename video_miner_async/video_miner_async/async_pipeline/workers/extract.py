"""
Extract Worker

Extracts frames from videos and passes frame folder to next stage.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from ..base_worker import BaseStageWorker
from ..messages import StageMessage
from ...config import ExtractionConfig
from ...modules.frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)


class ExtractWorker(BaseStageWorker):
    """
    Worker that extracts frames from videos.
    
    Input: StageMessage with video file path
    Output: StageMessage with frames folder path
    """
    
    def __init__(
        self,
        config: ExtractionConfig,
        cleanup_videos: bool = False,
        **kwargs,
    ):
        """
        Initialize extract worker.
        
        Args:
            config: Extraction configuration
            cleanup_videos: Delete video files after extraction
            **kwargs: Arguments passed to BaseStageWorker
        """
        super().__init__(**kwargs)
        self.config = config
        self.cleanup_videos = cleanup_videos
        self._extractor: Optional[FrameExtractor] = None
    
    @property
    def extractor(self) -> FrameExtractor:
        """Lazy-load extractor."""
        if self._extractor is None:
            self._extractor = FrameExtractor(self.config)
        return self._extractor
    
    def update_registry(self, video_id: str, success: bool, **kwargs) -> None:
        """Update registry after extraction."""
        if not self.registry:
            return
        
        from ...registry import VideoStatus
        
        if success:
            result = kwargs.get("result")
            frame_count = result.metadata.get("frame_count", 0) if result else 0
            output_dir = str(result.input_path) if result else None
            self.registry.update_extraction_stage(
                video_id=video_id,
                completed=True,
                total_frames=frame_count,
                output_dir=output_dir,
            )
            self.registry.update_status(video_id, VideoStatus.EXTRACTED)
        else:
            error = kwargs.get("error")
            self.registry.update_extraction_stage(
                video_id=video_id,
                completed=False,
                error=error,
            )
    
    def process(self, msg: StageMessage) -> Optional[StageMessage]:
        """
        Extract frames from a video.
        
        Args:
            msg: Message with video file path
            
        Returns:
            StageMessage with frames folder path, or None on failure
        """
        video_path = Path(msg.input_path)
        video_id = msg.video_id
        
        logger.info(f"[{self.name}] Extracting frames from {video_id}")
        
        result = self.extractor.extract_video(
            video_path=video_path,
            video_id=video_id,
            save_frames=True,
        )
        
        if not result.success:
            logger.warning(f"[{self.name}] Extraction failed for {video_id}: {result.error}")
            return None
        
        # Cleanup video file if configured
        if self.cleanup_videos and video_path.exists():
            try:
                video_path.unlink()
                logger.debug(f"[{self.name}] Cleaned up video file: {video_path}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to cleanup video {video_path}: {e}")
        
        return StageMessage(
            video_id=video_id,
            input_path=result.output_dir,
            metadata={
                **msg.metadata,
                "frame_count": result.frame_count,
            },
        )


