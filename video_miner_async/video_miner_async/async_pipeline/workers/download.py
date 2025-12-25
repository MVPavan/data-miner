"""
Download Worker

Downloads YouTube videos and passes video path to next stage.
"""

import logging
from pathlib import Path
from typing import Optional

from ..base_worker import BaseStageWorker
from ..messages import StageMessage
from ...config import DownloadConfig
from ...modules.downloader import YouTubeDownloader
from ...utils.io import get_video_id

logger = logging.getLogger(__name__)


class DownloadWorker(BaseStageWorker):
    """
    Worker that downloads YouTube videos.
    
    Input: StageMessage with video URL in input_path
    Output: StageMessage with video file path in input_path
    """
    
    def __init__(self, config: DownloadConfig, **kwargs):
        """
        Initialize download worker.
        
        Args:
            config: Download configuration
            **kwargs: Arguments passed to BaseStageWorker
        """
        super().__init__(**kwargs)
        self.config = config
        self._downloader: Optional[YouTubeDownloader] = None
    
    @property
    def downloader(self) -> YouTubeDownloader:
        """Lazy-load downloader."""
        if self._downloader is None:
            self._downloader = YouTubeDownloader(self.config)
        return self._downloader
    
    def update_registry(self, video_id: str, success: bool, **kwargs) -> None:
        """Update registry after download."""
        if not self.registry:
            return
        
        from ...registry import VideoStatus
        
        if success:
            result = kwargs.get("result")
            path = str(result.input_path) if result else None
            self.registry.update_download_stage(
                video_id=video_id,
                completed=True,
                path=path,
            )
            self.registry.update_status(video_id, VideoStatus.DOWNLOADED)
        else:
            error = kwargs.get("error")
            self.registry.update_download_stage(
                video_id=video_id,
                completed=False,
                error=error,
            )
    
    def process(self, msg: StageMessage) -> Optional[StageMessage]:
        """
        Download a video.
        
        Args:
            msg: Message with URL in input_path (as string)
            
        Returns:
            StageMessage with video file path, or None on failure
        """
        # url = str(msg.input_path)
        url = msg.metadata.get("url", "")
        video_id = msg.video_id
        
        logger.info(f"[{self.name}] Downloading {video_id}")
        
        result = self.downloader.download_single(url)
        
        if not result.success:
            logger.warning(f"[{self.name}] Download failed for {video_id}: {result.error}")
            return None
        
        return StageMessage(
            video_id=video_id,
            input_path=result.output_path,
            metadata={
                "title": result.title,
                "duration": result.duration,
                "url": url,
            },
        )

