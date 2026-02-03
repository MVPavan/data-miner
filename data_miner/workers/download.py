"""
Download worker - downloads videos from YouTube.
"""

import os
from pathlib import Path

from .base import BaseVideoWorker
from ..db.models import Video
from ..modules.downloader import YouTubeDownloader
from ..config import get_download_config, DownloadConfig, StageName
from ..logging import get_logger

logger = get_logger(__name__)


class DownloadWorker(BaseVideoWorker):
    """Worker that downloads videos."""
    
    stage_name = StageName.DOWNLOAD
    
    def __init__(self, config: DownloadConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        
        # Load config from YAML if not provided
        self.config = config or get_download_config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._downloader = YouTubeDownloader(self.config)
    
    def process(self, video: Video) -> dict:
        """Download video and return path."""
        if not video.url:
            raise ValueError("No URL to download")
        
        result = self._downloader.download_single(video.url)
        
        if not result.success:
            raise Exception(result.error or "Download failed")
        
        return {
            "video_path": str(result.output_path),
            "title": result.title,
        }


def main():
    """Entry point for download worker."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = DownloadWorker()
    worker.run()


if __name__ == "__main__":
    main()


