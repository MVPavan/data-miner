"""
Extract worker - extracts frames from videos.
"""

import os
from pathlib import Path

from .base import BaseVideoWorker
from ..db.models import Video
from ..modules.frame_extractor import FrameExtractor
from ..config import get_extraction_config, ExtractionConfig, StageName
from ..logging import get_logger

logger = get_logger(__name__)


class ExtractWorker(BaseVideoWorker):
    """Worker that extracts frames from videos."""
    
    stage_name = StageName.EXTRACT
    
    def __init__(self, config: ExtractionConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        
        # Load config from YAML if not provided
        self.config = config or get_extraction_config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._extractor = FrameExtractor(self.config)
    
    def process(self, video: Video) -> dict:
        """Extract frames and return output dir."""
        if not video.video_path:
            raise ValueError("No video path to extract from")
        
        video_path = Path(video.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        result = self._extractor.extract_video(
            video_path=video_path,
            video_id=video.video_id,
            save_frames=True,
        )
        
        if not result.success:
            raise Exception(result.error or "Extraction failed")
        
        return {
            "frames_dir": str(result.output_dir),
            "frame_count": result.frame_count,
        }


def main():
    """Entry point for extract worker."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = ExtractWorker()
    worker.run()


if __name__ == "__main__":
    main()


