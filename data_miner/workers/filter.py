"""
Filter worker - filters frames using SigLIP similarity.
"""

import logging
import os
from pathlib import Path

from .base import BaseProjectVideosWorker
from ..db.models import Video, ProjectVideo
from ..modules.frame_filter import FrameFilter
from ..config import get_filter_config, FilterConfig, StageName, init_hf_auth

logger = logging.getLogger(__name__)


class FilterWorker(BaseProjectVideosWorker):
    """Worker that filters frames."""
    
    stage_name = StageName.FILTER
    
    def __init__(self, config: FilterConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        
        # Load config from YAML if not provided
        self.config = config or get_filter_config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Authenticate with HuggingFace for private models
        init_hf_auth()
        
        self._filter = FrameFilter(self.config, device_map=self.config.device)
    
    def process(self, project_video: ProjectVideo, video: Video) -> dict:
        """Filter frames and return results."""
        from ..config import ProjectVideoStatus

        if not video.frames_dir:
            raise ValueError("No frames directory to filter")
        
        frames_dir = Path(video.frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
        
        # Get frame paths
        frame_paths = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        if not frame_paths:
            raise ValueError("No frames found in directory")
        
        result = self._filter.filter_frames(
            frame_paths=frame_paths,
            video_id=video.video_id,
        )
        
        # If no frames passed filter, signal skip status
        if result.passed_frames == 0:
            return {
                "_skip_status": ProjectVideoStatus.FILTERED_EMPTY,
                "filtered_dir": None,
                "passed_frames": 0,
            }
        
        return {
            "filtered_dir": str(self.config.output_dir / video.video_id),
            "passed_frames": result.passed_frames,
        }


def main():
    """Entry point for filter worker."""
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    # Set config path in env if provided
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = FilterWorker()
    worker.run()


if __name__ == "__main__":
    main()


