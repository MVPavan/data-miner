"""
Filter Worker

Filters frames using SigLIP and passes filtered folder to next stage.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from ..base_worker import BaseStageWorker
from ..messages import StageMessage
from ...config import FilterConfig
from ...modules.frame_filter import FrameFilter

logger = logging.getLogger(__name__)


class FilterWorker(BaseStageWorker):
    """
    Worker that filters frames using SigLIP.
    
    Input: StageMessage with raw frames folder path
    Output: StageMessage with filtered frames folder path
    
    Note: This worker should typically run as a single instance
    to avoid GPU memory contention.
    """
    
    def __init__(
        self,
        config: FilterConfig,
        classes: list[str],
        device_map: str = "auto",
        cleanup_raw_frames: bool = False,
        **kwargs,
    ):
        """
        Initialize filter worker.
        
        Args:
            config: Filter configuration
            classes: Target class names for filtering
            device_map: Device for model
            cleanup_raw_frames: Delete raw frames after filtering
            **kwargs: Arguments passed to BaseStageWorker
        """
        super().__init__(**kwargs)
        self.config = config
        self.classes = classes
        self.device_map = device_map
        self.cleanup_raw_frames = cleanup_raw_frames
        self._filter: Optional[FrameFilter] = None
    
    @property
    def filter(self) -> FrameFilter:
        """Lazy-load filter model."""
        if self._filter is None:
            self._filter = FrameFilter(self.config, device_map=self.device_map)
        return self._filter
    
    def update_registry(self, video_id: str, success: bool, **kwargs) -> None:
        """Update registry after filtering."""
        if not self.registry:
            return
        
        from ...registry import VideoStatus
        
        if success:
            result = kwargs.get("result")
            if result:
                metadata = result.metadata
                self.registry.update_filter_stage(
                    video_id=video_id,
                    completed=True,
                    input_frames=metadata.get("total_frames", 0),
                    passed_frames=metadata.get("passed_frames", 0),
                    output_dir=str(result.input_path),
                )
            self.registry.update_status(video_id, VideoStatus.FILTERED)
        else:
            error = kwargs.get("error")
            self.registry.update_filter_stage(
                video_id=video_id,
                completed=False,
                error=error,
            )
    
    def process(self, msg: StageMessage) -> Optional[StageMessage]:
        """
        Filter frames from a video.
        
        Args:
            msg: Message with raw frames folder path
            
        Returns:
            StageMessage with filtered frames folder path, or None if no frames pass
        """
        frames_dir = Path(msg.input_path)
        video_id = msg.video_id
        
        # Get all frame paths from folder
        frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        
        if not frame_paths:
            logger.warning(f"[{self.name}] No frames found in {frames_dir}")
            return None
        
        logger.info(f"[{self.name}] Filtering {len(frame_paths)} frames from {video_id}")
        
        result = self.filter.filter_frames(
            frame_paths=frame_paths,
            classes=self.classes,
            video_id=video_id,
            copy_files=True,
        )
        
        # Cleanup raw frames if configured
        if self.cleanup_raw_frames and frames_dir.exists():
            try:
                shutil.rmtree(frames_dir)
                logger.debug(f"[{self.name}] Cleaned up raw frames: {frames_dir}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to cleanup {frames_dir}: {e}")
        
        if result.passed_frames == 0:
            logger.info(f"[{self.name}] No frames passed filter for {video_id}")
            return None
        
        filtered_dir = self.config.output_dir / video_id
        
        return StageMessage(
            video_id=video_id,
            input_path=filtered_dir,
            metadata={
                **msg.metadata,
                "total_frames": result.total_frames,
                "passed_frames": result.passed_frames,
                "pass_rate": result.pass_rate,
            },
        )
    
    def cleanup(self) -> None:
        """Unload model to free GPU memory."""
        if self._filter is not None:
            self._filter.unload_model()
            self._filter = None
