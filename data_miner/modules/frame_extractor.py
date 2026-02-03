"""
Frame Extraction Module

Extracts frames from videos with configurable sampling strategies.
Uses PyAV for video decoding and PIL for image saving.
Generator-based iteration for memory efficiency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import av
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import ExtractionConfig, SamplingStrategy
from ..utils.io import ensure_dir
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class FrameInfo:
    """Information about an extracted frame."""
    video_path: Path
    video_id: str
    frame_number: int
    timestamp: float  # Seconds into video
    image: Image.Image  # PIL Image (RGB)
    output_path: Optional[Path] = None


@dataclass
class ExtractionResult:
    """Result of frame extraction for a video."""
    video_path: Path
    video_id: str
    success: bool
    frame_count: int = 0
    output_paths: list[Path] = None
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.output_paths is None:
            self.output_paths = []


class FrameExtractor:
    """
    Video frame extractor with configurable sampling strategies.
    
    Uses PyAV for video decoding (full codec support including AV1, VP9, HEVC)
    and PIL for image saving.
    
    Supports:
        - Interval-based sampling (every N frames)
        - Time-based sampling (every N seconds)
        - Keyframe extraction (scene changes)
        - Generator-based iteration for memory efficiency
    
    Example:
        >>> config = ExtractionConfig(strategy=SamplingStrategy.INTERVAL, interval_frames=30)
        >>> extractor = FrameExtractor(config)
        >>> result = extractor.extract_video(Path("video.mp4"), video_id="abc123")
    """
    
    def __init__(self, config: ExtractionConfig):
        """
        Initialize frame extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config
        ensure_dir(config.output_dir)
    
    def _get_frame_indices(
        self,
        total_frames: int,
        fps: float,
    ) -> Generator[int, None, None]:
        """
        Generate frame indices based on sampling strategy.
        
        Args:
            total_frames: Total number of frames in video
            fps: Video frames per second
            
        Yields:
            Frame indices to extract
        """
        count = 0
        max_frames = self.config.max_frames_per_video or float("inf")
        
        if self.config.strategy == SamplingStrategy.INTERVAL:
            # Every N frames
            for idx in range(0, total_frames, self.config.interval_frames):
                if count >= max_frames:
                    break
                yield idx
                count += 1
                
        elif self.config.strategy == SamplingStrategy.TIME_BASED:
            # Every N seconds
            frame_interval = int(fps * self.config.interval_seconds)
            frame_interval = max(1, frame_interval)
            for idx in range(0, total_frames, frame_interval):
                if count >= max_frames:
                    break
                yield idx
                count += 1
                
        elif self.config.strategy == SamplingStrategy.KEYFRAME:
            # Keyframe detection - will be filtered in iterate_frames
            yield 0  # Always include first frame
            count += 1
            
            for idx in range(1, total_frames):
                if count >= max_frames:
                    break
                yield idx
                count += 1
    
    def _compute_frame_difference(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> float:
        """
        Compute mean absolute difference between two images.
        Used for keyframe/scene change detection.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Mean difference value
        """
        # Convert to grayscale numpy arrays
        arr1 = np.array(img1.convert('L'), dtype=np.float32)
        arr2 = np.array(img2.convert('L'), dtype=np.float32)
        
        # Compute mean absolute difference
        diff = np.abs(arr1 - arr2)
        return float(np.mean(diff))
    
    def iterate_frames(
        self,
        video_path: Path,
        video_id: str,
    ) -> Generator[FrameInfo, None, None]:
        """
        Generator that yields frames from video based on sampling strategy.
        
        Uses PyAV for decoding with full codec support (AV1, VP9, HEVC, H.264).
        
        Args:
            video_path: Path to video file
            video_id: Identifier for the video
            
        Yields:
            FrameInfo objects for each sampled frame
        """
        try:
            container = av.open(str(video_path))
        except Exception as e:
            logger.error(f"Failed to open video {video_path}: {e}")
            return
        
        try:
            stream = container.streams.video[0]
            
            # Get video properties
            total_frames = stream.frames or 10000  # Estimate if not available
            fps = float(stream.average_rate or stream.base_rate or 30)
            
            logger.debug(f"Video {video_id}: ~{total_frames} frames at {fps:.1f} fps")
            
            # Get target frame indices
            frame_indices = set(self._get_frame_indices(total_frames, fps))
            
            if not frame_indices:
                logger.warning(f"No frames to extract from {video_id}")
                return
            
            frame_idx = 0
            prev_frame_gray = None
            scene_threshold = 30.0
            
            for frame in container.decode(video=0):
                if frame_idx in frame_indices:
                    # Convert to PIL Image (RGB)
                    pil_image = frame.to_image()
                    
                    # Keyframe detection (scene change)
                    if self.config.strategy == SamplingStrategy.KEYFRAME and frame_idx > 0:
                        if prev_frame_gray is not None:
                            mean_diff = self._compute_frame_difference(pil_image, prev_frame_gray)
                            if mean_diff < scene_threshold:
                                prev_frame_gray = pil_image
                                frame_idx += 1
                                continue
                        prev_frame_gray = pil_image
                    
                    timestamp = frame_idx / fps
                    
                    yield FrameInfo(
                        video_path=video_path,
                        video_id=video_id,
                        frame_number=frame_idx,
                        timestamp=timestamp,
                        image=pil_image,
                    )
                
                frame_idx += 1
                
                # Stop if we've passed all target frames
                if frame_idx > max(frame_indices):
                    break
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_id}: {e}")
        finally:
            container.close()
    
    def extract_video(
        self,
        video_path: Path,
        video_id: str,
        save_frames: bool = True,
        show_progress: bool = False,
        frame_counter: Optional[list] = None,  # Thread-safe counter [count, lock]
    ) -> ExtractionResult:
        """
        Extract and optionally save frames from a video.
        
        Args:
            video_path: Path to video file
            video_id: Identifier for the video
            save_frames: Whether to save frames to disk
            show_progress: Show progress bar
            frame_counter: Optional [count, lock] for shared frame counting
            
        Returns:
            ExtractionResult with saved frame paths
        """
        if not video_path.exists():
            return ExtractionResult(
                video_path=video_path,
                video_id=video_id,
                success=False,
                error=f"Video file not found: {video_path}",
            )
        
        # Create output directory for this video
        video_output_dir = self.config.output_dir / video_id
        ensure_dir(video_output_dir)
        
        output_paths = []
        frame_count = 0
        
        try:
            # Get frame iterator
            frame_iter = self.iterate_frames(video_path, video_id)
            
            if show_progress:
                frame_iter = tqdm(frame_iter, desc=f"Extracting {video_id}", unit="frame", leave=False)
            
            for frame_info in frame_iter:
                frame_count += 1
                
                if save_frames:
                    # Save frame to disk using PIL
                    # Use {video_id}_{frame_number}.ext for global traceability
                    filename = f"{video_id}_{frame_info.frame_number:06d}.{self.config.image_format}"
                    output_path = video_output_dir / filename
                    
                    # Save with quality setting
                    save_kwargs = {}
                    if self.config.image_format in ("jpg", "jpeg"):
                        save_kwargs["quality"] = self.config.quality
                        save_kwargs["optimize"] = True
                        # PIL uses "JPEG" format name
                        frame_info.image.save(str(output_path), format="JPEG", **save_kwargs)
                    elif self.config.image_format == "webp":
                        save_kwargs["quality"] = self.config.quality
                        frame_info.image.save(str(output_path), format="WEBP", **save_kwargs)
                    elif self.config.image_format == "png":
                        save_kwargs["compress_level"] = 6
                        frame_info.image.save(str(output_path), format="PNG", **save_kwargs)
                    else:
                        frame_info.image.save(str(output_path))
                    
                    output_paths.append(output_path)
                    
                    # Update shared frame counter every 10 frames (reduce lock contention)
                    if frame_counter is not None and frame_count % 20 == 0:
                        with frame_counter[1]:  # Lock
                            frame_counter[0] += 20
            
            # Update remaining frames not covered by batch
            if frame_counter is not None:
                remaining = frame_count % 20
                if remaining > 0:
                    with frame_counter[1]:
                        frame_counter[0] += remaining
            
            logger.info(f"Extracted {frame_count} frames from {video_id}")
            
            return ExtractionResult(
                video_path=video_path,
                video_id=video_id,
                success=True,
                frame_count=frame_count,
                output_paths=output_paths,
                output_dir=video_output_dir,
            )
            
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return ExtractionResult(
                video_path=video_path,
                video_id=video_id,
                success=False,
                frame_count=frame_count,
                output_paths=output_paths,
                output_dir=video_output_dir,
                error=str(e),
            )
    
    # extract_batch removed - supervisor handles parallelism via multiple workers
