"""
Frame Filter Module

Filters frames based on text prompts using SigLIP image-text similarity.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import numpy as np

from ..config import FilterConfig
from ..models.siglip_model import SigLIPModel
from ..utils.io import ensure_dir

logger = logging.getLogger(__name__)

positive_prompts = [
    "a glass door",
    "a french door",
    "a patio door",
    "a french patio door",
    "a photo of a glass door",
    "a glass entrance door",
    "a glass entrance door with handles",  # "Handle" is the #1 feature distinguishing doors from windows
    "commercial double glass doors",
    "a sliding glass door entrance",
    "a storefront entrance with a glass door",
    "a building entrance with a glass door",
    "framed glass door clearly visible",
    "a push bar on a glass door"           # Very specific to commercial doors (high precision)
]
negative_prompts = [
    # 1. The "Glass Trap" (Structure)
    "a glass wall", 
    "a fixed glass window", 
    "a large display window", 
    "a glass curtain wall",
    "a mirror", 
    "a reflective surface",
    "a shower door",           # Common false positive for "glass door"
    "a glass partition",

    # 2. The "Ceiling Trap" (Angle)
    "a skylight", 
    "a glass ceiling", 
    "an atrium roof",

    # 3. The "YouTube Trap" (Digital Artifacts - CRITICAL)
    "a split screen video",
    "text overlay on screen",
    "a video game screenshot", # Prevents confusion with realistic game graphics
    "a youtube thumbnail with text",
    "a close up of a security camera monitor",
    
    # 4. The "Quality Trap"
    "blurry out of focus image",
    "a solid color background",
    "dark grainy night shot",
    "a plain empty scene",
    "an empty background"

    # 5. Closeup Trap
    "extreme close up of glass door",
    "macro shot of a door handle",
    "zoomed in detail of door handle",
    "a close up texture of glass",
    "cropped view of metal frame",
    "partial view of door edge",
    "a photo showing only door handle",

]
DOOR_CLASSES = positive_prompts + negative_prompts
@dataclass
class FilteredFrame:
    """A frame that passed the filter."""
    source_path: Path
    output_path: Path
    video_id: str
    frame_number: int
    best_class: str
    score: float
    all_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of filtering operation."""
    total_frames: int
    passed_frames: int
    filtered_frames: list[FilteredFrame] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        return self.passed_frames / max(self.total_frames, 1)


class FrameFilter:
    """
    Filter frames based on text prompts using SigLIP.
    
    Computes similarity between frames and text prompts,
    keeping frames that exceed the similarity threshold.
    
    Example:
        >>> config = FilterConfig(threshold=0.3)
        >>> filter = FrameFilter(config)
        >>> result = filter.filter_frames(frame_paths, ["glass door", "window"])
    """
    
    def __init__(self, config: FilterConfig, device_map: str = "auto"):
        """
        Initialize frame filter.
        
        Args:
            config: Filter configuration
            device_map: Device: 'auto', 'cuda', 'cuda:0', 'cpu'
        """
        self.config = config
        self.model = SigLIPModel(
            model_id=config.model_id,
            device_map=device_map,
        )
        ensure_dir(config.output_dir)
    
    def filter_frames(
        self,
        frame_paths: list[Path],
        classes: list[str],
        video_id: str = "unknown",
        show_progress: bool = True,
        copy_files: bool = True,
    ) -> FilterResult:
        """
        Filter frames based on similarity to text prompts.
        
        Args:
            frame_paths: List of frame image paths
            classes: List of class names/text prompts
            video_id: Video identifier for organizing output
            show_progress: Show progress bar
            copy_files: Copy passing frames to output directory
            
        Returns:
            FilterResult with passing frames
        """
        #FIXME: overrwiting classes here, later fix it from config
        classes = DOOR_CLASSES
        
        if not frame_paths:
            return FilterResult(total_frames=0, passed_frames=0)
        
        logger.info(f"Filtering {len(frame_paths)} frames against {len(classes)} classes")
        
        # Load model
        self.model.load()
        
        # Compute similarities in batches
        scores = self.model.compute_similarity(
            images=frame_paths,
            texts=classes,
            batch_size=self.config.batch_size,
            show_progress=show_progress,
        )
        
        # Filter frames above threshold
        filtered_frames = []
        video_output_dir = self.config.output_dir / video_id
        ensure_dir(video_output_dir)
        
        iterator = enumerate(frame_paths)
        if show_progress:
            pass
            # iterator = tqdm(list(iterator), desc="Filtering frames", unit="frame", leave=False)

        # # NOTE: original processing where all classes should be valid
        # valid_frames_ids = np.where(np.all(scores > self.config.threshold, axis=1))[0].tolist()

        # Dual threshold parameters, positve and negative classes
        ABSOLUTE_THRESHOLD = 0.25  # Score must be at least this (0.0-1.0)
        SAFETY_MARGIN = 0.05  # Pos must beat Neg by this amount
        pos_scores = scores[:, :len(positive_prompts)]
        neg_scores = scores[:, len(positive_prompts):]
        pos_scores_max = pos_scores.max(axis=1, keepdims=True)
        neg_scores_max = neg_scores.max(axis=1, keepdims=True)
        margined_scores = pos_scores_max - neg_scores_max
        # Mask of frames that pass both absolute and margin thresholds
        mask = (pos_scores_max > ABSOLUTE_THRESHOLD) & (margined_scores > SAFETY_MARGIN)
        valid_frames_ids = np.where(mask[:,0])[0].tolist()

        # NOTE: vectorized processing where all classes should be valid
        for idx in valid_frames_ids:
            frame_path = frame_paths[idx]
            frame_scores = scores[idx]
            max_score = float(frame_scores.max())
            best_class_idx = int(frame_scores.argmax())
            best_class = classes[best_class_idx]
            
            # Determine output path
            output_path = video_output_dir / frame_path.name
            
            if copy_files and frame_path != output_path:
                shutil.copy2(frame_path, output_path)
            
            # Extract frame number from filename
            try:
                frame_number = int(frame_path.stem.split("_")[-1])
            except (ValueError, IndexError):
                frame_number = idx
            
            filtered_frames.append(FilteredFrame(
                source_path=frame_path,
                output_path=output_path,
                video_id=video_id,
                frame_number=frame_number,
                best_class=best_class,
                score=max_score,
                all_scores={classes[i]: float(frame_scores[i]) for i in range(len(classes))},
            ))
        
        result = FilterResult(
            total_frames=len(frame_paths),
            passed_frames=len(filtered_frames),
            filtered_frames=filtered_frames,
        )
        
        logger.info(
            f"Filter complete: {result.passed_frames}/{result.total_frames} frames passed "
            f"({result.pass_rate:.1%})"
        )
        
        return result
    
    def filter_batch(
        self,
        frame_groups: dict[str, list[Path]],  # video_id -> frame_paths
        classes: list[str],
        show_progress: bool = True,
        on_complete: callable = None,
    ) -> dict[str, FilterResult]:
        """
        Filter frames from multiple videos.
        
        Args:
            frame_groups: Dict mapping video_id to frame paths
            classes: List of class names/text prompts
            show_progress: Show progress bar
            on_complete: Optional callback called after each video with (video_id, FilterResult)
            
        Returns:
            Dict mapping video_id to FilterResult
        """
        results = {}
        
        iterator = frame_groups.items()
        if show_progress:
            iterator = tqdm(list(iterator), desc="Filtering videos", unit="video")
        
        for video_id, frame_paths in iterator:
            result = self.filter_frames(
                frame_paths=frame_paths,
                classes=classes,
                video_id=video_id,
                show_progress=True,  # Inner progress bar (leave=False)
                copy_files=True,
            )
            results[video_id] = result
            
            # Fire callback after each video
            if on_complete:
                try:
                    on_complete(video_id, result)
                except Exception as e:
                    logger.warning(f"Callback error for {video_id}: {e}")
        
        # Summary
        total = sum(r.total_frames for r in results.values())
        passed = sum(r.passed_frames for r in results.values())
        logger.info(f"Batch filter complete: {passed}/{total} frames passed ({passed/max(total,1):.1%})")
        
        return results
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        self.model.unload()
