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
    
    Supports:
    - Positive-only mode: frames pass if max positive score > threshold
    - Positive + Negative mode: positive must also beat negative by margin
    
    Example:
        >>> config = FilterConfig(
        ...     positive_prompts=["a glass door"],
        ...     threshold=0.25
        ... )
        >>> filter = FrameFilter(config)
        >>> result = filter.filter_frames(frame_paths)
    """
    
    def __init__(self, config: FilterConfig, device_map: str = "auto"):
        """
        Initialize frame filter.
        
        Args:
            config: Filter configuration with prompts and thresholds
            device_map: Device: 'auto', 'cuda', 'cuda:0', 'cpu'
        """
        self.config = config
        self.model = SigLIPModel(
            model_id=config.model_id,
            device_map=device_map,
        )
        ensure_dir(config.output_dir)
        
        # Validate config
        if not config.positive_prompts:
            raise ValueError("FilterConfig must have at least one positive_prompt")
    
    def filter_frames(
        self,
        frame_paths: list[Path],
        video_id: str = "unknown",
        show_progress: bool = True,
        copy_files: bool = True,
    ) -> FilterResult:
        """
        Filter frames based on similarity to text prompts.
        
        Args:
            frame_paths: List of frame image paths
            video_id: Video identifier for organizing output
            show_progress: Show progress bar
            copy_files: Copy passing frames to output directory
            
        Returns:
            FilterResult with passing frames
        """
        if not frame_paths:
            return FilterResult(total_frames=0, passed_frames=0)
        
        positive_prompts = self.config.positive_prompts
        negative_prompts = self.config.negative_prompts or []
        all_prompts = positive_prompts + negative_prompts
        
        logger.info(
            f"Filtering {len(frame_paths)} frames "
            f"({len(positive_prompts)} positive, {len(negative_prompts)} negative prompts)"
        )
        
        # Load model
        self.model.load()
        
        # Compute similarities in batches
        scores = self.model.compute_similarity(
            images=frame_paths,
            texts=all_prompts,
            batch_size=self.config.batch_size,
            show_progress=show_progress,
        )
        
        # Split scores into positive and negative
        num_positive = len(positive_prompts)
        pos_scores = scores[:, :num_positive]
        pos_max = pos_scores.max(axis=1)
        
        # Determine which frames pass
        if negative_prompts:
            # Positive + Negative mode: positive must beat negative by margin
            neg_scores = scores[:, num_positive:]
            neg_max = neg_scores.max(axis=1)
            margin = pos_max - neg_max
            mask = (pos_max > self.config.threshold) & (margin > self.config.margin_threshold)
        else:
            # Positive-only mode: just check threshold
            mask = pos_max > self.config.threshold
        
        valid_frame_ids = np.where(mask)[0].tolist()
        
        # Process passing frames
        filtered_frames = []
        video_output_dir = self.config.output_dir / video_id
        ensure_dir(video_output_dir)
        
        for idx in valid_frame_ids:
            frame_path = frame_paths[idx]
            frame_scores = scores[idx]
            
            # Best match among positive prompts only
            pos_frame_scores = frame_scores[:num_positive]
            best_class_idx = int(pos_frame_scores.argmax())
            best_class = positive_prompts[best_class_idx]
            max_score = float(pos_frame_scores[best_class_idx])
            
            # Output path
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
                all_scores={all_prompts[i]: float(frame_scores[i]) for i in range(len(all_prompts))},
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
        show_progress: bool = True,
        on_complete: callable = None,
    ) -> dict[str, FilterResult]:
        """
        Filter frames from multiple videos.
        
        Args:
            frame_groups: Dict mapping video_id to frame paths
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
                video_id=video_id,
                show_progress=True,
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

