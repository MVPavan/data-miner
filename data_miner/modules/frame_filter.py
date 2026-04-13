"""
Frame Filter Module

Filters frames based on text prompts using SigLIP image-text similarity.
Orchestrates caching: only passed frames' embeddings are cached.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from ..config import FilterConfig
from ..logging import get_logger
from ..models.siglip_model import SigLIPModel
from ..utils.embedding_cache import build_cache_dir, load_cached_embeddings, save_embeddings
from ..utils.io import ensure_dir

logger = get_logger(__name__)


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

    Orchestrates batch-wise processing with embedding caching:
    1. Precompute text embeddings once
    2. For each batch: check cache -> compute missing -> score -> filter -> cache passed

    Example:
        >>> config = FilterConfig(
        ...     positive_prompts=["a glass door"],
        ...     threshold=0.25
        ... )
        >>> filter = FrameFilter(config)
        >>> result = filter.filter_frames(frame_paths)
    """

    def __init__(self, config: FilterConfig, device_map: str = "auto"):
        self.config = config
        self.model = SigLIPModel(
            model_id=config.model_id,
            device_map=device_map,
        )
        ensure_dir(config.output_dir)

        if not config.positive_prompts:
            raise ValueError("FilterConfig must have at least one positive_prompt")

    def _get_cache_dir(self, input_dir: Path) -> Optional[Path]:
        """Build cache directory for SigLIP embeddings."""
        if not self.config.cache_embeddings or input_dir is None:
            return None
        # e.g. siglip2-so400m-patch14-384 or extract short name
        model_name = self.model.model_id.split("/")[-1]
        return build_cache_dir(input_dir, model_name, "image")

    def _apply_filter(
        self,
        scores: np.ndarray,
        num_positive: int,
        num_negative: int,
        positive_prompts: list[str],
        negative_prompts: list[str],
        junk_prompts: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply filter thresholds to scores.

        Args:
            scores: (batch_size, num_prompts) similarity scores.
            num_positive: Number of positive prompts.
            num_negative: Number of negative prompts.
            positive_prompts: List of positive prompt strings.
            negative_prompts: List of negative prompt strings.
            junk_prompts: List of junk prompt strings.

        Returns:
            (passed_mask, pos_max) — boolean mask and max positive scores.
        """
        pos_scores = scores[:, :num_positive]
        pos_max = pos_scores.max(axis=1)
        pos_mask = pos_max > self.config.positive_thr
        neg_mask = np.ones(len(scores), dtype=bool)
        junk_mask = np.ones(len(scores), dtype=bool)

        if negative_prompts:
            neg_scores = scores[:, num_positive:num_positive + num_negative]
            neg_max = neg_scores.max(axis=1)
            pos_neg_margin = pos_max - neg_max
            neg_mask = (neg_max < self.config.negative_thr) & (pos_neg_margin > self.config.pos_neg_margin_thr)

        if junk_prompts:
            junk_scores = scores[:, num_positive + num_negative:]
            junk_max = junk_scores.max(axis=1)
            pos_junk_margin = pos_max - junk_max
            junk_mask = (junk_max < self.config.junk_thr) & (pos_junk_margin > self.config.pos_junk_margin_thr)

        return pos_mask & neg_mask & junk_mask, pos_max

    def filter_frames(
        self,
        frame_paths: list[Path],
        video_id: str = "unknown",
        show_progress: bool = True,
        copy_files: bool = True,
        input_dir: Optional[Path] = None,
    ) -> FilterResult:
        """
        Filter frames based on similarity to text prompts.

        Single-pass batch loop — only one batch of embeddings in RAM at a time.
        Per batch: load cached from disk → compute uncached on GPU → score → filter → cache passed.

        Args:
            frame_paths: List of frame image paths.
            video_id: Video identifier for organizing output.
            show_progress: Show progress bar.
            copy_files: Copy passing frames to output directory.
            input_dir: Frames directory for cache resolution. None disables caching.

        Returns:
            FilterResult with passing frames.
        """
        if not frame_paths:
            return FilterResult(total_frames=0, passed_frames=0)

        frame_paths = sorted(frame_paths)
        positive_prompts = self.config.positive_prompts
        negative_prompts = self.config.negative_prompts or []
        junk_prompts = self.config.junk_prompts or []
        all_prompts = positive_prompts + negative_prompts + junk_prompts
        num_positive = len(positive_prompts)
        num_negative = len(negative_prompts)

        logger.info(
            f"Filtering {len(frame_paths)} frames "
            f"({num_positive} positive, {num_negative} negative, {len(junk_prompts)} junk prompts)"
        )

        self.model.load()

        # 1. Precompute text embeddings once
        text_embeds = self.model.get_text_embeddings(all_prompts)

        # 2. Single-pass: process frame_paths in constant-size batches
        cache_dir = self._get_cache_dir(input_dir)
        use_cache = cache_dir and not self.config.ignore_cache
        batch_size = self.config.batch_size
        video_output_dir = self.config.output_dir / video_id
        ensure_dir(video_output_dir)
        filtered_frames = []

        iterator = range(0, len(frame_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering", unit="batch")

        for start in iterator:
            batch_paths = frame_paths[start:start + batch_size]

            # 2a. Load cached embeddings from disk for this batch
            cached: dict[str, np.ndarray] = {}
            uncached: list[Path] = list(batch_paths)
            if use_cache:
                cached, uncached = load_cached_embeddings(batch_paths, cache_dir)

            # 2b. Compute uncached on GPU
            newly_computed_stems: set[str] = set()
            if uncached:
                computed = self.model.get_image_embeddings(
                    uncached, batch_size=len(uncached), show_progress=False,
                )
                for i, img in enumerate(uncached):
                    cached[img.stem] = computed[i]
                    newly_computed_stems.add(img.stem)

            # 2c. Assemble batch embeddings in order and score
            batch_embeds = np.stack([cached[img.stem] for img in batch_paths])
            batch_scores = self.model.compute_similarity(batch_embeds, text_embeds)

            # 2d. Apply filter thresholds
            passed_mask, _ = self._apply_filter(
                batch_scores, num_positive, num_negative,
                positive_prompts, negative_prompts, junk_prompts,
            )

            # 2e. Cache only newly computed passed frames
            if cache_dir and newly_computed_stems and np.any(passed_mask):
                new_passed = [
                    (batch_paths[i], batch_embeds[i])
                    for i in np.where(passed_mask)[0]
                    if batch_paths[i].stem in newly_computed_stems
                ]
                if new_passed:
                    imgs, embs = zip(*new_passed)
                    save_embeddings(list(imgs), np.stack(embs), cache_dir)

            # 2f. Collect results and copy files
            for local_idx in np.where(passed_mask)[0]:
                frame_path = batch_paths[local_idx]
                frame_scores = batch_scores[local_idx]

                pos_frame_scores = frame_scores[:num_positive]
                best_class_idx = int(pos_frame_scores.argmax())
                best_class = positive_prompts[best_class_idx]
                max_score = float(pos_frame_scores[best_class_idx])

                output_path = video_output_dir / frame_path.name
                if copy_files and frame_path != output_path:
                    shutil.copy2(frame_path, output_path)

                try:
                    frame_number = int(frame_path.stem.split("_")[-1])
                except (ValueError, IndexError):
                    frame_number = start + int(local_idx)

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
        frame_groups: dict[str, list[Path]],
        show_progress: bool = True,
        on_complete: callable = None,
        input_dir: Optional[Path] = None,
    ) -> dict[str, FilterResult]:
        """
        Filter frames from multiple videos.

        Args:
            frame_groups: Dict mapping video_id to frame paths.
            show_progress: Show progress bar.
            on_complete: Optional callback called after each video with (video_id, FilterResult).
            input_dir: Frames directory for cache resolution.

        Returns:
            Dict mapping video_id to FilterResult.
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
                input_dir=input_dir,
            )
            results[video_id] = result

            if on_complete:
                try:
                    on_complete(video_id, result)
                except Exception as e:
                    logger.warning(f"Callback error for {video_id}: {e}")

        total = sum(r.total_frames for r in results.values())
        passed = sum(r.passed_frames for r in results.values())
        logger.info(f"Batch filter complete: {passed}/{total} frames passed ({passed/max(total,1):.1%})")

        return results

    def unload_model(self) -> None:
        """Unload model to free memory."""
        self.model.unload()
