"""
Deduplication Module

Removes duplicate/similar frames using embeddings and cosine similarity.
Supports DINOv3 (default) or SigLIP2 (memory-efficient mode).
Uses FAISS for scalable cross-video deduplication.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..config import DeduplicationConfig, DedupModelType
from ..models.dinov3_model import DINOv3Model
from ..models.siglip_model import SigLIPModel
from ..utils.io import ensure_dir
from ..utils.device import get_model_device

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    total_frames: int
    unique_frames: int
    duplicates_removed: int
    unique_paths: list[Path] = field(default_factory=list)
    model_used: str = "dino"  # Track which model was used
    
    @property
    def dedup_rate(self) -> float:
        """Percentage of frames removed as duplicates."""
        return self.duplicates_removed / max(self.total_frames, 1)


class Deduplicator:
    """
    Frame deduplication using image embeddings.
    
    Supports two modes:
    - DINOv3/DINOv2: Best quality (default)
    - SigLIP2: Memory-efficient (reuses filter model)
    
    Example:
        >>> config = DeduplicationConfig(threshold=0.90)
        >>> dedup = Deduplicator(config)
        >>> result = dedup.deduplicate(frame_paths)
    """
    
    def __init__(
        self, 
        config: DeduplicationConfig, 
        device_map: str = "auto",
        use_fp16: bool = True,
        siglip_model: Optional[SigLIPModel] = None,  # Reuse from filter stage
    ):
        """
        Initialize deduplicator.
        
        Args:
            config: Deduplication configuration
            device_map: Device: 'auto', 'cuda', 'cuda:0', 'cpu'
            use_fp16: Use fp16 for memory efficiency
            siglip_model: Optional pre-loaded SigLIP model (for memory reuse)
        """
        self.config = config
        self.device_map = device_map
        self.use_fp16 = use_fp16
        self._model = None
        self._external_siglip = siglip_model
        
        # Initialize model based on config
        if config.model_type == DedupModelType.SIGLIP:
            if siglip_model is not None:
                self._model = siglip_model
                logger.info("Deduplicator: Reusing SigLIP model from filter stage")
            else:
                self._model = SigLIPModel(device_map=device_map)
                logger.info("Deduplicator: Using SigLIP2 for embeddings")
        else:
            self._model = DINOv3Model(
                model_id=config.dino_model_id,
                device_map=device_map,
                use_fp16=use_fp16,
            )
            logger.info("Deduplicator: Using DINO for embeddings")
        
        ensure_dir(config.output_dir)
    
    @property
    def model(self) -> Union[DINOv3Model, SigLIPModel]:
        """Get the embedding model."""
        return self._model
    
    def _get_embeddings(
        self,
        images: list[Path],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Get embeddings using the configured model."""
        self._model.load()
        
        if self.config.model_type == DedupModelType.SIGLIP:
            # SigLIP: Use image features only (no text)
            return self._get_siglip_embeddings(images, batch_size, show_progress)
        else:
            # DINO: Use standard embeddings
            return self._model.get_embeddings(
                images=images,
                batch_size=batch_size,
                show_progress=show_progress,
                normalize=True,
            )
    
    def _get_siglip_embeddings(
        self,
        images: list[Path],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Extract image embeddings from SigLIP (without text)."""
        import torch
        from PIL import Image as PILImage
        
        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing SigLIP embeddings", unit="batch", leave=False)
        
        for start_idx in iterator:
            batch_images = images[start_idx:start_idx + batch_size]
            pil_images = [self._model._load_image(img) for img in batch_images]
            
            image_inputs = self._model.processor(
                images=pil_images,
                return_tensors="pt",
            )
            device = get_model_device(self._model.model)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items() if k != "input_ids"}
            
            with torch.no_grad():
                image_features = self._model.model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _greedy_dedup(
        self,
        similarity_matrix: np.ndarray,
        threshold: float,
    ) -> list[int]:
        """
        Greedy deduplication: keep first occurrence, remove similar ones.
        
        Args:
            similarity_matrix: NxN symmetric similarity matrix
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of indices to keep
        """
        n = len(similarity_matrix)
        keep = []
        removed = set()
        
        for i in range(n):
            if i in removed:
                continue
            
            keep.append(i)
            
            # Mark all similar images as removed
            for j in range(i + 1, n):
                if j not in removed and similarity_matrix[i, j] >= threshold:
                    removed.add(j)
        
        return keep
    
    def _faiss_dedup(
        self,
        embeddings: np.ndarray,
        threshold: float,
        k_neighbors: int = 50,
    ) -> list[int]:
        """
        FAISS-based deduplication for large-scale datasets.
        
        Uses approximate nearest neighbor search which is O(N log N) instead of O(N²).
        
        Args:
            embeddings: (N, D) normalized embeddings
            threshold: Cosine similarity threshold for duplicates
            k_neighbors: Max neighbors to check per embedding
            
        Returns:
            List of indices to keep
        """
        n, d = embeddings.shape
        
        # Ensure embeddings are float32 and normalized for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index (Inner Product = Cosine Similarity for normalized vectors)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        
        # Search for k nearest neighbors for each embedding
        k = min(k_neighbors, n)
        similarities, indices = index.search(embeddings, k)
        
        # Greedy selection using neighbor info
        keep = []
        removed = set()
        
        for i in range(n):
            if i in removed:
                continue
            
            keep.append(i)
            
            # Vectorized neighbor processing
            neighbor_mask = (
                (indices[i] != i) & 
                (similarities[i] >= threshold) &
                np.array([idx not in removed for idx in indices[i]])
            )
            
            # Add all qualifying neighbors to removed set
            new_removed = indices[i][neighbor_mask]
            removed.update(new_removed)
            
        logger.debug(f"FAISS dedup: {n} → {len(keep)} frames (k={k})")
        return keep
    
    def deduplicate(
        self,
        frame_paths: list[Path],
        copy_files: bool = True,
        show_progress: bool = True,
    ) -> DeduplicationResult:
        """
        Deduplicate frames based on visual similarity.
        
        Args:
            frame_paths: List of frame image paths
            copy_files: Copy unique frames to output directory
            show_progress: Show progress bar
            
        Returns:
            DeduplicationResult with unique frame paths
        """
        if not frame_paths:
            return DeduplicationResult(
                total_frames=0,
                unique_frames=0,
                duplicates_removed=0,
                model_used=self.config.model_type,
            )
        
        if len(frame_paths) == 1:
            # Single frame, nothing to deduplicate
            output_path = self.config.output_dir / frame_paths[0].name
            if copy_files:
                ensure_dir(self.config.output_dir)
                shutil.copy2(frame_paths[0], output_path)
            return DeduplicationResult(
                total_frames=1,
                unique_frames=1,
                duplicates_removed=0,
                unique_paths=[output_path],
                model_used=self.config.model_type,
            )
        
        logger.info(f"Deduplicating {len(frame_paths)} frames using {self.config.model_type.upper()} + FAISS")
        
        # Get embeddings using configured model
        embeddings = self._get_embeddings(
            images=frame_paths,
            batch_size=self.config.batch_size,
            show_progress=show_progress,
        )
        
        # FAISS-based deduplication (scalable to 50k+ frames)
        keep_indices = self._faiss_dedup(
            embeddings, 
            self.config.threshold,
            k_neighbors=self.config.k_neighbors,
        )
        
        # Copy unique frames to output
        unique_paths = []
        
        if copy_files:
            ensure_dir(self.config.output_dir)
            
            iterator = keep_indices
            if show_progress:
                iterator = tqdm(iterator, desc="Copying unique frames", unit="frame")
            
            for idx in iterator:
                src_path = frame_paths[idx]
                dst_path = self.config.output_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                unique_paths.append(dst_path)
        else:
            unique_paths = [frame_paths[idx] for idx in keep_indices]
        
        result = DeduplicationResult(
            total_frames=len(frame_paths),
            unique_frames=len(keep_indices),
            duplicates_removed=len(frame_paths) - len(keep_indices),
            unique_paths=unique_paths,
            model_used=self.config.model_type,
        )
        
        logger.info(
            f"Deduplication complete: {result.unique_frames}/{result.total_frames} unique "
            f"({result.dedup_rate:.1%} duplicates removed)"
        )
        
        return result
    
    def deduplicate_cross_video(
        self,
        frame_groups: dict[str, list[Path]],  # video_id -> frame_paths
        show_progress: bool = True,
    ) -> DeduplicationResult:
        """
        Deduplicate frames across multiple videos (two-phase FAISS).
        
        Phase 1: Per-video dedup (removes temporal duplicates)
        Phase 2: Cross-video dedup (removes duplicates across videos)
        
        Args:
            frame_groups: Dict mapping video_id to frame paths
            show_progress: Show progress bar
            
        Returns:
            DeduplicationResult with unique frames across all videos
        """
        if not frame_groups:
            return DeduplicationResult(
                total_frames=0,
                unique_frames=0,
                duplicates_removed=0,
            )
        
        total_input = sum(len(paths) for paths in frame_groups.values())
        logger.info(f"Two-phase FAISS dedup: {total_input} frames from {len(frame_groups)} videos")
        
        # Phase 1: Per-video deduplication
        logger.info("Phase 1: Per-video deduplication...")
        phase1_paths = []
        phase1_removed = 0
        
        video_iterator = frame_groups.items()
        if show_progress:
            video_iterator = tqdm(list(video_iterator), desc="Per-video dedup", unit="video")
        
        for video_id, paths in video_iterator:
            if len(paths) <= 1:
                phase1_paths.extend(paths)
                continue
            
            # Get embeddings for this video
            embeddings = self._get_embeddings(
                images=paths,
                batch_size=self.config.batch_size,
                show_progress=False,
            )
            
            # FAISS dedup within video
            keep_indices = self._faiss_dedup(
                embeddings, 
                self.config.threshold,
                k_neighbors=self.config.k_neighbors,
            )
            
            # Keep only unique frames
            for idx in keep_indices:
                phase1_paths.append(paths[idx])
            
            phase1_removed += len(paths) - len(keep_indices)
        
        logger.info(f"Phase 1 complete: {total_input} → {len(phase1_paths)} frames ({phase1_removed} removed)")
        
        if len(phase1_paths) <= 1:
            # Nothing left to dedupe
            if phase1_paths:
                ensure_dir(self.config.output_dir)
                output_path = self.config.output_dir / phase1_paths[0].name
                shutil.copy2(phase1_paths[0], output_path)
                return DeduplicationResult(
                    total_frames=total_input,
                    unique_frames=1,
                    duplicates_removed=total_input - 1,
                    unique_paths=[output_path],
                    model_used=self.config.model_type,
                )
            return DeduplicationResult(
                total_frames=total_input,
                unique_frames=0,
                duplicates_removed=total_input,
            )
        
        # Phase 2: Cross-video deduplication
        logger.info(f"Phase 2: Cross-video deduplication ({len(phase1_paths)} frames)...")
        
        # Get embeddings for all remaining frames
        embeddings = self._get_embeddings(
            images=phase1_paths,
            batch_size=self.config.batch_size,
            show_progress=show_progress,
        )
        
        # FAISS dedup across all
        keep_indices = self._faiss_dedup(
            embeddings, 
            self.config.threshold,
            k_neighbors=self.config.k_neighbors,
        )
        
        # Copy unique frames to output
        ensure_dir(self.config.output_dir)
        unique_paths = []
        
        copy_iterator = keep_indices
        if show_progress:
            copy_iterator = tqdm(keep_indices, desc="Copying unique frames", unit="frame")
        
        for idx in copy_iterator:
            src_path = phase1_paths[idx]
            dst_path = self.config.output_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            unique_paths.append(dst_path)
        
        phase2_removed = len(phase1_paths) - len(keep_indices)
        total_removed = total_input - len(keep_indices)
        
        result = DeduplicationResult(
            total_frames=total_input,
            unique_frames=len(keep_indices),
            duplicates_removed=total_removed,
            unique_paths=unique_paths,
            model_used=self.config.model_type,
        )
        
        logger.info(
            f"Two-phase dedup complete: {result.unique_frames}/{result.total_frames} unique "
            f"({result.dedup_rate:.1%} removed, Phase1: {phase1_removed}, Phase2: {phase2_removed})"
        )
        
        return result
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        self.model.unload()
