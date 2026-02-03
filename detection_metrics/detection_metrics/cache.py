"""Cache manager for evaluation results."""

import hashlib
import pickle
from pathlib import Path
from typing import Optional

from detection_metrics.evaluator import EvaluationResult
from detection_metrics.logging import logger


class CacheManager:
    """Manages caching of evaluation results using pickle files."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize CacheManager.
        
        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def generate_cache_key(
        model_name: str,
        gt_path: Path,
        iou_threshold: float,
        conf_threshold: float,
        class_ids: Optional[list] = None
    ) -> str:
        """
        Generate a unique cache key based on evaluation parameters.
        
        Args:
            model_name: Name of the model.
            gt_path: Path to ground truth file.
            iou_threshold: IoU threshold used.
            conf_threshold: Confidence threshold used.
            class_ids: Optional list of class IDs evaluated.
        
        Returns:
            Hash string for the cache key.
        """
        key_parts = [
            model_name,
            str(gt_path),
            f"iou_{iou_threshold}",
            f"conf_{conf_threshold}",
        ]
        if class_ids:
            key_parts.append(f"classes_{'_'.join(map(str, sorted(class_ids)))}")
        
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def exists(self, cache_key: str) -> bool:
        """Check if a cached result exists."""
        return self.get_cache_path(cache_key).exists()
    
    def load(self, cache_key: str) -> Optional[EvaluationResult]:
        """
        Load cached evaluation result.
        
        Args:
            cache_key: Cache key to load.
        
        Returns:
            EvaluationResult if found, None otherwise.
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"Loaded cached result: {cache_key}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def save(self, cache_key: str, result: EvaluationResult) -> None:
        """
        Save evaluation result to cache.
        
        Args:
            cache_key: Cache key for the result.
            result: EvaluationResult to cache.
        """
        cache_path = self.get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Saved result to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def clear(self) -> int:
        """
        Clear all cached results.
        
        Returns:
            Number of files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cached results")
        return count
