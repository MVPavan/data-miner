"""
Embedding Cache Utilities

Shared caching logic for per-file .npy embedding storage.
Used by both SigLIP2 filter and DINOv3 deduplicator.

Cache is stem-name based: each image's embedding is stored as {stem}.npy.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from ..logging import get_logger

logger = get_logger(__name__)


def build_cache_dir(input_dir: Path, model_name: str, stage_name: str) -> Path:
    """Build cache directory path.

    Returns: {input_dir.parent}/{model_name}_{stage_name}_embeddings/
    """
    return input_dir.parent / f"{model_name}_{stage_name}_embeddings"


def load_cached_embeddings(
    images: list[Path],
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    """Load cached .npy embeddings by stem name.

    Args:
        images: List of image paths (stems used as cache keys).
        cache_dir: Directory containing .npy files.

    Returns:
        (cached_dict, missing_images) where cached_dict maps stem -> embedding,
        and missing_images is the list of image paths without cache.
    """
    if not cache_dir.exists():
        return {}, list(images)

    cached = {}
    missing = []

    for img in images:
        cache_file = cache_dir / f"{img.stem}.npy"
        if cache_file.exists():
            cached[img.stem] = np.load(cache_file)
        else:
            missing.append(img)

    return cached, missing


def save_embeddings(
    images: list[Path],
    embeddings: np.ndarray,
    cache_dir: Path,
) -> None:
    """Save per-image .npy embedding files. Overwrites existing.

    Args:
        images: Image paths (stems used as filenames).
        embeddings: Array of shape (len(images), dim).
        cache_dir: Target directory for .npy files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        np.save(cache_dir / f"{img.stem}.npy", embeddings[i])


def assemble_embeddings(
    images: list[Path],
    cached: dict[str, np.ndarray],
    computed_images: list[Path],
    computed_embeddings: np.ndarray,
) -> np.ndarray:
    """Assemble final embedding array from cached + computed, ordered by images list.

    Args:
        images: Full ordered list of image paths.
        cached: Dict of stem -> embedding for cached items.
        computed_images: List of image paths that were computed.
        computed_embeddings: Embeddings for computed_images.

    Returns:
        (len(images), dim) numpy array in same order as images.
    """
    # Build lookup for computed embeddings
    computed_lookup = {img.stem: computed_embeddings[i] for i, img in enumerate(computed_images)}

    # Determine dim from either source
    if computed_lookup:
        dim = next(iter(computed_lookup.values())).shape[0]
    elif cached:
        dim = next(iter(cached.values())).shape[0]
    else:
        raise ValueError("No embeddings to assemble")

    result = np.zeros((len(images), dim), dtype=np.float32)
    for i, img in enumerate(images):
        stem = img.stem
        if stem in computed_lookup:
            result[i] = computed_lookup[stem]
        elif stem in cached:
            result[i] = cached[stem]

    return result


def get_embeddings_cached(
    images: list[Path],
    compute_fn: Callable[[list[Path]], np.ndarray],
    batch_size: int,
    cache_dir: Optional[Path] = None,
    ignore_cache: bool = False,
    show_progress: bool = True,
    progress_desc: str = "Computing embeddings",
) -> np.ndarray:
    """Full cache-aware embedding pipeline.

    Loads cached embeddings by stem name, computes missing ones in batches,
    saves each batch immediately for crash resilience, and assembles results.

    Used by deduplicator where ALL embeddings are needed upfront for FAISS.

    Args:
        images: List of image paths.
        compute_fn: Callable that takes list[Path] and returns (N, dim) numpy array.
        batch_size: Batch size for compute_fn.
        cache_dir: Cache directory. None disables caching.
        ignore_cache: If True, recompute all and overwrite cache.
        show_progress: Show progress bar.
        progress_desc: Description for progress bar.

    Returns:
        (len(images), dim) numpy array of embeddings in same order as images.
    """
    # Try loading from cache
    if cache_dir and not ignore_cache:
        cached, missing_images = load_cached_embeddings(images, cache_dir)
        if not missing_images:
            logger.info(f"Loaded all {len(images)} embeddings from cache")
            return assemble_embeddings(images, cached, [], np.empty((0,)))
        logger.info(
            f"Cache hit: {len(cached)}/{len(images)}, "
            f"computing {len(missing_images)} missing"
        )
    else:
        cached = {}
        missing_images = list(images)

    # Compute in batches, saving each batch to cache immediately
    all_computed = []
    all_computed_images = []
    iterator = range(0, len(missing_images), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc, unit="batch", leave=False)

    for start in iterator:
        batch_imgs = missing_images[start : start + batch_size]
        batch_embeds = compute_fn(batch_imgs)
        all_computed.append(batch_embeds)
        all_computed_images.extend(batch_imgs)

        # Save batch to cache immediately
        if cache_dir:
            save_embeddings(batch_imgs, batch_embeds, cache_dir)

    computed_embeddings = np.vstack(all_computed) if all_computed else np.empty((0,))

    return assemble_embeddings(images, cached, all_computed_images, computed_embeddings)
