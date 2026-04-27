"""Embedding extraction pipelines + LanceDB-backed store.

The store is a centralized local-disk LanceDB root that all data_miner
datasets write into as sibling tables (or subdirs). Default layout::

    /mnt/data/data_miner_lance/
        laion_good2/embeddings.lance/
        coco/embeddings.lance/
        ...

The schema reserves columns for SigLIP2 and DINOv3 embeddings so later
runs can fill the DINOv3 slot without rewriting existing SigLIP2 data.
"""

from .lance_store import (
    DINOV3_DIM,
    DINOV3_DIMS,
    LANCE_ROOT_DEFAULT,
    LanceEmbeddingWriter,
    SIGLIP2_DIM,
    SIGLIP2_DIMS,
    TABLE_DEFAULT,
    build_errors_schema,
    build_schema,
)

__all__ = [
    "SIGLIP2_DIM",
    "SIGLIP2_DIMS",
    "DINOV3_DIM",
    "DINOV3_DIMS",
    "LANCE_ROOT_DEFAULT",
    "TABLE_DEFAULT",
    "build_schema",
    "build_errors_schema",
    "LanceEmbeddingWriter",
]
