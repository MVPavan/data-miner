"""SigLIP2 prompt-scoring pass over YOLO-labeled datasets.

Reuses image embeddings already written to LanceDB (see
``data_miner.embeddings``); this module only encodes text prompts and
runs the pos/neg/junk threshold rule from
``data_miner.modules.frame_filter``.
"""

from .config import ClassPrompts, Thresholds, ValidationConfig, load_config

__all__ = [
    "ClassPrompts",
    "Thresholds",
    "ValidationConfig",
    "load_config",
]
