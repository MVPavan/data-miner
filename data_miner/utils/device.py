"""
Device Management Utilities

Handles device resolution and GPU memory management.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(device_map: str = "auto") -> str:
    """
    Resolve device_map for HuggingFace model loading.
    
    If device_map is 'auto' and multiple GPUs available, returns 'auto' 
    for HuggingFace multi-GPU distribution. Otherwise resolves to specific device.
    
    Args:
        device_map: 'auto', 'cuda', 'cuda:N', 'cpu'
        
    Returns:
        str: Device string for HuggingFace device_map parameter
    """
    num_gpus = torch.cuda.device_count()
    
    if device_map == "auto":
        if num_gpus > 1:
            logger.info(f"Multi-GPU detected ({num_gpus} GPUs), using device_map='auto'")
            return "auto"
        elif num_gpus == 1:
            return "cuda"
        else:
            return "cpu"
    else:
        return device_map


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model from its parameters."""
    return next(model.parameters()).device


def clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared")


def get_gpu_memory_info() -> Optional[dict]:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return None
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    info = get_gpu_memory_info()
    if info:
        msg = f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, Reserved: {info['reserved_gb']:.2f}GB"
        if prefix:
            msg = f"{prefix}: {msg}"
        logger.debug(msg)
