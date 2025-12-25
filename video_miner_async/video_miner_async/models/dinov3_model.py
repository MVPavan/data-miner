"""
DINOv3 Model Wrapper

Wrapper for Meta's DINOv3 (or DINOv2 fallback) for image embeddings.
Used for deduplication via cosine similarity.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from ..constants import DINO_MODELS, DINO_DEFAULT
from ..utils.device import resolve_device, get_model_device
from .base import BaseModel, create_batch_iterator

logger = logging.getLogger(__name__)


class DINOv3Model(BaseModel):
    """
    DINOv3/DINOv2 model wrapper for image embeddings.
    
    Used for computing image embeddings for deduplication.
    
    Example:
        >>> model = DINOv3Model()
        >>> model.load()
        >>> embeddings = model.get_embeddings(image_paths)
    """
    
    # Reference to centralized model registry
    MODELS = DINO_MODELS
    
    def __init__(
        self,
        model_id: str = None,
        device_map: str = "auto",
        use_fp16: bool = True,
    ):
        """
        Initialize DINO model wrapper.
        
        Args:
            model_id: HuggingFace model ID (defaults to DINO_DEFAULT)
            device_map: Device: 'auto' (multi-GPU), 'cuda', 'cuda:0', 'cpu'
            use_fp16: Use fp16 for memory efficiency
        """
        super().__init__()
        
        self.model_id = model_id or DINO_MODELS[DINO_DEFAULT]
        self.device_map = resolve_device(device_map)
        self.use_fp16 = use_fp16
    
    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return
        
        from transformers import AutoModel, AutoImageProcessor
        
        logger.info(f"Loading DINO model: {self.model_id}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            
            # Use HuggingFace device_map directly
            torch_dtype = torch.float16 if self.use_fp16 and self.device_map != "cpu" else None
            self.model = AutoModel.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
            )
            
            self.model.eval()
            self._loaded = True
            logger.info(f"Loaded: {self.model_id} on {self.device_map} (fp16: {self.use_fp16})")
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO model {self.model_id}: {e}")
    
    @torch.no_grad()
    def get_embeddings(
        self,
        images: list[Union[Path, str, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute embeddings for a list of images.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            batch_size: Batch size for inference
            show_progress: Show progress bar
            normalize: L2-normalize embeddings
            
        Returns:
            numpy array of shape (num_images, embedding_dim)
        """
        self._ensure_loaded()
        
        # Get device from model
        device = get_model_device(self.model)
        
        all_embeddings = []
        
        for start_idx, batch_images in create_batch_iterator(
            images, batch_size, show_progress, "Computing embeddings"
        ):
            # Load and process images
            pil_images = [self._load_image(img) for img in batch_images]
            
            inputs = self.processor(images=pil_images, return_tensors="pt")
            
            # Move inputs to model device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if self.use_fp16 and device.type != "cpu":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Get CLS token embeddings
            outputs = self.model(**inputs)
            
            # Use pooler output if available, else use CLS token from last hidden state
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0]
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        
        return embeddings
    
    def get_embedding(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """Get embedding for a single image."""
        return self.get_embeddings([image], normalize=normalize)[0]
