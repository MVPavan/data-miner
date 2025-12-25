"""
SigLIP Model Wrapper

Wrapper for Google's SigLIP model for image-text similarity scoring.
Used for filtering frames based on text prompts/class names.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from ..constants import SIGLIP2_MODELS, SIGLIP2_DEFAULT
from ..utils.device import resolve_device, get_model_device
from .base import BaseModel, create_batch_iterator

logger = logging.getLogger(__name__)


class SigLIPModel(BaseModel):
    """
    SigLIP model wrapper for image-text similarity.
    
    Uses Google's SigLIP (Sigmoid Loss for Language-Image Pre-training)
    for computing similarity scores between images and text prompts.
    
    Example:
        >>> model = SigLIPModel()
        >>> model.load()
        >>> scores = model.compute_similarity(images, ["glass door", "window"])
    """
    
    def __init__(
        self,
        model_id: str = None,
        device_map: str = "auto",
    ):
        """
        Initialize SigLIP model wrapper.
        
        Args:
            model_id: HuggingFace model ID (defaults to SIGLIP2_SO400M)
            device_map: Device: 'auto', 'cuda', 'cuda:0', 'cpu'
        """
        super().__init__()
        self.model_id = model_id or SIGLIP2_MODELS[SIGLIP2_DEFAULT]
        self.device_map = resolve_device(device_map)
    
    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return
        
        logger.info(f"Loading SigLIP model: {self.model_id}")
        
        try:
            from transformers import AutoModel, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            if self.device_map == "auto" and torch.cuda.is_available():
                self.device_map = "cuda"
            self.model = AutoModel.from_pretrained(
                self.model_id,
                device_map=self.device_map,
            )
            self.model.eval()
            
            self._loaded = True
            logger.info(f"SigLIP model loaded on {self.device_map}")
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")
            raise
    
    @torch.no_grad()
    def compute_similarity(
        self,
        images: list[Union[Path, str, Image.Image, np.ndarray]],
        texts: list[str],
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Compute similarity scores between images and texts.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            texts: List of text prompts
            batch_size: Batch size for inference
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (num_images, num_texts) with similarity scores
        """
        self._ensure_loaded()
        
        # Get device from model
        device = get_model_device(self.model)
        
        # Precompute text features (done once for all images)
        text_inputs = self.processor(
            text=texts,
            padding="max_length",
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Process images in batches
        all_scores = []
        
        for start_idx, batch_images in create_batch_iterator(
            images, batch_size, show_progress, "Computing similarities"
        ):
            # Load and process images
            pil_images = [self._load_image(img) for img in batch_images]
            
            image_inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(device) for k, v in image_inputs.items() if k != "input_ids"}
            
            # Compute image features
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (using sigmoid for SigLIP)
            logits = (image_features @ text_features.T) * self.model.logit_scale.exp() + self.model.logit_bias
            scores = torch.sigmoid(logits).cpu().numpy()

            ############ for debug ###############
            # for s,p in zip(scores,batch_images):
            #     if np.any(s>0.5):
            #         for i,_confs in enumerate(s.tolist()):print(i+1,_confs,texts[i])
            #         print(p.resolve(),'\n\n')
            #######################################
            all_scores.append(scores)
        
        return np.vstack(all_scores)
    
    def get_best_class(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        texts: list[str],
    ) -> tuple[str, float]:
        """Get the best matching class for a single image."""
        scores = self.compute_similarity([image], texts)
        best_idx = np.argmax(scores[0])
        return texts[best_idx], float(scores[0, best_idx])
