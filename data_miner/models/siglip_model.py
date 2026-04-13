"""
SigLIP Model Wrapper

Wrapper for Google's SigLIP model for image-text similarity scoring.
Used for filtering frames based on text prompts/class names.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from ..config import SIGLIP2_MODELS, SIGLIP2_DEFAULT
from ..utils.device import resolve_device, get_model_device
from .base import BaseModel, create_batch_iterator
from ..logging import get_logger

logger = get_logger(__name__)


class SigLIPModel(BaseModel):
    """
    SigLIP model wrapper for image-text similarity.

    Pure compute layer — no caching, no file I/O.

    Example:
        >>> model = SigLIPModel()
        >>> model.load()
        >>> text_embeds = model.get_text_embeddings(["glass door", "window"])
        >>> image_embeds = model.get_image_embeddings(image_paths, batch_size=32)
        >>> scores = model.compute_similarity(image_embeds, text_embeds)
    """

    def __init__(
        self,
        model_id: str = None,
        device_map: str = "auto",
    ):
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
    def get_text_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute normalized text embeddings.

        Args:
            texts: List of text prompts.

        Returns:
            (num_texts, dim) float16 numpy array of L2-normalized text features.
        """
        self._ensure_loaded()
        device = get_model_device(self.model)

        text_inputs = self.processor(
            text=[t.lower() for t in texts],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().to(torch.float16).numpy()

    @torch.no_grad()
    def get_image_embeddings(
        self,
        images: list[Union[Path, str, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Compute normalized image embeddings.

        Args:
            images: List of images (paths, PIL Images, or numpy arrays).
            batch_size: Batch size for inference.
            show_progress: Show progress bar.

        Returns:
            (num_images, dim) float16 numpy array of L2-normalized image features.
        """
        self._ensure_loaded()
        device = get_model_device(self.model)

        all_embeddings = []

        for start_idx, batch_images in create_batch_iterator(
            images, batch_size, show_progress, "Computing image embeddings"
        ):
            pil_images = [self._load_image(img) for img in batch_images]

            image_inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(device) for k, v in image_inputs.items() if k != "input_ids"}

            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().to(torch.float16).numpy())

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute SigLIP2 similarity scores from pre-computed embeddings.

        Uses sigmoid scoring: sigmoid(image @ text.T * logit_scale + logit_bias)

        Args:
            image_embeddings: (num_images, dim) normalized image features.
            text_embeddings: (num_texts, dim) normalized text features.

        Returns:
            (num_images, num_texts) float32 similarity scores.
        """
        self._ensure_loaded()
        device = get_model_device(self.model)

        img_tensor = torch.from_numpy(image_embeddings.astype(np.float32)).to(device)
        txt_tensor = torch.from_numpy(text_embeddings.astype(np.float32)).to(device)

        logits = (img_tensor @ txt_tensor.T) * self.model.logit_scale.exp() + self.model.logit_bias
        scores = torch.sigmoid(logits).cpu().numpy()

        return scores

    def get_best_class(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        texts: list[str],
    ) -> tuple[str, float]:
        """Get the best matching class for a single image."""
        text_embeds = self.get_text_embeddings(texts)
        image_embeds = self.get_image_embeddings([image])
        scores = self.compute_similarity(image_embeds, text_embeds)
        best_idx = np.argmax(scores[0])
        return texts[best_idx], float(scores[0, best_idx])
