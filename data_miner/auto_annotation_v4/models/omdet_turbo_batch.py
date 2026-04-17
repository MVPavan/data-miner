"""Multi-image batch predictor for OmDet-Turbo open-vocabulary detection.

Wraps ``OmDetTurboForObjectDetection`` with a split forward pass that
separates vision encoding from decoding, analogous to
``sam3_dart_batch.Sam3MultiClassPredictorBatch`` for DART/SAM3.

The HuggingFace model's ``forward()`` runs four stages sequentially:
  1. Vision backbone (Swin-Tiny)
  2. Hybrid encoder (FPN+PAN+deformable attention)
  3. Language embedding (CLIP text + LRU cache)
  4. Decoder (6 deformable transformer layers)

This wrapper exposes them as reusable stages:
  - ``set_classes()``   -> pre-compute (3) once
  - ``set_images()``    -> run (1)+(2) once for B images, cache features
  - ``predict_batch()`` -> run (4) with cached features, post-process per image

Zero modifications to the transformers source -- all access is via public
attributes of the model.

Note: This is the v4 copy of the batch predictor, moved from
``data_miner.auto_annotation_v3.omdet_batch`` into the v4 models package
for self-contained deployment. No LitServe dependency.

Usage:
    from data_miner.auto_annotation_v4.models.omdet_turbo_batch import (
        OmDetTurboBatchPredictor,
    )

    predictor = OmDetTurboBatchPredictor(device="cuda")
    predictor.set_classes(["person", "forklift", "pallet jack"])

    # Option A: two-step
    state = predictor.set_images([img1, img2, img3])
    results = predictor.predict_batch(state, threshold=0.3)

    # Option B: convenience one-liner
    results = predictor.predict_images([img1, img2, img3])

    # Each result dict:
    # {"boxes": Tensor(K,4), "scores": Tensor(K,), "class_ids": Tensor(K,),
    #  "class_names": list[str]}
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import PIL.Image
import timm
import torch

from transformers import AutoProcessor, OmDetTurboForObjectDetection
from transformers.models.omdet_turbo.modeling_omdet_turbo import (
    OmDetTurboObjectDetectionOutput,
)

logger = logging.getLogger(__name__)


class OmDetTurboBatchPredictor:
    """Multi-image batch predictor for OmDet-Turbo.

    Splits OmDet-Turbo's monolithic forward pass into reusable stages:
      - ``set_classes(names)``  -- pre-compute CLIP text embeddings (cheap)
      - ``set_images(images)``  -- run Swin backbone + encoder (heavy, cached)
      - ``predict_batch(state)``  -- run decoder + post-process per image
      - ``predict_images(images)`` -- convenience: set_images + predict_batch

    When all images share the same class list (the annotation pipeline case),
    the text encoding runs once and the vision backbone runs once for the
    entire batch.  No B*N fan-out is needed -- the decoder natively handles
    the class dimension via batched matmuls.
    """

    def __init__(
        self,
        model_id: str = "omlab/omdet-turbo-swin-tiny-hf",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            OmDetTurboForObjectDetection.from_pretrained(
                model_id, torch_dtype=dtype
            )
            .to(self.device)
            .eval()
        )

        # Workaround: transformers' TimmBackbone doesn't forward img_size
        # to timm.create_model(), so the Swin PatchEmbed is initialized at
        # 224x224 instead of the config's 640x640.  Recreate the timm model
        # with the correct img_size and load the existing weights.
        self._fix_backbone_img_size()

        # Cached language state (populated by set_classes)
        self._class_names: Optional[List[str]] = None
        self._task: Optional[str] = None
        self._class_features: Optional[torch.Tensor] = None  # (N, 1, 512)
        self._task_features: Optional[torch.Tensor] = None   # (seq, 1, hidden)
        self._task_mask: Optional[torch.Tensor] = None        # (1, seq)

    # ------------------------------------------------------------------
    # Backbone fix (timm img_size bug workaround)
    # ------------------------------------------------------------------

    def _fix_backbone_img_size(self) -> None:
        """Recreate the timm backbone with the correct img_size.

        The transformers ``TimmBackbone`` class doesn't forward
        ``config.img_size`` to ``timm.create_model()``, so the Swin
        PatchEmbed defaults to 224x224.  This method recreates the timm
        model with the correct resolution and loads the existing weights.
        """
        bb_config = self.model.config.backbone_config
        target_size = getattr(bb_config, "img_size", None) or self.model.config.image_size

        timm_backbone = self.model.vision_backbone.vision_backbone._backbone
        current_size = getattr(timm_backbone.patch_embed, "img_size", None)

        if current_size is not None and current_size == (target_size, target_size):
            return  # already correct

        logger.info(
            "Fixing timm backbone img_size: %s -> (%d, %d)",
            current_size, target_size, target_size,
        )

        old_state = timm_backbone.state_dict()
        new_backbone = timm.create_model(
            bb_config.backbone,
            pretrained=False,
            features_only=True,
            out_indices=list(bb_config.out_indices),
            img_size=target_size,
        )
        new_backbone.load_state_dict(old_state, strict=True)
        new_backbone = new_backbone.to(
            device=self.device, dtype=self.dtype
        ).eval()

        self.model.vision_backbone.vision_backbone._backbone = new_backbone

    # ------------------------------------------------------------------
    # Text embedding (stage 3 -- pre-computed, reused across batches)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def set_classes(
        self,
        class_names: List[str],
        task: Optional[str] = None,
    ) -> None:
        """Pre-compute and cache CLIP text embeddings for a class set.

        Args:
            class_names: List of N class name strings.
            task: Optional task description; defaults to
                  ``"Detect {class1}, {class2}, ..."``.
        """
        task = task or f"Detect {', '.join(class_names)}."

        # No-op if unchanged
        if self._class_names == class_names and self._task == task:
            return

        # Tokenize classes (flattened list) and task (single string)
        # using the same kwargs the processor uses internally.
        tok_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 77,
            "return_tensors": "pt",
        }
        classes_encoding = self.processor.tokenizer(
            text=class_names, **tok_kwargs
        )
        tasks_encoding = self.processor.tokenizer(
            text=[task], **tok_kwargs
        )

        classes_input_ids = classes_encoding["input_ids"].to(self.device)
        classes_attention_mask = classes_encoding["attention_mask"].to(self.device)
        tasks_input_ids = tasks_encoding["input_ids"].to(self.device)
        tasks_attention_mask = tasks_encoding["attention_mask"].to(self.device)
        classes_structure = torch.tensor(
            [len(class_names)], dtype=torch.long, device=self.device
        )

        # Run the model's language embedding path (uses its own LRU cache)
        class_features, task_features, task_mask = (
            self.model.get_language_embedding(
                classes_input_ids,
                classes_attention_mask,
                tasks_input_ids,
                tasks_attention_mask,
                classes_structure,
            )
        )

        # Cache -- shapes are (N, 1, 512), (seq, 1, hidden), (1, seq)
        self._class_features = class_features
        self._task_features = task_features
        self._task_mask = task_mask
        self._class_names = list(class_names)
        self._task = task

    # ------------------------------------------------------------------
    # Vision encoding (stages 1+2 -- run once for the full batch)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def set_images(
        self,
        images: List[Union[PIL.Image.Image, np.ndarray]],
    ) -> Dict:
        """Encode a batch of images through vision backbone + encoder.

        All images are resized/normalized by the HuggingFace image processor
        (default 640x640).  The backbone and encoder run ONCE for the entire
        batch, producing cached features for ``predict_batch()``.

        Args:
            images: List of B input images (PIL or ndarray).

        Returns:
            State dict with encoder features, original sizes, and batch size.
        """
        if not images:
            raise ValueError("images must be a non-empty list")

        B = len(images)
        original_sizes = []
        for img in images:
            if isinstance(img, PIL.Image.Image):
                original_sizes.append((img.height, img.width))
            elif isinstance(img, np.ndarray):
                original_sizes.append((img.shape[0], img.shape[1]))
            else:
                raise ValueError(
                    f"Each image must be a PIL Image or ndarray, got {type(img)}"
                )

        # Preprocess images (resize, normalize, pad)
        encoding = self.processor.image_processor(
            images, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].to(
            self.device, dtype=self.dtype
        )  # (B, 3, 640, 640)

        # Stage 1: Vision backbone (Swin-Tiny)
        with torch.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            image_features = self.model.vision_backbone(pixel_values)

            # Stage 2: Hybrid encoder (FPN + PAN + deformable attention)
            encoder_outputs = self.model.encoder(
                image_features, return_dict=True
            )

        return {
            "encoder_extracted_states": encoder_outputs.extracted_states,
            "original_sizes": original_sizes,
            "batch_size": B,
        }

    # ------------------------------------------------------------------
    # Decoding + post-processing (stage 4 -- uses cached features)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_batch(
        self,
        state: Dict,
        threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[Dict]:
        """Run the decoder with cached vision+text features, post-process.

        Args:
            state: State dict from ``set_images()``.
            threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for NMS.

        Returns:
            List of B result dicts, each with ``boxes`` (K,4 pixel xyxy),
            ``scores`` (K,), ``class_ids`` (K,), ``class_names`` (list[str]).
        """
        if self._class_names is None:
            raise RuntimeError("Call set_classes() before predict_batch()")
        if "encoder_extracted_states" not in state:
            raise RuntimeError("Call set_images() before predict_batch()")

        B = state["batch_size"]
        N = len(self._class_names)
        encoder_extracted_states = state["encoder_extracted_states"]

        # Expand pre-computed language features to match batch size
        # (N, 1, 512) -> (N, B, 512)   [view, no copy]
        class_features = self._class_features.expand(-1, B, -1)
        # (seq, 1, hidden) -> (seq, B, hidden)
        task_features = self._task_features.expand(-1, B, -1)
        # (1, seq) -> (B, seq)
        task_mask = self._task_mask.expand(B, -1)

        # Stage 4: Decoder
        with torch.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            decoder_outputs = self.model.decoder(
                encoder_extracted_states,
                class_features,
                task_features,
                task_mask,
                return_dict=True,
            )

        # Last-layer outputs
        # decoder_coords is a stacked tensor: (num_layers, B, 900, 4)
        # decoder_classes is a stacked tensor: (num_layers, B, 900, N)
        decoder_coord_logits = decoder_outputs.decoder_coords[-1]  # (B, 900, 4)
        decoder_class_logits = decoder_outputs.decoder_classes[-1]  # (B, 900, N)
        classes_structure = torch.tensor(
            [N] * B, dtype=torch.long, device=self.device
        )

        # Build a mock output for the processor's post-processing
        mock_output = OmDetTurboObjectDetectionOutput(
            decoder_coord_logits=decoder_coord_logits,
            decoder_class_logits=decoder_class_logits,
            classes_structure=classes_structure,
        )

        target_sizes = state["original_sizes"]  # list of (h, w)
        results_raw = self.processor.post_process_grounded_object_detection(
            mock_output,
            text_labels=[self._class_names] * B,
            target_sizes=target_sizes,
            threshold=threshold,
            nms_threshold=nms_threshold,
        )

        # Convert to our standard result format
        results = []
        for r in results_raw:
            results.append({
                "boxes": r["boxes"],                # (K, 4) pixel xyxy
                "scores": r["scores"],              # (K,)
                "class_ids": r["labels"],            # (K,) int indices
                "class_names": r["text_labels"] or [],
            })
        return results

    # ------------------------------------------------------------------
    # Convenience one-liner
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_images(
        self,
        images: List[Union[PIL.Image.Image, np.ndarray]],
        threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[Dict]:
        """Convenience: ``set_images`` + ``predict_batch`` in one call.

        Args:
            images: List of B input images.
            threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for NMS.

        Returns:
            List of B result dicts.
        """
        state = self.set_images(images)
        return self.predict_batch(
            state, threshold=threshold, nms_threshold=nms_threshold
        )
