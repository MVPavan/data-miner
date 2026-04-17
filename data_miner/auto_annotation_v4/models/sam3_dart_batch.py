"""Multi-image batching extension for DART's Sam3MultiClassPredictorFast.

Subclasses Sam3MultiClassPredictorFast to add image-axis batching on top of
the existing class-axis batching.  Instead of looping set_image -> predict per
image, runs the backbone ONCE for B images and encoder+decoder ONCE at
bs=B*N, then post-processes per image.

Zero modifications to the DART source -- all new code lives here.

Note: This is the v4 copy of the batch predictor, moved from
``data_miner.auto_annotation_v3.dart_batch`` into the v4 models package
for self-contained deployment. No LitServe dependency.

Usage:
    from data_miner.auto_annotation_v4.models.sam3_dart_batch import (
        Sam3MultiClassPredictorBatch,
    )

    predictor = Sam3MultiClassPredictorBatch(model, device="cuda", use_fp16=True)
    predictor.set_classes(["person", "forklift", "pallet jack"])

    # Option A: two-step
    state = predictor.set_images([img1, img2, img3, img4])
    results = predictor.predict_batch(state, confidence_threshold=0.3)

    # Option B: convenience one-liner
    results = predictor.predict_images([img1, img2, img3, img4])

    # Each result is the same dict as predict() returns:
    # {"boxes", "masks", "masks_logits", "scores", "class_ids", "class_names"}
"""

from typing import Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from torchvision.transforms import v2

from sam3.model.model_misc import inverse_sigmoid
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


class Sam3MultiClassPredictorBatch(Sam3MultiClassPredictorFast):
    """Extends Sam3MultiClassPredictorFast with multi-image batching.

    Adds three new public methods:
      - set_images(images) -> state dict with batched backbone features
      - predict_batch(state) -> List[Dict] of per-image results
      - predict_images(images) -> List[Dict] (convenience: set_images + predict_batch)

    All parent methods (set_image, predict, predict_image, set_classes) are
    inherited and work unchanged for single-image use.

    Constraint: all images are resized to self.resolution (1008x1008) internally,
    so the backbone always sees a uniform batch.  Original sizes are tracked for
    correct output coordinate scaling.

    Note: torch.compile with dynamic=False (the parent's default for backbone)
    may trigger recompilation for different batch sizes.  For best results with
    batching, either disable compile_mode or set it after construction.
    """

    @torch.inference_mode()
    def set_images(
        self,
        images: List[Union[PIL.Image.Image, torch.Tensor, np.ndarray]],
        state: Optional[Dict] = None,
    ) -> Dict:
        """Encode a batch of images through the vision backbone in one call.

        All images are resized to self.resolution (1008x1008) and stacked into
        a single (B, 3, H, W) tensor.  The backbone runs ONCE for the entire
        batch, producing FPN features at (B, C, H_l, W_l) per level.

        Args:
            images: List of B input images (PIL, tensor, or ndarray).
            state: Optional state dict to update (creates new if None).

        Returns:
            State dict with backbone_out, original_sizes, and batch_size.
        """
        if not images:
            raise ValueError("images must be a non-empty list")

        if state is None:
            state = {}

        B = len(images)
        original_sizes = []
        tensors = []

        for image in images:
            # Record original size
            if isinstance(image, PIL.Image.Image):
                width, height = image.size
            elif isinstance(image, (torch.Tensor, np.ndarray)):
                height, width = image.shape[-2:]
            else:
                raise ValueError("Each image must be a PIL image, tensor, or ndarray")
            original_sizes.append((height, width))

            # Fast path: resize PIL on CPU first (transfers ~3MB instead of full res)
            if isinstance(image, PIL.Image.Image):
                image = image.resize(
                    (self.resolution, self.resolution), PIL.Image.Resampling.BILINEAR
                )

            t = v2.functional.to_image(image).to(self.device)
            t = self.transform(t)
            tensors.append(t)

        batch = torch.stack(tensors, dim=0)  # (B, 3, 1008, 1008)

        self._ensure_compiled()

        if self._trt_engine_path is not None:
            state["backbone_out"] = self._backbone_fn(batch)
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
                state["backbone_out"] = self._backbone_fn(batch)

        state["original_sizes"] = original_sizes
        state["batch_size"] = B

        return state

    @torch.inference_mode()
    def predict_batch(
        self,
        state: Dict,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
    ) -> List[Dict]:
        """Run batched encoder+decoder at bs=B*N and post-process per image.

        Expands per-image features (H*W, B, d) to (H*W, B*N, d) and text
        prompts (seq, N, d) to (seq, B*N, d), then runs the full
        encoder->decoder->scoring->boxes pipeline once.  Results are reshaped
        back to (B, N, ...) and post-processed per image using the parent's
        _postprocess method.

        Args:
            state: State dict from set_images().
            confidence_threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for NMS.
            per_class_nms: Per-class (True) or cross-class (False) NMS.

        Returns:
            List of B result dicts, each with "boxes", "masks",
            "masks_logits", "scores", "class_ids", "class_names".
        """
        if self._class_names is None:
            raise RuntimeError("Call set_classes() before predict_batch()")
        if "backbone_out" not in state:
            raise RuntimeError("Call set_images() before predict_batch()")

        self._ensure_compiled()

        B = state["batch_size"]
        N = self._num_classes
        backbone_out = state["backbone_out"]

        # --- Extract features for all B images ---
        img_ids_all = torch.arange(B, device=self.device, dtype=torch.long)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = (
            self.model._get_img_feats(backbone_out, img_ids_all)
        )
        # Keep potentially updated backbone_out in state
        state["backbone_out"] = backbone_out

        # --- Expand image features: (H*W, B, d) -> (H*W, B*N, d) ---
        BN = B * N
        batched_img_feats = [
            f.unsqueeze(2).expand(-1, -1, N, -1).reshape(f.shape[0], BN, -1)
            for f in img_feats
        ]
        batched_img_pos = [
            p.unsqueeze(2).expand(-1, -1, N, -1).reshape(p.shape[0], BN, -1)
            for p in img_pos_embeds
        ]

        # --- Expand text prompts: (seq, N, d) -> (seq, B*N, d) ---
        prompt = self._batched_text.repeat(1, B, 1)  # (seq, B*N, d)
        prompt_mask = self._batched_mask.repeat(B, 1)  # (B*N, seq)

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
            # --- Encoder (bs=B*N) ---
            prompt_pos_embed = torch.zeros_like(prompt)
            memory = self._encoder_fn(
                src=batched_img_feats,
                src_key_padding_mask=None,
                src_pos=batched_img_pos,
                prompt=prompt,
                prompt_pos=prompt_pos_embed,
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=vis_feat_sizes,
            )

            encoder_hidden_states = memory["memory"]  # (total_tokens, B*N, d)

            # --- Decoder (bs=B*N) ---
            query_embed = self.model.transformer.decoder.query_embed.weight
            tgt = query_embed.unsqueeze(1).expand(-1, BN, -1)

            hs, reference_boxes, dec_presence_out, _ = self._decoder_fn(
                tgt=tgt,
                memory=encoder_hidden_states,
                memory_key_padding_mask=memory["padding_mask"],
                pos=memory["pos_embed"],
                reference_boxes=None,
                level_start_index=memory["level_start_index"],
                spatial_shapes=memory["spatial_shapes"],
                valid_ratios=memory["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )
            # hs: (layers, Q, B*N, d) -> (layers, B*N, Q, d)
            hs = hs.transpose(1, 2)
            reference_boxes = reference_boxes.transpose(1, 2)

            # --- Scoring (batched) ---
            scores = self.model.dot_prod_scoring(hs, prompt, prompt_mask)
            # scores: (layers, B*N, Q, 1)

            # --- Box prediction ---
            box_offsets = self.model.transformer.decoder.bbox_embed(hs)
            ref_inv = inverse_sigmoid(reference_boxes)
            outputs_coord = (ref_inv + box_offsets).sigmoid()

        # --- Presence probabilities ---
        presence_probs_all = None
        if dec_presence_out is not None:
            presence_logits = dec_presence_out[-1]
            presence_probs_all = presence_logits.sigmoid().squeeze(0)  # (B*N,)

        # Last layer outputs
        scores_last = scores[-1]  # (B*N, Q, 1)
        boxes_last = outputs_coord[-1]  # (B*N, Q, 4)

        # --- Per-image post-processing ---
        results = []
        for i in range(B):
            start = i * N
            end = start + N

            scores_i = scores_last[start:end]  # (N, Q, 1)
            boxes_i = boxes_last[start:end]  # (N, Q, 4)

            presence_i = None
            if presence_probs_all is not None:
                presence_i = presence_probs_all[start:end]  # (N,)

            # Per-image presence check
            if presence_i is not None and self.presence_threshold > 0.0:
                present_mask = presence_i > self.presence_threshold
            else:
                present_mask = torch.ones(N, dtype=torch.bool, device=self.device)

            present_indices = present_mask.nonzero(as_tuple=True)[0]

            orig_h, orig_w = state["original_sizes"][i]

            if len(present_indices) == 0:
                results.append(self._empty_result(orig_h, orig_w))
                continue

            # Slice decoder hidden states and encoder memory for this image
            hs_i = hs[:, start:end]  # (layers, N, Q, d)
            enc_hs_i = encoder_hidden_states[:, start:end]  # (tokens, N, d)

            batched_dict = {
                "scores_all": scores_i,  # (N, Q, 1)
                "boxes_all": boxes_i,  # (N, Q, 4)
                "hs_all": hs_i,  # (layers, N, Q, d)
                "encoder_hidden_states": enc_hs_i,  # (tokens, N, d)
                "prompt": self._batched_text,  # (seq, N, d)
                "prompt_mask": self._batched_mask,  # (N, seq)
                "presence_probs": presence_i,  # (N,) or None
                "present_indices": present_indices,  # (K,)
            }

            # img_ids for seg head: indexes into backbone FPN at (B, C, H, W)
            img_ids_i = torch.tensor([i], device=self.device, dtype=torch.long)

            result_i = self._postprocess(
                batched=batched_dict,
                backbone_out=backbone_out,
                img_ids=img_ids_i,
                orig_h=orig_h,
                orig_w=orig_w,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                per_class_nms=per_class_nms,
            )
            results.append(result_i)

        return results

    @torch.inference_mode()
    def predict_images(
        self,
        images: List[Union[PIL.Image.Image, torch.Tensor, np.ndarray]],
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
    ) -> List[Dict]:
        """Convenience: set_images + predict_batch in one call.

        Args:
            images: List of B input images.
            confidence_threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for NMS.
            per_class_nms: Per-class (True) or cross-class (False) NMS.

        Returns:
            List of B result dicts.
        """
        state = self.set_images(images)
        return self.predict_batch(
            state,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )
