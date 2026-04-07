from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from falcon_perception import PERCEPTION_MODEL_ID, build_prompt_for_task, load_from_hf_export
from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate

from ..config import ClassPackConfig
from ..contracts import BoundingBox, Candidate, ReviewDecision
from ..registry import register_adapter
from ..utils import clamp
from .base import AnnotationAdapter


def _to_bytes_rle(rle: dict[str, Any]) -> dict[str, Any]:
    out = dict(rle)
    if isinstance(out.get("counts"), str):
        out["counts"] = out["counts"].encode("utf-8")
    return out


def _bbox_from_rle(rle: dict[str, Any], image_size: tuple[int, int]) -> BoundingBox | None:
    mask = mask_utils.decode(_to_bytes_rle(rle))
    if mask is None or not np.any(mask):
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    width, height = image_size
    return BoundingBox(
        x1=clamp(float(cols[0]) / width),
        y1=clamp(float(rows[0]) / height),
        x2=clamp(float(cols[-1] + 1) / width),
        y2=clamp(float(rows[-1] + 1) / height),
    )


@register_adapter("falcon")
class FalconAdapter(AnnotationAdapter):
    capabilities = {"proposal", "semantic_retry"}

    def __init__(self, name, config):
        super().__init__(name, config)
        self.model = None
        self.tokenizer = None
        self.engine = None

    def _ensure_loaded(self) -> None:
        if self.engine is not None:
            return
        model, tokenizer, _ = load_from_hf_export(hf_model_id=self.config.model_id or PERCEPTION_MODEL_ID)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.engine = BatchInferenceEngine(self.model, self.tokenizer)

    def propose(self, image: Image.Image, class_pack: ClassPackConfig, expression: str, params: dict[str, Any]) -> list[Candidate]:
        self._ensure_loaded()
        prompt = build_prompt_for_task(expression, params.get("task", "segmentation"))
        batch_inputs = process_batch_and_generate(
            self.tokenizer,
            [(image.convert("RGB"), prompt)],
            max_length=int(params.get("max_length", 4096)),
            min_dimension=int(params.get("min_dimension", 256)),
            max_dimension=int(params.get("max_dimension", 1024)),
        )
        batch_inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch_inputs.items()}
        stop_ids = [self.tokenizer.eos_token_id]
        end_query = getattr(self.tokenizer, "end_of_query_token_id", None)
        if end_query is not None:
            stop_ids.append(end_query)
        _, aux_out = self.engine.generate(
            **batch_inputs,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=0.0,
            stop_token_ids=stop_ids,
            seed=int(params.get("seed", 42)),
        )
        aux = aux_out[0]
        candidates: list[Candidate] = []
        masks = getattr(aux, "masks_rle", []) or []
        for index, rle in enumerate(masks, start=1):
            box = _bbox_from_rle(rle, image.size)
            if box is None:
                continue
            candidates.append(
                Candidate(
                    candidate_id=f"{self.name}:{class_pack.name}:{index}",
                    class_name=class_pack.name,
                    label=class_pack.name,
                    source_model=self.name,
                    expression=expression,
                    bbox=box,
                    score=1.0,
                    mask=rle,
                    metadata={
                        "task": params.get("task", "segmentation"),
                        "binary_support_only": True,
                        "confidence_type": "binary_support",
                    },
                )
            )
        return candidates

    def refine(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any], request: ReviewDecision | None = None) -> Candidate | None:
        expression = request.retry_expression if request and request.retry_expression else candidate.expression
        proposals = self.propose(image, class_pack, expression, params)
        if not proposals:
            return None
        best = max(proposals, key=lambda item: item.score)
        return best.model_copy(update={"candidate_id": candidate.candidate_id, "notes": [*candidate.notes, f"falcon_retry:{expression}"]})

    def verify(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any]) -> ReviewDecision:
        raise NotImplementedError