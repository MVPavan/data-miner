"""Proposal stage: run detection/segmentation models to generate candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import AutoAnnotationV2Config, ClassPackConfig, DetectionModelConfig
from ..contracts import BoundingBox, Candidate, CandidateStatus
from ..log_utils import get_logger
from ..utils import clamp

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model loaders (lazy, cached per-process)
# ---------------------------------------------------------------------------

_loaded_models: dict[str, Any] = {}


def _resolve_device(cfg: DetectionModelConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def _load_falcon(cfg: DetectionModelConfig) -> dict[str, Any]:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    from falcon_perception import PERCEPTION_MODEL_ID, load_from_hf_export
    from falcon_perception.batch_inference import BatchInferenceEngine

    model_id = cfg.model_id or PERCEPTION_MODEL_ID
    model, tokenizer, _ = load_from_hf_export(hf_model_id=model_id)
    device = _resolve_device(cfg)
    dtype_name = cfg.params.get("dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
    model = model.to(dtype=torch_dtype, device=device).eval()
    engine = BatchInferenceEngine(model, tokenizer)
    return {"model": model, "tokenizer": tokenizer, "engine": engine, "device": device}


def _load_grounding_dino(cfg: DetectionModelConfig) -> dict[str, Any]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = cfg.model_id or "IDEA-Research/grounding-dino-base"
    device = _resolve_device(cfg)
    processor = AutoProcessor.from_pretrained(model_id)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    )
    return {"model": model, "processor": processor, "device": device}


def _load_sam(cfg: DetectionModelConfig) -> dict[str, Any]:
    from transformers import Sam3Model, Sam3Processor

    model_id = cfg.model_id or "facebook/sam3"
    device = _resolve_device(cfg)
    processor = Sam3Processor.from_pretrained(model_id)
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (
        Sam3Model.from_pretrained(model_id, torch_dtype=torch_dtype).to(device).eval()
    )
    return {"model": model, "processor": processor, "device": device}


_LOADERS = {
    "falcon": _load_falcon,
    "grounding_dino": _load_grounding_dino,
    "sam": _load_sam,
}


def _get_model(name: str, cfg: DetectionModelConfig) -> dict[str, Any]:
    if name not in _loaded_models:
        loader = _LOADERS.get(cfg.kind)
        if loader is None:
            raise ValueError(f"Unknown model kind: {cfg.kind}")
        logger.info("Loading model %s (kind=%s)", name, cfg.kind)
        _loaded_models[name] = loader(cfg)
    return _loaded_models[name]


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------


def _run_falcon(
    loaded: dict[str, Any],
    image: Image.Image,
    class_pack: ClassPackConfig,
    expression: str,
    params: dict[str, Any],
    model_name: str,
) -> list[Candidate]:
    from falcon_perception import build_prompt_for_task
    from falcon_perception.batch_inference import process_batch_and_generate
    from pycocotools import mask as mask_utils

    engine = loaded["engine"]
    tokenizer = loaded["tokenizer"]
    device = loaded["device"]

    prompt = build_prompt_for_task(expression, params.get("task", "segmentation"))
    batch_inputs = process_batch_and_generate(
        tokenizer,
        [(image.convert("RGB"), prompt)],
        max_length=int(params.get("max_length", 4096)),
        min_dimension=int(params.get("min_dimension", 256)),
        max_dimension=int(params.get("max_dimension", 1024)),
    )
    batch_inputs = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_inputs.items()
    }
    stop_ids = [tokenizer.eos_token_id]
    end_query = getattr(tokenizer, "end_of_query_token_id", None)
    if end_query is not None:
        stop_ids.append(end_query)

    _, aux_out = engine.generate(
        **batch_inputs,
        max_new_tokens=int(params.get("max_new_tokens", 2048)),
        temperature=0.0,
        stop_token_ids=stop_ids,
        seed=int(params.get("seed", 42)),
    )

    candidates: list[Candidate] = []
    masks = getattr(aux_out[0], "masks_rle", []) or []
    for idx, rle in enumerate(masks, start=1):
        rle_bytes = dict(rle)
        if isinstance(rle_bytes.get("counts"), str):
            rle_bytes["counts"] = rle_bytes["counts"].encode("utf-8")
        decoded = mask_utils.decode(rle_bytes)
        if decoded is None or not np.any(decoded):
            continue
        rows = np.where(decoded.any(axis=1))[0]
        cols = np.where(decoded.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        w, h = image.size
        box = BoundingBox(
            x1=clamp(float(cols[0]) / w),
            y1=clamp(float(rows[0]) / h),
            x2=clamp(float(cols[-1] + 1) / w),
            y2=clamp(float(rows[-1] + 1) / h),
        )
        candidates.append(
            Candidate(
                candidate_id=f"{model_name}:{class_pack.name}:{expression}:{idx}",
                class_name=class_pack.name,
                label=class_pack.name,
                source_model=model_name,
                expression=expression,
                bbox=box,
                score=1.0,
                mask_rle=rle,
                metadata={"task": params.get("task", "segmentation")},
            )
        )
    return candidates


def _run_grounding_dino(
    loaded: dict[str, Any],
    image: Image.Image,
    class_pack: ClassPackConfig,
    expression: str,
    params: dict[str, Any],
    model_name: str,
) -> list[Candidate]:
    model = loaded["model"]
    processor = loaded["processor"]
    device = loaded["device"]

    text = " . ".join(class_pack.all_names())
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=float(params.get("box_threshold", 0.25)),
        text_threshold=float(params.get("text_threshold", 0.2)),
        target_sizes=[image.size[::-1]],
    )[0]

    w, h = image.size
    candidates: list[Candidate] = []
    for idx, (box_t, score_t, label_str) in enumerate(
        zip(results["boxes"], results["scores"], results["labels"]), start=1
    ):
        x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
        candidates.append(
            Candidate(
                candidate_id=f"{model_name}:{class_pack.name}:{expression}:{idx}",
                class_name=class_pack.name,
                label=str(label_str),
                source_model=model_name,
                expression=expression,
                bbox=BoundingBox(
                    x1=clamp(x1 / w),
                    y1=clamp(y1 / h),
                    x2=clamp(x2 / w),
                    y2=clamp(y2 / h),
                ),
                score=float(score_t),
            )
        )
    return candidates


def _run_sam(
    loaded: dict[str, Any],
    image: Image.Image,
    class_pack: ClassPackConfig,
    expression: str,
    params: dict[str, Any],
    model_name: str,
) -> list[Candidate]:
    model = loaded["model"]
    processor = loaded["processor"]
    device = loaded["device"]

    inputs = processor(images=image, text=expression, return_tensors="pt")
    inputs = inputs.to(device=device, dtype=model.dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(params.get("threshold", 0.5)),
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    boxes = results.get("boxes")
    scores = results.get("scores")
    if boxes is None or scores is None:
        return []

    w, h = image.size
    candidates: list[Candidate] = []
    for idx, (box_t, score_t) in enumerate(zip(boxes, scores), start=1):
        x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
        candidates.append(
            Candidate(
                candidate_id=f"{model_name}:{class_pack.name}:{expression}:{idx}",
                class_name=class_pack.name,
                label=class_pack.name,
                source_model=model_name,
                expression=expression,
                bbox=BoundingBox(
                    x1=clamp(x1 / w),
                    y1=clamp(y1 / h),
                    x2=clamp(x2 / w),
                    y2=clamp(y2 / h),
                ),
                score=float(score_t),
            )
        )
    return candidates


_RUNNERS = {
    "falcon": _run_falcon,
    "grounding_dino": _run_grounding_dino,
    "sam": _run_sam,
}


def _refine_with_sam(
    loaded: dict[str, Any],
    image: Image.Image,
    candidate: Candidate,
    points: list[tuple[int, int, int]] | None = None,
    params: dict[str, Any] | None = None,
) -> Candidate | None:
    """Refine a candidate's bbox using SAM with box or point prompts."""
    from ..utils import bbox_to_pixels

    model = loaded["model"]
    processor = loaded["processor"]
    device = loaded["device"]
    params = params or {}

    pixel_box = list(bbox_to_pixels(candidate.bbox, image.size))

    if points:
        input_points = [[[p[0], p[1]] for p in points]]
        input_labels = [[p[2] for p in points]]
        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=[[pixel_box]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(device=device, dtype=model.dtype)
    else:
        inputs = processor(
            images=image,
            input_boxes=[[pixel_box]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(device=device, dtype=model.dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(params.get("threshold", 0.5)),
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    boxes = results.get("boxes")
    scores = results.get("scores")
    if boxes is None or scores is None or len(boxes) == 0:
        return None

    best_idx = int(torch.argmax(scores).item()) if torch.is_tensor(scores) else 0
    x1, y1, x2, y2 = [float(v) for v in boxes[best_idx].tolist()]
    w, h = image.size

    return candidate.model_copy(
        update={
            "bbox": BoundingBox(
                x1=clamp(x1 / w),
                y1=clamp(y1 / h),
                x2=clamp(x2 / w),
                y2=clamp(y2 / h),
            ),
            "score": max(candidate.score, float(scores[best_idx])),
            "source_model": f"{candidate.source_model}+sam",
            "status": CandidateStatus.REFINED,
            "notes": [*candidate.notes, "sam_refined"],
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_proposal(
    image: Image.Image,
    config: AutoAnnotationV2Config,
) -> list[Candidate]:
    """Run all configured detection models across all classes and expressions."""
    all_candidates: list[Candidate] = []

    for model_name in config.proposal.models:
        model_cfg = config.detection_models[model_name]
        if not model_cfg.enabled:
            continue

        loaded = _get_model(model_name, model_cfg)
        runner = _RUNNERS.get(model_cfg.kind)
        if runner is None:
            logger.warning(
                "No runner for kind=%s, skipping %s", model_cfg.kind, model_name
            )
            continue

        for class_pack in config.classes:
            expressions = class_pack.prompt_variants or [class_pack.name]
            for expression in expressions:
                logger.info(
                    "Proposing: model=%s class=%s expression=%s",
                    model_name,
                    class_pack.name,
                    expression,
                )
                try:
                    candidates = runner(
                        loaded,
                        image,
                        class_pack,
                        expression,
                        model_cfg.params,
                        model_name,
                    )
                    logger.info(
                        "Got %d candidates from %s for %s/%s",
                        len(candidates),
                        model_name,
                        class_pack.name,
                        expression,
                    )
                    all_candidates.extend(candidates)
                except Exception:
                    logger.exception(
                        "Proposal failed: model=%s class=%s expr=%s",
                        model_name,
                        class_pack.name,
                        expression,
                    )

    logger.info("Total proposal candidates: %d", len(all_candidates))
    return all_candidates


def get_loaded_model(name: str, config: AutoAnnotationV2Config) -> dict[str, Any]:
    """Get or load a model by name. Used by refinement stage."""
    cfg = config.detection_models[name]
    return _get_model(name, cfg)
