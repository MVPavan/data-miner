"""LitServe server for Falcon-Perception zero-shot detection / segmentation.

Endpoint: POST /predict
Request:
    {
        "image_path": "/abs/path/to/image.jpg",
        "text_prompt": "person . forklift . palletjack",
        "task": "detection"   # optional, default "segmentation"
    }
Response:
    {
        "boxes":  [[x1,y1,x2,y2], ...],   # normalized 0-1
        "scores": [float, ...],
        "labels": ["person", ...]
    }

Launch:
    python serve_falcon.py --port 3002 --device 1 --max-batch-size 4
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import litserve as ls
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "tiiuae/Falcon-Perception"

# Default generation / image-resize params (can be overridden per-request)
_DEFAULT_PARAMS: dict[str, Any] = {
    "task": "segmentation",
    "dtype": "bfloat16",
    "max_length": 4096,
    "min_dimension": 256,
    "max_dimension": 512,
    "max_new_tokens": 2048,
    "seed": 42,
}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _center_size_to_xyxy(
    x_center: float, y_center: float, box_w: float, box_h: float
) -> tuple[float, float, float, float]:
    """Convert normalized center-size bbox to normalized xyxy."""
    x1 = _clamp(x_center - box_w / 2)
    y1 = _clamp(y_center - box_h / 2)
    x2 = _clamp(x_center + box_w / 2)
    y2 = _clamp(y_center + box_h / 2)
    return x1, y1, x2, y2


def _mask_rle_to_xyxy(
    mask_rle: dict,
) -> tuple[float, float, float, float] | None:
    """Decode a Falcon mask RLE and return normalized xyxy bounding box."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        logger.warning("pycocotools not available — skipping mask-to-bbox conversion")
        return None

    if not isinstance(mask_rle, dict):
        return None
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not isinstance(size, list) or len(size) != 2 or counts is None:
        return None

    try:
        encoded = {
            "size": size,
            "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
        }
        decoded = mask_utils.decode(encoded)
    except Exception:
        return None

    if decoded is None or decoded.size == 0:
        return None

    foreground = np.argwhere(decoded.astype(bool))
    if foreground.size == 0:
        return None

    y_coords, x_coords = foreground[:, 0], foreground[:, 1]
    img_h, img_w = decoded.shape[:2]
    return (
        _clamp(float(x_coords.min()) / img_w),
        _clamp(float(y_coords.min()) / img_h),
        _clamp(float(x_coords.max() + 1) / img_w),
        _clamp(float(y_coords.max() + 1) / img_h),
    )


class FalconApi(ls.LitAPI):
    """LitAPI wrapper around Falcon-Perception for zero-shot detection/segmentation."""

    # ------------------------------------------------------------------
    # LitServe lifecycle
    # ------------------------------------------------------------------

    def setup(self, device: str) -> None:
        """Load Falcon model and BatchInferenceEngine onto *device*."""
        self.device = device

        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            from falcon_perception import load_and_prepare_model, setup_torch_config
            from falcon_perception.batch_inference import BatchInferenceEngine
        except ImportError as exc:
            raise RuntimeError(
                "falcon_perception package not installed. "
                "Install it before starting this server."
            ) from exc

        logger.info("Configuring torch for Falcon on device=%s", device)
        setup_torch_config()

        dtype_name = _DEFAULT_PARAMS["dtype"]
        logger.info("Loading Falcon model %s (dtype=%s)", _MODEL_ID, dtype_name)
        try:
            model, tokenizer, model_args = load_and_prepare_model(
                hf_model_id=_MODEL_ID,
                device=device,
                dtype=dtype_name,
                compile=False,
            )
        except Exception as exc:
            logger.exception("Failed to load Falcon model")
            raise RuntimeError(f"Falcon model load failed: {exc}") from exc

        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.engine = BatchInferenceEngine(model, tokenizer)
        self.stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.end_of_query_token_id,
        ]

        # Determine effective task (model may not support segmentation)
        self._supports_segmentation = bool(model_args.do_segmentation)
        logger.info(
            "Falcon ready | supports_segmentation=%s device=%s",
            self._supports_segmentation,
            device,
        )

    def decode_request(self, request: dict) -> dict:
        """Open image and prepare batch input tensors."""
        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import process_batch_and_generate

        image_path: str = request["image_path"]
        text: str = request["text_prompt"]
        task: str = request.get("task", _DEFAULT_PARAMS["task"])

        # Downgrade task if model doesn't support segmentation
        if task == "segmentation" and not self._supports_segmentation:
            task = "detection"

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        prompt = build_prompt_for_task(text, task)
        batch_inputs = process_batch_and_generate(
            self.tokenizer,
            [(image, prompt)],
            max_length=int(request.get("max_length", _DEFAULT_PARAMS["max_length"])),
            min_dimension=int(
                request.get("min_dimension", _DEFAULT_PARAMS["min_dimension"])
            ),
            max_dimension=int(
                request.get("max_dimension", _DEFAULT_PARAMS["max_dimension"])
            ),
        )
        return {
            "batch_inputs": batch_inputs,
            "task": task,
            "text": text,
            "max_new_tokens": int(
                request.get("max_new_tokens", _DEFAULT_PARAMS["max_new_tokens"])
            ),
            "seed": int(request.get("seed", _DEFAULT_PARAMS["seed"])),
        }

    def predict(self, batch: list[dict]) -> list[dict]:
        """Run Falcon generation per-item in the batch.

        LitServe collates concurrent requests into a list. We iterate to avoid
        the complexity of tensor-stacking with variable image/text sizes.
        """
        results = []
        for item in batch:
            batch_inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in item["batch_inputs"].items()
            }
            _, aux_out = self.engine.generate(
                **batch_inputs,
                max_new_tokens=item["max_new_tokens"],
                temperature=0.0,
                stop_token_ids=self.stop_token_ids,
                seed=item["seed"],
                task=item["task"],
            )
            results.append({"aux": aux_out[0], "task": item["task"], "text": item["text"]})
        return results

    def encode_response(self, result: dict) -> dict:
        """Extract bboxes from Falcon aux output; normalize to [0, 1]."""
        from falcon_perception.visualization_utils import pair_bbox_entries

        aux = result["aux"]
        task = result["task"]
        text = result["text"]

        paired_bboxes = pair_bbox_entries(aux.bboxes_raw)
        masks_rle: list = aux.masks_rle if task == "segmentation" else []

        # Parse class names from prompt (period-separated)
        class_names = [c.strip() for c in text.split(".") if c.strip()]

        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []

        for idx, bbox_entry in enumerate(paired_bboxes):
            mask_rle = masks_rle[idx] if idx < len(masks_rle) else None
            mask_bbox = _mask_rle_to_xyxy(mask_rle) if mask_rle else None

            # Fall back to coordinate-based bbox
            coord_bbox: tuple[float, float, float, float] | None = None
            xy = bbox_entry.get("x"), bbox_entry.get("y")
            hw = bbox_entry.get("h"), bbox_entry.get("w")
            if all(v is not None for v in (*xy, *hw)):
                try:
                    coord_bbox = _center_size_to_xyxy(
                        float(xy[0]), float(xy[1]), float(hw[1]), float(hw[0])
                    )
                except (TypeError, ValueError):
                    coord_bbox = None

            primary_bbox = mask_bbox or coord_bbox
            if primary_bbox is None:
                continue

            x1, y1, x2, y2 = primary_bbox
            boxes.append([x1, y1, x2, y2])
            scores.append(1.0)  # Falcon doesn't emit per-detection scores

            # Best-effort label: use class index from bbox_entry if available
            label_idx = bbox_entry.get("class_id")
            if label_idx is not None and int(label_idx) < len(class_names):
                label = class_names[int(label_idx)]
            else:
                label = class_names[0] if class_names else "object"
            labels.append(label)

        return {"boxes": boxes, "scores": scores, "labels": labels}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Falcon-Perception LitServe server")
    parser.add_argument("--port", type=int, default=3002, help="HTTP port to listen on")
    parser.add_argument("--device", type=int, default=1, help="CUDA device index")
    parser.add_argument(
        "--max-batch-size", type=int, default=4, help="Dynamic batch ceiling"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.1,
        help="Max seconds to wait before dispatching an incomplete batch",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=_MODEL_ID,
        help="HuggingFace model ID override",
    )
    args = parser.parse_args()

    if args.model_id != _MODEL_ID:
        _MODEL_ID = args.model_id  # noqa: F811

    logger.info(
        "Starting Falcon server | port=%d device=%d batch=%d timeout=%.3f",
        args.port,
        args.device,
        args.max_batch_size,
        args.batch_timeout,
    )

    api = FalconApi()
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=[args.device],
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
