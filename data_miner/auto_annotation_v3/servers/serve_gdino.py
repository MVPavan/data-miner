"""LitServe server for GroundingDINO zero-shot object detection.

Endpoint: POST /predict
Request:
    {
        "image_path": "/abs/path/to/image.jpg",
        "text_prompt": "person . forklift . palletjack ."
    }
Response:
    {
        "boxes":  [[x1,y1,x2,y2], ...],   # normalized 0-1
        "scores": [float, ...],
        "labels": ["person", ...]
    }

Launch:
    python serve_gdino.py --port 3001 --device 0 --max-batch-size 8
"""

from __future__ import annotations

import logging
import sys

import litserve as ls
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger(__name__)

_MODEL_ID = "IDEA-Research/grounding-dino-base"


class GDINOApi(ls.LitAPI):
    """LitAPI wrapper around GroundingDINO for zero-shot detection."""

    # ------------------------------------------------------------------
    # LitServe lifecycle
    # ------------------------------------------------------------------

    def setup(self, device: str) -> None:
        """Load processor and model onto *device*."""
        self.device = device
        logger.info("Loading GroundingDINO processor from %s", _MODEL_ID)
        try:
            self.processor = AutoProcessor.from_pretrained(_MODEL_ID)
        except Exception as exc:
            logger.exception("Failed to load GroundingDINO processor")
            raise RuntimeError(f"GroundingDINO processor load failed: {exc}") from exc

        logger.info("Loading GroundingDINO model onto device=%s", device)
        try:
            self.model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)
                .to(device)
                .eval()
            )
        except Exception as exc:
            logger.exception("Failed to load GroundingDINO model")
            raise RuntimeError(f"GroundingDINO model load failed: {exc}") from exc

        logger.info("GroundingDINO ready on %s", device)

    def decode_request(self, request: dict) -> dict:
        """Open image, run processor, return tensor dict + metadata."""
        image_path: str = request["image_path"]
        text: str = request["text_prompt"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        # Processor expects text as a string (period-separated classes)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        w, h = image.size
        return {
            "inputs": inputs,
            "input_ids": inputs["input_ids"],
            "image_size": (w, h),
            "text": text,
        }

    def predict(self, batch: dict) -> dict:
        """Run model forward pass (called with a *batched* item from LitServe)."""
        # Move all tensors to the model device
        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch["inputs"].items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)
        return {
            "outputs": outputs,
            "input_ids": batch["input_ids"],
            "image_size": batch["image_size"],
        }

    def encode_response(self, result: dict) -> dict:
        """Post-process detections and normalize boxes to [0, 1]."""
        w, h = result["image_size"]
        post = self.processor.post_process_grounded_object_detection(
            result["outputs"],
            result["input_ids"],
            threshold=0.25,
            text_threshold=0.2,
            target_sizes=[(h, w)],
        )[0]

        raw_boxes = post["boxes"].cpu().tolist()   # absolute pixel coords
        scores = post["scores"].cpu().tolist()
        labels = post["labels"]                    # list[str]

        # Normalize to [0, 1]
        norm_boxes = [
            [
                max(0.0, min(1.0, x1 / w)),
                max(0.0, min(1.0, y1 / h)),
                max(0.0, min(1.0, x2 / w)),
                max(0.0, min(1.0, y2 / h)),
            ]
            for x1, y1, x2, y2 in raw_boxes
        ]

        return {
            "boxes": norm_boxes,
            "scores": scores,
            "labels": labels,
        }


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

    parser = argparse.ArgumentParser(description="GroundingDINO LitServe server")
    parser.add_argument("--port", type=int, default=3001, help="HTTP port to listen on")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--max-batch-size", type=int, default=8, help="Dynamic batch ceiling"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.05,
        help="Max seconds to wait before dispatching an incomplete batch",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=_MODEL_ID,
        help="HuggingFace model ID override",
    )
    args = parser.parse_args()

    # Allow model-id override at launch time
    if args.model_id != _MODEL_ID:
        _MODEL_ID = args.model_id  # noqa: F811  (module-level reassign)

    logger.info(
        "Starting GroundingDINO server | port=%d device=%d batch=%d timeout=%.3f",
        args.port,
        args.device,
        args.max_batch_size,
        args.batch_timeout,
    )

    api = GDINOApi()
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=[args.device],
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
