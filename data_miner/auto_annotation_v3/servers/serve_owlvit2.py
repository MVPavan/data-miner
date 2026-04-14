"""LitServe server for OWLv2 (OWL-ViT v2) zero-shot object detection.

Endpoint: POST /predict
Request:
    {
        "image_path": "/abs/path/to/image.jpg",
        "text_queries": ["a photo of a person", "a photo of a forklift"],
        "threshold": 0.1   # optional, default 0.1
    }
Response:
    {
        "boxes":  [[x1,y1,x2,y2], ...],   # normalized 0-1
        "scores": [float, ...],
        "labels": ["a photo of a person", ...]
    }

Launch:
    python serve_owlvit2.py --port 3004 --device 0 --max-batch-size 8
"""

from __future__ import annotations

import logging
import sys

import litserve as ls
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

logger = logging.getLogger(__name__)

_MODEL_ID = "google/owlv2-base-patch16-ensemble"
_DEFAULT_THRESHOLD = 0.1


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class OWLv2Api(ls.LitAPI):
    """LitAPI wrapper around OWLv2 for zero-shot open-vocabulary detection."""

    # ------------------------------------------------------------------
    # LitServe lifecycle
    # ------------------------------------------------------------------

    def setup(self, device: str) -> None:
        """Load OWLv2 processor and model onto *device*."""
        self.device = device

        logger.info("Loading OWLv2 processor from %s", _MODEL_ID)
        try:
            self.processor = Owlv2Processor.from_pretrained(_MODEL_ID)
        except Exception as exc:
            logger.exception("Failed to load OWLv2 processor")
            raise RuntimeError(f"OWLv2 processor load failed: {exc}") from exc

        torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32
        logger.info(
            "Loading OWLv2 model onto device=%s dtype=%s", device, torch_dtype
        )
        try:
            self.model = (
                Owlv2ForObjectDetection.from_pretrained(
                    _MODEL_ID, torch_dtype=torch_dtype
                )
                .to(device)
                .eval()
            )
        except Exception as exc:
            logger.exception("Failed to load OWLv2 model")
            raise RuntimeError(f"OWLv2 model load failed: {exc}") from exc

        logger.info("OWLv2 ready on %s", device)

    # ------------------------------------------------------------------
    # Request / predict / response
    # ------------------------------------------------------------------

    def decode_request(self, request: dict) -> dict:
        """Open image, build processor inputs."""
        image_path: str = request["image_path"]
        text_queries: list[str] = request["text_queries"]
        threshold: float = float(request.get("threshold", _DEFAULT_THRESHOLD))

        if not text_queries:
            raise ValueError("text_queries must be a non-empty list of strings")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        w, h = image.size

        # OWLv2 expects a nested list: [[query0, query1, ...]] (one set per image)
        inputs = self.processor(
            text=[text_queries],
            images=image,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        target_sizes = torch.tensor([[h, w]], device=self.device)

        return {
            "inputs": inputs,
            "target_sizes": target_sizes,
            "image_size": (w, h),
            "text_queries": text_queries,
            "threshold": threshold,
        }

    def predict(self, x: dict) -> dict:
        """Run OWLv2 forward pass."""
        with torch.no_grad():
            outputs = self.model(**x["inputs"])
        return {
            "outputs": outputs,
            "target_sizes": x["target_sizes"],
            "image_size": x["image_size"],
            "text_queries": x["text_queries"],
            "threshold": x["threshold"],
        }

    def encode_response(self, result: dict) -> dict:
        """Post-process OWLv2 outputs; normalize boxes to [0, 1]."""
        w, h = result["image_size"]
        text_queries: list[str] = result["text_queries"]
        threshold: float = result["threshold"]

        post = self.processor.post_process_grounded_object_detection(
            outputs=result["outputs"],
            target_sizes=result["target_sizes"],
            threshold=threshold,
        )[0]

        raw_boxes = post.get("boxes")
        scores_t = post.get("scores")
        label_idxs = post.get("labels")

        if raw_boxes is None or len(raw_boxes) == 0:
            return {"boxes": [], "scores": [], "labels": []}

        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []

        scores_list = (
            scores_t.cpu().tolist() if torch.is_tensor(scores_t) else list(scores_t)
        )
        label_list = (
            label_idxs.cpu().tolist()
            if torch.is_tensor(label_idxs)
            else list(label_idxs)
        )

        for box_t, score, label_idx in zip(raw_boxes, scores_list, label_list):
            x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
            boxes.append(
                [_clamp(x1 / w), _clamp(y1 / h), _clamp(x2 / w), _clamp(y2 / h)]
            )
            scores.append(float(score))
            idx = int(label_idx)
            labels.append(
                text_queries[idx] if idx < len(text_queries) else "object"
            )

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

    parser = argparse.ArgumentParser(description="OWLv2 LitServe server")
    parser.add_argument("--port", type=int, default=3004, help="HTTP port to listen on")
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help="Default detection score threshold",
    )
    args = parser.parse_args()

    if args.model_id != _MODEL_ID:
        _MODEL_ID = args.model_id  # noqa: F811
    if args.threshold != _DEFAULT_THRESHOLD:
        _DEFAULT_THRESHOLD = args.threshold  # noqa: F811

    logger.info(
        "Starting OWLv2 server | port=%d device=%d batch=%d timeout=%.3f threshold=%.3f",
        args.port,
        args.device,
        args.max_batch_size,
        args.batch_timeout,
        args.threshold,
    )

    api = OWLv2Api()
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=[args.device],
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
