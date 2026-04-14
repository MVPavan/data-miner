"""LitServe server for SAM3 (Segment-Anything Model 3).

Single endpoint, dual-mode via request["mode"]:

  mode="proposal"  — text-prompted segmentation → candidate detections
    Request:
        {
            "mode": "proposal",
            "image_path": "/abs/path/image.jpg",
            "text_prompt": "person",
            "threshold": 0.5           # optional
        }
    Response:
        {
            "boxes":  [[x1,y1,x2,y2], ...],   # normalized 0-1
            "scores": [float, ...],
            "labels": ["person", ...]
        }

  mode="refine"  — box+point-prompted mask refinement → tighter bbox
    Request:
        {
            "mode": "refine",
            "image_path": "/abs/path/image.jpg",
            "bbox": [x1, y1, x2, y2],          # normalized 0-1
            "points": [[px, py, label], ...],   # optional; pixel coords, label 0/1
            "threshold": 0.5                    # optional
        }
    Response:
        {
            "box":   [x1, y1, x2, y2],   # normalized 0-1  (best mask)
            "score": float
        }

Launch:
    python serve_sam3.py --port 3003 --device 2 --max-batch-size 8
"""

from __future__ import annotations

import logging
import sys

import litserve as ls
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/sam3"


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _norm_to_pixels(
    bbox_norm: list[float], img_w: int, img_h: int
) -> list[float]:
    """Convert normalized [0,1] bbox to absolute pixel coordinates."""
    x1, y1, x2, y2 = bbox_norm
    return [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]


class SAM3Api(ls.LitAPI):
    """LitAPI wrapper around SAM3 supporting proposal and refinement modes."""

    # ------------------------------------------------------------------
    # LitServe lifecycle
    # ------------------------------------------------------------------

    def setup(self, device: str) -> None:
        """Load SAM3 processor and model onto *device*."""
        self.device = device

        logger.info("Loading SAM3 processor from %s", _MODEL_ID)
        try:
            from transformers import Sam3Model, Sam3Processor

            self.processor = Sam3Processor.from_pretrained(_MODEL_ID)
        except Exception as exc:
            logger.exception("Failed to load SAM3 processor")
            raise RuntimeError(f"SAM3 processor load failed: {exc}") from exc

        torch_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
        logger.info(
            "Loading SAM3 model onto device=%s dtype=%s", device, torch_dtype
        )
        try:
            from transformers import Sam3Model

            self.model = (
                Sam3Model.from_pretrained(_MODEL_ID, torch_dtype=torch_dtype)
                .to(device)
                .eval()
            )
        except Exception as exc:
            logger.exception("Failed to load SAM3 model")
            raise RuntimeError(f"SAM3 model load failed: {exc}") from exc

        logger.info("SAM3 ready on %s", device)

    # ------------------------------------------------------------------
    # Request / predict / response
    # ------------------------------------------------------------------

    def decode_request(self, request: dict) -> dict:
        mode: str = request.get("mode", "proposal")
        if mode not in ("proposal", "refine"):
            raise ValueError(f"Unknown SAM3 mode '{mode}'. Use 'proposal' or 'refine'.")

        image_path: str = request["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        w, h = image.size
        threshold: float = float(request.get("threshold", 0.5))

        if mode == "proposal":
            text: str = request["text_prompt"]
            inputs = self.processor(
                images=image, text=text, return_tensors="pt"
            )
            inputs = inputs.to(device=self.device, dtype=self.model.dtype)
            return {
                "mode": "proposal",
                "inputs": inputs,
                "image_size": (w, h),
                "label": text,
                "threshold": threshold,
            }

        # mode == "refine"
        bbox_norm: list[float] = request["bbox"]   # [x1,y1,x2,y2] in [0,1]
        pixel_box = _norm_to_pixels(bbox_norm, w, h)

        raw_points: list[list] = request.get("points") or []
        if raw_points:
            input_points = [[[p[0], p[1]] for p in raw_points]]
            input_labels = [[int(p[2]) for p in raw_points]]
            inputs = self.processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=[[pixel_box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device=self.device, dtype=self.model.dtype)
        else:
            inputs = self.processor(
                images=image,
                input_boxes=[[pixel_box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device=self.device, dtype=self.model.dtype)

        return {
            "mode": "refine",
            "inputs": inputs,
            "image_size": (w, h),
            "threshold": threshold,
        }

    def predict(self, batch: list[dict]) -> list[dict]:
        """Run SAM3 forward pass per-item in the batch.

        LitServe collates concurrent requests into a list. We iterate to avoid
        the complexity of tensor-stacking with variable image/text sizes.
        """
        results = []
        for item in batch:
            with torch.no_grad():
                outputs = self.model(**item["inputs"])
            results.append({
                "mode": item["mode"],
                "outputs": outputs,
                "inputs": item["inputs"],
                "image_size": item["image_size"],
                "label": item.get("label", "object"),
                "threshold": item["threshold"],
            })
        return results

    def encode_response(self, result: dict) -> dict:
        """Post-process SAM3 outputs; normalize boxes to [0, 1]."""
        mode: str = result["mode"]
        outputs = result["outputs"]
        inputs = result["inputs"]
        w, h = result["image_size"]
        threshold: float = result["threshold"]

        post = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        raw_boxes = post.get("boxes")
        scores_t = post.get("scores")

        if raw_boxes is None or scores_t is None or len(raw_boxes) == 0:
            if mode == "proposal":
                return {"boxes": [], "scores": [], "labels": []}
            else:
                return {"box": None, "score": 0.0}

        scores_list = (
            scores_t.cpu().tolist()
            if torch.is_tensor(scores_t)
            else list(scores_t)
        )

        def _norm_box(box_t) -> list[float]:
            x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
            return [_clamp(x1 / w), _clamp(y1 / h), _clamp(x2 / w), _clamp(y2 / h)]

        if mode == "proposal":
            label = result["label"]
            return {
                "boxes": [_norm_box(b) for b in raw_boxes],
                "scores": scores_list,
                "labels": [label] * len(raw_boxes),
            }

        # refine: pick the highest-scoring mask
        if torch.is_tensor(scores_t):
            best_idx = int(torch.argmax(scores_t).item())
        else:
            best_idx = int(max(range(len(scores_list)), key=lambda i: scores_list[i]))

        return {
            "box": _norm_box(raw_boxes[best_idx]),
            "score": scores_list[best_idx],
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

    parser = argparse.ArgumentParser(description="SAM3 LitServe server (proposal + refine)")
    parser.add_argument("--port", type=int, default=3003, help="HTTP port to listen on")
    parser.add_argument("--device", type=int, default=2, help="CUDA device index")
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

    if args.model_id != _MODEL_ID:
        _MODEL_ID = args.model_id  # noqa: F811

    logger.info(
        "Starting SAM3 server | port=%d device=%d batch=%d timeout=%.3f",
        args.port,
        args.device,
        args.max_batch_size,
        args.batch_timeout,
    )

    api = SAM3Api()
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=[args.device],
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )
    server.run(port=args.port)
