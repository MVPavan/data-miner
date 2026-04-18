"""GDINO memory/throughput harness — standalone, no pipeline / no LitServe.

Why this exists:
    The production server OOMs at batch=1 on 24 GB 3090s because the current
    batched predictor does one forward pass with pixel_values expanded to
    (N_prompts, 3, H, W). With N=43 that runs the Swin backbone on 43 copies
    of the same image; memory peaks inside Swin window-attention.

    This harness benchmarks four strategies on the SAME image + prompts to
    find a setting that fits in 24 GB. It runs on GPU 0 (idle) so it does
    not interfere with the production GDINO server on GPUs 6,7.

Strategies:
    sequential       -- 1 prompt per forward, N forwards (the safety net).
    chunked[k]       -- k prompts per forward, ceil(N/k) forwards.
    joint            -- dot-concat "a. b. c. ..." into one text, 1 forward
                        (GroundingDINO's intended multi-class form). Faster
                        but may degrade recall on long prompt lists.
    small_image[s]   -- chunked with the image processor resized so the
                        shortest edge is s px (default 800). Reduces memory
                        quadratically.

Outputs one line per run:
    <strategy>  wall=<ms>  peak=<MiB>  dets=<n>  [OOM|OK]

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m data_miner.auto_annotation_v4.scripts.harness_gdino \
        --image /media/data_2/datasets/datasets_pavan/DataTang_val/002wu_f00518.jpg

    # Add --fp16 to load the model in float16 (halves weight+activation memory;
    #   GDINO's BERT text backbone works in fp16 for pure inference).
    # Add --warmup 2 to run each strategy twice and report the second timing
    #   (first call pays allocator / cuDNN-autotune overhead).
    # Use --only chunked16,joint to pick specific strategies.
    # Use --chunks 1,2,4,8,16,32 to override the default chunk sweep.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger("harness_gdino")


# ---------------------------------------------------------------------------
# Prompt loading -- re-uses the pipeline config so we measure the exact list
# production would send.
# ---------------------------------------------------------------------------
def load_prompts() -> list[str]:
    from data_miner.auto_annotation_v4.configs.loader import load_config
    from data_miner.auto_annotation_v4.utils import normalize_class_alias

    cfg = load_config()
    seen: set[str] = set()
    prompts: list[str] = []
    for c in cfg.classes.values():
        for p in (*c.prompts, *c.synonyms):
            k = normalize_class_alias(p)
            if k in seen:
                continue
            seen.add(k)
            prompts.append(p)
    return prompts


# ---------------------------------------------------------------------------
# Memory/timing helpers
# ---------------------------------------------------------------------------
def reset_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ---------------------------------------------------------------------------
# Forward-pass strategies
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_chunked(model, processor, image, prompts, *, chunk: int,
                threshold: float, text_threshold: float,
                image_processor_kwargs: dict | None = None) -> int:
    """Split the prompt list into chunks and run one forward pass per chunk.

    Returns the total number of surviving detections (post-threshold).
    chunk=1 -> sequential mode (one forward per prompt).
    """
    w, h = image.size
    image_processor_kwargs = image_processor_kwargs or {}

    # Image preprocessing is the same for every chunk -- do it once.
    image_inputs = processor.image_processor(
        images=[image], return_tensors="pt", **image_processor_kwargs
    )
    pv_single = image_inputs["pixel_values"]   # (1, 3, H, W)
    pm_single = image_inputs["pixel_mask"]     # (1, H, W)

    device = next(model.parameters()).device
    total = 0
    for start in range(0, len(prompts), chunk):
        sub = prompts[start:start + chunk]
        texts = [f"{p.strip()} ." for p in sub]
        text_inputs = processor.tokenizer(
            text=texts, padding=True, return_tensors="pt",
            return_token_type_ids=True,
        )
        n = len(sub)
        pixel_values = pv_single.expand(n, -1, -1, -1).contiguous().to(device)
        pixel_mask = pm_single.expand(n, -1, -1).contiguous().to(device)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        token_type_ids = text_inputs["token_type_ids"].to(device)

        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        post = processor.post_process_grounded_object_detection(
            outputs, input_ids=input_ids,
            threshold=threshold, text_threshold=text_threshold,
            target_sizes=[(h, w)] * n,
        )
        for p in post:
            boxes = p["boxes"]
            total += int(boxes.shape[0]) if hasattr(boxes, "shape") else len(boxes)

        # Drop per-chunk tensors before the next chunk so peak is bounded
        # by the chunk size rather than by the sum of chunks.
        del outputs, pixel_values, pixel_mask, input_ids
    return total


@torch.inference_mode()
def run_multi_image_chunked(model, processor, images, prompts, *,
                            chunk: int, threshold: float,
                            text_threshold: float) -> int:
    """Multi-image chunked: per chunk, fuse B images x k prompts -> B*k forward.

    Different from ``run_chunked`` (which loops images sequentially): here we
    stack (B images x chunk prompts) into one (B*chunk, 3, H, W) Swin pass,
    then chunk through the prompt list. Total forwards = ceil(N/chunk),
    each at batch B*chunk -- amortising backbone launch overhead across B
    images.

    All images must share the prompt list (the pipeline case). Pad-to-largest
    is handled by the image processor; pixel_mask carries the per-image
    valid region so deformable attention sampling is unaffected.

    Returns total detections across all (image, prompt) cells.
    """
    B = len(images)
    if B == 0:
        return 0
    device = next(model.parameters()).device

    # Pad images to a common (H, W) ONCE so the per-chunk expand() is cheap.
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    pv_all = image_inputs["pixel_values"]   # (B, 3, H, W)
    pm_all = image_inputs["pixel_mask"]     # (B, H, W)

    sizes = [(im.size[1], im.size[0]) for im in images]  # (h, w) per image
    total = 0
    for start in range(0, len(prompts), chunk):
        sub = prompts[start:start + chunk]
        k = len(sub)

        texts = [f"{p.strip()} ." for p in sub]
        text_inputs = processor.tokenizer(
            text=texts, padding=True, return_tensors="pt",
            return_token_type_ids=True,
        )

        # Image side: tile each image across k prompts -> (B*k, 3, H, W).
        # repeat_interleave keeps image-order stable so post_process can
        # use sizes[i] for cell (i, j).
        pv = pv_all.repeat_interleave(k, dim=0).contiguous().to(device)
        pm = pm_all.repeat_interleave(k, dim=0).contiguous().to(device)

        # Text side: tile k prompts B times -> (B*k, seq).
        ids = text_inputs["input_ids"].repeat(B, 1).to(device)
        am = text_inputs["attention_mask"].repeat(B, 1).to(device)
        tt = text_inputs["token_type_ids"].repeat(B, 1).to(device)

        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            outputs = model(pixel_values=pv, pixel_mask=pm,
                            input_ids=ids, attention_mask=am,
                            token_type_ids=tt)

        # target_sizes mirrors the tiling: image i appears at rows i*k..i*k+k-1
        target_sizes = [sizes[i] for i in range(B) for _ in range(k)]
        post = processor.post_process_grounded_object_detection(
            outputs, input_ids=ids,
            threshold=threshold, text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        for p in post:
            boxes = p["boxes"]
            total += int(boxes.shape[0]) if hasattr(boxes, "shape") else len(boxes)

        del outputs, pv, pm, ids, am, tt
    return total


@torch.inference_mode()
def run_joint(model, processor, image, prompts, *, threshold: float,
              text_threshold: float) -> int:
    """GroundingDINO's intended multi-class form: one text, one forward pass."""
    w, h = image.size
    device = next(model.parameters()).device
    # GDINO's post_process is very particular about delimiters -- `. ` between
    # classes and a trailing `.` is what the tokenizer was trained on.
    joint = ". ".join(p.strip() for p in prompts) + " ."
    inputs = processor(images=image, text=joint, return_tensors="pt")
    moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=device.type == "cuda"):
        outputs = model(**moved)
    post = processor.post_process_grounded_object_detection(
        outputs, input_ids=moved["input_ids"],
        threshold=threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]
    boxes = post["boxes"]
    return int(boxes.shape[0]) if hasattr(boxes, "shape") else len(boxes)


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------
def bench(name, fn, *, warmup: int) -> tuple[float, float, int, str | None]:
    """Run *fn* warmup+1 times; return (wall_ms, peak_mib, dets, error?)."""
    reset_mem()
    err: str | None = None
    dets = 0
    wall = 0.0
    try:
        for i in range(warmup + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dets = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000.0
            if i == 0:
                reset_mem()
    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM: {e}"
    except RuntimeError as e:
        err = f"{type(e).__name__}: {e}"
    peak = peak_mib()
    reset_mem()
    return wall, peak, dets, err


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark GDINO forward-pass strategies on one image."
    )
    ap.add_argument("--image", required=True, help="Path to a test image (single-image strategies).")
    ap.add_argument("--images-dir", default="",
                    help="Directory of images for the multi-image strategies. "
                         "First --multi-images files are loaded.")
    ap.add_argument("--multi-images", type=int, default=0,
                    help="Run multi-image strategies with B in {1,2,4,8} "
                         "capped to this number. 0 disables. Requires --images-dir.")
    ap.add_argument("--multi-batches", default="1,2,4,8",
                    help="Comma-separated B values for multi-image batching.")
    ap.add_argument("--multi-chunks", default="1,2,4",
                    help="Comma-separated chunk sizes for multi-image batching.")
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.2)
    ap.add_argument("--fp16", action="store_true",
                    help="Cast model weights to float16 (not just autocast).")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup iterations discarded before timed run.")
    ap.add_argument("--chunks", default="1,2,4,8,16,32",
                    help="Comma-separated chunk sizes to sweep.")
    ap.add_argument("--only", default="",
                    help="Optional comma-separated subset of strategy names "
                         "(e.g. 'chunked16,joint,small800_chunked8'). "
                         "Empty means run everything.")
    ap.add_argument("--small-edge", type=int, default=800,
                    help="Shortest-edge pixels for the 'small_image[s]' variant.")
    ap.add_argument("--json", action="store_true",
                    help="Emit one JSON line per result (for tooling).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        return 2

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}", file=sys.stderr)
        return 2

    # --- Load image + prompts ---------------------------------------------
    image = Image.open(img_path).convert("RGB")
    prompts = load_prompts()

    # Optional: load extra images for the multi-image strategies.
    multi_images: list[Image.Image] = []
    if args.multi_images > 0:
        if not args.images_dir:
            print("ERROR: --multi-images requires --images-dir", file=sys.stderr)
            return 2
        ddir = Path(args.images_dir)
        files = sorted(ddir.glob("*.jpg"))[: args.multi_images]
        if len(files) < args.multi_images:
            print(f"WARN: only {len(files)} images found, requested "
                  f"{args.multi_images}", file=sys.stderr)
        multi_images = [Image.open(f).convert("RGB") for f in files]
        print(f"# multi imgs  : {len(multi_images)} from {ddir}")
    print(f"# image       : {img_path}  size={image.size}")
    print(f"# prompts     : N={len(prompts)}")
    print(f"# gpu         : {torch.cuda.get_device_name(0)}  "
          f"cuda_visible={torch.cuda.device_count()} device(s)")
    print(f"# model       : {args.model_id}  fp16={args.fp16}")
    print()

    # --- Load model -------------------------------------------------------
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(args.model_id, torch_dtype=dtype)
        .to("cuda:0")
        .eval()
    )

    # Record baseline model memory so "peak" below is addition beyond weights.
    torch.cuda.synchronize()
    base_mib = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"# weights     : {base_mib:.0f} MiB resident after load\n")

    # --- Build strategy table --------------------------------------------
    chunks = [int(c) for c in args.chunks.split(",") if c.strip()]
    strategies: list[tuple[str, callable]] = []

    for k in chunks:
        label = "sequential" if k == 1 else f"chunked{k}"
        strategies.append((
            label,
            lambda k=k: run_chunked(
                model, processor, image, prompts, chunk=k,
                threshold=args.threshold, text_threshold=args.text_threshold,
            ),
        ))

    strategies.append((
        "joint",
        lambda: run_joint(
            model, processor, image, prompts,
            threshold=args.threshold, text_threshold=args.text_threshold,
        ),
    ))

    # small-image variant at two chunk sizes (8 and 16) to see the tradeoff.
    for k in (8, 16):
        strategies.append((
            f"small{args.small_edge}_chunked{k}",
            lambda k=k: run_chunked(
                model, processor, image, prompts, chunk=k,
                threshold=args.threshold, text_threshold=args.text_threshold,
                image_processor_kwargs={
                    "size": {"shortest_edge": args.small_edge,
                             "longest_edge": int(args.small_edge * 4 / 3)},
                },
            ),
        ))

    # Multi-image batched: only added when --multi-images is set.
    multi_specs: list[tuple[int, int]] = []
    if multi_images:
        bs = sorted({int(b) for b in args.multi_batches.split(",") if b.strip()})
        ks = sorted({int(k) for k in args.multi_chunks.split(",") if k.strip()})
        for B in bs:
            B = min(B, len(multi_images))
            for k in ks:
                multi_specs.append((B, k))
        # Dedupe (B,k) once B was capped to len(multi_images).
        seen = set()
        for B, k in multi_specs:
            tag = (B, k)
            if tag in seen:
                continue
            seen.add(tag)
            sub_imgs = multi_images[:B]
            strategies.append((
                f"multi{B}_chunk{k}",
                lambda B=B, k=k, sub_imgs=sub_imgs: run_multi_image_chunked(
                    model, processor, sub_imgs, prompts, chunk=k,
                    threshold=args.threshold, text_threshold=args.text_threshold,
                ),
            ))

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        strategies = [s for s in strategies if s[0] in wanted]
        if not strategies:
            print("ERROR: --only did not match any strategy name.", file=sys.stderr)
            return 2

    # --- Run --------------------------------------------------------------
    header = (f"{'strategy':<22} {'wall(ms)':>10} {'ms/img':>9} "
              f"{'peak(MiB)':>11} {'dets':>6}  status")
    print(header)
    print("-" * len(header))
    rows = []
    for name, fn in strategies:
        wall, peak, dets, err = bench(name, fn, warmup=args.warmup)
        # Infer batch size from name for ms/img column ('multi{B}_...' else 1).
        b = 1
        if name.startswith("multi"):
            try:
                b = int(name.split("_")[0].removeprefix("multi"))
            except ValueError:
                b = 1
        ms_per_img = wall / b if wall > 0 else 0.0
        status = err if err else "OK"
        row = {"strategy": name, "wall_ms": wall, "ms_per_img": ms_per_img,
               "peak_mib": peak, "dets": dets, "batch": b, "status": status}
        rows.append(row)
        if args.json:
            print(json.dumps(row))
        else:
            print(f"{name:<22} {wall:>10.1f} {ms_per_img:>9.1f} "
                  f"{peak:>11.1f} {dets:>6}  {status}")

    # --- Summary ----------------------------------------------------------
    ok = [r for r in rows if r["status"] == "OK"]
    if not ok:
        print("\nAll strategies OOMed -- try --fp16 or a smaller --small-edge.")
        return 1
    fastest = min(ok, key=lambda r: r["wall_ms"])
    leanest = min(ok, key=lambda r: r["peak_mib"])
    best_throughput = min(ok, key=lambda r: r["ms_per_img"])
    print()
    print(f"fastest OK : {fastest['strategy']}  "
          f"{fastest['wall_ms']:.0f} ms total / {fastest['peak_mib']:.0f} MiB / "
          f"{fastest['dets']} dets")
    print(f"leanest OK : {leanest['strategy']}  "
          f"{leanest['wall_ms']:.0f} ms total / {leanest['peak_mib']:.0f} MiB / "
          f"{leanest['dets']} dets")
    print(f"best/img OK: {best_throughput['strategy']}  "
          f"{best_throughput['ms_per_img']:.0f} ms/img "
          f"(batch {best_throughput['batch']}) "
          f"/ {best_throughput['peak_mib']:.0f} MiB / "
          f"{best_throughput['dets']} total dets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
