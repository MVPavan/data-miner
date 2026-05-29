"""Standalone SAM3-DART batch inference runner (YOLO output).

Runs sam3_dart as a single in-process model (no HTTP, no workers, no pipeline
orchestration) and writes one YOLO-format ``.txt`` per image into an output
dir — naturally parallel across GPUs because each image_id is a distinct
filename.

Design goals:
  * GPU fed continuously — PyTorch DataLoader with num_workers does image
    decode + resize on CPU threads so the GPU is never waiting on I/O.
  * Multi-image batching — uses Sam3MultiClassPredictorBatch.set_images +
    predict_batch which runs the backbone ONCE per batch and encoder+decoder
    ONCE at bs=B*N.
  * Filesystem output — no SQLite write contention on multi-GPU runs; resume
    is just "skip image_ids whose .txt already exists".

YOLO format:
  ``class_id cx cy w h score`` (6-col, default; standard ultralytics
  prediction format) or ``class_id cx cy w h`` with ``--no-yolo-with-conf``.
  Class ids come from ``config.yaml``'s ``class_registry[*].id`` unless
  ``--renumber-classes`` is passed.

To populate a pipeline.db from these .txt files, use a separate converter
script.

Usage:
    python -m data_miner.auto_annotation_v4.models.run_sam3_dart_standalone \\
        --image-dir /path/to/images \\
        --job-dir  /path/to/job_out \\
        [--config /path/to/config.yaml] \\
        [--out-dir DIR]  [--classes NAME,NAME]  [--renumber-classes] \\
        [--batch-size 8] [--num-workers 6] [--confidence 0.5] [--nms 0.7] \\
        [--resume] [--limit N] [--shard-index I --shard-count K]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_miner.auto_annotation_v4.models.sam3_dart import SAM3DartModel  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # large laion images

log = logging.getLogger("sam3_dart_standalone")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Dataset: decode + resize on worker processes
# ---------------------------------------------------------------------------


class ImagePathDataset(Dataset):
    """Decodes and resizes images on worker processes.

    Returns dicts so the main process pays pickle cost for the resized
    (~3MB) PIL image rather than the full-resolution source.
    """

    def __init__(self, items: list[tuple[str, str]], resolution: int) -> None:
        self.items = items
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id, path = self.items[idx]
        try:
            img = Image.open(path)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            img_resized = img.resize(
                (self.resolution, self.resolution), Image.Resampling.BILINEAR,
            )
            return {"image_id": image_id, "image": img_resized,
                    "orig_w": w, "orig_h": h, "ok": True, "err": ""}
        except Exception as e:
            return {"image_id": image_id, "image": None,
                    "orig_w": 0, "orig_h": 0, "ok": False,
                    "err": f"{type(e).__name__}: {e}"}


def _collate_passthrough(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


# ---------------------------------------------------------------------------
# Class registry loading
# ---------------------------------------------------------------------------


def _normalize_alias(name: str) -> str:
    return " ".join(
        name.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def load_class_registry(
    config_path: Path | None,
    job_dir: Path,
    selected: list[str] | None = None,
    renumber: bool = False,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Return ``(class_names, all_prompts, prompt_idx_to_class, class_id_of)``.

    * ``class_names``: canonical names in (filtered) class-id order.
    * ``all_prompts``: flat prompt list (canonical prompts + synonyms) fed to
      the detector; dedup is case/whitespace-insensitive.
    * ``prompt_idx_to_class``: same length as ``all_prompts``; maps each
      prompt index back to its canonical class_name.
    * ``class_id_of``: canonical ``name -> class_id`` used when writing YOLO
      .txt. Default comes from config's ``class_registry[*].id``; with
      ``renumber=True`` reassigned to ``0..N-1`` in filtered order.

    ``selected`` filters the registry to those class names (normalized match).
    Priority: --config, then <job_dir>/config.yaml, then <job_dir>/classes.txt.
    """
    selected_norm: set[str] | None = None
    if selected:
        selected_norm = {_normalize_alias(s) for s in selected if s.strip()}

    for candidate in (config_path, job_dir / "config.yaml"):
        if candidate is None or not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("could not parse %s as JSON: %s", candidate, e)
            continue
        reg = (data or {}).get("class_registry") or {}
        if not isinstance(reg, dict) or not reg:
            continue
        ordered = sorted(
            reg.items(),
            key=lambda kv: int(kv[1].get("id", 0)) if isinstance(kv[1], dict) else 0,
        )
        if selected_norm is not None:
            before = len(ordered)
            ordered = [(n, c) for n, c in ordered
                       if _normalize_alias(n) in selected_norm]
            picked = {_normalize_alias(n) for n, _ in ordered}
            missing = selected_norm - picked
            if missing:
                raise SystemExit(
                    f"--classes contained unknown names: {sorted(missing)}. "
                    f"Available: {sorted(_normalize_alias(n) for n in reg)}"
                )
            log.info("class filter: %d/%d kept (%s)",
                     len(ordered), before, [n for n, _ in ordered])

        class_names = [name for name, _ in ordered]
        all_prompts: list[str] = []
        prompt_idx_to_class: list[str] = []
        seen: set[str] = set()
        for name, cls in ordered:
            prompts_raw = list(cls.get("prompts") or []) if isinstance(cls, dict) else []
            synonyms_raw = list(cls.get("synonyms") or []) if isinstance(cls, dict) else []
            if not prompts_raw:
                prompts_raw = [name]
            for p in (*prompts_raw, *synonyms_raw):
                key = _normalize_alias(str(p))
                if not key or key in seen:
                    continue
                seen.add(key)
                all_prompts.append(str(p))
                prompt_idx_to_class.append(name)
        if renumber:
            class_id_of = {n: i for i, n in enumerate(class_names)}
        else:
            class_id_of = {
                name: int(cls["id"]) for name, cls in ordered
                if isinstance(cls, dict) and "id" in cls
            }
            for i, n in enumerate(class_names):
                class_id_of.setdefault(n, i)
        log.info(
            "loaded %d classes (%d prompts incl %d synonyms) from %s (renumber=%s)",
            len(class_names), len(all_prompts),
            len(all_prompts) - len(class_names), candidate, renumber,
        )
        return class_names, all_prompts, prompt_idx_to_class, class_id_of

    classes_txt = job_dir / "classes.txt"
    if classes_txt.exists():
        names = [
            ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if selected_norm is not None:
            before = len(names)
            names = [n for n in names if _normalize_alias(n) in selected_norm]
            log.info("class filter: %d/%d kept", len(names), before)
        log.info("loaded %d classes from %s (no synonyms)", len(names), classes_txt)
        return names, list(names), list(names), {n: i for i, n in enumerate(names)}

    raise SystemExit(
        f"No class registry found. Provide --config, or place "
        f"config.yaml / classes.txt under {job_dir}"
    )


# ---------------------------------------------------------------------------
# DART result -> YOLO .txt
# ---------------------------------------------------------------------------


def write_yolo_txt(
    out_dir: Path,
    image_id: str,
    orig_w: int,
    orig_h: int,
    res: dict[str, Any] | None,
    prompt_idx_to_class: list[str],
    class_id_of: dict[str, int],
    with_conf: bool,
) -> int:
    """Write one YOLO .txt per image. Returns kept-candidate count.

    DART returns pixel boxes already rescaled to ``(orig_w, orig_h)`` by
    ``_postprocess`` plus class_ids indexing into the current prompt list.
    ``prompt_idx_to_class`` collapses synonyms back to canonical class names
    so the ``class_id_of`` lookup uses canonical ids.

    Empty or failed images still produce an empty .txt file to preserve the
    one-file-per-image invariant.
    """
    lines: list[str] = []
    if res and "scores" in res and len(res["scores"]) > 0:
        def _tolist(x: Any) -> list:
            return x.detach().cpu().tolist() if torch.is_tensor(x) else list(x)

        w = max(1, orig_w)
        h = max(1, orig_h)
        for bx, sc, cid in zip(
            _tolist(res["boxes"]), _tolist(res["scores"]), _tolist(res["class_ids"])
        ):
            idx = int(cid)
            if idx < 0 or idx >= len(prompt_idx_to_class):
                continue
            cls_name = prompt_idx_to_class[idx]
            out_cid = class_id_of.get(cls_name)
            if out_cid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in bx)
            x1n = max(0.0, min(1.0, x1 / w))
            y1n = max(0.0, min(1.0, y1 / h))
            x2n = max(0.0, min(1.0, x2 / w))
            y2n = max(0.0, min(1.0, y2 / h))
            if x2n <= x1n or y2n <= y1n:
                continue
            cx = (x1n + x2n) / 2.0
            cy = (y1n + y2n) / 2.0
            bw = x2n - x1n
            bh = y2n - y1n
            if with_conf:
                lines.append(f"{out_cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(sc):.6f}")
            else:
                lines.append(f"{out_cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    (out_dir / f"{image_id}.txt").write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8",
    )
    return len(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def scan_images(root: Path) -> list[tuple[str, str]]:
    """Return [(image_id=stem, absolute_path)] sorted by image_id."""
    items: list[tuple[str, str]] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            items.append((p.stem, str(p.resolve())))
    items.sort(key=lambda t: t[0])
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--job-dir", type=Path, required=True,
                    help="Job dir; default output is <job-dir>/labels_standalone")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="YOLO .txt output dir (default: <job-dir>/labels_standalone)")
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to config.yaml (default: <job-dir>/config.yaml)")
    ap.add_argument("--classes", default="",
                    help="Comma-separated subset of class names (default: all)")
    ap.add_argument("--renumber-classes", action="store_true",
                    help="Renumber filtered classes to 0..N-1 in YOLO output "
                         "(default: keep config.yaml class ids)")
    ap.add_argument("--yolo-with-conf", action=argparse.BooleanOptionalAction,
                    default=True, help="Write confidence as 6th column (default: on)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.7)
    ap.add_argument("--presence-threshold", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true",
                    help="Skip images whose .txt already exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N images after sharding")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1,
                    help="Stride-slice items[shard_index::shard_count] per process")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Preflight ---
    if not args.image_dir.exists():
        log.error("image dir not found: %s", args.image_dir)
        return 2
    args.job_dir.mkdir(parents=True, exist_ok=True)
    out_dir = (args.out_dir or (args.job_dir / "labels_standalone")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("out dir: %s (with_conf=%s)", out_dir, args.yolo_with_conf)

    # --- Scan ---
    log.info("scanning images under %s ...", args.image_dir)
    t0 = time.perf_counter()
    items = scan_images(args.image_dir)
    log.info("found %d image files in %.1fs", len(items), time.perf_counter() - t0)
    if not items:
        log.error("no images found")
        return 3

    # --- Shard ---
    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        log.error("invalid shard settings: index=%d count=%d",
                  args.shard_index, args.shard_count)
        return 4
    if args.shard_count > 1:
        items = items[args.shard_index :: args.shard_count]
        log.info("shard %d/%d: %d images assigned",
                 args.shard_index, args.shard_count, len(items))

    # --- Classes + synonyms ---
    selected_classes = (
        [s.strip() for s in args.classes.split(",") if s.strip()]
        if args.classes else None
    )
    class_names, all_prompts, prompt_idx_to_class, class_id_of = load_class_registry(
        args.config, args.job_dir,
        selected=selected_classes, renumber=args.renumber_classes,
    )

    # --- Resume ---
    if args.resume:
        before = len(items)
        done = {p.stem for p in out_dir.glob("*.txt")}
        items = [it for it in items if it[0] not in done]
        log.info("resume: %d already done; %d remain", before - len(items), len(items))
    if args.limit > 0:
        items = items[: args.limit]
        log.info("limiting to %d images", len(items))
    if not items:
        log.info("nothing to do")
        return 0

    # --- Model load ---
    log.info("loading SAM3DartModel on %s ...", args.device)
    model = SAM3DartModel()
    model.load(
        device=args.device,
        detection_only=True,
        presence_threshold=args.presence_threshold,
    )
    with model._class_lock:
        model.predictor.set_classes(all_prompts)
        model._current_classes = tuple(all_prompts)
    log.info("model ready (resolution=%d, %d classes, %d prompts, ids=%s)",
             model.predictor.resolution, len(class_names), len(all_prompts),
             [class_id_of[n] for n in class_names])

    # --- DataLoader ---
    dataset = ImagePathDataset(items, model.predictor.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_passthrough,
        pin_memory=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        shuffle=False,
    )

    # --- Main loop ---
    total = len(items)
    n_done = n_empty = n_failed = n_detections = batch_idx = 0
    window_t0 = time.perf_counter()
    window_done = 0
    log.info("starting inference: %d images, batch=%d, workers=%d",
             total, args.batch_size, args.num_workers)
    overall_t0 = time.perf_counter()

    def write_empty_for(items_: list[dict]) -> None:
        for b in items_:
            write_yolo_txt(out_dir, b["image_id"], b["orig_w"], b["orig_h"],
                           None, prompt_idx_to_class, class_id_of,
                           args.yolo_with_conf)

    for batch in loader:
        batch_idx += 1
        ok_items = [b for b in batch if b["ok"]]
        bad_items = [b for b in batch if not b["ok"]]

        if bad_items:
            for b in bad_items:
                log.warning("decode failed: %s (%s)", b["image_id"], b["err"])
            write_empty_for(bad_items)
            n_failed += len(bad_items)
        if not ok_items:
            continue

        # True original sizes — dataset pre-resized to model.resolution, so
        # without this override DART would scale boxes back to the resized
        # frame instead of the true image frame.
        pil_images = [b["image"] for b in ok_items]
        true_sizes = [(b["orig_h"], b["orig_w"]) for b in ok_items]
        t0 = time.perf_counter()
        try:
            state = model.predictor.set_images(pil_images)
            state["original_sizes"] = true_sizes
            results = model.predictor.predict_batch(
                state,
                confidence_threshold=float(args.confidence),
                nms_threshold=float(args.nms),
            )
        except Exception as e:
            log.exception("batch predict failed (%d images): %s", len(pil_images), e)
            write_empty_for(ok_items)
            n_failed += len(ok_items)
            continue
        batch_latency_ms = (time.perf_counter() - t0) * 1000

        for b, res in zip(ok_items, results):
            kept = write_yolo_txt(
                out_dir, b["image_id"], b["orig_w"], b["orig_h"], res,
                prompt_idx_to_class, class_id_of, args.yolo_with_conf,
            )
            n_detections += kept
            if kept == 0:
                n_empty += 1

        n_done += len(ok_items)
        window_done += len(ok_items)

        if batch_idx % args.log_every == 0:
            dt = time.perf_counter() - window_t0
            rate = window_done / dt if dt > 0 else 0.0
            overall_rate = n_done / (time.perf_counter() - overall_t0)
            eta_s = (total - n_done - n_failed) / max(overall_rate, 1e-9)
            log.info(
                "[%d/%d] det=%d empty=%d fail=%d | win=%.1f img/s "
                "overall=%.1f img/s | batch_latency=%.0fms eta=%.1fmin",
                n_done + n_failed, total, n_detections, n_empty, n_failed,
                rate, overall_rate, batch_latency_ms, eta_s / 60.0,
            )
            window_t0 = time.perf_counter()
            window_done = 0

    total_s = time.perf_counter() - overall_t0
    log.info("=== done ===")
    log.info("images processed: %d", n_done + n_failed)
    log.info("  with detections: %d", n_done - n_empty)
    log.info("  empty:           %d", n_empty)
    log.info("  failed decode:   %d", n_failed)
    log.info("total detections:  %d", n_detections)
    log.info("wall time:         %.1fs  (avg %.1f img/s)",
             total_s, (n_done + n_failed) / max(total_s, 1e-9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
