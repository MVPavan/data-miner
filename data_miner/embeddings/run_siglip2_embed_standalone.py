"""Standalone SigLIP2 embedding runner (multi-GPU via stable hash sharding).

Mirrors the design of
``data_miner/auto_annotation_v4/models/run_sam3_dart_standalone.py`` but
writes into a LanceDB table instead of YOLO .txt files.

Launch one process per GPU via ``scripts_bench/run_embed_4gpu.sh``;
each process pins to a single GPU through ``CUDA_VISIBLE_DEVICES=i``
and processes the subset of items whose ``blake2s(image_id) % N ==
shard_index``. That's stable across runs even if the on-disk file set
changes between invocations.

Usage (single GPU)::

    python -m data_miner.embeddings.run_siglip2_embed_standalone \\
        --image-dir /mnt/data/deepak/DATASET/Laion/Good2/images \\
        --lance-uri /mnt/data/data_miner_lance/laion_good2 \\
        --device cuda:0 --batch-size 48 --num-workers 8 --resume

For the 4-GPU production run, use the shell launcher — it performs a
single scan-and-write-items-snapshot step up front and then fans out to
4 shards via ``--items-file``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_miner.config import SIGLIP2_MODELS  # noqa: E402
from data_miner.embeddings.lance_store import (  # noqa: E402
    DINOV3_DIM,
    ERRORS_TABLE_DEFAULT,
    LANCE_ROOT_DEFAULT,
    LanceEmbeddingWriter,
    SIGLIP2_DIMS,
    TABLE_DEFAULT,
    build_errors_schema,
    build_schema,
    utcnow,
)
from data_miner.models import SigLIPModel  # noqa: E402

# Re-enable decompression bomb protection via an explicit byte-size guard
# below. Leaving MAX_IMAGE_PIXELS at the PIL default means we still refuse
# absurdly large images before decoding, instead of disabling the guard.
MAX_IMAGE_BYTES = 200 * 1024 * 1024  # 200 MB on disk → don't try to decode
# Below this pixel side length, SigLIP2 (trained on 384²) will upscale a
# thumbnail and produce a semantically meaningless embedding with L2 norm 1.
# Better to flag it as an error than silently write garbage.
MIN_IMAGE_SIDE = 96

log = logging.getLogger("siglip2_embed_standalone")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Scan + shard
# ---------------------------------------------------------------------------


def scan_images(root: Path) -> list[tuple[str, str, str]]:
    """Return [(image_id=stem, absolute_path, subset=parent_dir_name)] sorted.

    ``subset`` is the name of the image's parent directory (e.g. ``Subset100``).
    De-duplicates by ``image_id``; on collision keeps the lexicographically
    smallest path (stable across filesystems and reruns — ``rglob`` order is
    not guaranteed). Writes the dropped-path audit to a sidecar file
    ``<root>/.embed_stem_collisions.log`` if any collisions occur.

    LanceDB has no primary-key constraint, so duplicate image_ids would
    silently produce duplicate rows in the embeddings table.
    """
    # Collect all candidate paths first so we can sort before dedup.
    paths: list[Path] = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    paths.sort()   # lexicographic (bytes-stable across filesystems)

    seen: dict[str, tuple[str, str]] = {}    # image_id -> (path, subset)
    collision_log: list[tuple[str, str, str]] = []   # (id, kept_path, dropped_path)
    for p in paths:
        image_id = p.stem
        path = str(p.resolve())
        subset = p.parent.name
        if image_id in seen:
            collision_log.append((image_id, seen[image_id][0], path))
            continue
        seen[image_id] = (path, subset)

    items = [(iid, path, subset) for iid, (path, subset) in seen.items()]
    items.sort(key=lambda t: t[0])

    if collision_log:
        audit_path = root / ".embed_stem_collisions.log"
        try:
            with audit_path.open("w", encoding="utf-8") as f:
                f.write("# image_id\tkept_path\tdropped_path\n")
                for iid, kept, dropped in collision_log:
                    f.write(f"{iid}\t{kept}\t{dropped}\n")
        except OSError as e:
            log.warning("could not write collision audit to %s: %s", audit_path, e)
        log.warning(
            "image_id (filename stem) collisions: %d paths dropped; "
            "audit log: %s. Consider a composite id if this matters.",
            len(collision_log), audit_path,
        )
    return items


def items_read_file(path: Path) -> list[tuple[str, str, str]]:
    """Read TSV: image_id\\tabspath\\tsubset per line."""
    out: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"malformed items line in {path}: {line!r}")
            out.append((parts[0], parts[1], parts[2]))
    return out


def items_write_file(path: Path, items: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for iid, p, subset in items:
            f.write(f"{iid}\t{p}\t{subset}\n")
    tmp.replace(path)


def shard_key(image_id: str, num_shards: int) -> int:
    """Stable hash -> shard index. blake2s is fast and non-cryptographic-hash-avoiding."""
    h = hashlib.blake2s(image_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") % num_shards


def filter_to_shard(
    items: list[tuple[str, str, str]], shard_index: int, shard_count: int,
) -> list[tuple[str, str, str]]:
    if shard_count <= 1:
        return items
    return [it for it in items if shard_key(it[0], shard_count) == shard_index]


# ---------------------------------------------------------------------------
# Dataset: decode RGB PIL on worker processes. Workers are spawned (not
# forked) so they don't inherit the parent's CUDA context or lancedb
# connection — neither of which is fork-safe.
# ---------------------------------------------------------------------------


class ImagePathDataset(Dataset):
    def __init__(self, items: list[tuple[str, str, str]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id, path, subset = self.items[idx]
        try:
            p = Path(path)
            try:
                sz = p.stat().st_size
            except OSError:
                sz = 0
            if sz > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"image file too large ({sz} bytes > {MAX_IMAGE_BYTES})",
                )
            img = Image.open(path)
            img.load()
            w, h = img.size
            if min(w, h) < MIN_IMAGE_SIDE:
                raise ValueError(
                    f"image too small ({w}x{h} < {MIN_IMAGE_SIDE}); "
                    "SigLIP2 would produce a meaningless embedding",
                )
            if img.mode != "RGB":
                img = img.convert("RGB")
            return {"image_id": image_id, "image_path": path, "subset": subset,
                    "image": img, "ok": True, "err": ""}
        except Exception as e:
            return {"image_id": image_id, "image_path": path, "subset": subset,
                    "image": None, "ok": False,
                    "err": f"{type(e).__name__}: {e}"}


def _collate_passthrough(batch):
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="Root directory of images (scanned recursively). "
                         "Either --image-dir OR --items-file is required.")
    ap.add_argument("--items-file", type=Path, default=None,
                    help="Pre-computed TSV of (image_id, abspath, subset). "
                         "Skips the scan; recommended for multi-shard runs "
                         "so the scan happens exactly once in the launcher.")
    ap.add_argument("--lance-uri", type=str, default=f"{LANCE_ROOT_DEFAULT}/laion_good2",
                    help="LanceDB root directory (per-dataset subdir)")
    ap.add_argument("--table", type=str, default=TABLE_DEFAULT,
                    help=f"Table name within --lance-uri (default: {TABLE_DEFAULT})")
    ap.add_argument("--errors-table", type=str, default=ERRORS_TABLE_DEFAULT)
    ap.add_argument("--model", type=str, default="siglip2-giant",
                    choices=list(SIGLIP2_MODELS.keys()),
                    help=(
                        "SigLIP2 variant key from SIGLIP2_MODELS. "
                        "Default 'siglip2-giant' (1536-dim); 'siglip2-so400m' "
                        "gives 1152-dim. A table created with one variant "
                        "cannot be appended to by another (schema dim mismatch)."
                    ))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--buffer-rows", type=int, default=1024,
                    help="Rows buffered before flushing to Lance (trade-off: "
                         "smaller = tighter crash recovery, more fragments)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="Skip image_ids already in the table / errors table")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-run images previously marked as errors")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N images (after sharding)")
    ap.add_argument("--log-every", type=int, default=50,
                    help="Log progress every N batches")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s[s%(shard)s]: %(message)s",
    )
    # inject shard index into log records for multi-shard log grepping
    old_factory = logging.getLogRecordFactory()
    shard_tag = str(args.shard_index)

    def _factory(*a, **kw):
        rec = old_factory(*a, **kw)
        rec.shard = shard_tag
        return rec
    logging.setLogRecordFactory(_factory)

    # --- Preflight --------------------------------------------------------
    if args.image_dir is None and args.items_file is None:
        log.error("one of --image-dir or --items-file is required")
        return 2
    if args.image_dir is not None and not args.image_dir.exists():
        log.error("image dir not found: %s", args.image_dir)
        return 2
    if args.items_file is not None and not args.items_file.exists():
        log.error("items file not found: %s", args.items_file)
        return 2
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        log.error("invalid shard: index=%d count=%d",
                  args.shard_index, args.shard_count)
        return 3

    # --- Items (scan or read) --------------------------------------------
    t0 = time.perf_counter()
    if args.items_file is not None:
        items = items_read_file(args.items_file)
        log.info("loaded %d items from %s in %.1fs",
                 len(items), args.items_file, time.perf_counter() - t0)
    else:
        log.info("scanning %s ...", args.image_dir)
        items = scan_images(args.image_dir)
        log.info("found %d images in %.1fs",
                 len(items), time.perf_counter() - t0)
    if not items:
        log.error("no items to process")
        return 4

    # --- Hash shard -------------------------------------------------------
    before = len(items)
    items = filter_to_shard(items, args.shard_index, args.shard_count)
    log.info("shard %d/%d: %d/%d items assigned (blake2s hash-sharded)",
             args.shard_index, args.shard_count, len(items), before)

    # --- Writers ----------------------------------------------------------
    if args.model not in SIGLIP2_DIMS:
        log.error("unknown embedding dim for model key %r; "
                  "add it to SIGLIP2_DIMS in lance_store.py", args.model)
        return 5
    siglip2_dim = SIGLIP2_DIMS[args.model]
    main_schema = build_schema(siglip2_dim=siglip2_dim, dinov3_dim=DINOV3_DIM)
    log.info("lance schema: siglip2_embedding=fp16[%d], dinov3_embedding=fp16[%d]",
             siglip2_dim, DINOV3_DIM)
    writer = LanceEmbeddingWriter(
        uri=args.lance_uri, table=args.table,
        schema=main_schema, buffer_rows=args.buffer_rows,
    )
    existing_dim = writer.table.schema.field("siglip2_embedding").type.list_size
    if existing_dim != siglip2_dim:
        log.error(
            "schema mismatch: existing table has siglip2_embedding dim=%d "
            "but --model=%s produces dim=%d. Use a different --table or "
            "--lance-uri for this model.",
            existing_dim, args.model, siglip2_dim,
        )
        return 6
    err_writer = LanceEmbeddingWriter(
        uri=args.lance_uri, table=args.errors_table,
        schema=build_errors_schema(), buffer_rows=256,
    )

    # --- Signal handlers: flush on SIGTERM/SIGINT/SIGHUP -----------------
    def _graceful(signum, _frame):
        log.warning("received signal %d; flushing writers and exiting", signum)
        try:
            writer.close()
        except Exception:
            log.exception("main writer close failed")
        try:
            err_writer.close()
        except Exception:
            log.exception("error writer close failed")
        sys.exit(128 + int(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _graceful)
        except Exception:  # e.g. SIGHUP unavailable on non-POSIX
            pass

    # --- Resume -----------------------------------------------------------
    if args.resume:
        t0 = time.perf_counter()
        shard_ids = {it[0] for it in items}
        done_ids = writer.existing_image_ids(shard_filter=shard_ids)
        err_ids = (set() if args.retry_errors
                   else err_writer.existing_image_ids(shard_filter=shard_ids))
        skip = done_ids | err_ids
        items = [it for it in items if it[0] not in skip]
        log.info("resume: %d done (%d errored) skipped in %.1fs; %d remain",
                 len(done_ids), len(err_ids),
                 time.perf_counter() - t0, len(items))
    if args.limit > 0:
        items = items[: args.limit]
        log.info("limit: processing first %d", len(items))
    if not items:
        log.info("nothing to do")
        writer.close()
        err_writer.close()
        return 0

    # --- Model ------------------------------------------------------------
    siglip2_model_id = SIGLIP2_MODELS[args.model]
    log.info("loading SigLIP2 (%s) on %s ...", args.model, args.device)
    model = SigLIPModel(model_id=siglip2_model_id, device_map=args.device)
    model.load()

    # --- DataLoader (spawn context so workers don't inherit CUDA / lance)
    dataset = ImagePathDataset(items)
    mp_ctx = None
    if args.num_workers > 0:
        import torch.multiprocessing as torch_mp
        mp_ctx = torch_mp.get_context("spawn")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_passthrough,
        pin_memory=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        shuffle=False,
        multiprocessing_context=mp_ctx,
    )

    # --- Main loop --------------------------------------------------------
    total = len(items)
    n_done = n_failed = 0
    window_t0 = time.perf_counter()
    window_done = 0
    overall_t0 = window_t0
    dim_verified = False
    log.info("starting inference: %d images | bs=%d workers=%d",
             total, args.batch_size, args.num_workers)

    for batch_idx, batch in enumerate(loader, start=1):
        ok_items = [b for b in batch if b["ok"]]
        bad_items = [b for b in batch if not b["ok"]]

        if bad_items:
            now = utcnow()
            for b in bad_items:
                log.warning("decode failed: %s (%s)", b["image_id"], b["err"])
                err_writer.add({
                    "image_id":   b["image_id"],
                    "image_path": b["image_path"],
                    "subset":     b["subset"],
                    "stage":      "decode",
                    "error":      b["err"],
                    "created_at": now,
                })
            n_failed += len(bad_items)

        batch_latency_ms = 0.0
        if ok_items:
            pil_images = [b["image"] for b in ok_items]
            t0 = time.perf_counter()
            try:
                embs = model.get_image_embeddings(
                    pil_images, batch_size=args.batch_size, show_progress=False,
                )
            except Exception as e:
                log.exception("batch infer failed (%d images): %s",
                              len(pil_images), e)
                now = utcnow()
                for b in ok_items:
                    err_writer.add({
                        "image_id":   b["image_id"],
                        "image_path": b["image_path"],
                        "subset":     b["subset"],
                        "stage":      "infer",
                        "error":      f"{type(e).__name__}: {e}",
                        "created_at": now,
                    })
                n_failed += len(ok_items)
                continue
            batch_latency_ms = (time.perf_counter() - t0) * 1000

            # First-batch sanity check — catches wrong pooler field, wrong
            # processor output, or a future refactor that returns per-token
            # features `(N, tokens, dim)` instead of pooled `(N, dim)`.
            if not dim_verified:
                if embs.ndim != 2:
                    log.error("SigLIP2 returned ndim=%d (shape=%s), expected 2D "
                              "(N, dim). Aborting before writes.",
                              embs.ndim, tuple(embs.shape))
                    writer.close()
                    err_writer.close()
                    return 7
                got_dim = int(embs.shape[1])
                if got_dim != siglip2_dim:
                    log.error("SigLIP2 returned dim=%d, expected %d for model=%s. "
                              "Aborting before writes.",
                              got_dim, siglip2_dim, args.model)
                    writer.close()
                    err_writer.close()
                    return 7
                log.info("verified embedding shape=%s matches schema",
                         tuple(embs.shape))
                dim_verified = True

            now = utcnow()
            rows = []
            for b, emb in zip(ok_items, embs):
                rows.append({
                    "image_id":          b["image_id"],
                    "image_path":        b["image_path"],
                    "subset":            b["subset"],
                    "siglip2_embedding": emb,
                    "dinov3_embedding":  None,
                    "siglip2_model_id":  siglip2_model_id,
                    "dinov3_model_id":   None,
                    "created_at":        now,
                })
            writer.extend(rows)
            n_done += len(ok_items)
            window_done += len(ok_items)
            # Release PIL image references now; otherwise they stay alive
            # until the next iteration's batch list replaces `batch`.
            for b in ok_items:
                b["image"] = None

        if batch_idx % args.log_every == 0:
            dt = time.perf_counter() - window_t0
            rate = window_done / dt if dt > 0 else 0.0
            overall_rate = n_done / (time.perf_counter() - overall_t0)
            remaining = total - n_done - n_failed
            eta_min = remaining / max(overall_rate, 1e-9) / 60.0
            log.info(
                "[%d/%d] ok=%d fail=%d | win=%.1f img/s overall=%.1f img/s "
                "| batch_latency=%.0fms eta=%.1fmin",
                n_done + n_failed, total, n_done, n_failed,
                rate, overall_rate, batch_latency_ms, eta_min,
            )
            window_t0 = time.perf_counter()
            window_done = 0

    # --- Final flush ------------------------------------------------------
    writer.close()
    err_writer.close()

    total_s = time.perf_counter() - overall_t0
    log.info("=== done ===")
    log.info("images processed: %d", n_done + n_failed)
    log.info("  embedded:       %d", n_done)
    log.info("  failed:         %d", n_failed)
    log.info("wall time:        %.1fs  (avg %.1f img/s)",
             total_s, (n_done + n_failed) / max(total_s, 1e-9))
    log.info("table rows total (incl. other shards): %d", writer.count_rows())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
