"""Concurrent load test for the GDINO LitServe server.

Single-GPU `harness_gdino.py` measured per-forward speed in isolation. This
script measures end-to-end production throughput against the running server
on `:3001`, including: HTTP overhead, LitServe batching window, multi-worker
GPU parallelism, image-load I/O.

It dispatches `--total` requests across `--concurrency` aiohttp tasks
(matching the pipeline's `aiohttp.ClientSession` shape), pinning the same
prompt list production sends. Outputs req/s, latency percentiles, OK/error
counts, average detection count per image.

Usage:
    # Sweep concurrency 1,2,4,8,16,32 against ./DataTang_val
    python -m data_miner.auto_annotation_v4.scripts.loadtest_gdino \
        --images-dir /media/data_2/datasets/datasets_pavan/DataTang_val \
        --total 64

    # Single fixed concurrency (e.g. mimic 4 detect_model workers)
    python -m data_miner.auto_annotation_v4.scripts.loadtest_gdino \
        --images-dir /media/data_2/datasets/datasets_pavan/DataTang_val \
        --total 200 --concurrency 4

    # Different host/port
    python -m data_miner.auto_annotation_v4.scripts.loadtest_gdino \
        --images-dir ... --url http://localhost:3001/predict --total 64
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import time
from pathlib import Path

import aiohttp


def load_prompts() -> list[str]:
    """Pull the exact prompt list the pipeline would send (canonical+synonyms)."""
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


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    image_path: str,
    prompts: list[str],
    timeout: float,
) -> tuple[float, int, str | None]:
    """Send one /predict, return (latency_ms, det_count, error?)."""
    payload = {"image_path": image_path, "prompts": prompts}
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            ms = (time.perf_counter() - t0) * 1000.0
            if resp.status != 200:
                body = await resp.text()
                return ms, 0, f"HTTP {resp.status}: {body[:200]}"
            data = await resp.json()
            n = len(data.get("boxes", []))
            return ms, n, None
    except (TimeoutError, asyncio.TimeoutError):
        return (time.perf_counter() - t0) * 1000.0, 0, "timeout"
    except Exception as e:
        return (time.perf_counter() - t0) * 1000.0, 0, f"{type(e).__name__}: {e}"


async def run_level(
    url: str,
    images: list[str],
    prompts: list[str],
    *,
    total: int,
    concurrency: int,
    timeout: float,
) -> dict:
    """Send `total` requests with `concurrency` parallel workers; collect stats."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    dets: list[int] = []
    errors: list[str] = []

    async def worker(i: int) -> None:
        img = images[i % len(images)]
        async with sem:
            ms, n, err = await fire_one(session, url, img, prompts, timeout)
            latencies.append(ms)
            dets.append(n)
            if err:
                errors.append(err)

    timeout_cfg = aiohttp.ClientTimeout(total=None, connect=10)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total)))
        wall = time.perf_counter() - t0

    ok = total - len(errors)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20 else max(latencies, default=0.0)
    )
    avg = statistics.mean(latencies) if latencies else 0.0
    avg_dets = statistics.mean(dets) if dets else 0.0
    return {
        "concurrency": concurrency,
        "total": total,
        "ok": ok,
        "errors": len(errors),
        "wall_s": wall,
        "rps": ok / wall if wall > 0 else 0.0,
        "latency_avg_ms": avg,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "avg_dets": avg_dets,
        "first_errors": errors[:3],
    }


async def warmup(url: str, image: str, prompts: list[str]) -> None:
    """One sequential request per LitServe worker so loaders+CUDA contexts settle.

    Without this, the first measured run is dominated by lazy model load on
    each worker and the result is unrepresentative.
    """
    async with aiohttp.ClientSession() as session:
        # Empirically, 8 sequential requests are enough to hit each worker
        # in the round-robin LitServe scheduler with 8 GPUs.
        for _ in range(8):
            await fire_one(session, url, image, prompts, timeout=300)


async def main_async(args) -> int:
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 2
    files = sorted(str(p) for p in images_dir.glob("*.jpg"))
    if not files:
        print(f"ERROR: no .jpg in {images_dir}", file=sys.stderr)
        return 2

    pool = files[: max(args.total, args.warmup_pool)]
    prompts = load_prompts()

    print(f"# url         : {args.url}")
    print(f"# images dir  : {images_dir}  ({len(files)} files, using {len(pool)})")
    print(f"# prompts     : N={len(prompts)}")
    print(f"# total reqs  : {args.total}")

    if not args.skip_warmup:
        print("# warmup      : sending 8 sequential requests to fill all workers...")
        t0 = time.perf_counter()
        await warmup(args.url, pool[0], prompts)
        print(f"#               warmup done in {time.perf_counter() - t0:.1f}s")
    print()

    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
    header = (f"{'conc':>5} {'ok/total':>10} {'wall(s)':>8} {'rps':>7} "
              f"{'avg(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} "
              f"{'dets':>6} {'errors':>7}")
    print(header)
    print("-" * len(header))
    for c in levels:
        stats = await run_level(
            args.url, pool, prompts,
            total=args.total, concurrency=c, timeout=args.timeout,
        )
        print(
            f"{stats['concurrency']:>5} "
            f"{stats['ok']:>4}/{stats['total']:<5} "
            f"{stats['wall_s']:>8.1f} {stats['rps']:>7.2f} "
            f"{stats['latency_avg_ms']:>9.0f} "
            f"{stats['latency_p50_ms']:>9.0f} "
            f"{stats['latency_p95_ms']:>9.0f} "
            f"{stats['avg_dets']:>6.1f} "
            f"{stats['errors']:>7}"
        )
        if stats['errors'] and stats['first_errors']:
            for e in stats['first_errors']:
                print(f"      first-err: {e}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:3001/predict",
                    help="GDINO /predict endpoint.")
    ap.add_argument("--images-dir", required=True,
                    help="Directory of .jpg images to send.")
    ap.add_argument("--total", type=int, default=64,
                    help="Requests per concurrency level.")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32",
                    help="Comma-separated concurrency levels to sweep.")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="Per-request timeout in seconds.")
    ap.add_argument("--warmup-pool", type=int, default=16,
                    help="Min images to load even when --total is small.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Skip the 8-request warmup (don't use on cold server).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
