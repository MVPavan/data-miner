"""
Thorough vLLM benchmark for the Qwen3.5-27B-FP8 annotation validator.

Shapes a realistic VLM request (system prompt + overview image + optional
crop image + short user text) matching `stages/evaluate.py`. Runs a
concurrency sweep, records per-level throughput / latency / token-rate /
errors, and snapshots vLLM's /metrics endpoint before and after each level
to surface preemption counts and KV-cache pressure.

Usage:
    # Sweep the LB (round-robins across 4 replicas)
    python -m data_miner.auto_annotation_v4.tests.bench_vllm \\
        --url http://localhost:8956 \\
        --model Qwen/Qwen3.5-27B-FP8 \\
        --concurrency 1,2,4,8,16,24,32,48,64 \\
        --n-requests 40 \\
        --output docs/aa_v4_bench_results.json

    # Single replica (for isolated per-GPU tuning)
    python -m data_miner.auto_annotation_v4.tests.bench_vllm \\
        --url http://localhost:8957 --direct-replica --concurrency 1,4,16,32

    # Sustained load test (memory-leak / long-tail check)
    python -m data_miner.auto_annotation_v4.tests.bench_vllm \\
        --url http://localhost:8956 --sustained 600 --concurrency 24
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Payload synthesis — matches stages/evaluate.py request shape
# --------------------------------------------------------------------------- #

# ~900 input tokens of realistic annotation-validator system prompt.
SYSTEM_PROMPT = (
    "You are an expert annotation quality reviewer. You receive an overview "
    "image with exactly one bounding box highlighted in red (the TARGET) and "
    "optionally a zoomed crop of the target region. Your job is to decide "
    "whether the highlighted TARGET is correctly labelled as the proposed "
    "class, wrongly labelled, partially captured, or an invalid box.\n\n"
    "Valid classes and their disambiguation rules:\n"
    "  - person: any human figure, standing, seated, or partially occluded. "
    "Hands, arms and legs alone do not qualify unless the torso is visible.\n"
    "  - car: passenger vehicles with 4 wheels, including sedans, hatchbacks, "
    "coupes, SUVs. Does not include pickup trucks, vans, or buses.\n"
    "  - truck: pickup trucks, box trucks, and delivery trucks. Semi-trailers "
    "are a separate class.\n"
    "  - bus: large passenger vehicles with multiple windows along their side.\n"
    "  - bicycle: human-powered two-wheeled vehicle, with or without a rider.\n"
    "  - motorbike: motorised two-wheeled vehicle, with or without a rider.\n"
    "  - traffic_light: signal fixtures with red/yellow/green lights for "
    "road or pedestrian control.\n"
    "  - traffic_sign: static signage such as speed limits, stop signs, yield "
    "signs, and direction arrows.\n\n"
    "Annotation rules:\n"
    "  1. The bounding box must tightly enclose the object with <5% padding.\n"
    "  2. An occluded object is still valid if at least 40% visible.\n"
    "  3. Reflections, photographs, and screens do not count as instances.\n"
    "  4. Ambiguous cases should be marked reject with a reason.\n"
    "  5. Prefer the most specific applicable class (e.g. truck over vehicle).\n\n"
    "Respond with strict JSON ONLY, no prose or markdown fences, matching:\n"
    "  {\n"
    '    "decision": "accept" | "reject_wrong_class" | "reject_bad_box" | '
    '"reject_not_object",\n'
    '    "confidence": 0.0..1.0,\n'
    '    "corrected_class": "<class or null>",\n'
    '    "reasoning": "<one short sentence>"\n'
    "  }\n"
)

USER_TEXT_TEMPLATE = (
    "Classify the TARGET (proposed class: {cls}). Remember: strict JSON only."
)

TEST_CLASSES = [
    "person", "car", "truck", "bus", "bicycle",
    "motorbike", "traffic_light", "traffic_sign",
]


def _make_synthetic_image(
    width: int, height: int, seed: int,
) -> Image.Image:
    """Build a synthetic RGB image with random shapes. Deterministic by seed.

    Vision encoder cost tracks pixel count; shapes vary so prefix cache on
    image tokens does not give a false high hit-rate.
    """
    rng = random.Random(seed)
    img = Image.new(
        "RGB", (width, height),
        (rng.randrange(32, 96), rng.randrange(32, 96), rng.randrange(32, 96)),
    )
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randrange(8, 20)):
        x0 = rng.randrange(0, width - 40)
        y0 = rng.randrange(0, height - 40)
        x1 = x0 + rng.randrange(20, 200)
        y1 = y0 + rng.randrange(20, 200)
        fill = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline="red", width=3)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=fill)
    # Red "TARGET" box so the system prompt cue is satisfied
    tx0, ty0 = rng.randrange(50, width // 2), rng.randrange(50, height // 2)
    tx1, ty1 = tx0 + rng.randrange(80, 200), ty0 + rng.randrange(80, 200)
    draw.rectangle([tx0, ty0, tx1, ty1], outline="red", width=5)
    return img


def _pil_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def build_payload(
    model: str, seed: int, with_crop: bool, max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """Build one realistic VLM chat/completions payload."""
    overview = _make_synthetic_image(960, 720, seed)
    content = [
        {"type": "image_url",
         "image_url": {"url": _pil_to_data_url(overview)}},
    ]
    if with_crop:
        crop = _make_synthetic_image(224, 224, seed + 10_000)
        content.append(
            {"type": "image_url",
             "image_url": {"url": _pil_to_data_url(crop)}},
        )
    cls = TEST_CLASSES[seed % len(TEST_CLASSES)]
    content.append({"type": "text", "text": USER_TEXT_TEMPLATE.format(cls=cls)})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


# --------------------------------------------------------------------------- #
# vLLM /metrics scraping (Prometheus text format)
# --------------------------------------------------------------------------- #

_METRIC_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)')


async def _scrape_metrics(session: aiohttp.ClientSession, url: str) -> dict:
    """Pull a small subset of vLLM Prometheus metrics. Returns {} on failure.

    We only aggregate the fields relevant to tuning: KV usage, running,
    waiting, preemptions, prompt/gen throughput totals.
    """
    try:
        async with session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
            if r.status != 200:
                return {}
            text = await r.text()
    except Exception:
        return {}

    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:num_requests_swapped": "swapped",
        "vllm:gpu_cache_usage_perc": "kv_cache_pct",
        "vllm:num_preemptions_total": "preemptions",
        "vllm:prompt_tokens_total": "prompt_tokens_total",
        "vllm:generation_tokens_total": "generation_tokens_total",
    }
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in wanted:
            continue
        try:
            val = float(m.group("value"))
        except ValueError:
            continue
        key = wanted[name]
        # Multiple replicas behind an LB would each report — here we target
        # a single host, so aggregation is sum for counters, max for gauges.
        if key in ("running", "waiting", "swapped"):
            out[key] = max(out.get(key, 0.0), val)
        elif key == "kv_cache_pct":
            out[key] = max(out.get(key, 0.0), val)
        else:
            out[key] = out.get(key, 0.0) + val
    return out


# --------------------------------------------------------------------------- #
# Single request
# --------------------------------------------------------------------------- #

@dataclass
class ReqResult:
    ok: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


async def _one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    timeout_s: float,
) -> ReqResult:
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            body = await resp.json()
            lat = time.perf_counter() - t0
            if resp.status != 200:
                return ReqResult(False, lat, error=f"HTTP {resp.status}: {str(body)[:200]}")
            usage = body.get("usage") or {}
            return ReqResult(
                ok=True,
                latency_s=lat,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
    except asyncio.TimeoutError:
        return ReqResult(False, time.perf_counter() - t0, error="timeout")
    except Exception as e:
        return ReqResult(False, time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Sweep driver
# --------------------------------------------------------------------------- #

@dataclass
class LevelResult:
    concurrency: int
    n_requests: int
    wall_s: float
    req_per_s: float
    lat_avg_s: float
    lat_p50_s: float
    lat_p95_s: float
    lat_p99_s: float
    gen_tok_per_s: float
    prompt_tok_per_s: float
    avg_completion_tokens: float
    errors: int
    error_samples: list[str]
    temperature: float = 0.0
    metrics_delta: dict = field(default_factory=dict)
    metrics_end: dict = field(default_factory=dict)


async def _run_level(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    concurrency: int,
    n_requests: int,
    with_crop: bool,
    max_tokens: int,
    timeout_s: float,
    seed_offset: int,
    temperature: float = 0.0,
) -> LevelResult:
    sem = asyncio.Semaphore(concurrency)

    async def limited(i: int) -> ReqResult:
        async with sem:
            payload = build_payload(
                model, i + seed_offset, with_crop, max_tokens,
                temperature=temperature,
            )
            return await _one_request(session, url, payload, timeout_s)

    m_start = await _scrape_metrics(session, url)
    t0 = time.perf_counter()
    results = await asyncio.gather(*[limited(i) for i in range(n_requests)])
    wall = time.perf_counter() - t0
    m_end = await _scrape_metrics(session, url)

    oks = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]
    lats = [r.latency_s for r in oks] or [0.0]
    gen_tokens = sum(r.completion_tokens for r in oks)
    prompt_tokens = sum(r.prompt_tokens for r in oks)

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
        return s[k]

    metrics_delta = {
        k: m_end.get(k, 0.0) - m_start.get(k, 0.0)
        for k in m_start.keys() | m_end.keys()
        if k in ("preemptions", "prompt_tokens_total", "generation_tokens_total")
    }

    return LevelResult(
        concurrency=concurrency,
        n_requests=n_requests,
        wall_s=wall,
        req_per_s=len(oks) / wall if wall > 0 else 0.0,
        lat_avg_s=statistics.fmean(lats) if lats else 0.0,
        lat_p50_s=_pct(lats, 50),
        lat_p95_s=_pct(lats, 95),
        lat_p99_s=_pct(lats, 99),
        gen_tok_per_s=gen_tokens / wall if wall > 0 else 0.0,
        prompt_tok_per_s=prompt_tokens / wall if wall > 0 else 0.0,
        avg_completion_tokens=gen_tokens / len(oks) if oks else 0.0,
        errors=len(errs),
        error_samples=[e.error for e in errs[:3] if e.error],
        temperature=temperature,
        metrics_delta=metrics_delta,
        metrics_end=m_end,
    )


async def _warmup(
    session: aiohttp.ClientSession, url: str, model: str,
    with_crop: bool, max_tokens: int, n: int = 4,
) -> None:
    """Fire a few serial requests to prime CUDA graphs + prefix cache."""
    for i in range(n):
        await _one_request(
            session, url, build_payload(model, i, with_crop, max_tokens),
            timeout_s=180,
        )


def _print_header() -> None:
    print(
        f"{'C':>3} {'temp':>5} {'wall_s':>8} {'req/s':>7} "
        f"{'lat_avg':>8} {'p50':>8} {'p95':>8} {'p99':>8} "
        f"{'gen_tok/s':>10} {'avg_out':>8} "
        f"{'preempt':>8} {'err':>4}"
    )
    print("-" * 110)


def _print_row(r: LevelResult) -> None:
    preempt = int(r.metrics_delta.get("preemptions", 0))
    print(
        f"{r.concurrency:>3} {r.temperature:>5.2f} {r.wall_s:>8.1f} {r.req_per_s:>7.2f} "
        f"{r.lat_avg_s:>8.2f} {r.lat_p50_s:>8.2f} {r.lat_p95_s:>8.2f} {r.lat_p99_s:>8.2f} "
        f"{r.gen_tok_per_s:>10.1f} {r.avg_completion_tokens:>8.1f} "
        f"{preempt:>8d} {r.errors:>4d}"
    )
    if r.error_samples:
        for e in r.error_samples:
            print(f"       err: {e[:180]}")


async def run_sweep(args: argparse.Namespace) -> dict:
    connector = aiohttp.TCPConnector(limit=max(args.concurrency) * 2 + 16)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"endpoint: {args.url}  model: {args.model}  with_crop: {args.with_crop}")
        m0 = await _scrape_metrics(session, args.url)
        if m0:
            print(f"initial /metrics: {m0}")
        else:
            print("initial /metrics: unavailable (behind LB or not exposed) — continuing")

        if args.warmup > 0:
            print(f"warmup: {args.warmup} serial requests ...")
            await _warmup(session, args.url, args.model, args.with_crop, args.max_tokens, args.warmup)

        print("\nconcurrency sweep:")
        _print_header()
        level_results: list[LevelResult] = []
        seed = 0
        for temp in args.temperatures:
            for c in args.concurrency:
                n = args.n_requests if args.n_requests else max(c * 4, 20)
                r = await _run_level(
                    session, args.url, args.model, c, n,
                    with_crop=args.with_crop,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout,
                    seed_offset=seed,
                    temperature=temp,
                )
                seed += n
                _print_row(r)
                level_results.append(r)
                await asyncio.sleep(args.cooldown)

        peak = max(level_results, key=lambda r: r.req_per_s)
        print("\nPeak throughput:")
        print(
            f"  C={peak.concurrency}  {peak.req_per_s:.2f} req/s  "
            f"p50={peak.lat_p50_s:.2f}s  p95={peak.lat_p95_s:.2f}s  "
            f"gen_tok/s={peak.gen_tok_per_s:.1f}"
        )

        return {
            "endpoint": args.url,
            "model": args.model,
            "with_crop": args.with_crop,
            "max_tokens": args.max_tokens,
            "levels": [asdict(r) for r in level_results],
            "peak": asdict(peak),
        }


async def run_sustained(args: argparse.Namespace) -> dict:
    """Run a fixed-concurrency workload for N seconds. Useful for memory
    leak / long-tail checks. Single concurrency value required.
    """
    c = args.concurrency[0]
    duration = args.sustained
    print(f"sustained test: C={c} for {duration}s")
    sem = asyncio.Semaphore(c)
    lats: list[float] = []
    errors: list[str] = []
    gen_tok = 0
    prompt_tok = 0
    t_end = time.perf_counter() + duration
    checkpoint_interval = 30
    next_checkpoint = time.perf_counter() + checkpoint_interval
    seed = 0

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=c * 2 + 16),
    ) as session:
        if args.warmup:
            await _warmup(session, args.url, args.model, args.with_crop, args.max_tokens, args.warmup)

        async def worker(i: int) -> None:
            nonlocal gen_tok, prompt_tok
            async with sem:
                payload = build_payload(args.model, i, args.with_crop, args.max_tokens)
                r = await _one_request(session, args.url, payload, args.timeout)
                if r.ok:
                    lats.append(r.latency_s)
                    gen_tok += r.completion_tokens
                    prompt_tok += r.prompt_tokens
                elif r.error:
                    errors.append(r.error)

        pending: set[asyncio.Task] = set()
        i = 0
        while time.perf_counter() < t_end:
            while len(pending) < c * 2 and time.perf_counter() < t_end:
                pending.add(asyncio.create_task(worker(i)))
                i += 1
                seed += 1
            done, pending = await asyncio.wait(pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            now = time.perf_counter()
            if now >= next_checkpoint:
                elapsed = now - (t_end - duration)
                if lats:
                    print(
                        f"  t={elapsed:>5.0f}s  completed={len(lats)}  "
                        f"req/s={len(lats)/elapsed:.2f}  "
                        f"gen_tok/s={gen_tok/elapsed:.1f}  errors={len(errors)}"
                    )
                next_checkpoint = now + checkpoint_interval
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    total = duration
    if not lats:
        print("no successful requests")
        return {"sustained": True, "n": 0}
    lats_sorted = sorted(lats)
    return {
        "sustained": True,
        "endpoint": args.url,
        "concurrency": c,
        "duration_s": total,
        "completed": len(lats),
        "errors": len(errors),
        "req_per_s": len(lats) / total,
        "gen_tok_per_s": gen_tok / total,
        "prompt_tok_per_s": prompt_tok / total,
        "lat_p50_s": lats_sorted[len(lats_sorted) // 2],
        "lat_p95_s": lats_sorted[min(len(lats_sorted) - 1, int(len(lats_sorted) * 0.95))],
        "lat_p99_s": lats_sorted[min(len(lats_sorted) - 1, int(len(lats_sorted) * 0.99))],
        "error_samples": errors[:5],
    }


def _parse_concurrency(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_temperatures(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8956",
                    help="Base URL (LB or single replica). Default: %(default)s")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8",
                    help="Model name to pass in the payload")
    ap.add_argument("--concurrency", type=_parse_concurrency,
                    default=[1, 2, 4, 8, 16, 24, 32, 48, 64],
                    help="Comma-separated concurrency levels. Default: %(default)s")
    ap.add_argument("--n-requests", type=int, default=0,
                    help="Requests per level (0 = auto, 4×C or min 20)")
    ap.add_argument("--warmup", type=int, default=4,
                    help="Serial warmup requests before sweep")
    ap.add_argument("--with-crop", action="store_true", default=True,
                    help="Include crop image (matches evaluate.py default)")
    ap.add_argument("--no-crop", dest="with_crop", action="store_false",
                    help="Overview image only (lower prefill cost)")
    ap.add_argument("--max-tokens", type=int, default=384,
                    help="max_tokens in payload. Default matches evaluate.py")
    ap.add_argument("--temperatures", type=_parse_temperatures, default=[0.0],
                    help="Comma-separated temperatures to sweep. Each temp "
                         "is run across the full --concurrency list.")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="Per-request timeout in seconds")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="Sleep between concurrency levels (lets vLLM settle)")
    ap.add_argument("--sustained", type=int, default=0,
                    help="If >0, run sustained-load mode for N seconds "
                         "(requires single --concurrency value)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write results JSON to this path")
    ap.add_argument("--direct-replica", action="store_true",
                    help="Informational flag: we're hitting a single replica, "
                         "so /metrics should be accurate")
    args = ap.parse_args()

    if args.sustained and len(args.concurrency) != 1:
        ap.error("--sustained requires exactly one --concurrency value")

    if args.sustained:
        result = asyncio.run(run_sustained(args))
    else:
        result = asyncio.run(run_sweep(args))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nwrote results: {args.output}")


if __name__ == "__main__":
    main()
