"""bench_vllm_3090.py — concurrency sweep against a single vLLM replica.

Use AFTER bringing up ONE replica of docker-compose-vllm-3090.yml to map the
throughput vs. latency curve on a 3090, so we know the right --max-num-seqs
to bake into the compose file for all 8 replicas.

Key idea: bring the replica up with a generous max-num-seqs (say 48), then
sweep CLIENT-side concurrency — the effective cap is min(client, server), so a
single model-load cycle covers the full curve. Much faster than restarting the
replica per config.

Typical flow from the host (not inside this container):

  cd data_miner/auto_annotation_v4
  # bring up ONE replica with a generous seq cap
  VALIDATOR_MAX_NUM_SEQS=48 VALIDATOR_GPU_MEM_UTIL=0.88 \\
    docker compose -f docker-compose-vllm-3090.yml up -d qwen35-validator-0

  # wait for model load (~60-120s) until /health returns 200:
  until curl -sf http://localhost:8000/health >/dev/null; do sleep 5; done

  # run the sweep (from host or from any box with httpx installed):
  python scripts_bench/bench_vllm_3090.py \\
      --base-url http://localhost:8000/v1 \\
      --model Qwen/Qwen3.5-27B-GPTQ-Int4 \\
      --concurrencies 1,2,4,8,16,24,32,48 \\
      --num-requests 96 --output-tokens 200

The script prints a table per concurrency level plus a recommended
--max-num-seqs derived from the throughput knee under a p99-latency budget.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

# Realistic annotation-validator prompt (~1.6k tokens). Edit to match your
# actual downstream validator if you care about exact prefill cost.
BASE_PROMPT = """You are validating auto-generated image annotations for an industrial warehouse dataset.
Given a candidate bounding-box label, decide whether to accept, reject, or send to a VLM for evaluation.

Class definitions (relevant subset):
- forklift: Powered truck with vertical MAST, counterweight body, overhead guard. MUST have visible mast.
- palletjack: LOW-PROFILE, no mast, forks slide under pallets at ground level. Steering handle/tiller.
- shopping cart: Wheeled cart with handle, flat platform or basket. No forks, no mast.
- truck: Large road vehicle for cargo. No mast, no forks.
- person: Full or partial visible human body in any pose.
- head: Head annotated separately only when body is not fully visible.

Decision rules:
- A forklift MUST have a visible mast — without a mast the correct class is pallet jack.
- Reject any candidate with confidence < 0.25.
- Accept without VLM check ONLY when:
    (a) confidence >= 0.70,
    (b) at least TWO independent detectors agree,
    (c) no confusion-class detection overlaps the bbox with IoU > 0.6.
- Otherwise return decision="evaluate" and provide a reason.

Candidate:
{
  "image_id": "wh_042318_frame_0192",
  "bbox_normalized": [0.312, 0.458, 0.521, 0.784],
  "class": "forklift",
  "confidence": 0.72,
  "detectors_agreed": ["sam3_dart", "grounding_dino"],
  "confusion_hits": {"palletjack": {"iou": 0.18, "conf": 0.31}, "truck": {"iou": 0.02, "conf": 0.11}},
  "context_tags": ["warehouse", "daytime", "pallets_visible", "operator_seated", "indoor"],
  "visible_features": ["mast", "counterweight", "overhead_guard", "forks_raised", "load_backrest"]
}

Respond with strict JSON ONLY (no prose before or after):
{"decision": "accept"|"reject"|"evaluate", "reason": "<one sentence>"}
"""


@dataclass
class Result:
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str = ""


async def one_request(client: httpx.AsyncClient, url: str, model: str,
                      prompt: str, max_tokens: int) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/chat/completions", json=payload,
                              timeout=300.0)
        r.raise_for_status()
        j = r.json()
        usage = j.get("usage", {}) or {}
        return Result(
            ok=True,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
    except Exception as e:
        return Result(False, time.perf_counter() - t0, 0, 0,
                      error=f"{type(e).__name__}: {e}")


async def run_concurrency(url: str, model: str, conc: int, n_req: int,
                          max_tokens: int) -> dict:
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient() as client:
        # Warm-up: 2 serial requests so KV cache + kernel JIT settle.
        for _ in range(2):
            await one_request(client, url, model, BASE_PROMPT, max_tokens)

        results: list[Result] = []

        async def worker() -> None:
            async with sem:
                results.append(await one_request(
                    client, url, model, BASE_PROMPT, max_tokens))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(n_req)))
        wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lat = sorted(r.latency_s for r in ok)
    ct = sum(r.completion_tokens for r in ok)
    pt = sum(r.prompt_tokens for r in ok) / max(1, len(ok))

    def q(frac: float) -> float | None:
        if not lat:
            return None
        idx = min(len(lat) - 1, int(round(frac * (len(lat) - 1))))
        return lat[idx]

    return {
        "concurrency": conc,
        "requests_total": n_req,
        "ok": len(ok),
        "failed": len(failed),
        "wall_s": round(wall, 1),
        "throughput_req_s": round(len(ok) / wall, 2) if wall else 0.0,
        "throughput_output_tok_s": round(ct / wall, 1) if wall else 0.0,
        "avg_prompt_tokens": round(pt, 1),
        "latency_p50_s": round(q(0.50), 2) if lat else None,
        "latency_p90_s": round(q(0.90), 2) if lat else None,
        "latency_p99_s": round(q(0.99), 2) if lat else None,
        "errors_sample": [r.error for r in failed[:3]],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible endpoint root")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-GPTQ-Int4")
    ap.add_argument("--concurrencies", default="1,2,4,8,16,24,32,48",
                    help="Comma-separated concurrency levels to sweep.")
    ap.add_argument("--num-requests", type=int, default=96,
                    help="Requests per concurrency level (min 2x concurrency).")
    ap.add_argument("--output-tokens", type=int, default=200,
                    help="max_tokens for generation.")
    ap.add_argument("--p99-budget-mult", type=float, default=2.5,
                    help="Allow p99 up to this multiple of conc=1 p99 when "
                         "picking the recommended max-num-seqs.")
    args = ap.parse_args()

    concs = [int(c) for c in args.concurrencies.split(",")]
    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print(f"output   : {args.output_tokens} tokens/request")
    print()
    print(f"{'conc':>5} {'ok':>4} {'fail':>5} {'wall_s':>8} {'req/s':>7} "
          f"{'out_tok/s':>10} {'p50_s':>7} {'p90_s':>7} {'p99_s':>7}")
    print("-" * 72)

    rows: list[dict] = []
    for c in concs:
        n_req = max(c * 2, args.num_requests)
        r = await run_concurrency(args.base_url, args.model, c, n_req,
                                  args.output_tokens)
        print(f"{r['concurrency']:>5} {r['ok']:>4} {r['failed']:>5} "
              f"{r['wall_s']:>8} {r['throughput_req_s']:>7} "
              f"{r['throughput_output_tok_s']:>10} "
              f"{(r['latency_p50_s'] or 0):>7} "
              f"{(r['latency_p90_s'] or 0):>7} "
              f"{(r['latency_p99_s'] or 0):>7}")
        rows.append(r)
        if r["failed"] > 0 and r["errors_sample"]:
            print(f"       ! errors: {r['errors_sample']}")

    # Recommend max-num-seqs = highest concurrency where p99 latency is within
    # p99-budget-mult × the serial p99 AND there are zero failures.
    print()
    serial = next((r for r in rows if r["concurrency"] == min(concs)), rows[0])
    serial_p99 = (serial.get("latency_p99_s") or serial.get("latency_p90_s")
                  or serial.get("latency_p50_s") or 1.0)
    budget = args.p99_budget_mult * serial_p99
    print(f"serial p99 baseline : {serial_p99:.2f}s   "
          f"p99 budget @ {args.p99_budget_mult}× : {budget:.2f}s")

    eligible = [
        r for r in rows
        if r["failed"] == 0
        and (r.get("latency_p99_s") or r.get("latency_p90_s")
             or r.get("latency_p50_s") or 0.0) <= budget
    ]
    if eligible:
        best = max(eligible, key=lambda r: r["throughput_output_tok_s"])
        print(f"recommended --max-num-seqs : {best['concurrency']}  "
              f"(~{best['throughput_output_tok_s']:.0f} tok/s aggregate, "
              f"p99={best.get('latency_p99_s') or best.get('latency_p90_s')}s)")
    else:
        print("no eligible concurrency — all levels exceed the p99 budget.")

    print()
    print("raw rows:")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
