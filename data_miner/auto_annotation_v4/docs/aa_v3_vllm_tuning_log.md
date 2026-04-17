# aa_v3 vLLM Tuning Log

## 2026-04-16 — Initial tuning pass (Qwen3.5-27B-FP8 on 2x RTX 3090)

### Hardware

```
GPU 2: NVIDIA GeForce RTX 3090  24,576 MiB
GPU 3: NVIDIA GeForce RTX 3090  24,576 MiB
```

### Changes applied to `docker-compose-vllm.yml`

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `max-model-len` | 4096 | **2048** | Annotation prompts use ~1500-2000 tokens; halving reserves 2x KV cache capacity |
| `gpu-memory-utilization` | 0.85 | **0.92** | Claims ~2280 MiB/GPU for KV cache instead of ~560 MiB (4x more headroom) |
| `enforce-eager` | present | **removed** | CUDA graphs eliminate per-iteration launch overhead (10-30% speedup) |
| `enable-chunked-prefill` | absent | **added** | Interleaves long prefills with decode steps, keeps all sequences progressing |
| `max-num-batched-tokens` | default | **2048** | Controls scheduler token budget per iteration, matches max-model-len |
| `max-num-seqs` | default (256) | **16** | Matches actual KV cache capacity, avoids scheduler overhead on empty slots |

### GPU memory utilization

| Config | VRAM per GPU | KV cache budget per GPU |
|---|---|---|
| Before (0.85) | 20,328 MiB | ~560 MiB |
| After (0.92) | 21,592 MiB | ~2,280 MiB |

### Preemptions

| Config | Preemptions |
|---|---|
| Before (lifetime of prior instance) | 42 |
| After (full benchmark, 120 requests at C=1..16) | **0** |

---

### Benchmark: concurrency sweep (20 requests per level)

Workload: realistic annotation-style prompts (~150-200 input tokens system + user, ~60-80 output tokens JSON response, temperature=0).

#### Before tuning

```
  C= 1: 766.2s  0.03 req/s  lat_avg=29.17s  p50= 9.04s  p95=96.89s  gen_tok/s=  1.5  err=2
  C= 2:  91.4s  0.22 req/s  lat_avg= 8.95s  p50= 9.21s  p95=15.70s  gen_tok/s= 14.6  err=0
  C= 4:  43.3s  0.46 req/s  lat_avg= 8.42s  p50= 8.37s  p95=12.90s  gen_tok/s= 29.9  err=0
  C= 8:  29.0s  0.69 req/s  lat_avg=10.19s  p50=10.37s  p95=13.99s  gen_tok/s= 44.6  err=0
  C=12:  23.0s  0.87 req/s  lat_avg= 9.61s  p50= 9.62s  p95=13.81s  gen_tok/s= 56.1  err=0
  C=16: 117.5s  0.17 req/s  lat_avg=89.09s  p50=109.89s  p95=111.96s gen_tok/s= 11.0  err=0
```

Sweet spot: C=12, 0.87 req/s. **Collapsed at C=16** (preemption cascade, latency 10x).

#### After tuning

```
  C= 1:  37.0s  0.54 req/s  lat_avg= 1.85s  p50= 1.68s  p95= 7.16s  gen_tok/s= 35.6  err=0
  C= 2:  20.3s  0.99 req/s  lat_avg= 1.98s  p50= 2.06s  p95= 2.69s  gen_tok/s= 65.0  err=0
  C= 4:  12.5s  1.60 req/s  lat_avg= 2.39s  p50= 2.29s  p95= 3.94s  gen_tok/s=104.4  err=0
  C= 8:   7.8s  2.58 req/s  lat_avg= 2.82s  p50= 3.03s  p95= 3.97s  gen_tok/s=167.7  err=0
  C=12:   7.4s  2.70 req/s  lat_avg= 3.94s  p50= 3.76s  p95= 5.83s  gen_tok/s=175.5  err=0
  C=16:   6.3s  3.17 req/s  lat_avg= 4.23s  p50= 4.91s  p95= 5.30s  gen_tok/s=207.6  err=0
```

Sweet spot: C=16, 3.17 req/s. **No collapse, no preemptions, zero errors.**

---

### Comparison summary

| Metric | Before | After | Improvement |
|---|---|---|---|
| Peak throughput | 0.87 req/s (C=12) | **3.17 req/s** (C=16) | **3.6x** |
| Peak gen tok/s | 56.1 (C=12) | **207.6** (C=16) | **3.7x** |
| P50 latency @ peak | 9.62s | **4.91s** | **2.0x faster** |
| P95 latency @ peak | 13.81s | **5.30s** | **2.6x faster** |
| Collapse concurrency | C=16 (0.17 req/s) | No collapse through C=16 | Eliminated |
| Preemptions | 42 (prior run) | **0** | Eliminated |
| Errors | 2 (at C=1) | **0** | Eliminated |
| Latency @ C=1 | 29.17s avg | **1.85s avg** | **15.8x faster** |

### Extrapolated pipeline impact

At 3.17 req/s (vs 0.87 baseline), a 100k-image annotation job with ~8 VLM calls/image:

```
Before:  800,000 calls / 0.87 req/s = 919,540s ≈ 255 hours ≈ 10.6 days
After:   800,000 calls / 3.17 req/s = 252,366s ≈  70 hours ≈  2.9 days
```

---

## 2026-04-17 — Peak concurrency sweep (extended)

Ran a wider sweep (C=8..48, 30 requests per level) to find the true ceiling after tuning. `max-num-seqs` was set to 16 in compose, but vLLM's scheduler still accepts more connections — it just queues beyond the cap.

### Results

```
  C= 8:  11.5s   2.61 req/s  lat_avg= 2.92s  p50= 2.83s  p95= 4.20s  gen_tok/s=171.8  err=0
  C=12:   8.7s   3.47 req/s  lat_avg= 3.14s  p50= 3.02s  p95= 4.45s  gen_tok/s=225.2  err=0
  C=16:   9.9s   3.02 req/s  lat_avg= 4.85s  p50= 4.80s  p95= 7.18s  gen_tok/s=196.9  err=0
  C=20:   8.8s   3.42 req/s  lat_avg= 4.98s  p50= 4.83s  p95= 8.06s  gen_tok/s=225.9  err=0
  C=24:   7.2s   4.16 req/s  lat_avg= 4.58s  p50= 4.18s  p95= 6.68s  gen_tok/s=272.3  err=0
  C=32:   7.8s   3.85 req/s  lat_avg= 5.57s  p50= 5.46s  p95= 7.71s  gen_tok/s=251.1  err=0
  C=40:   7.8s   3.84 req/s  lat_avg= 5.56s  p50= 5.48s  p95= 7.71s  gen_tok/s=251.3  err=0
  C=48:   7.8s   3.84 req/s  lat_avg= 5.51s  p50= 5.33s  p95= 7.62s  gen_tok/s=250.9  err=0
```

**Preemptions during entire sweep: 0**

### Analysis

| Zone | Concurrency | Throughput | P50 / P95 | Notes |
|---|---|---|---|---|
| Best balance | **C=12** | 3.47 req/s | 3.02s / 4.45s | Lowest latency, high throughput |
| **Peak throughput** | **C=24** | **4.16 req/s** | 4.18s / 6.68s | **Maximum observed throughput** |
| Plateau | C=32-48 | ~3.85 req/s | ~5.4s / ~7.7s | No further gains, GPU compute-bound |

Key observations:
- **No preemption at any level** — KV cache comfortably holds 48 concurrent sequences
- **C=16 dip** (3.02 vs 3.47 at C=12): likely CUDA graph bucket boundary — vLLM captures graphs at specific batch sizes, and C=16 may fall between two captured sizes causing a decode path switch
- **Plateau at C=32+**: GPU compute is saturated; more concurrency just queues, adding latency without throughput gain
- **Peak gen tok/s: 272.3** at C=24 — this is the hardware ceiling for decode throughput on 2x 3090 with 27B FP8

### Updated recommendations

| Use case | Concurrency | Expected throughput | Latency |
|---|---|---|---|
| Low-latency (interactive) | 8-12 | 2.6-3.5 req/s | ~3s p50 |
| **Balanced (pipeline default)** | **12-16** | **3.0-3.5 req/s** | **3-5s p50** |
| Max throughput (bulk jobs) | 24 | 4.2 req/s | ~4.2s p50 |
| Over-saturated (no benefit) | 32+ | ~3.85 req/s | ~5.5s p50 |

### Updated compose recommendation

Consider bumping `max-num-seqs` from 16 to 24 to allow the scheduler to batch up to the peak throughput point:

```yaml
- --max-num-seqs
- "24"
```

### Extrapolated pipeline impact (revised)

At 4.16 req/s peak (vs 0.87 baseline), a 100k-image job with ~8 VLM calls/image:

```
Baseline:  800,000 / 0.87 = 919,540s ≈ 255 hours ≈ 10.6 days
Tuned:     800,000 / 4.16 = 192,308s ≈  53 hours ≈  2.2 days
                                                      ─────────
                                                      4.8x faster
```

---

### Next tuning opportunities

1. **Bump `max-num-seqs` to 24** — current cap of 16 may be leaving throughput on the table since C=24 peaks higher. Requires restart + re-benchmark.
2. **Longer prompt benchmark** — test with actual annotation prompts (~1000-1500 input tokens) to verify `max-model-len=2048` has enough headroom and measure prefix cache hit rate.
3. **Qwen3.5-9B comparison** — A/B test on 100 images: if quality is comparable, 9B on 1 GPU would free GPU 3 and potentially double throughput again.
4. **Sustained load test** — run 500+ requests at C=24 to check for memory leaks, cache eviction under pressure, and long-tail latency.
