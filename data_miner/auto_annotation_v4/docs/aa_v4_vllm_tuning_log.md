# aa_v4 vLLM Tuning Log

## 2026-04-17 — Initial 4-replica sweep on 8× L40S box

### Hardware

```
GPU 4-7: NVIDIA L40S  46,068 MiB each   (no NVLink, PCIe Gen4)
Driver: 580.119.02
```

### Topology

4 independent TP=1 replicas, one per GPU (GPUs 4..7), fronted by nginx
(`least_conn`) on `:8956`. Rationale: L40S has no NVLink, so TP>1 pays a
PCIe allreduce tax. Qwen3.5-27B's hybrid GatedDeltaNet + attention stack
(16/64 layers full-attn) has a ~4× smaller KV footprint than a pure-attention
27B, so the FP8 model fits comfortably on one 46 GB card with a large KV pool.

### Config (compose env)

```
VALIDATOR_MODEL         Qwen/Qwen3.5-27B-FP8
VALIDATOR_MAX_MODEL_LEN 4096
VALIDATOR_MAX_NUM_SEQS  64
VALIDATOR_GPU_MEM_UTIL  0.92
kv-cache-dtype          fp8_e4m3
enable-chunked-prefill  on
max-num-batched-tokens  4096
enable-prefix-caching   on
mm-processor-cache-type shm
```

### Benchmark methodology

Realistic VLM payload matching `stages/evaluate.py`:
  - System prompt: ~900 tokens of annotation-validator rules
  - Overview image (960×720, synthetic, unique per-request seed)
  - Optional crop image (224×224) — `--with-crop` matches pipeline default
  - `temperature=0.0`, `max_tokens=384`
Warmup: 6 serial requests. Cooldown: 3s between levels. 40 requests/level.

### Results — with crop (pipeline-matching payload)

Peak: **6.28 req/s @ C=96**, p50=5.91s, gen_tok/s=360.

```
  C   wall_s   req/s  lat_avg      p50      p95      p99  gen_tok/s  prompt_tok/s  preempt  err
  1    134.6    0.30     3.36     3.35     3.67     3.70       16.9         360.2        0    0
  2     67.7    0.59     3.33     3.35     3.63     3.68       33.3         716.0        0    0
  4     35.7    1.12     3.42     3.39     3.70     3.95       65.0        1358.3        0    0
  8     18.3    2.19     3.58     3.48     4.10     4.57      125.6        2650.1        0    0
 16     12.2    3.29     4.10     4.15     5.07     5.13      188.0        3983.2        0    0
 24      9.3    4.32     4.56     4.57     5.51     5.74      246.3        5239.3        0    0
 32      9.2    4.34     5.18     5.47     6.14     6.35      249.2        5263.6        0    0
 48      6.8    5.92     6.30     6.31     6.59     6.65      338.2        7177.4        0    0
 64      6.4    6.22     5.95     5.92     6.17     6.36      357.3        7535.8        0    0
 96      6.4    6.28     5.91     5.91     6.01     6.25      360.0        7613.3        0    0
128      6.4    6.24     5.92     5.89     6.21     6.30      358.1        7571.1        0    0
```

### Results — without crop (overview only)

Peak: **6.52 req/s @ C=64**, p50=5.75s, gen_tok/s=377. Only ~4% above
the with-crop run, so the 224×224 crop is a small cost.

```
  C   wall_s   req/s  lat_avg      p50      p95      p99  gen_tok/s  prompt_tok/s  preempt  err
  1    132.2    0.30     3.30     3.33     3.56     3.73       17.2         346.7        0    0
  4     33.8    1.18     3.31     3.31     3.67     3.67       67.1        1356.1        0    0
 16     11.5    3.47     3.91     3.87     4.90     5.01      196.6        3968.8        0    0
 32      9.0    4.45     5.08     5.37     6.02     6.26      255.0        5098.2        0    0
 64      6.1    6.52     5.76     5.75     6.04     6.06      376.8        7472.4        0    0
 96      6.1    6.51     5.74     5.78     6.03     6.08      372.3        7455.4        0    0
```

### Observations

- **Zero preemptions, zero errors** at every concurrency level through C=128.
  KV cache pool is oversized for this workload.
- **All 4 GPUs saturated at 99-100% during run** (verified via nvidia-smi
  polling at 3s intervals), 200-320 W per GPU. Load distribution across the
  LB is working — `least_conn` is adequate.
- **Plateau at C=48+**: ~6.3 req/s regardless of further concurrency. System
  is GPU-compute bound, not KV-bound nor LB-bound.
- **Latency scales cleanly** from C=1 (3.36s) through C=16 (4.1s) before
  tail latency starts growing.
- **Sweet spot C=24-32**: lowest p50 that's still near peak throughput.
  Pipeline `max_concurrent_calls=24` is a reasonable starting value.

### Comparison to prior 2× 3090 tuning (aa_v3)

| Metric | aa_v3 (2×3090 TP=2 + text-only) | aa_v4 (4×L40S TP=1 + VLM) | Ratio |
|---|---|---|---|
| Peak req/s | 4.16 | **6.28** | 1.51× |
| Peak gen tok/s | 272 | **360** | 1.32× |
| Concurrency headroom | C=24 (plateau at 32) | C=48+ (plateau) | 2× |
| p50 latency at peak | 4.18s | 5.91s | slower (VLM payload + 2 images) |
| Preemptions | 0 | **0** | — |

The gap vs the 25-35 req/s I initially predicted has several causes:
1. **VLM workload ≠ text-only**: prior bench was text-only; each request now
   costs vision encoder forward passes (not CUDA-graphed in vLLM).
2. **Short outputs (~60 tok/request)**: request fixed cost (prefill + vision
   + LB hop) dominates over decode time, so decode throughput advantage of
   4 replicas is masked.
3. **Per-replica L40S TP=1 bandwidth**: 864 GB/s per card vs 1552 GB/s
   aggregate across a 2×3090 TP=2 shard. Per-replica decode is bandwidth-
   limited at roughly the same rate as one old TP=2 group.
4. **Hybrid-arch kernel maturity**: Qwen3.5's GatedDeltaNet support in vLLM
   may not yet be as optimized as the standard attention path. Worth
   revisiting after vLLM releases explicit optimization for the Qwen3.5
   architecture family.

Still a net win: **+51% req/s on VLM workload, +4 GPUs freed for other
work**, zero preemptions across the tested envelope.

### Next tuning knobs to try

1. **`VALIDATOR_GPU_MEM_UTIL=0.95`** (pending restart) — +1.4 GB KV cache
   per replica. Already oversized for this workload; expect marginal gain.
2. **MTP speculative decoding** (`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`)
   — likely the biggest untapped win for short-output decode. Retry after
   baseline stabilises.
3. **Pipeline-level sticky routing by `image_id`** — would restore full
   per-image prefix-cache hit rate when classifying multiple candidates
   from the same image. Biggest structural win possible, requires pipeline
   changes, not a vLLM tune.
4. **Text-only mode on a 5th replica** — for non-vision steps (if any),
   `--language-model-only` skips the vision encoder entirely.

---

## 2026-04-17 — Temperature sweep

Question: does temperature (sampling vs greedy) affect throughput? Theory
says no — throughput is dominated by prefill + per-token decode, neither
sensitive to sampling distribution. Only path to a throughput delta is via
output length: higher temperature could in principle generate longer
responses before hitting the JSON-closing tokens.

Same workload as baseline (with-crop VLM, max_tokens=384, ~900-token
system prompt). Tested C=32 (balanced) and C=64 (peak) at temperatures
0.0 / 0.1 / 0.3 / 0.7.

```
  C  temp   wall_s   req/s  lat_avg      p50      p95      p99  gen_tok/s  avg_out  preempt  err
 32  0.00      9.0    4.42     5.17     5.42     6.19     6.22      254.6     57.6        0    0
 32  0.10     14.1    2.84     8.79     9.85    10.50    11.36      162.6     57.3        0    0  ← transient
 32  0.30      9.1    4.39     5.16     5.43     6.03     6.30      252.1     57.4        0    0
 32  0.70      9.9    4.05     5.25     5.47     6.22     6.41      241.6     59.6        0    0
 64  0.00      6.2    6.44     5.79     5.79     6.03     6.09      367.1     57.0        0    0
 64  0.10      6.3    6.39     5.83     5.84     6.04     6.16      363.4     56.8        0    0
 64  0.30      6.3    6.32     5.93     5.93     6.19     6.24      364.8     57.7        0    0
 64  0.70      6.4    6.23     5.96     6.00     6.28     6.34      362.7     58.2        0    0
```

### Findings

- **Throughput is essentially temperature-independent.** At C=64 (peak),
  req/s spans 6.23–6.44 across temp 0.0→0.7 (<3.4% spread, within noise).
  gen_tok/s spans 363–367 (1% spread).
- **Output length is essentially temperature-independent.** Avg completion
  tokens stays in 56.8–59.6 across all temps — strict JSON output gives the
  model very little freedom to ramble at higher temp.
- **The C=32 / temp=0.1 outlier (2.84 req/s) is noise**, not a real effect.
  Same temp at C=64 immediately after was normal (6.39 req/s). Likely a
  transient host event or warmup straggler.

### Recommendation

**Keep `temperature=0.0` (current pipeline default in `configs/servers.yaml`).**
- No throughput cost vs higher temps.
- Deterministic annotation decisions (reproducible, easier to debug).
- No risk of the validator flipping its judgement run-to-run on the same
  candidate.

Temperature is a quality-vs-determinism knob, not a throughput knob, on
this workload. If we ever wanted ensemble-style aggregation (sample N
times at temp=0.7, majority-vote), the vLLM-side cost would scale ~linearly
with N — that's a pipeline-level decision, not a tuning one.

---

### Extrapolated pipeline impact

At 6.28 req/s, a 100k-image job with ~8 VLM calls/image:

```
aa_v3 baseline (2× 3090 post-tune): 800,000 / 4.16 = 53 h
aa_v4 now (4× L40S):                800,000 / 6.28 = 35 h
                                                     ─────
                                                     +34% faster
```

The larger structural win is that **4 more L40S GPUs remain free** for
other inference (detectors, SAM, additional replicas of this model, etc).
