# aa_v3 — vLLM Tuning Guide (Qwen3.5 on 2× RTX 3090)

## Hardware baseline

```
GPU 2: NVIDIA GeForce RTX 3090  24,576 MiB  (used for VLM via TP shard 0)
GPU 3: NVIDIA GeForce RTX 3090  24,576 MiB  (used for VLM via TP shard 1)
```

At idle with `Qwen/Qwen3.5-27B-FP8` loaded (`tensor-parallel-size=2`):

```
GPU 2: 20,328 MiB / 24,576 MiB  (82.7% consumed by model + runtime)
GPU 3: 20,330 MiB / 24,576 MiB  (82.7% consumed by model + runtime)
```

The ~20.3 GB per GPU breaks down as:
- ~13.5 GB model weights (27B FP8 ÷ 2 GPUs)
- ~6.8 GB CUDA context, activation buffers, vLLM runtime, torch allocator

---

## Current config (before tuning)

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --enable-prefix-caching \
    --dtype auto \
    --port 8955
```

---

## How vLLM concurrency works

vLLM uses continuous batching — it doesn't process one request at a time.
The scheduler dynamically packs sequences into each GPU iteration based on
available KV cache memory. There is no single "max concurrent requests"
setting. The actual concurrency is an emergent property of three things:

1. How much VRAM is left for KV cache after model weights are loaded
2. How many tokens each active sequence consumes (prompt + generated so far)
3. The `max_num_seqs` cap (upper bound on scheduler's batch size)

The KV cache is the memory that stores intermediate attention states for
all active sequences. More KV cache = more concurrent sequences = higher
throughput.

---

## Memory budget analysis

### KV cache budget with current config

```
Per GPU total VRAM:                         24,576 MiB
vLLM allocation at gpu_memory_utilization=0.85:  20,890 MiB
Already consumed at idle (model + runtime):      20,330 MiB
────────────────────────────────────────────────────────────
KV cache budget per GPU:                          ~560 MiB
KV cache total (2 GPUs):                        ~1,120 MiB
```

### KV cache per token

For Qwen3.5-27B (GQA architecture, ~4 KV heads, 48 layers, head_dim 128,
FP16 KV cache, tensor-parallel across 2 GPUs):

```
per token per GPU = 2 (K+V) × 128 (head_dim) × 2 (kv_heads/GPU) × 48 (layers) × 2 bytes
                  = 48 KB per token per GPU
```

### Concurrent sequences with current config

```
max_model_len = 4096 tokens
KV per sequence per GPU = 48 KB × 4096 = 192 MiB

Max concurrent at max_model_len:
  560 MiB / 192 MiB ≈ 2-3 sequences

At actual annotation length (~2000 tokens):
  560 MiB / (48 KB × 2000) ≈ 5-6 sequences
```

**With current settings, you can run approximately 3-6 concurrent VLM
requests.** This is the bottleneck for pipeline throughput.

### How to verify

Check the vLLM startup logs. It prints the exact KV cache capacity:

```
INFO: # GPU blocks: XXXX, # CPU blocks: YYYY
INFO: Maximum concurrency for 4096 tokens per request: Z
```

Or at runtime, look for preemption warnings:

```
WARNING: Sequence group 0 is preempted by PreemptionMode.RECOMPUTE ...
```

If you see preemption, you're exceeding the KV cache capacity. Reduce
pipeline concurrency or increase KV cache budget per the tuning below.

---

## Tuning parameters (ordered by impact)

### 1. Lower max-model-len (biggest single win)

Annotation prompts are ~1000-1500 input tokens + ~200-500 output tokens.
Total ~1500-2000. Setting `max_model_len=4096` reserves 4096 tokens of
KV cache per sequence, but you never use the upper half. Every unused token
of headroom is KV cache wasted.

```
max_model_len=4096:  ~3 concurrent sequences  (current)
max_model_len=2048:  ~6 concurrent sequences  (2× improvement)
max_model_len=1536:  ~8 concurrent sequences  (2.7× improvement)
```

**Recommendation**: `--max-model-len 2048`

If any annotation prompt + response exceeds 2048 tokens, vLLM returns an
error. Check your longest prompts before lowering this. The evaluate stage
system prompt + image tokens + candidate description + JSON response
typically totals 1500-1800 tokens.

### 2. Raise gpu-memory-utilization (more KV cache from unused VRAM)

At 0.85 utilization, vLLM claims 20,890 MiB per GPU. But 4,200 MiB per
GPU sits unused. Raising to 0.92 gives ~1,720 MiB more per GPU for KV cache.

```
gpu_memory_utilization=0.85:  ~560 MiB KV cache per GPU
gpu_memory_utilization=0.92:  ~2,280 MiB KV cache per GPU  (4× more)
```

Combined with `max_model_len=2048`:

```
Sequences = 2,280 MiB / (48 KB × 2048) ≈ 24 concurrent sequences
```

**Recommendation**: `--gpu-memory-utilization 0.92`

Don't go above 0.93 — leave ~1.2 GB headroom per GPU for activation
spikes during prefill of long prompts. If you see OOM errors at 0.92,
drop to 0.90.

### 3. Drop enforce-eager (enable CUDA graphs)

The current config has `--enforce-eager` which disables CUDA graph capture.
CUDA graphs eliminate Python-side and CUDA launch overhead on each forward
pass by replaying a captured graph. The savings are most noticeable at
small batch sizes (3-16 sequences), which is exactly your operating range.

CUDA graphs consume some memory for the captured graphs (~200-500 MiB),
but the per-iteration speedup (10-30%) is worth it.

**Recommendation**: Remove `--enforce-eager`

If this causes stability issues (some model+vLLM version combinations have
CUDA graph bugs), add it back. Test without it first.

### 4. Enable chunked prefill (better GPU utilization)

Without chunked prefill, each new request's prompt (1000-1500 tokens) is
processed in one shot, blocking decode steps for all other active sequences.
With chunked prefill, long prefills are broken into smaller chunks and
interleaved with decode steps. This keeps all active sequences making
progress and improves GPU compute utilization.

**Recommendation**: `--enable-chunked-prefill --max-num-batched-tokens 2048`

`max_num_batched_tokens` controls how many tokens are processed per
scheduler iteration across all sequences. The default with chunked prefill
is 2048, which is a good starting point. Higher values (4096) improve
prefill throughput but may increase decode latency. For batch annotation
where latency isn't critical, 2048-4096 is fine.

### 5. Set max-num-seqs explicitly

The default `max_num_seqs` is 256 or 1024 depending on vLLM version.
Your KV cache supports ~6-24 sequences (depending on other tuning).
Setting `max_num_seqs` much higher than actual capacity wastes scheduler
overhead managing empty slots.

**Recommendation**: `--max-num-seqs 16`

Set to 1.5-2× your pipeline's `max_concurrent_calls`. If the pipeline
sends at most 8 concurrent requests, `max_num_seqs=16` gives scheduling
headroom without waste.

---

## Optimized vLLM command

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --dtype auto \
    --port 8955 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --mm-processor-cache-type shm
```

For the docker-compose env overrides:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
  - NVIDIA_DISABLE_REQUIRE=1
command:
  - --model
  - Qwen/Qwen3.5-27B-FP8
  - --tensor-parallel-size
  - "2"
  - --max-model-len
  - "2048"
  - --gpu-memory-utilization
  - "0.92"
  - --max-num-seqs
  - "16"
  - --max-num-batched-tokens
  - "2048"
  - --enable-prefix-caching
  - --enable-chunked-prefill
  - --reasoning-parser
  - qwen3
  - --dtype
  - auto
  - --port
  - "8000"
  - --default-chat-template-kwargs
  - '{"enable_thinking": false}'
  - --mm-processor-cache-type
  - shm
```

---

## Expected improvement

| Setting | Concurrent seqs | Throughput est | Change |
|---------|----------------|----------------|--------|
| Current config (baseline) | 3 | ~1.0 req/s | — |
| + `max-model-len 2048` | 6 | ~1.8 req/s | +80% |
| + `gpu-memory-utilization 0.92` | 16-24 | ~3.5 req/s | +250% |
| + chunked prefill | same count | ~4.0 req/s | +300% |
| + drop enforce-eager | same count | ~4.5 req/s | +350% |

At 4.5 req/s vs 1.0 req/s, a 100k image job with ~8 VLM calls per image
drops from ~222 hours to ~49 hours.

---

## Pipeline-side concurrency tuning

### Semaphore in the evaluate worker

The pipeline must limit how many concurrent requests it sends to vLLM.
Sending more than the KV cache can handle causes preemption (vLLM evicts
sequences and recomputes them later), which destroys throughput.

```yaml
# aa_v3 config
reasoning:
  max_concurrent_calls: 8
```

```python
# In pipeline.py or evaluate worker
vlm_semaphore = asyncio.Semaphore(config.reasoning.max_concurrent_calls)

# In evaluate worker's _call_vlm method
async def _call_vlm(self, prompt, images):
    async with self.vlm_semaphore:
        async with asyncio.timeout(120):
            async with self._session.post(vlm_url, json=payload) as resp:
                return await resp.json()
```

### Finding the right concurrency

Run a sweep on a sample of 50 images to find where throughput plateaus:

```python
import asyncio, aiohttp, time, json

VLM_URL = "http://localhost:8955/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B-FP8"

async def call_vlm(session, prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.0,
    }
    async with session.post(VLM_URL, json=payload) as resp:
        return await resp.json()

async def bench(concurrency, n_requests=50):
    sem = asyncio.Semaphore(concurrency)

    async def limited(session, prompt):
        async with sem:
            return await call_vlm(session, prompt)

    # Use a realistic annotation-length prompt
    prompt = (
        "You are an annotation quality reviewer. "
        "Evaluate whether this bounding box correctly captures the object. "
        "Respond with JSON: {decision, confidence, reasoning}. "
    ) * 10  # ~1000 tokens

    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        tasks = [limited(session, prompt) for _ in range(n_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - t0

        errors = sum(1 for r in results if isinstance(r, Exception))
        print(
            f"  concurrency={concurrency:>2}: "
            f"{elapsed:.1f}s  "
            f"{n_requests/elapsed:.1f} req/s  "
            f"errors={errors}"
        )

async def main():
    print("vLLM concurrency sweep (50 requests each):")
    for c in [1, 2, 4, 8, 12, 16]:
        await bench(c)
        await asyncio.sleep(2)  # let vLLM settle between runs

asyncio.run(main())
```

Expected output pattern:

```
vLLM concurrency sweep (50 requests each):
  concurrency= 1: 102.3s  0.5 req/s  errors=0
  concurrency= 2:  55.1s  0.9 req/s  errors=0
  concurrency= 4:  30.2s  1.7 req/s  errors=0
  concurrency= 8:  14.8s  3.4 req/s  errors=0    ← sweet spot
  concurrency=12:  13.1s  3.8 req/s  errors=0    ← marginal gain
  concurrency=16:  14.5s  3.4 req/s  errors=0    ← preemption starts
```

Set `max_concurrent_calls` to the concurrency where throughput plateaus
(likely 8-12 for your setup after tuning).

### Prefix caching effectiveness

Prefix caching is already enabled. To maximize its benefit, the evaluate
worker should group VLM calls by image — all candidates for the same image
use the same system prompt + image, so the prefix is computed once and
cached for subsequent candidates.

The current v3 evaluate stage already does this (processes all candidates
per image before moving to the next image). No change needed, but be aware
that random-order processing would break prefix caching effectiveness.

---

## Monitoring during pipeline runs

### vLLM metrics to watch

If vLLM is started with `--disable-log-stats false` (or not explicitly
disabled), it logs throughput metrics periodically:

```
INFO: Avg prompt throughput: 1234.5 tokens/s,
      Avg generation throughput: 567.8 tokens/s,
      Running: 8, Swapped: 0, Pending: 2, GPU KV cache usage: 67.3%,
      CPU KV cache usage: 0.0%.
```

Key indicators:

| Metric | Good | Bad | Action |
|--------|------|-----|--------|
| GPU KV cache usage | 40-75% | >90% | Reduce pipeline concurrency |
| Swapped | 0 | >0 | KV cache exhausted, reduce concurrency |
| Pending | 0-2 | >10 growing | Pipeline sending faster than vLLM processes |
| Running | near max_num_seqs | 1-2 | Pipeline concurrency too low |
| Preemption warnings | none | any | Reduce concurrency or raise gpu_memory_utilization |

### nvidia-smi during runs

```bash
watch -n1 nvidia-smi
```

- GPU utilization should be 60-90% during active processing
- Memory should be stable (no growth = no leaks)
- If utilization is <30%, pipeline isn't sending enough concurrent requests

---

## Alternative: Qwen3.5-9B for higher throughput

If annotation quality from 9B is acceptable (worth A/B testing on 100
images), switching to the smaller model dramatically changes the math:

```
Qwen3.5-9B BF16 on 1× RTX 3090:
  Model weights:         ~18 GB
  Available KV cache:    ~3.5 GB (at 0.92 utilization)
  KV per token:          ~24 KB (fewer layers, fewer KV heads)
  Concurrent seqs:       ~40 at max_model_len=2048
  Per-token latency:     ~3× faster than 27B
  Throughput:            ~8-10 req/s
```

Comparison:

| Model | GPUs | Concurrent seqs | Throughput | Quality |
|-------|------|----------------|------------|---------|
| Qwen3.5-27B-FP8 | 2× 3090 | 16-24 | ~4.5 req/s | Higher |
| Qwen3.5-9B BF16 | 1× 3090 | ~40 | ~8-10 req/s | Lower (test it) |

The 9B model frees GPU 3 entirely — you could use it for an additional
detection model server or double up on SAM3 capacity.

**How to A/B test**: Run both models on the same 100 images, compare
accept/reject agreement rate. If they agree on >95% of candidates, the 9B
is good enough for your annotation pipeline and you get 2× throughput.

```bash
# 27B run
vllm serve Qwen/Qwen3.5-27B-FP8 --tensor-parallel-size 2 --port 8955
python -m data_miner.auto_annotation_v3 --config fl_pj.yaml

# 9B run (same images, different VLM)
vllm serve Qwen/Qwen3.5-9B --tensor-parallel-size 1 --port 8955
python -m data_miner.auto_annotation_v3 --config fl_pj.yaml

# Compare
python -m data_miner.auto_annotation_v3.compare --job-a job_27b --job-b job_9b
```

---

## Quick reference: parameter cheat sheet

| Parameter | Current | Optimized | Effect |
|-----------|---------|-----------|--------|
| `max-model-len` | 4096 | 2048 | 2× more concurrent sequences |
| `gpu-memory-utilization` | 0.85 | 0.92 | 4× more KV cache budget |
| `enforce-eager` | true | false | 10-30% faster per iteration |
| `enable-chunked-prefill` | off | on | Better GPU utilization |
| `max-num-batched-tokens` | default | 2048 | Throughput-optimized scheduling |
| `max-num-seqs` | default (256+) | 16 | Matches actual capacity |
| Pipeline `max_concurrent_calls` | ? | 8 (tune via sweep) | Matches vLLM sweet spot |
