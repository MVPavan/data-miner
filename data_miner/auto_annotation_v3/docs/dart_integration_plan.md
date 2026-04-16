# DART → aa v3 Integration Plan (v2, post-review)

Concrete phased implementation plan. Companion to [dart_integration_study.md](dart_integration_study.md) (research + empirical findings). Revised after an independent review flagged two showstoppers in the v1 plan; changes marked ⚠ below. **SAM3 refine stage + `_sam_presence` are literally untouched by routing them through the existing HF-transformers path.**

## Scope

- **Add** a new `proposal_multiclass` endpoint backed by DART's `Sam3MultiClassPredictorFast`, loaded independently.
- **Preserve** the existing HF-transformers `Sam3Model` + `Sam3Processor` for `refine` and `proposal` modes verbatim. No code changes to those handlers.
- **Accept** ~2× SAM3 weight VRAM cost in the server (one HF-transformers model, one DART model) — both fit in 24 GB with headroom (benchmark confirmed <5 GB total for SAM3 weights × 2).
- **Leave** GDINO, Falcon, OWLv2 servers and their per-class fan-out unchanged.

## ⚠ v1 → v2 key corrections

| v1 claim | Reality | v2 approach |
|---|---|---|
| "Share the same model instance" | `serve_sam3.py:78-99` loads `transformers.Sam3Model` (HF class); DART's predictors expect `build_sam3_image_model()` output (DART class with `.backbone.forward_text`, `.dot_prod_scoring`). **Different classes; cannot be shared.** | Load both models side-by-side in the server process. HF-transformers for `proposal`/`refine`; DART for `proposal_multiclass` only. |
| "Single-class `proposal` routes through Fast with N=1" | [refine.py:559](../stages/refine.py#L559) builds `text_prompt = ". ".join(load_vocab)` — a single compound prompt. DART's `set_classes(["a. b. c"])` vs stock SAM3's `". "`-joined-phrases have different semantics (merged presence vs N-way class-conditioned). Would silently change `_sam_presence`. | `proposal` endpoint stays on the HF path. DART is only reachable via the new `proposal_multiclass` endpoint. |
| "Refine untouched" | Only true at HTTP layer; would silently change at semantics layer if single-class path routed through DART. | Now literally true — `refine` and `proposal` server code paths are not edited. |

## Phase 0 — Prep

- [ ] **Install DART without numpy downgrade.** DART's `pyproject.toml` pins `numpy<2, torch==2.7`; aa v3's venv has `numpy==2.4.2, torch==2.11`. Install with `--no-deps` and add only the missing direct imports:
  ```bash
  uv pip install iopath timm ftfy regex decord
  # Then add DART to venv site-packages via .pth file (no pip install -e):
  echo "$(pwd)/scratchpad/DART" > .venv/lib/python3.12/site-packages/dart.pth
  ```
  Empirically validated for proposal + refine paths ([benchmark_dart_modes.py](../tests/benchmark_dart_modes.py)).
- [ ] Verify `sam3.pt` auto-downloads via `huggingface_hub` for DART, and the HF `facebook/sam3` weights continue to load for the existing HF path.
- [ ] **Skip text-cache pre-warm in Phase 0.** DART's cache invalidates on *exact* `class_names` list mismatch ([sam3_multiclass_fast.py:381](../../../scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L381)), and per-job `detect_classes` is usually a subset of `class_registry`. Cache is built lazily on first request per job's class list instead.

## Phase 1 — Additive `proposal_multiclass` endpoint

**File:** [serve_sam3.py](../servers/serve_sam3.py)

**Structurally additive.** The existing `proposal` (stock SAM3 single-class) and `refine` (stock SAM3 box-prompted) handlers at lines 107-110 are **not modified**. Add a third branch:

```python
elif mode == "proposal_multiclass":
    # handled by dart_predictor (loaded separately at startup)
```

**Server startup:** load two models:

```python
# Existing HF path — unchanged
self.hf_model = transformers.Sam3Model.from_pretrained(...)
self.hf_processor = Sam3Processor.from_pretrained(...)

# New DART path — additive
sys.path.insert(0, str(SCRATCH / "DART"))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
self.dart_model = build_sam3_image_model(device=..., eval_mode=True)
self.dart_predictor = Sam3MultiClassPredictorFast(
    self.dart_model, device=..., use_fp16=True,
    presence_threshold=args.presence_threshold,
    detection_only=args.detection_only,
)
self._dart_class_lock = threading.Lock()
self._dart_current_classes: tuple[str, ...] | None = None
```

**New endpoint contract:**

| Mode | Input | Output | Backend |
|---|---|---|---|
| `proposal` (existing, **unchanged**) | `{image_path, text_prompt, threshold}` | `{boxes, scores, labels}` | HF-transformers `Sam3Model` |
| `refine` (existing, **unchanged**) | `{image_path, bbox, points?}` | `{box, score}` | HF-transformers `Sam3Model` |
| `proposal_multiclass` (**new**) | `{image_path, text_prompts: [str,...], threshold?}` | `{boxes, scores, class_ids, labels}` | DART `Sam3MultiClassPredictorFast` |

⚠ Concurrency: `set_classes` mutates predictor state ([fast.py:373, 393](../../../scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L373)). With LitServe worker batching, different class lists in the same batch would clobber each other.

**Enforce same-key batching:**

```python
def predict(self, batch):
    # batch: list of requests collated by LitServe
    dart_items = [r for r in batch if r["mode"] == "proposal_multiclass"]
    if dart_items:
        # Reject the whole DART sub-batch if class lists differ
        first_key = tuple(dart_items[0]["text_prompts"])
        if any(tuple(r["text_prompts"]) != first_key for r in dart_items):
            raise ValueError("proposal_multiclass batch with mixed class lists; use same detect_classes per job")
        with self._dart_class_lock:
            if self._dart_current_classes != first_key:
                cache_path = _text_cache_path(first_key)
                self.dart_predictor.set_classes(list(first_key), text_cache=cache_path)
                self._dart_current_classes = first_key
            # now run self.dart_predictor on each request's image
```

**Text cache path:** per-(class-list-hash), atomic write:

```python
def _text_cache_path(classes: tuple[str, ...]) -> str:
    import hashlib, os
    h = hashlib.sha256("|".join(sorted(classes)).encode()).hexdigest()[:12]
    os.makedirs("/var/cache/aav3", exist_ok=True)
    return f"/var/cache/aav3/sam3_text_{h}.pt"
```

(DART's `torch.save` inside `set_classes` is not atomic; accept the small race on first write, or wrap with `torch.save` → `os.replace`.)

## Phase 2 — Detect stage client changes

**File:** [detect.py](../stages/detect.py) — SAM3 path only (lines 265-379).

Add config-gated branch; other detectors untouched:

```python
if config.servers.sam3.mode == "multiclass":
    # ONE call per image
    prompts = [cls.prompt for cls in detect_class_configs]  # ordered by detect_classes
    response = await http_post(sam3_url, {
        "image_path": path,
        "text_prompts": prompts,
        "mode": "proposal_multiclass",
    })
    # Split by class_ids (the predictor's index into the prompts list we sent)
    for det_i in range(len(response["scores"])):
        cls_idx = response["class_ids"][det_i]     # int index into prompts
        cls_cfg = detect_class_configs[cls_idx]    # aa v3 ClassConfig
        candidates[cls_cfg.name].append(
            Candidate(bbox=response["boxes"][det_i], score=response["scores"][det_i], ...)
        )
else:  # "perclass" — existing behavior
    ...
```

⚠ **Use `class_ids` (index into our prompt list), NOT `labels[i]` as class_name.** DART's `labels` is the predictor's own string ordering and may not preserve aa v3's canonical class name; the existing comment at [detect.py:275-276](../stages/detect.py#L275-L276) warns about label noise. The contract is: aa v3 sends ordered prompts, receives `class_ids` as integer indices back, maps via `detect_class_configs[class_id].name`.

Preserve existing `ProposalResult` / `Candidate` construction — dedup/filter stages downstream see an identical shape. Keep `source_model="sam3"`.

## Phase 3 — Config plumbing (all three layers)

### 3a. Pydantic ([config.py](../config.py))

Extend `ServerConfig` (lines 22-32, `extra="forbid"`):

```python
class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    port: int
    gpu: int
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    model_id: str | None = None
    # New — only read for sam3:
    mode: Literal["perclass", "multiclass"] = "perclass"
    detection_only: bool = True
    presence_threshold: float = 0.05
```

Since `extra="forbid"`, missing keys in YAML fall back to defaults — old configs keep working.

### 3b. Pipeline YAML ([configs/default.yaml](../configs/default.yaml))

```yaml
servers:
  sam3:
    port: 3003
    gpu: 1
    max_batch_size: 8
    batch_timeout_ms: 50
    mode: perclass                  # perclass | multiclass
    detection_only: true
    presence_threshold: 0.05
```

### 3c. Launcher ([serve_config.yaml](../servers/serve_config.yaml) + [launch_all.py](../servers/launch_all.py))

⚠ Two distinct files with different schemas — launcher reads `serve_config.yaml`, pipeline reads `configs/default.yaml`. Update both.

Add to `serve_config.yaml` under `sam3`:

```yaml
sam3:
  mode: perclass
  detection_only: true
  presence_threshold: 0.05
```

Extend `launch_all.py`:

- **`apply_overrides`** currently coerces only `int / float / str` (lines 102-110). Add bool handling, or avoid the issue by using `action="store_true"` CLI flags.
- **`build_command`** currently forwards `--port --device --max-batch-size --batch-timeout` (lines 130-138). Add `--mode`, `--detection-only`, `--presence-threshold`.
- **`serve_sam3.py` argparse** (lines 244-271): add matching CLI flags with the same defaults.

Concrete edit for `launch_all.py` bool handling (recommended):

```python
cmd.extend(["--mode", str(cfg.get("mode", "perclass"))])
if cfg.get("detection_only", True):
    cmd.append("--detection-only")
cmd.extend(["--presence-threshold", str(cfg.get("presence_threshold", 0.05))])
```

With `--detection-only` as `action="store_true"` in serve_sam3 argparse — sidesteps the string coercion bug entirely.

## Phase 4 — Refine stage: zero changes

[refine.py](../stages/refine.py) lines 522-584 continue to call:
- `mode: refine` → routes to HF-transformers `Sam3Model` (unchanged)
- `mode: proposal` (for `_sam_presence`, [line 559](../stages/refine.py#L559)) → routes to HF-transformers `Sam3Model` (unchanged)

No config flags for refine. No behavioral change.

## Phase 5 — Verification

### Critical missing test ⚠

- [ ] **Stock-vs-DART single-class parity** — before any routing changes, sanity-check that DART with N=1 produces similar boxes to the HF `proposal` endpoint on the same prompt. This isn't on the integration path (we don't route single-class through DART), but it guards against regressions if someone later tries to consolidate. Acceptance: per-detection IoU ≥ 0.85, score correlation ≥ 0.95 on fl_pj_sample. File: new `tests/test_dart_vs_hf_singleclass.py`.

### Integration tests

- [ ] **Parity (perclass vs multiclass, both HF vs DART respectively)** — extend [test_multiclass_proposal.py](../tests/test_multiclass_proposal.py) to compare HF per-class fan-out (N calls) vs DART multiclass (1 call) on the 8-image fixture. Accept: per-matched-detection IoU ≥ 0.9, score delta < 0.05.
- [ ] **Batch invariant with same class list** — extend [test_batch_accuracy.py](../tests/test_batch_accuracy.py) with a multi-class batch; assert batching doesn't perturb output.
- [ ] **Mixed class-list rejection** — new test: POST two `proposal_multiclass` requests with different `text_prompts` concurrently. Confirm the server rejects rather than silently clobbers (Phase 1 same-key-batching enforcement).
- [ ] **Latency** — run a full pipeline on a 100-image job with 5 classes; require SAM3 wall-time drop ≥ 2× (benchmark measured 2.3× standalone).
- [ ] **Refine regression** — run full pipeline with refine enabled; confirm refine checkpoints + traces are byte-identical between `mode=perclass` and `mode=multiclass` runs (refine must not see any difference).
- [ ] **NMS-flip post-finalize check** ⚠ — not just IoU on matched detections. Count surviving detections per image per class after cluster_and_collapse, before and after switching to `mode=multiclass`. Assert no class disappears at finalize; report duplicate-proposal counts fed into cluster_and_collapse.
- [ ] **End-to-end diff** — run the same `job_id` with both modes; use [compare.py](../compare.py) to diff final YOLO outputs. Accept: label-set match ≥ 95%, mean per-class bbox IoU ≥ 0.9.
- [ ] **Fallback drill** — set `mode=perclass`, confirm it still works and matches the pre-change baseline.

## Phase 6 — Gradual rollout

1. Land Phases 1-4 with `mode=perclass` default. Zero behavior change (new endpoint exists but isn't called).
2. Run all of Phase 5; flip one dev job to `mode=multiclass`. Inspect traces/viewer.
3. Diff finalize outputs (label-set match ≥ 95%, mean per-class bbox IoU ≥ 0.9).
4. Flip default to `mode=multiclass` once two consecutive dev jobs look clean.
5. Keep the `perclass` code path for at least one release cycle as a rollback.

## Risks + mitigations (updated)

| Risk | Mitigation |
|---|---|
| ~~Sharing one SAM3 model weight load~~ (deleted; not applicable) | Two models loaded; ~2× SAM3 weights fits comfortably in 24 GB RTX 3090 |
| Concurrent `proposal_multiclass` requests with different class lists | Same-key-batching enforcement + `_dart_class_lock` in server predict path (Phase 1) |
| `class_name` extracted from DART's `labels` field instead of our prompt index | Use returned `class_ids` as index into aa v3's sent prompt list (Phase 2) |
| Text cache race on first write / multi-worker deploy | Per-hash filename, atomic `os.replace`; first-request cost acceptable |
| `numpy<2` pin in DART pyproject breaking editable install | `--no-deps` install + `.pth` file; validated at `numpy==2.4.2` |
| Config bool coercion in launch_all.apply_overrides | Use `action="store_true"` CLI flags to sidestep; or patch apply_overrides for bool |
| VRAM on GPUs smaller than 24 GB | Add a config escape hatch to disable DART loading when `mode=perclass`; skip the DART model load entirely if no worker in the pool has `multiclass` |
| NMS-flip shifts proposals fed to dedup | Verified in Phase 5 post-finalize check, not just matched-IoU |

## What NOT to change

- Don't touch [refine.py](../stages/refine.py) at all.
- Don't change the HF-transformers code path in [serve_sam3.py](../servers/serve_sam3.py) — `proposal` and `refine` handlers stay verbatim.
- Don't change GDINO, Falcon, OWLv2 servers or their per-class fan-out.
- Don't vendor DART into `data_miner/` — it remains under `scratchpad/DART/` reachable via `.pth`.
- Don't add video / TRT / single-pass / student-backbone modes in Phase 1-6.

## Critical files to modify (final list)

| File | Change |
|---|---|
| [servers/serve_sam3.py](../servers/serve_sam3.py) | Additive third mode `proposal_multiclass`; load DART model alongside HF model; add same-key-batching + lock; add CLI flags |
| [servers/serve_config.yaml](../servers/serve_config.yaml) | Add `sam3.mode / detection_only / presence_threshold` |
| [servers/launch_all.py](../servers/launch_all.py) | Forward `--mode --detection-only --presence-threshold`; handle bool in `apply_overrides` or use store_true |
| [stages/detect.py](../stages/detect.py) (265-379) | SAM3-only gated branch; emit single multiclass call; decode by `class_ids` |
| [config.py](../config.py) | `ServerConfig` gains `mode`, `detection_only`, `presence_threshold` (defaults preserve old behavior) |
| [configs/default.yaml](../configs/default.yaml) | Add `servers.sam3.mode=perclass` etc. |
| [tests/test_multiclass_proposal.py](../tests/test_multiclass_proposal.py) | Add perclass-vs-multiclass parity test |
| [tests/test_batch_accuracy.py](../tests/test_batch_accuracy.py) | Add multi-class batch invariant |
| **New** `tests/test_dart_vs_hf_singleclass.py` | Stock-vs-DART N=1 parity (sanity guard) |
| **New** `tests/test_multiclass_mixed_batch_rejection.py` | Server rejects mixed class lists |
| `dart.pth` in venv site-packages | Makes DART importable without pip install |

## Reference: why this is worth doing

- **2.3× SAM3 wall-time speedup** measured standalone ([benchmark_dart_modes.py](../tests/benchmark_dart_modes.py), documented in [dart_integration_study.md § Empirical benchmark](dart_integration_study.md#empirical-benchmark-on-aa-v3-sample-images)).
- Closes the long-standing per-class fan-out workaround for SAM3 specifically.
- **Refine stage is literally untouched** (v2 correction) — the highest-risk stage stays on its proven code path.

## Verdict on plan readiness

✅ Ready to execute after v2 edits. Two showstoppers from v1 are resolved (separate model loads; `proposal` endpoint stays on HF path). Remaining work is additive + config plumbing; the riskiest stage (refine) is never touched.
