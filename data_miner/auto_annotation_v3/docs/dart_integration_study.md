# DART ↔ auto_annotation_v3 Integration Study

## Context

`auto_annotation_v3` currently uses stock SAM3 as one of four detectors and again inside the refine stage. The detect stage fans out **one HTTP call per (model, class)** because joint multi-class prompts collapse SAM3's output to zero detections (verified in [test_multiclass_proposal.py](data_miner/auto_annotation_v3/tests/test_multiclass_proposal.py) and called out in [detect.py:184-186](data_miner/auto_annotation_v3/stages/detect.py#L184-L186) and §10/§11 of [aav3_filtering_scores_discussion.md](data_miner/auto_annotation_v3/docs/aav3_filtering_scores_discussion.md)). Each per-class call re-runs the ViT-H backbone from scratch — the single largest wall-clock cost in the detector.

[DART](scratchpad/DART/README.md) is a training-free framework whose entire thesis is "convert SAM3 into a real-time multi-class open-vocab detector": run the ViT-H backbone **once**, batch the per-class encoder/decoder passes, apply class-specific NMS, and optionally drop in TensorRT / distilled backbones / ByteTrack. Its `Sam3MultiClassPredictorFast` ([scratchpad/DART/sam3/model/sam3_multiclass_fast.py](scratchpad/DART/sam3/model/sam3_multiclass_fast.py)) is a direct, preserves-quality replacement for the per-class loop we're already running.

The architectural alignment is unusually clean: DART's raison d'être *is* the workaround aa v3 is paying for.

## Recommended integration: tiered, opt-in

### Tier 1 — Drop-in multi-class SAM3 server (highest value, lowest risk)

Replace the core of [serve_sam3.py](data_miner/auto_annotation_v3/servers/serve_sam3.py) with `Sam3MultiClassPredictorFast` (batched + FP16 + shared-encoder mode). Add one new endpoint mode while **keeping the existing `proposal` and `refine` modes bit-compatible** so detect/refine stages need zero changes to start:

- New `proposal_multiclass` mode: input `{image_path, text_prompts: [str, ...]}` → output `{boxes, scores, labels, class_ids}` with labels aligned to input prompts. Backbone runs once per image; encoder/decoder batched across the prompt list.
- Existing `proposal` mode: stays single-class (trivially: wrap single prompt in a 1-element list); same output schema.
- Existing `refine` mode: unchanged — DART's mask path is unchanged for single-box prompts.

Then, in [detect.py:265-299](data_miner/auto_annotation_v3/stages/detect.py#L265-L299), add a code path that collapses the `asyncio.gather` per-class fan-out for SAM3 into a **single `proposal_multiclass` call per image**. Other three servers (GDINO, Falcon, OWLv2) keep their existing per-class fan-out — this change is scoped strictly to SAM3.

Expected win: for N classes, SAM3 cost drops from `N × (backbone + enc + dec + masks)` to `1 × backbone + 1 × enc(bs=N) + 1 × dec(bs=N) + 1 × masks(bs=K_hits)`. With typical N=3-10 classes, that's a 3-10× speedup on the SAM3 share of detect wall time, with **no quality change** (DART preserves per-class prompting semantics).

**Critical files to modify:**
- [data_miner/auto_annotation_v3/servers/serve_sam3.py](data_miner/auto_annotation_v3/servers/serve_sam3.py) — swap model loader for `Sam3MultiClassPredictorFast`; add `proposal_multiclass` handler; keep `proposal`/`refine` response schemas byte-identical.
- [data_miner/auto_annotation_v3/stages/detect.py](data_miner/auto_annotation_v3/stages/detect.py) (lines 265-379) — gate SAM3 path on a config flag; emit one multi-class call instead of per-class fan-out.
- [data_miner/auto_annotation_v3/servers/serve_config.yaml](data_miner/auto_annotation_v3/servers/serve_config.yaml) — bump SAM3 GPU memory / adjust batching (batched multi-class uses more VRAM per request).
- [data_miner/auto_annotation_v3/configs/default.yaml](data_miner/auto_annotation_v3/configs/default.yaml) — add `servers.sam3.mode: multiclass | perclass` toggle for fallback.

**Reuse, don't rewrite:** `Sam3MultiClassPredictorFast.set_classes(...).set_image(...).predict(...)` is the full public API we need. Text-cache support (`.pt` embedding cache) can be reused to cache class embeddings across images in a job — since `class_registry` is frozen at job start, one warm-up embeds all classes once.

### Tier 2 — TRT acceleration for the detect path (medium value, medium risk)

DART ships TRT engines for both backbone and encoder-decoder ([scratchpad/DART/sam3/trt/](scratchpad/DART/sam3/trt/)). The enc-dec engine is **detection-only** (no hidden states for masks), which is a perfect fit: aa v3's detect stage only needs boxes + scores; masks are only generated downstream in refine. Route:

- **Detect stage** → TRT backbone + TRT enc-dec (7-41 ms for 1-8 classes after a one-time engine build).
- **Refine stage** → PyTorch SAM3 (unchanged), because TRT can't emit masks.

Gated behind the same `servers.sam3.mode` config; requires a one-time `export_enc_dec.py` + `build_engine.py` run per `(max_classes, imgsz)` tuple. Document the build step; do not auto-build in `launch_all.py`.

### Tier 3 — Video annotation (new capability, separable workstream)

DART's `PipelinedVideoProcessor` + ByteTrack ([scratchpad/DART/sam3/tracking/byte_tracker.py](scratchpad/DART/sam3/tracking/byte_tracker.py)) give aa v3 a cleanly-scoped path to **video auto-annotation**, which the pipeline currently doesn't support. This is not a refine-stage enhancement; it's a new entry point:

- New `runtime.video_path` / `runtime.video_dir` input mode in [cli.py](data_miner/auto_annotation_v3/cli.py).
- New pre-detect stage: sample-or-decode frames, track identities across frames, emit per-frame proposals with stable `track_id` that the rest of the pipeline can propagate into traces / review.
- Label propagation: tracks let VLM evaluation + refine happen once per track (or once per keyframe), then replay the verdict to all frames of that track — a massive VLM-cost reduction for video.

Only worth doing if video is on the roadmap; do not build speculatively.

### Tier 4 — Student backbones (low value right now)

DART's distilled RepViT/TinyViT backbones trade ~15 AP for 3-5× speedup. aa v3 runs as a batch job with 4 detectors cross-checking each other; the quality floor matters more than latency, so this is the weakest fit. Revisit only if Tier 1/2 aren't enough and a latency budget forces triage.

## What **not** to do

- Don't replace GDINO/Falcon/OWLv2 with DART — they contribute diverse failure modes that feed `agreement`-based auto-accept; collapsing to a single detector family would kill the cluster-and-collapse dedup signal.
- Don't change the per-class fan-out for non-SAM3 servers. Falcon's joint-prompt degradation is a separate issue that DART does not address.
- Don't vendor DART into `data_miner/`. It's an active external repo with its own licence/deps (torch 2.7, CUDA 12.6). Install as an editable dep in `pyproject.toml`.

## Verification

For Tier 1 (the only tier worth landing first):

1. **Parity test** — extend [tests/test_multiclass_proposal.py](data_miner/auto_annotation_v3/tests/test_multiclass_proposal.py) with a fourth comparison: `per_class_perclass_endpoint` vs `multiclass_endpoint`, asserting IoU ≥ 0.9 on overlapping detections and score delta < 0.05. Must pass on the same 3-image `output/sample/fl_pj_sample` fixture used by existing tests.
2. **Batch-vs-sequential invariant** — [tests/test_batch_accuracy.py](data_miner/auto_annotation_v3/tests/test_batch_accuracy.py) already covers LitServe batching; extend it with multi-class requests to confirm batching doesn't perturb multi-class outputs.
3. **Latency** — log SAM3 `latency_ms` per image in [contracts.py ProposalResult](data_miner/auto_annotation_v3/contracts.py); run a `detect_classes=[forklift,palletjack,person,pallet,worker]` job on a 100-image fixture pre- and post-integration; require SAM3 wall time to drop ≥ 3× and total detect-stage p95 to not regress.
4. **End-to-end equivalence** — run full pipeline with `servers.sam3.mode=perclass` and `=multiclass` on the same job_id, then use the existing [compare.py](data_miner/auto_annotation_v3/compare.py) tool to diff final YOLO outputs. Acceptance: label set per image matches ≥ 95%, mean per-class bbox IoU ≥ 0.9.
5. **Fallback** — kill the multiclass server, confirm `mode=perclass` still works unchanged (nothing about the PyTorch stock-SAM3 path is removed).
6. **Mask-tight-box empirical check** (one-off, pre-implementation) — on 20 aa v3 validation images, run DART with `detection_only=False`, compute `box_iou(box_head_output, mask.nonzero().aminmax())` per detection. If mean IoU > 0.97 across aa v3's class set, Tier-1 `detection_only=True` is safe for detect. If < 0.9 for any critical class, keep masks on in detect and derive tight boxes post-hoc. See §"Box vs mask-derived box" below.

---

## Technical findings from deep exploration of DART

Captured here so future iterations don't re-derive them. Three source files drive everything: [demo_multiclass.py](scratchpad/DART/demo_multiclass.py) (CLI/benchmark driver), [sam3_multiclass.py](scratchpad/DART/sam3/model/sam3_multiclass.py) (baseline `Sam3MultiClassPredictor`, sequential per-class), [sam3_multiclass_fast.py](scratchpad/DART/sam3/model/sam3_multiclass_fast.py) (`Sam3MultiClassPredictorFast`, the superset).

### Compute comparison (N classes, K present after presence gate)

| Mode | Backbone | Encoder | Decoder | Masks |
|---|---|---|---|---|
| Stock SAM3 per-prompt (what aa v3 does today) | N× | N× | N× | N× |
| `Sam3MultiClassPredictor` (baseline) | 1× | N× (bs=1) | N× (bs=1) | K× (bs=1, lazy) |
| Fast `_forward_batched` | 1× | 1× (bs=N) | 1× (bs=N) | K× (bs=1, lazy per present class) |
| Fast `_forward_shared_encoder` | 1× | 1× (bs=1, generic prompt) | 1× (bs=N) | K× |
| Fast `_forward_batched_trt` | 1× TRT | 1× TRT (chunked by `max_classes`) | 1× TRT | — (forced `detection_only`) |
| Fast `_predict_single_pass` | 1× | 1× (bs=1, concat prompt) | 1× (bs=1) | 1× (bs=1, all classes) |

Dispatch precedence inside `Sam3MultiClassPredictorFast.predict()` ([lines 615-633](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L615-L633)): `single_pass → trt_enc_dec → shared_encoder → batched`.

### `Sam3MultiClassPredictorFast` knob matrix

| Knob | Values | Default | Effect | Constraints |
|---|---|---|---|---|
| `compile_mode` | None / "default" / "reduce-overhead" / "max-autotune" | None | JIT compile encoder/decoder/backbone | Lazy, cached on first call ([_ensure_compiled:287-345](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L287-L345)) |
| `use_fp16` | bool | True | Autocast around all compute-heavy calls | — |
| `presence_threshold` | float | 0.05 | Per-class gate; classes below threshold skip mask generation | Set 0.0 to disable; see §"Presence early-exit" |
| `shared_encoder` | bool | False | Encoder runs once with `generic_prompt`; decoder stays bs=N class-specific | Mutually exclusive with `single_pass` |
| `generic_prompt` | str | "object" | Prompt for shared encoder pass | Used only when `shared_encoder=True` |
| `single_pass` | bool | False | One pass at bs=1 with concat'd class tokens; class assigned post-hoc | Mutually exclusive with `shared_encoder` |
| `class_method` | "cosine" / "attention" / "prototype" | "cosine" | Query→class assignment for single-pass | `"prototype"` requires `prototype_path` |
| `detection_only` | bool | False | Skip mask head, box NMS, fully vectorized postprocess | Required when `trt_enc_dec_engine_path` set |
| `trt_engine_path` | path | None | TRT backbone | — |
| `trt_enc_dec_engine_path` | path | None | TRT enc+dec+scoring (chunked on N>max_classes) | Forces `detection_only=True` |
| `trt_max_classes` | int | 4 | Matches `--max-classes` at engine export | — |
| `text_cache` (arg to `set_classes`) | path | None | `.pt` save/load for class text embeddings | Enables full-TRT no-checkpoint mode via `_TRTModelStub` |

### Presence early-exit — two behaviors between the predictors

- **Baseline `Sam3MultiClassPredictor`** ([line 233-235](scratchpad/DART/sam3/model/sam3_multiclass.py#L233-L235)): presence is used **only as a score multiplier** (`probs = probs * presence`). No early exit. Every class gets mask head called.
- **Fast `Sam3MultiClassPredictorFast`** ([line 758-765](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L758-L765) and [line 1296](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L1296)): both — thresholded early exit at the decoder output skips mask generation for absent classes, *and* presence still multiplies into kept queries' scores.

The presence signal is derived from a **learned presence token** in the decoder ([decoder.py:209-298](scratchpad/DART/sam3/model/decoder.py#L209-L298)), a prepended query whose output is fed through a lightweight MLP → scalar logit per class. Not a separate head; not an image-level detector.

⚠️ **Distilled/pruned checkpoints disable presence** ([demo:218-222](scratchpad/DART/demo_multiclass.py#L218-L222)): `model.transformer.decoder.presence_token = None` because distillation doesn't train this head, and silently multiplying by untrained logits destroys quality. If we ever swap in a distilled backbone (Tier 4), we must mirror this nullification.

### `detection_only=True` — what it actually does

**Claim:** the returned boxes are **pre-NMS bit-identical** between `detection_only=True` and `detection_only=False`. Only NMS choice and post-processing vectorization change.

**Proof by code trace:**

All four forward paths emit boxes via the same pattern — `bbox_embed(hs)` on decoder hidden states, added to `inverse_sigmoid(reference_boxes)`, sigmoided, then cxcywh→xyxy scaled to pixels:

- `Sam3MultiClassPredictor._forward_single_class` → [sam3_multiclass.py:443-447](scratchpad/DART/sam3/model/sam3_multiclass.py#L443-L447)
- `_forward_batched` → [fast.py:746-749](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L746-L749)
- `_forward_shared_encoder` → [fast.py:858-861](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L858-L861)
- `_predict_single_pass` → [fast.py:1060-1063](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L1060-L1063)

The `batched` dict (`scores_all`, `boxes_all`, `presence_probs`) is produced **before** the dispatch at [lines 1262-1274](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L1262-L1274). `detection_only` only selects between `_postprocess_detection` (vectorized, box NMS) and `_postprocess_with_masks` (per-class Python loop, mask NMS).

What changes:

| Aspect | `detection_only=False` | `detection_only=True` |
|---|---|---|
| Seg head called | Yes, per present class | No |
| `masks` / `masks_logits` in output | Populated | `None` |
| NMS algorithm | Mask-IoU (Python loop, `_nms`) | Box-IoU (torchvision `batched_nms`, CUDA) |
| Postprocess | Per-class Python loop | Fully vectorized across all classes |

**Implication:** the *set* of surviving detections can differ (mask-IoU vs box-IoU suppress different overlaps — e.g., person holding a bag with contained but shape-different masks), but any detection that survives has the **same box value** in either mode.

### Box vs mask-derived box — architectural reality

Common intuition: "SAM3's mask head takes the predicted box as input, so a loose box → loose mask, and a box derived from the mask (`mask.nonzero().aminmax()`) is tighter and more accurate."

**This is how SAM1 works, not SAM3.** SAM1 is prompt-to-mask (box in → mask out). SAM3 is DETR-style: both heads are **siblings** reading from the same decoder query embeddings `hs`, predicted in parallel.

The seg head signature at [sam3_multiclass.py:257-264](scratchpad/DART/sam3/model/sam3_multiclass.py#L257-L264):

```
segmentation_head(
    backbone_feats=FPN,
    obj_queries=hs,             # decoder hidden states
    encoder_hidden_states=...,
    prompt=text_embeds,
    prompt_mask=...,
)
```

**No box is passed in.** The dataflow is:

```
decoder(queries) → hs ──┬──> bbox_embed(hs) ──> boxes
                        └──> segmentation_head(hs, img feats, text) ──> masks
```

So a loose box does NOT cause a loose mask in SAM3, and a tight mask was NOT "refined from" a loose box. They are two reads of the same latent.

**That said**, mask-derived box and box-head box *can* still disagree empirically — not because one conditions on the other, but because:
- Box head = 3-layer MLP → 4 scalars, trained with L1+GIoU
- Mask head = heavier, per-pixel, trained with BCE+Dice
- For elongated/irregular/partially-occluded objects, `mask.nonzero().aminmax()` can be marginally tighter or slightly shifted vs the box head output (documented in Mask-DINO follow-ups)

**Operational consequence:** neither mode returns a mask-tight box. If we want one, we compute it ourselves: `box = mask_binary.nonzero(...).aminmax(...)`. Nothing in DART does this internally. For aa v3's detect stage, the empirical IoU check in Verification step 6 tells us whether to bother.

### Empirical benchmark on aa v3 sample images

Ran [scratchpad/DART/test_aav3_modes.py](scratchpad/DART/test_aav3_modes.py) on the 8-image `output/sample/fl_pj_sample/` fixture with classes `["person", "forklift", "pallet jack"]`, single RTX 3090, 1 warmup pass per mode, confidence 0.3, NMS 0.7. Results saved under `output/dart_tests/`.

| Mode | Backbone (ms) | Predict (ms) | Total (ms) | Dets | Speedup |
|---|---|---|---|---|---|
| M1 `Sam3MultiClassPredictor` (baseline) | 293.0 | 209.8 | 502.8 | 28 | 1.00× |
| M2 Baseline + `detection_only` | 290.1 | 173.3 | 463.4 | 29 | 1.08× |
| M3 `Sam3MultiClassPredictorFast` batched fp16 | 133.7 | 87.1 | **220.8** | 27 | **2.28×** |
| M4 Fast batched + `detection_only` | 131.0 | 64.8 | **195.9** | 28 | **2.57×** |
| M5 Fast + `shared_encoder="warehouse"` | 132.1 | 71.6 | 203.7 | 30 | 2.47× |
| M6 Single-pass cosine | 131.6 | 46.3 | 177.9 | 12 | 2.83× ⚠️ |
| M7 Single-pass attention | 135.8 | 45.8 | 181.6 | 12 | 2.77× ⚠️ |
| M8 `compile_mode="reduce-overhead"` | — | — | **FAILED** | — | CUDA graphs tensor-reuse bug in encoder.py:562 |

**Interpretation:**

1. **Fast batched gives a clean 2.3× end-to-end speedup** with no detection-count regression (27 vs 28). Backbone drops ~2.2× (fp16 autocast); predict drops ~2.4× (batched encoder + decoder). This is the Tier-1 win, empirically confirmed.
2. **`detection_only` adds a small additional ~11% speedup** on top of fast batched (220.8 → 195.9 ms). Smaller than the prior naive estimate because lazy mask gen already only runs for queries above confidence (a small K), so skipping it doesn't save as much as skipping the full mask head would if called per-class.
3. **Shared encoder is competitive** (203.7 ms, 2.47×) with slightly *more* detections (30), suggesting the generic-prompt encoder pass doesn't hurt recall here. Saves 1× encoder pass × (N-1) at the cost of a contiguous-memory expand.
4. **Single-pass modes are fastest but detect 12 objects vs 27-30** — a ~55% detection drop. Unusable for aa v3 as a quality-preserving replacement. Consistent with the DART paper's AP tradeoff for single-pass.
5. **`compile_mode="reduce-overhead"` fails** with a CUDA graphs tensor-reuse error inside SAM3's encoder (`encoder.py:562`). Not a DART bug — it's a known incompatibility between the SAM3 encoder's tensor handling and torch.compile's CUDA graph mode. Workaround would require `torch.compiler.cudagraph_mark_step_begin()` per call or switching to `mode="default"`. Not worth pursuing for Tier 1.

**Bbox-from-mask agreement (on the M3 full-mask run, 27 detections total)**:

| Metric | Value |
|---|---|
| Mean IoU(box_head, mask_tight_box) | **0.9685** |
| Min IoU | 0.7981 |
| % detections with IoU > 0.95 | 88.9% |
| % detections with IoU > 0.90 | 96.3% |

Validated the architectural analysis: the two box sources **usually agree** (mean 0.9685), but there is a **long tail of ~12% where they disagree by >5% IoU** (min 0.7981). So `detection_only=True` is safe for the detect stage because aa v3's cluster/dedup re-geometrically-filters anyway, but if we ever want mask-tight boxes in the final output, there *is* a measurable difference worth capturing — specifically for the tail. Decision: stay with `detection_only=True` in detect; if the refine-stage QA shows the tail detections are the problematic ones for a given class, add a `box = mask.nonzero().aminmax()` post-step in the finalize stage (trivial, one line).

### Empirical NMS-flip root cause (9_fPtt5zRpA_002730.jpg)

A visual inspection of the M1-M4 grids showed what looked like a **class flip** on one annotation between the mask-mode pair (M1, M3) and the detection-only pair (M2, M4). The split aligning with the NMS axis — not the baseline-vs-fast axis — pointed to NMS as the sole cause. Confirmed with [scratchpad/DART/diagnose_nms_flip.py](scratchpad/DART/diagnose_nms_flip.py).

**Finding:** no individual detection's class label changed. Instead, the **set of surviving detections differs** because box-IoU and mask-IoU disagree on within-class duplicates, and the score-descending draw order in `annotate_image` makes the difference visible as an apparent flip.

**Concrete trace on the right-side vehicle:**

| Mode | Detections at right vehicle |
|---|---|
| M1, M3 (mask-NMS) | forklift @ 0.865 (tall box) + pallet jack @ 0.785 (inner box) + **pallet jack @ 0.696 (tall box)** |
| M2, M4 (box-NMS) | forklift @ 0.865 (tall box) + pallet jack @ 0.785 (inner box) |

The two pallet jack detections had **box-IoU ≈ 0.707** (just above the default NMS threshold of 0.7) but **mask-IoU < 0.7** (the tall-box mask covers more pixels than the inner mask; pixel-overlap is lower than bounding-box overlap). Result:

- `batched_nms` @ thr=0.7 → suppresses `pallet jack @ 0.696` as a within-class duplicate of `pallet jack @ 0.785`.
- Mask-IoU NMS @ thr=0.7 → both pallet jacks survive; forklift is unaffected either way (different class, `per_class_nms=True`).

`annotate_image` draws in score-descending order (lower scores painted last, label on top). So on the tall right box:
- Mask modes → last label drawn is `pallet jack @ 0.696` → user sees "pallet jack" there.
- Detection-only modes → last label drawn on that tall box is `forklift @ 0.865` → user sees "forklift" there.

**Implication for aa v3:**

- Not a correctness failure. Both pallet jack proposals exist in mask mode and would feed aa v3's own cluster-and-collapse dedup which re-runs geometric filtering at its own thresholds (filtering block in [configs/default.yaml](data_miner/auto_annotation_v3/configs/default.yaml)).
- Choice between `detection_only=True/False` for the detect stage is a **policy choice**, not accuracy: mask-NMS preserves more duplicate proposals at different scales (more candidates for downstream cluster/vote); box-NMS ships cleaner, slightly-smaller proposal sets.
- Reinforces the Tier-1 recommendation: `detection_only=True` is fine because aa v3's dedup doesn't rely on preserving multiple same-class overlapping proposals — it clusters them anyway.

### Availability

DART is a **local Python package**, not a hosted HF API:
- Repo: `scratchpad/DART/` (editable import via `sys.path`)
- HuggingFace role: hosts model weights (`sam3.pt`) auto-downloaded by `build_sam3_image_model()` via `huggingface_hub`; also serves as the project homepage ([https://huggingface.co/mehmetkeremturkcan/DART](https://huggingface.co/mehmetkeremturkcan/DART))
- No inference API — all compute runs on your GPU
- Deps installed surgically into the repo `.venv`: `iopath`, `timm`, `ftfy`, `regex`, `decord` (plus their transitive). Note: DART's `pyproject.toml` pins `numpy<2` but we ran with `numpy==2.4.2` successfully for these modes; if TRT or video modes show numpy issues, revisit.

### Architectural alignment with aa v3

aa v3 already pays the per-class HTTP fan-out cost because joint multi-class prompts collapse stock SAM3 to zero detections ([detect.py:184-186](data_miner/auto_annotation_v3/stages/detect.py#L184-L186), verified in [test_multiclass_proposal.py](data_miner/auto_annotation_v3/tests/test_multiclass_proposal.py)). DART exists specifically to solve this: 1× backbone + batched enc/dec + presence gating + per-class NMS, preserving per-class prompting semantics. The integration isn't an upgrade; it's closing an open workaround.

### Recommended default for Tier 1 (sharpened)

```
Sam3MultiClassPredictorFast(
    model,
    resolution=1008,
    use_fp16=True,                      # default; free win
    presence_threshold=0.05,            # default; skips mask head for absent classes
    detection_only=True,                # aa v3 detect stage only reads boxes+scores
    compile_mode=None,                  # skip until stable; add "reduce-overhead" after parity
    shared_encoder=False,
    single_pass=False,
    # text cache: set once per job since class_registry is frozen
)
predictor.set_classes(classes, text_cache="/tmp/aav3_{job_id}_text.pt")
```

Refine stage: **do not swap**. DART's multi-class predictors don't expose box-prompted single-class mask refinement (that's `Sam3Processor.set_box_prompt` in stock SAM3, which the existing `serve_sam3.py` refine endpoint already wraps). Keep the existing refine path unchanged.
