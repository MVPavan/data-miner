# Multi-Image Batching for DART's Sam3MultiClassPredictorFast

## Context

DART currently batches on the **class axis** (B images × N classes → one backbone + one encoder/decoder at bs=N per image). But in a LitServe server handling concurrent requests, images arrive in batches too. Today the server loops `set_image → predict` per image — backbone runs B times sequentially. Multi-image batching would run the backbone ONCE for B images, then the encoder/decoder ONCE at bs=B*N, amortizing both the heaviest compute stages.

Architecture verification confirms this is feasible: backbone, `_get_img_feats`, encoder, decoder, deformable attention, scoring, and seg head ALL treat the batch dim as a purely parallel axis with zero cross-batch dependencies.

## Feasibility trace (validated against source)

| Component | Current shape | Multi-image shape | Cross-batch ops? | Change needed |
|---|---|---|---|---|
| `backbone.forward_image` | (1, 3, 1008, 1008) → FPN list of (1, C, H, W) | (B, 3, 1008, 1008) → FPN list of (B, C, H, W) | None | Already works |
| `_get_img_feats` [sam3_image.py:114-132](scratchpad/DART/sam3/model/sam3_image.py#L114-L132) | `img_ids=[0]` → `x[img_ids].flatten(2).permute(2,0,1)` → (H\*W, 1, d) | `img_ids=arange(B)` → (H\*W, B, d) | None | Change img_ids |
| Encoder [encoder.py] | src: (H\*W, N, d), valid_ratios computed internally at (N, levels, 2) | src: (H\*W, B\*N, d), valid_ratios at (B\*N, levels, 2) | None — all ops are per-element/broadcast | Works if we pass B\*N features |
| Decoder [decoder.py] | tgt: (Q, N, d), memory: (tokens, N, d) | tgt: (Q, B\*N, d), memory: (tokens, B\*N, d) | None — deformable attn is per-batch | Works at B\*N |
| `dot_prod_scoring` [model_misc.py:68-93] | hs: (layers, N, Q, d), prompt: (seq, N, d) → batched matmul | hs: (layers, B\*N, Q, d) → batched matmul | None | Works at B\*N |
| Seg head [maskformer_segmentation.py] | `feat[image_ids]` with image_ids=[0] | `feat[image_ids]` with image_ids=[0,0,..,1,1,..,B-1,...] | None — per-element indexing | Pass correct image_ids |
| Presence token | (layers, 1, N) | (layers, 1, B\*N) → reshape to (B, N) | None | Reshape in postprocess |

**Constraint:** all images must be same resolution. Since `set_image` resizes to `self.resolution` (1008×1008), this is always true.

## Implementation plan

**New file:** `data_miner/auto_annotation_v3/dart_batch.py`

Create a subclass `Sam3MultiClassPredictorBatch(Sam3MultiClassPredictorFast)` that adds multi-image batching. Lives in aa v3's tree (not in DART), since DART is installed in the uv venv and importable via `from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast`. Zero changes to DART source. All parent internals (`self._batched_text`, `self._encoder_fn`, `self.model._get_img_feats`, `self._postprocess`, `self._empty_result`, `self._zeros_like_cached`, etc.) are single-underscore attributes — fully accessible from the subclass.

### Step 1: `set_images(images) → Dict` (batched backbone)

```python
@torch.inference_mode()
def set_images(
    self,
    images: List[Union[PIL.Image.Image, torch.Tensor, np.ndarray]],
    state: Optional[Dict] = None,
) -> Dict:
```

- Resize all images to `self.resolution` (PIL resize on CPU first — fast path from line 506-509).
- Transform each → stack into (B, 3, 1008, 1008).
- Run backbone ONCE: `self._backbone_fn(batched_tensor)` → FPN features at (B, C, H, W).
- Store per-image original sizes (list of (h, w) tuples).
- Return state dict with `backbone_out`, `original_sizes: [(h,w), ...]`, `batch_size: B`.

### Step 2: `predict_batch(state) → List[Dict]` (B×N encoder/decoder)

```python
@torch.inference_mode()
def predict_batch(
    self,
    state: Dict,
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.7,
    per_class_nms: bool = True,
) -> List[Dict]:
```

**Feature expansion:** `_get_img_feats(backbone_out, img_ids=arange(B))` → img_feats as list of (H\*W, B, d). Expand to (H\*W, B\*N, d):

```python
B = state["batch_size"]
N = self._num_classes
img_ids = torch.arange(B, device=self.device, dtype=torch.long)
backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = (
    self.model._get_img_feats(state["backbone_out"], img_ids)
)
# (H*W, B, d) → (H*W, B, 1, d) → (H*W, B, N, d) → (H*W, B*N, d)
batched_img_feats = [
    f.unsqueeze(2).expand(-1, -1, N, -1).reshape(f.shape[0], B * N, -1)
    for f in img_feats
]
batched_img_pos = [
    p.unsqueeze(2).expand(-1, -1, N, -1).reshape(p.shape[0], B * N, -1)
    for p in img_pos_embeds
]
```

**Text prompt expansion:** (seq, N, d) → (seq, B\*N, d) via tile:

```python
prompt = self._batched_text.repeat(1, B, 1)       # (seq, B*N, d)
prompt_mask = self._batched_mask.repeat(B, 1)      # (B*N, seq)
```

**Forward pass:** call `_forward_batched_multi_image()` (new private method) which is identical to existing `_forward_batched` but with the expanded tensors. The encoder and decoder don't know or care that bs=B\*N instead of bs=N — same API.

**Postprocess:** reshape outputs from (B\*N, ...) back to (B, N, ...) and loop per-image:

```python
# scores: (B*N, Q, 1) → (B, N, Q, 1)
scores_BN = scores_last.reshape(B, N, Q, 1)
boxes_BN = boxes_last.reshape(B, N, Q, 4)
presence_BN = presence_probs.reshape(B, N) if presence_probs is not None else None

results = []
for i in range(B):
    # Slice per-image tensors: (N, Q, ...) — same shape existing _postprocess expects
    per_img_batched = {
        "scores_all": scores_BN[i],          # (N, Q, 1)
        "boxes_all": boxes_BN[i],            # (N, Q, 4)
        "presence_probs": presence_BN[i] if presence_BN is not None else None,
        "present_indices": ...,               # recompute from per-image presence
        # For masks: hs_all, encoder_hidden_states sliced at [i*N:(i+1)*N]
        "hs_all": hs_all[:, i*N:(i+1)*N] if hs_all is not None else None,
        "encoder_hidden_states": enc_hs[:, i*N:(i+1)*N] if enc_hs is not None else None,
        "prompt": self._batched_text,         # (seq, N, d) — same for all images
        "prompt_mask": self._batched_mask,    # (N, seq) — same for all images
    }
    orig_h, orig_w = state["original_sizes"][i]
    # Reuse existing _postprocess with per-image img_ids for seg head
    img_ids_i = torch.tensor([i], device=self.device, dtype=torch.long)
    result_i = self._postprocess(
        batched=per_img_batched,
        backbone_out=state["backbone_out"],   # seg head indexes via image_ids
        img_ids=img_ids_i,
        orig_h=orig_h, orig_w=orig_w,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        per_class_nms=per_class_nms,
    )
    results.append(result_i)
return results
```

### Step 3: `predict_images()` convenience method

```python
@torch.inference_mode()
def predict_images(
    self,
    images: List[Union[PIL.Image.Image, torch.Tensor, np.ndarray]],
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.7,
    per_class_nms: bool = True,
) -> List[Dict]:
    """Convenience: set_images + predict_batch in one call."""
    state = self.set_images(images)
    return self.predict_batch(state, confidence_threshold, nms_threshold, per_class_nms)
```

### Step 4: `_forward_batched_multi_image()` (private)

Nearly identical to existing `_forward_batched` ([lines 675-776](scratchpad/DART/sam3/model/sam3_multiclass_fast.py#L675-L776)) but accepts pre-expanded (H\*W, B\*N, d) features and (seq, B\*N, d) prompts. The encoder/decoder calls are identical — only the input shapes differ.

Key difference: `query_embed.unsqueeze(1).expand(-1, B*N, -1)` instead of `expand(-1, N, -1)`.

### What NOT to change

- **DART repo (`scratchpad/DART/`) — zero edits.** All new code lives in `data_miner/auto_annotation_v3/dart_batch.py`.
- Existing `set_image()`, `predict()`, `predict_image()` — inherited, untouched.
- `_forward_batched`, `_postprocess`, `_postprocess_detection`, `_postprocess_with_masks` — inherited, reused as-is for the per-image post-processing loop.
- Encoder, decoder, backbone, seg head internals — zero changes.

## Expected speedup

Based on the benchmark results (RTX 3090, N=3 classes, `detection_only=True`):

| Setup | Per-image | 4-image batch | Total | Speedup |
|---|---|---|---|---|
| Current (sequential set_image→predict) | 196 ms | 4 × 196 ms | 784 ms | 1.0× |
| Multi-image backbone + sequential predict | ~45 ms backbone + 4 × 65 ms predict | — | ~305 ms | ~2.6× |
| Full B\*N batching (this plan) | ~45 ms backbone + ~75 ms predict(B\*N=12) | — | ~120 ms | **~6.5×** |

Backbone at B=4 should scale sub-linearly (~45 ms vs 131 ms ×1 — GPU parallelism). Encoder/decoder at bs=12 vs bs=3 also sub-linear (~75 ms vs 65 ms — mostly kernel launch overhead).

## Verification

### Test 1: Parity (single-image API vs batch API, same results)

```python
# Run same images through both APIs, assert identical results
for img in images:
    single_result = predictor.predict(predictor.set_image(img))

batch_results = predictor.predict_images(images)

for single, batched in zip(single_results, batch_results):
    assert torch.allclose(single["boxes"], batched["boxes"], atol=1e-3)
    assert torch.allclose(single["scores"], batched["scores"], atol=1e-3)
    assert single["class_names"] == batched["class_names"]
```

Run on all 8 fl_pj_sample images. Acceptance: boxes match within atol=1e-3 (fp16 accumulation may cause tiny differences), scores within atol=1e-3, identical class names.

### Test 2: Latency benchmark

Run `predict_images([img]*B)` for B=1,2,4,8 with N=3 classes, `detection_only=True`. Report backbone_ms and predict_ms. Confirm:
- B=4 total < 2× single-image total (i.e., >2× throughput improvement).

### Test 3: Batch size 1 regression

`predict_images([single_img])` must produce identical results to `predict_image(single_img)` — edge case where B=1 should degenerate cleanly.

### Test 4: Variable image sizes

Pass images of different original sizes (all get resized to 1008×1008 internally, but original_sizes differ). Confirm output boxes are correctly scaled per-image to their respective original (h, w).

### Test 5: With masks (detection_only=False)

Run with `detection_only=False` on B=2 images. Confirm masks are at correct original resolution per image and seg head indexes the right backbone FPN features via `image_ids`.

## Critical files

| File | Change |
|---|---|
| `scratchpad/DART/` | **No changes** — upstream stays clean |
| **New** [data_miner/auto_annotation_v3/dart_batch.py](data_miner/auto_annotation_v3/dart_batch.py) | `Sam3MultiClassPredictorBatch` subclass with `set_images()`, `predict_batch()`, `predict_images()`, `_forward_batched_multi_image()` |
| **New** [data_miner/auto_annotation_v3/tests/test_dart_image_batching.py](data_miner/auto_annotation_v3/tests/test_dart_image_batching.py) | Parity, latency, edge-case tests |
| [data_miner/auto_annotation_v3/tests/benchmark_dart_modes.py](data_miner/auto_annotation_v3/tests/benchmark_dart_modes.py) | Add batch-mode benchmarks alongside existing M1-M8 |
