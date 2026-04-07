# Auto Annotation — Critical Review

## What's Done Well

**Clean separation of concerns.** The registry/adapter/stage layering is sound. Config-driven stage ordering, OmegaConf + Pydantic validation, and the decorator-based registry pattern make the system genuinely pluggable without subclass gymnastics.

**Immutable-ish state flow.** `PipelineState` passed through stages with `model_copy()` for mutations is a good pattern. The `extra="forbid"` on Pydantic models catches config typos early.

**Lazy model loading.** The `_ensure_loaded()` pattern in adapters avoids loading multi-GB models until first use—important when not all adapters are exercised.

---

## Design-Level Issues

### 1. The consensus stage has a flawed quality heuristic

`consensus.py` `_quality()` computes quality purely from bbox geometry (area, aspect ratio, edge proximity). This has **nothing to do with detection quality**—a perfectly detected small object near the image edge will score poorly. The score from the actual model is completely ignored. This means the auto-accept/reject thresholds in `LimitConfig` gate on a geometric proxy, not on model confidence. The `_uncertainty()` function compounds this by blending the geometric quality score with agreement count using arbitrary fixed weights (0.6/0.4).

**Impact:** The consensus stage will systematically reject valid small or edge-adjacent detections and accept large, centered, low-confidence ones.

### 2. Proposals are truncated before consensus, not after

In `proposal.py`, `class_candidates[:per_class_limit]` is applied **per-class across all models and prompt variants combined**, before any quality-based filtering. If Falcon returns 12 mediocre candidates first, GroundingDINO's better candidates are silently dropped. The truncation should happen after scoring/ranking, or at least interleave model results.

### 3. Consensus clustering is greedy and order-dependent

`consensus.py` uses a greedy sequential clustering: pop first element, absorb anything within IoU threshold, repeat. This means **cluster membership depends on the order candidates appear in the list**, which is determined by which model/expression ran first. Two identical runs with models in different order can produce different clusters. A proper approach would use agglomerative clustering or a union-find structure.

### 4. The retry loop is hardcoded to the pipeline runner, not the stage system

`pipeline.py` has a `while` loop that manually searches for the refinement and verification stages by `kind` string, then re-runs them. This breaks the abstraction—the pipeline runner "knows" about specific stage types. If someone adds a second refinement variant or renames the kind, the retry loop silently stops working. The retry logic should either be a stage-level concern or a first-class pipeline concept, not an ad-hoc loop in `run_image`.

### 5. Single-model verification creates a bottleneck with no fallback

`verification.py` iterates models but `break`s after the first one that supports verification. If that model (Qwen) is down or returns garbage, every flagged candidate falls through to the `flagged.append` path with no review decision, meaning they'll hit escalation as if the verifier never ran. There's no retry, no timeout handling at the stage level, and no distinction between "verifier said uncertain" and "verifier crashed."

---

## Implementation Issues

### 6. `urllib.request` for HTTP calls — no retries, no connection pooling

`qwen.py` uses raw `urllib.request` for the verification API. No retry logic, no connection reuse, no backoff. A single transient error (503, timeout, connection reset) fails the candidate permanently. For a pipeline that may process thousands of images, this will cause silent data loss. Use `requests.Session` or `httpx` with retry middleware.

### 7. Base64-encoded full images in every verification request

`qwen.py` sends **two** base64-encoded images (full overlay + crop) per candidate per verification call. For a 1920×1080 image, that's ~5–10 MB of base64 per request. With 12 candidates per class across multiple classes, this will saturate the network and the LLM endpoint. Consider resizing the overlay, or sending only the crop with textual bounding box coordinates.

### 8. `_resolve_device` is duplicated across every adapter

The exact same function appears in `falcon.py`, `grounding_dino.py`, and `sam.py`. Should live in the base adapter or utils.

### 9. Relabel in verification can create orphan class names

`verification.py`: when the verifier says `relabel_to: "pallet_truck"`, the code blindly sets both `label` and `class_name` to whatever string the LLM returns. If that string doesn't match any configured class name, `save_result` will hit a `KeyError` on the `label_map` lookup in `utils.py`. There's no validation that relabel targets exist in the config's class list.

### 10. `save_result` loses data on relabeled candidates

`utils.py`: `label_map[candidate.class_name]` will crash for any accepted candidate whose `class_name` was relabeled to a value not in the original `class_names` list (see issue 9). Even if the relabel target is a valid synonym, it won't be in the map since the map is built from `config.classes[*].name` only.

### 11. No error handling in the stage loop

`pipeline.py`: if any stage raises an exception (model OOM, corrupt image region, network error), the entire image is lost with no partial result saved. For a batch pipeline, this means one bad candidate in one stage can lose all work done on that image. At minimum, `run_image` should catch per-image errors and return a partial/failed result.

### 12. The `Candidate` model uses `extra="allow"` while everything else uses `extra="forbid"`

`contracts.py`: `Candidate` allows arbitrary extra fields. This means typos in field names when constructing candidates (e.g., `sourec_model`) will silently create extra fields instead of raising validation errors. This is inconsistent with the defensive `extra="forbid"` used everywhere else.

### 13. No logging anywhere in the pipeline

None of the adapters, stages, or the pipeline runner use any logger. For a multi-model pipeline that processes batches of images, the complete absence of logging means debugging production issues requires adding print statements. The parent `data_miner` project has a `get_logger` convention that this package ignores entirely.

### 14. Verification prompt is fragile and not schema-enforced

`prompts.py`: the prompt asks the LLM to return raw JSON with a specific schema, but there's no JSON-mode enforcement (`response_format`), no schema passed to the API, and the `_extract_json` fallback in the Qwen adapter silently degrades to an "escalate" decision if parsing fails. The prompt itself includes the JSON template as a flat string, making it easy for the LLM to hallucinate extra fields or skip required ones.

### 15. `PipelineState` carries a full PIL Image — not serializable

`contracts.py`: `PipelineState` holds `image: Image.Image` with `arbitrary_types_allowed=True`. This means the state can't be serialized for checkpointing, debugging dumps, or distributed processing. If any future stage tries to `model_dump()` the state, it will fail.

---

## Structural Gaps

### 16. No batching support

The pipeline processes one image at a time. The Falcon adapter internally uses `BatchInferenceEngine` but is called with a batch of 1. For GPU-bound models, this leaves significant throughput on the table. The pipeline architecture doesn't have a concept of batch processing—each adapter call is per-image, per-class, per-expression.

### 17. No progress reporting or callback hooks

For large runs, there's no way to monitor progress, get intermediate counts, or hook into lifecycle events (image started, stage completed, image failed). The CLI just loops silently.

### 18. `save_result` is a standalone function, not integrated with the pipeline

Output writing is done in the CLI loop (`cli.py`), not in the pipeline. If someone uses `AutoAnnotationPipeline` as a library (which the `__init__.py` exports suggest is intended), they have to manually call `save_result` with the correct arguments. The output config is in the pipeline config but isn't used by the pipeline itself.

---

## Summary

| Priority | Issue | Risk |
|----------|-------|------|
| **High** | Consensus quality heuristic ignores model scores | Systematic mis-classification of detections |
| **High** | Relabel creates invalid class names → crash | Runtime crash on accepted candidates |
| **High** | No error handling per image or per candidate | Single failure loses entire image |
| **High** | No HTTP retries for verification endpoint | Silent data loss at scale |
| **Medium** | Order-dependent greedy clustering | Non-deterministic results |
| **Medium** | Proposal truncation before ranking | Good candidates silently dropped |
| **Medium** | Base64 full images per verification call | Throughput bottleneck |
| **Medium** | No logging | Undebuggable in production |
| **Low** | No batching | Suboptimal GPU utilization |
| **Low** | Non-serializable state | Blocks future distributed/checkpoint work |

---

## Response To Review

This review is substantially correct. The main correctness risks are real and should be fixed before treating the package as a serious annotation pipeline.

The most important confirmed issues are:

1. `consensus.py` uses geometry as the main quality signal instead of model evidence.
2. `proposal.py` truncates before any meaningful cross-model comparison.
3. `verification.py` allows raw LLM relabel strings to become `class_name`.
4. `pipeline.py` has no per-image failure envelope and no per-candidate isolation.
5. `qwen.py` has no retry or transport error classification.

Some review points need narrower framing:

1. The relabel issue does not currently crash in `save_result`; accepted candidates with invalid class names are silently dropped from YOLO output because of the `if candidate.class_name in label_map` filter.
2. `urllib` is not the core defect by itself. The defect is lack of retry, backoff, and failure classification around the verification transport.
3. Non-serializable `PipelineState` is a real architectural limitation, but not a first-order blocker for v1 correctness.

---

## Agreed Priorities

### Top Priority

1. Fix consensus scoring.
2. Fix proposal truncation.
3. Fix relabel canonicalization.
4. Add per-image and per-candidate failure handling.
5. Add verification transport resilience.

### Move Up From Original Priority

1. Add minimal structured logging while changing scoring and retry behavior.
2. Tighten `Candidate` from `extra="allow"` to `extra="forbid"`.

### Lower Priority After Correctness

1. Deterministic clustering cleanup.
2. Retry-controller refactor.
3. Payload-size optimization for verifier images.
4. State serialization improvements.
5. Batching and lifecycle hooks.

---

## Concrete Remediation Plan

### Phase 1 — Correctness Core

#### 1. Tighten candidate validation

Files:
- `contracts.py`

Changes:
- Change `Candidate` to `extra="forbid"`.
- Keep flexible per-model data only inside `metadata`.

Acceptance criteria:
- Typos in top-level candidate fields fail validation immediately.
- Existing adapters still validate without hidden extra fields.

#### 2. Add minimal structured logging

Files:
- `pipeline.py`
- `stages/proposal.py`
- `stages/consensus.py`
- `stages/refinement.py`
- `stages/verification.py`
- `stages/escalation.py`

Changes:
- Log image start and end.
- Log proposal counts by class and model.
- Log consensus route counts.
- Log retry rounds.
- Log verification transport failures and schema failures.
- Log escalations.

Acceptance criteria:
- A single image run provides enough logs to understand why candidates were accepted, flagged, rejected, or escalated.

#### 3. Fix proposal truncation

Files:
- `stages/proposal.py`

Changes:
- Stop slicing `class_candidates[:per_class_limit]` on flat insertion order.
- Group proposals by source model and expression.
- For v1, interleave proposals by source and defer strict truncation until after interleaving.

Notes:
- Do not invent a pre-consensus ranking function that pretends Falcon has calibrated confidence.
- Falcon should be treated as binary support at this stage.

Acceptance criteria:
- Later models and prompt variants are not starved by earlier ones.
- Proposal order is deterministic for identical input.

#### 4. Redesign consensus scoring

Files:
- `stages/consensus.py`

Changes:
- Stop using geometry as the primary quality score.
- Treat Falcon as semantic support, not calibrated confidence.
- Use GroundingDINO score as actual confidence when available.
- Use agreement count and source diversity as support signals.
- Use geometry as sanity veto or penalty only.
- Incorporate SAM refinement delta once refinement is available.

Recommended v1 decision model:
- `hard_reject` only for obviously invalid geometry.
- `accept` when multi-model support exists and at least one scored proposer is strong.
- `flag` otherwise.

Acceptance criteria:
- Valid small or edge detections are not systematically downgraded.
- Large centered low-confidence detections are not automatically upgraded.

#### 5. Make clustering deterministic

Files:
- `stages/consensus.py`

Changes:
- Replace greedy pop-and-absorb clustering with union-find or connected-components over IoU edges.

Acceptance criteria:
- Cluster membership no longer depends on proposal insertion order.

### Phase 2 — Verification And Output Correctness

#### 6. Canonicalize relabels

Files:
- `stages/verification.py`
- `utils.py`

Changes:
- Build a canonical alias map from `ClassPackConfig.names()`.
- Resolve `relabel_to` through that alias map.
- Keep `class_name` canonical.
- Allow `label` to remain display-oriented if needed.
- If relabel cannot be resolved, convert to `escalate`.

Acceptance criteria:
- Synonym relabels such as `lift truck` map to `forklift`.
- Unknown relabels never become accepted canonical class names.
- Accepted sidecars and YOLO labels remain consistent.

#### 7. Make export failures explicit

Files:
- `utils.py`

Changes:
- Do not silently drop accepted candidates that cannot be exported.
- Record structured export warnings or escalate such candidates.

Acceptance criteria:
- No silent disagreement between accepted sidecar results and YOLO output.

#### 8. Tighten verifier outcome handling

Files:
- `adapters/qwen.py`

Changes:
- Distinguish transport failure, invalid JSON, schema failure, and model uncertainty.
- Encode those separately in the fallback `ReviewDecision` or `metadata`.

Acceptance criteria:
- Logs and sidecars reveal whether the issue was the verifier server, the parser, or the model.

### Phase 3 — Failure Isolation

#### 9. Add per-candidate failure handling

Files:
- `stages/refinement.py`
- `stages/verification.py`

Changes:
- Wrap per-candidate adapter calls.
- If one candidate fails, preserve the rest of the image state.
- Keep the failed candidate in pre-refinement or pre-verification state and mark it for escalation or retry.

Acceptance criteria:
- One failing candidate does not lose the other candidates in the same image.

#### 10. Add per-image failure envelopes

Files:
- `pipeline.py`

Changes:
- Catch stage-level exceptions.
- Return partial image results where possible.
- Include failed stage and reason in sidecar output.

Acceptance criteria:
- One image can fail without crashing the full run.

#### 11. Add verification retries

Files:
- `adapters/qwen.py`

Changes:
- Catch `URLError`, `HTTPError`, timeout-related exceptions, and connection failures.
- Retry bounded times on transient failures.
- Escalate with explicit failure metadata after retry exhaustion.

Notes:
- In this environment, local vLLM failures and connection refused are likely more important than rate limits.

Acceptance criteria:
- Transient verifier outages do not immediately fail candidates.

### Phase 4 — Architectural Cleanup

#### 12. Refactor retry control

Files:
- `pipeline.py`

Changes:
- Move away from hard-coded `verification` and `refinement` stage inspection.
- Introduce a cleaner controller or stage-directed next-action model.

Acceptance criteria:
- Retry behavior no longer depends on magic stage kinds.

#### 13. Deduplicate shared adapter helpers

Files:
- `adapters/falcon.py`
- `adapters/grounding_dino.py`
- `adapters/sam.py`

Changes:
- Move repeated `_resolve_device` logic into a shared utility or the base adapter.

Acceptance criteria:
- No repeated device resolution helpers remain.

#### 14. Revisit verifier payload size

Files:
- `adapters/qwen.py`
- `utils.py`

Changes:
- Resize context images or reduce duplicated full-frame evidence once correctness is stable.

Acceptance criteria:
- Transport size becomes measured and tunable rather than fixed.

---

## Recommended Edit Order

1. `contracts.py` strictness and minimal logging.
2. `proposal.py` truncation fix.
3. `consensus.py` scoring and deterministic clustering.
4. `verification.py` canonical relabel mapping.
5. `utils.py` export consistency.
6. `qwen.py` transport resilience and failure classification.
7. `refinement.py` and `verification.py` candidate-level isolation.
8. `pipeline.py` image-level failure envelopes.
9. `pipeline.py` retry-controller cleanup.

---

## Acceptance Checklist

1. Same input produces deterministic clusters and routes.
2. Valid small or edge detections are no longer systematically penalized.
3. Accepted detections always map to canonical configured class names.
4. Accepted sidecars and YOLO outputs stay aligned.
5. One candidate failure does not lose the rest of the image.
6. One image failure does not crash the whole batch.
7. Verifier transport failures are retried or escalated explicitly.
8. Logs are sufficient to debug routing behavior without print statements.
