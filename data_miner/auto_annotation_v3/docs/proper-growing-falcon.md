# Proper Agentic Auto-Annotation Plan: Falcon + SAM + Qwen

## Context

**Goal**: Build a production-grade agentic auto-annotation system where Qwen 3.5 VL orchestrates Falcon Perception and SAM 3.1 to maximize annotation accuracy, while staying measurable, debuggable, and cost-aware.

**Current state**: The repo already has the raw components needed for this:
- Falcon open-vocabulary grounding in `data_miner/models/falcon_perception.py`
- SAM segmentation and box refinement in `data_miner/models/sam.py`
- Qwen-based classification and rating baselines in `data_miner/models/qwen3_vl*.py`
- An OpenAI-compatible vLLM validator path in `annotation-validator/validator.py`
- A strong upstream grounded agent template in `output/temp/Falcon-Perception-upstream/demo/agent/`

**Problem with the current flow**: Detection and validation are still mostly separated. The repo can detect, and it can judge, but it does not yet run a controlled reasoning loop that decides when to re-ground, when to refine, and when to stop.

**Design objective**: Preserve the good parts of `silly-growing-falcon.md` - detector consensus, agentic review, and mode separation - while making the plan more explicit about tool contracts, uncertainty routing, provenance, and evaluation.

---

## Research-Informed Position

### What should drive the architecture

1. **Falcon should propose first**.
   Falcon is the strongest candidate in this repo for open-vocabulary semantic grounding, spatial language, and prompt-driven region discovery.

2. **SAM should refine, not lead**.
   SAM is strong when given a good box, mask, or short concept phrase. It is weaker as the primary semantic selector for arbitrary classes.

3. **Qwen should orchestrate and verify**.
   Qwen is best used to arbitrate ambiguity, generate better follow-up actions, and validate results with structured outputs. It should not be the default first-stage detector.

4. **Agentic loops should be selective**.
   Full agentic loops are expensive. The default production path should call the VLM only on the uncertainty bucket.

5. **Quantitative gates should outrank free-form opinions**.
   If a detection fails clear geometric and consistency checks, the pipeline should reject or route it before asking the VLM to improvise.

### Core conclusion

The best default architecture is:

**Falcon propose -> optional consensus check -> SAM refine -> Qwen verify -> accept, relabel, reject, or retry**

This is close to the spirit of `silly-growing-falcon.md`, but the main improvement is that the loop is treated as an explicit state machine with typed tools and measurable stopping rules instead of a mainly prompt-centric retry system.

---

## Architecture

### Mode A: Hybrid Consensus + Targeted Agentic Review

This should be the default mode.

```
Image
  -> Falcon proposal
  -> Optional secondary detectors for agreement scoring
  -> Consensus and uncertainty routing
  -> Auto-accept clear detections
  -> Agentic review only for flagged detections
  -> Final annotation + provenance
```

**How it works**:
- Falcon proposes candidates for target classes.
- Optional detectors such as SAM text prompting, GroundingDINO, OWLv2, or direct Qwen detection can contribute agreement signals.
- Detections are clustered by IoU and label compatibility.
- High-confidence, high-agreement detections pass without VLM review.
- Only conflicts, weak single-source detections, or poor-quality boxes go to the agent loop.

**Why this should be the default**:
- Lowest VLM cost.
- Best chance of production throughput.
- Keeps agentic reasoning focused on cases where it actually adds value.

### Mode B: Full Agentic Loop

This should be the experimental mode.

```
Image + target classes
  -> Qwen plans which class or expression to try first
  -> Falcon proposes
  -> SAM refines
  -> Qwen verifies
  -> Loop until accept, reject, or budget exhausted
```

**How it works**:
- Qwen actively chooses the next action.
- Falcon and SAM are exposed as tools.
- Every class or ambiguous scene can trigger multiple agent turns.

**When to use it**:
- Small datasets.
- Rare classes.
- Compositional or relation-heavy prompts.
- Cases where annotation quality matters more than runtime.

**Why it should not be the default**:
- More expensive.
- Harder to debug at scale.
- More chances for VLM drift or unnecessary retries.

---

## State Machines

### Hybrid Mode State Machine

```
PROPOSE -> SCORE -> ACCEPT
                -> REVIEW -> REFINE -> RECHECK -> DECIDE
```

**States**:
- `PROPOSE`: Falcon generates candidate detections.
- `SCORE`: Consensus and quality gates assign `accepted`, `flagged`, or `rejected`.
- `REVIEW`: Qwen inspects evidence for a flagged candidate.
- `REFINE`: The agent chooses a follow-up action such as Falcon re-grounding or SAM refinement.
- `RECHECK`: Metrics are recomputed.
- `DECIDE`: Accept, relabel, reject, or escalate.

### Full Agentic Mode State Machine

```
PLAN -> PROPOSE -> REFINE -> VERIFY -> DECIDE
```

**States**:
- `PLAN`: Qwen chooses the next class expression or strategy.
- `PROPOSE`: Falcon proposes candidates.
- `REFINE`: SAM or Falcon follow-up tool calls improve a selected candidate.
- `VERIFY`: Qwen verifies class identity and box quality using structured evidence.
- `DECIDE`: Accept current result, retry, or stop.

**Budget rules**:
- Cap tool calls per image.
- Cap retries per candidate.
- Prefer early acceptance once hard gates are met.

---

## Tool Contracts

### Tool 1: `detect_candidates`

**Purpose**: Generate candidate detections.

**Inputs**:
```json
{
  "image_id": "string",
  "class_name": "string",
  "expression": "string",
  "preferred_model": "falcon|sam3|grounding_dino|owlv2|qwen",
  "threshold": 0.3,
  "max_candidates": 20
}
```

**Outputs**:
```json
{
  "tool_status": "ok|error",
  "candidate_ids": [1, 2],
  "candidates": [
    {
      "candidate_id": 1,
      "label": "door",
      "source_model": "falcon",
      "bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.9],
      "mask_ref": "optional",
      "raw_score": 0.91,
      "metadata": {
        "coord_bbox": [0.11, 0.22, 0.41, 0.88],
        "mask_bbox": [0.10, 0.20, 0.40, 0.90],
        "bbox_iou": 0.87,
        "prompt_used": "left entrance door"
      }
    }
  ]
}
```

### Tool 2: `refine_candidate`

**Purpose**: Improve a selected candidate.

**Inputs**:
```json
{
  "image_id": "string",
  "candidate_id": 1,
  "refinement_mode": "sam_from_box|sam_from_mask|falcon_reground|broaden_expression|narrow_expression",
  "refinement_prompt": "string",
  "input_bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.9]
}
```

**Outputs**:
```json
{
  "tool_status": "ok|error",
  "refined_candidate": {
    "candidate_id": 1,
    "bbox_xyxy_norm": [0.12, 0.2, 0.39, 0.89],
    "mask_ref": "optional",
    "quality_gate_score": 0.92
  },
  "delta_metrics": {
    "old_vs_new_iou": 0.83,
    "area_change_ratio": 0.91,
    "edge_contact_change": -1
  }
}
```

### Tool 3: `get_candidate_views`

**Purpose**: Produce evidence images for verification.

**Inputs**:
```json
{
  "image_id": "string",
  "candidate_id": 1,
  "requested_views": ["full_image_overlay", "tight_crop", "padded_crop", "mask_isolated_crop"]
}
```

**Outputs**:
```json
{
  "tool_status": "ok|error",
  "views": {
    "full_image_overlay": "ref",
    "tight_crop": "ref",
    "padded_crop": "ref",
    "mask_isolated_crop": "ref"
  },
  "derived_metrics": {
    "crop_fill_ratio": 0.74,
    "touches_border": false,
    "truncation_risk": "low"
  }
}
```

### Tool 4: `verify_candidate`

**Purpose**: Structured semantic and box-quality verification.

**Inputs**:
```json
{
  "image_id": "string",
  "candidate_id": 1,
  "expected_class": "door",
  "class_pack_ref": "door_v1",
  "view_refs": ["tight_crop_ref", "full_overlay_ref"],
  "candidate_metadata": {
    "source_models": ["falcon", "sam3"],
    "quality_gate_score": 0.84,
    "uncertainty_score": 0.38
  }
}
```

**Outputs**:
```json
{
  "schema_version": "1.0",
  "candidate_id": 1,
  "predicted_class": "door",
  "semantic_match": "yes",
  "object_complete": "yes",
  "bbox_tight": "tight",
  "wrong_instance": false,
  "likely_confusions": [],
  "needs_refinement": false,
  "recommended_action": "accept",
  "confidence_band": "high",
  "rationale_short": "Full doorway visible with frame and seam."
}
```

### Tool 5: `compare_candidates`

**Purpose**: Resolve confusion between nearby candidates.

**Inputs**:
```json
{
  "image_id": "string",
  "candidate_ids": [1, 2],
  "relation_query": "which is leftmost"
}
```

### Tool 6: `finalize_annotations`

**Purpose**: Save accepted, rejected, and review-required outputs with provenance.

---

## Internal Schemas

### Candidate Schema

```json
{
  "candidate_id": 1,
  "label": "door",
  "source_model": "falcon",
  "prompt_used": "left entrance door",
  "bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.9],
  "mask_ref": "optional",
  "raw_score": 0.91,
  "calibrated_score": 0.78,
  "quality_gate_score": 0.84,
  "uncertainty_score": 0.38,
  "review_required": true,
  "metadata": {}
}
```

### Verification Schema

```json
{
  "schema_version": "1.0",
  "image_id": "string",
  "candidate_id": 1,
  "expected_class": "door",
  "predicted_class": "door",
  "semantic_match": "yes|no|uncertain",
  "object_complete": "yes|partial|no|uncertain",
  "bbox_tight": "tight|loose|too_small|uncertain",
  "wrong_instance": false,
  "truncation_present": false,
  "adjacent_object_intrusion": "low",
  "likely_confusions": [],
  "needs_refinement": false,
  "recommended_action": "accept|relabel|refine|reject|escalate",
  "confidence_band": "high|medium|low",
  "rationale_short": "string"
}
```

### Provenance Sidecar

Every output image should have a sidecar that records:
- pipeline mode
- detector config
- consensus summary
- agent decisions
- accepted and rejected candidates
- model versions
- tool-call counts

This is one of the main improvements over the sillier draft: the output should be replayable and auditable.

---

## Implementation Plan

### Step 1: Add a detector adapter layer

**Create** `data_miner/models/annotation_adapters.py`

Purpose:
- Normalize outputs from Falcon, SAM, GroundingDINO, OWLv2, and optional Qwen detection.
- Preserve model-specific metadata.

### Step 2: Build a consensus router

**Create** `data_miner/models/consensus_router.py`

Purpose:
- Cluster detections by IoU and label compatibility.
- Compute `accepted`, `flagged`, and `rejected` sets.
- Assign uncertainty and quality scores.

### Step 3: Build the agentic annotator

**Create** `data_miner/models/agentic_annotator.py`

Purpose:
- Implement hybrid and full-agentic state machines.
- Expose the typed tools.
- Route follow-up actions through Falcon and SAM.

### Step 4: Add class packs

**Create** `data_miner/config/class_packs/`

Each class pack should contain:
- canonical class name
- synonyms
- hard negatives
- must-have cues
- common confusions
- completeness rules
- prompt variants

### Step 5: Add a CLI runner

**Create** `scripts/run_agentic_annotate.py`

Modes:
- `consensus`
- `hybrid`
- `full`

### Step 6: Extend config

**Modify**:
- `data_miner/config/config.py`
- `data_miner/config/constants.py`

Add:
- annotation mode
- detector selection
- iteration budgets
- VLM serving config
- acceptance thresholds

### Step 7: Add evaluation harness

**Create** `scripts/eval_agentic_annotate.py`

Compare:
- Falcon only
- Falcon + SAM
- Falcon + SAM + single-pass validator
- Hybrid agentic mode
- Full agentic mode
- Direct Qwen detection baseline

---

## Quality Gates

The following checks should run before or alongside VLM review:

1. Minimum area.
2. Maximum area.
3. Minimum width and height.
4. Border contact and truncation risk.
5. Duplicate overlap.
6. Falcon mask-box disagreement.
7. Cross-model label conflict.
8. SAM refinement improvement or regression.

These should feed a single `quality_gate_score`, but the raw checks should also be preserved.

---

## Decision Graph And Thresholds

This is the concrete routing logic for the method you proposed:

**Falcon + GroundingDINO consensus -> SAM refinement -> Qwen corrective prompting if needed -> structured re-judge -> accept or human review**

### Recommended role ordering

1. **Falcon**: primary semantic proposer.
2. **GroundingDINO**: secondary semantic proposer and cross-check.
3. **SAM**: geometric refiner attached to an existing candidate or cluster.
4. **Qwen 3.5 VL**: action selector and structured verifier for unresolved cases.
5. **Human review**: final sink for persistent ambiguity.

### High-level flow

```
Image
  -> Falcon proposals
  -> GroundingDINO proposals
  -> Cluster by IoU + label compatibility
  -> SAM refines viable clusters
  -> Compute quality and uncertainty
  -> If clear: accept
  -> If semantic issue: Qwen rewrites Falcon prompt and retry once
  -> If geometry issue: SAM refinement retry once
  -> Re-judge with structured output
  -> Accept, relabel, reject, or send to human review
```

### Step-by-step routing

#### Step 1: Generate proposals

- Run Falcon with class-pack prompt variants for the target class.
- Run GroundingDINO with the canonical class label and, if needed, a synonym-expanded label set.
- Normalize all detections into a common candidate schema.

**Starting thresholds**:
- Falcon proposal threshold: `0.15` to `0.25` for recall-oriented proposal generation.
- GroundingDINO proposal threshold: `0.20` to `0.30`.
- Per-class max proposals before routing: `10` to `20`.

#### Step 2: Build consensus clusters

Cluster detections if:
- IoU `>= 0.5`
- and labels match exactly or through the class pack synonym map.

Each cluster should record:
- member detections
- source models
- fused box
- label agreement state
- per-source raw scores

**Cluster types**:
1. `strong_agreement`
  Falcon and GroundingDINO both support the same object.
2. `weak_single_source`
  Only one proposer found it.
3. `label_conflict`
  Two proposers overlap but disagree semantically.
4. `duplicate_conflict`
  Multiple clusters compete for the same instance.

#### Step 3: Run SAM refinement on viable clusters

SAM should refine:
- every `strong_agreement` cluster that survives basic size checks
- every `weak_single_source` cluster that is still plausible
- every cluster where box quality is uncertain

SAM should usually receive:
- the fused box from the cluster
- or the Falcon box if Falcon is the only proposer

SAM should output:
- refined mask
- refined box
- refinement diagnostics such as area change and border contact change

#### Step 4: Score quality and uncertainty

Compute a `quality_gate_score` from raw checks such as:
- min area
- max area
- min width and height
- aspect ratio sanity
- border contact
- duplicate overlap
- Falcon mask-vs-coord disagreement
- SAM refinement improvement

Compute an `uncertainty_score` from signals such as:
- cross-model disagreement
- low fused confidence
- label conflict
- Falcon mask/coord mismatch
- SAM making the box worse rather than better

**Suggested initial bands**:
- `quality_gate_score >= 0.80` and `uncertainty_score <= 0.25`: clear
- `quality_gate_score 0.50 to 0.79` or `uncertainty_score 0.26 to 0.60`: ambiguous
- `quality_gate_score < 0.50` or `uncertainty_score > 0.60`: poor or risky

These are starting thresholds, not final tuned values.

### Routing decisions

#### Route A: Auto-accept

Accept immediately if all are true:
- cluster type is `strong_agreement`
- post-SAM `quality_gate_score >= 0.80`
- `uncertainty_score <= 0.25`
- no hard-negative rule is triggered from the class pack

This is the main production path.

#### Route B: Qwen -> Falcon retry

Use this when the issue is **semantic**, not geometric.

Trigger if any are true:
- `label_conflict` between Falcon and GroundingDINO
- cluster is `weak_single_source` but geometry is acceptable
- hard-negative confusion is likely
- the object seems to be the right region category but the description may be too broad or too narrow

Qwen should then produce a structured retry instruction such as:

```json
{
  "recommended_action": "retry_falcon",
  "new_expression": "low-profile pallet jack without mast near floor",
  "reason": "Likely confusion with forklift because original prompt was too broad."
}
```

**Rule**:
- allow at most **one semantic retry** per candidate in v1

#### Route C: SAM-only retry

Use this when the issue is **geometry**, not semantics.

Trigger if all are roughly true:
- semantic label is probably correct
- bounding box is loose, too small, or truncated
- Falcon box and SAM-refined box differ meaningfully
- no strong label conflict remains

Practical triggers:
- Falcon mask-vs-coord IoU `< 0.75`
- border contact increases risk
- SAM can refine from a box or local crop

Qwen is optional here. If the failure is purely geometric, do not ask Qwen to solve it with language unless the geometry issue appears to come from wrong-instance selection.

**Rule**:
- allow at most **one geometry retry** per candidate in v1

#### Route D: Structured Qwen verification

After any retry, or for unresolved ambiguous cases, call Qwen with a minimal verifier schema.

For v1, keep it small:

```json
{
  "semantic_match": "yes|no|uncertain",
  "bbox_tight": "tight|loose|too_small|uncertain",
  "recommended_action": "accept|relabel|refine|reject|escalate",
  "confidence_band": "high|medium|low",
  "rationale_short": "string"
}
```

This is enough to drive the loop without overfitting to a large schema too early.

#### Route E: Human review

Escalate to human review if any are true after retries:
- semantic retry already used and conflict remains
- geometry retry already used and box quality is still poor
- Qwen returns `recommended_action = escalate`
- Qwen confidence is `low`
- two proposers still disagree after retry
- hard-negative confusion remains unresolved

Human review should be explicit, not treated as a silent reject.

#### Route F: Reject

Reject if:
- quality is clearly bad
- the candidate violates class-pack negative rules
- Qwen semantic match is `no`
- both semantic and geometry evidence point away from the target class

### Practical v1 policy

For the first implementation, use this simple control policy:

1. Falcon and GroundingDINO propose.
2. Merge into clusters.
3. SAM refines each viable cluster.
4. Auto-accept only high-quality strong-agreement clusters.
5. If semantic ambiguity remains, let Qwen rewrite the Falcon prompt once.
6. If geometry ambiguity remains, run one SAM retry.
7. Re-judge using the minimal structured verifier.
8. If uncertainty remains high, send to human review.

That gives you a clean and bounded loop instead of an open-ended agent that keeps inventing actions.

---

## Key Files

| File | Action | Purpose |
|------|--------|---------|
| `data_miner/models/annotation_adapters.py` | Create | Common detection schema across models |
| `data_miner/models/consensus_router.py` | Create | Consensus clustering and uncertainty routing |
| `data_miner/models/agentic_annotator.py` | Create | Hybrid and full agent loops |
| `scripts/run_agentic_annotate.py` | Create | Main CLI entry point |
| `scripts/eval_agentic_annotate.py` | Create | Benchmark and ablation runner |
| `data_miner/config/config.py` | Modify | Agentic config |
| `data_miner/config/constants.py` | Modify | Annotation mode enum |
| `data_miner/config/class_packs/` | Create | Per-class knowledge packs |

---

## Verification Plan

1. **Schema validation**:
   Confirm that all tools and sidecars emit schema-valid JSON.

2. **Hybrid sanity run**:
   Run on 5 images and confirm that only the uncertainty bucket hits Qwen.

3. **Ablation study**:
   Compare Falcon-only, Falcon+SAM, hybrid, and full-agentic on the same gold set.

4. **Class-pack sensitivity**:
   Measure whether class packs reduce relabel or reject errors on confusing categories.

5. **Cost check**:
   Track VLM calls per accepted annotation and per image.

---

## Risks And Mitigations

| Risk | Mitigation |
|------|-----------|
| VLM overthinks easy cases | Default to hybrid mode and auto-accept clear detections |
| Falcon false positives | Use uncertainty routing and optional detector agreement |
| SAM refinement makes boxes worse | Recompute metrics and allow refinement rollback |
| Free-form validator drift | Use rigid JSON schemas and enumerated decisions |
| Hard classes remain inconsistent | Add class packs and evaluate per class, not just globally |
| Full agentic mode becomes too slow | Keep it experimental and budgeted |

---

## Differences From `silly-growing-falcon.md`

### What the two plans share

1. Both prefer a **hybrid mode** over a full agentic loop for production.
2. Both treat **Qwen as the orchestrator** and **Falcon plus SAM as perception tools**.
3. Both favor **consensus or quality gates** before trusting the VLM.
4. Both suggest a **CLI-driven implementation** and reuse the current repo structure.

### What this plan changes

1. **Sharper role separation**.
   The original draft is correct directionally, but this plan is stricter about Falcon-first proposal and SAM-second refinement.

2. **Typed tools and schemas**.
   The original draft mentions tools, but this plan pushes much harder on formal contracts, replayability, and sidecar provenance.

3. **Class packs as a core requirement**.
   The original draft is more model-centric. This plan makes class-specific ontology and confusion rules part of the architecture.

4. **Evaluation-first rollout**.
   The original draft includes verification, but this plan is more explicit about ablations, cost checks, and proving that the agent loop is worth keeping.

5. **Fewer early file commitments**.
   The original draft names several concrete files immediately. This plan still proposes concrete files, but starts with interfaces and adapter boundaries first.

### Pros and cons of `silly-growing-falcon.md`

**Pros**:
1. Easy to read and fast to act on.
2. Strong intuition for a hybrid production mode.
3. Good emphasis on consensus and quality gates.
4. Good practical instinct that full agentic mode should be optional.

**Cons**:
1. Some tool boundaries are still a little loose.
2. It under-specifies provenance and replayability.
3. It does not emphasize class packs enough.
4. It risks turning into implementation before interface design is stable.

### Pros and cons of this plan

**Pros**:
1. More rigorous and easier to implement safely.
2. Better for long-term maintainability and offline evaluation.
3. Makes auditing and debugging first-class.
4. More explicit about why Falcon-first is the right default.

**Cons**:
1. Heavier upfront design cost.
2. More files and schemas to maintain.
3. Slower to prototype if the goal is just a quick experiment.
4. May feel over-structured for a very small dataset or one-off run.

### Recommended use of each

Use `silly-growing-falcon.md` as the exploratory draft and high-level argument.

Use this file as the implementation handoff plan.
