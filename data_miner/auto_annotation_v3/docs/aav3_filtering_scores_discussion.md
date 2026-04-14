# aa_v3 — Detection Scoring, Filtering & Dedup Discussion Log

Session notes, decisions, open items. To be continued.

---

## 1. Class registry `prompt` vs evaluation `annotation_rules` vs `refinement.class_rules`

Three different substitution points, often confused:

| Config field | Stage | Goes into | Example |
|---|---|---|---|
| `class_registry[].prompt` | **Detect** (stage 1) | Text fed to GDINO / OWLv2 | `"person"`, `"pallet jack"` |
| `evaluation_groups[].annotation_rules` | **Evaluate** (stage 2, VLM) | `{annotation_rules}` substitution in `prompts/v1/classify_*.yaml` | `forklift: "Include any load on the forks in bbox"` |
| `refinement.class_rules[].strategy` | **Refine** (stage 3, SAM) | Selects SAM refinement strategy | `forklift: {strategy: load_extension}` |

Code refs:
- Detector prompt usage: [stages/detect.py:263-275](../stages/detect.py#L263)
- VLM rule injection: [stages/evaluate.py:249-254](../stages/evaluate.py#L249)
- Refinement strategy: [configs/default.yaml:180-182](../configs/default.yaml#L180)

### Are detectors trained to follow compositional prompts like "pallet jack including any load"?

**No.** GDINO and OWLv2 are open-vocabulary *phrase grounding* models — they align noun phrases to regions, not follow compositional instructions. "including any load" is noise to them; any expansion behavior is incidental (correlation in training captions), not guaranteed.

### Decision: simplify detector prompts (option 2)

Updated [configs/default.yaml](../configs/default.yaml):
```yaml
- {id: 33, name: forklift,    tier: 3, prompt: "forklift"}        # was "forklift including any load"
- {id: 34, name: palletjack,  tier: 3, prompt: "pallet jack"}     # was "pallet jack including any load"
```

Flow becomes:
1. **Detect** — bare prompts → tight bbox on the machine
2. **Evaluate** — VLM sees `annotation_rules: "Include any load on the forks"` → flags tight bbox as `needs_expansion`
3. **Refine** — SAM `load_extension` strategy extends the bbox

`annotation_rules` stay; they're the trigger for refinement.

---

## 2. Filtering pipeline — orchestration & hyperparams

Orchestrator: [`DetectWorker.process`](../stages/detect.py#L65-L135). Seven steps between parallel detector calls and VLM routing:

| # | Step | Location | Hyperparam |
|---|---|---|---|
| 3 | Area filter | `utils.passes_area_filter` | `filtering.min_area: 0.0005`, `filtering.max_area: 0.95` |
| 3 | Aspect ratio | `utils.passes_aspect_ratio_filter` | `filtering.min_aspect_ratio: 0.1`, `filtering.max_aspect_ratio: 10.0` |
| 3 | Edge distance | `utils.passes_edge_distance_filter` | `filtering.min_edge_distance: 0.0` |
| 4 | IoU dedup | `utils.dedup_by_iou` | `filtering.iou_dedup_threshold: 0.7` |
| 5 | Per-class cap | `utils.limit_per_class` | `filtering.max_per_class: 30` |
| 6 | Cross-class suppression | `utils.apply_cross_class_rules` | `co_existence.globally_exempt`, `co_existence.confusion_pairs` |
| 7 | Agreement compute | `utils.compute_agreement` | (no hyperparam; IoU>0.5 + same class + different model) |
| 8 | Auto-accept routing | `utils.route_candidates` | `auto_accept.tiers`, `.min_model_agreement`, `.min_score` |

Stats recorded in `DetectResult.filter_stats`.

---

## 3. Score semantics per detector (research findings)

Scores across our four detectors are **NOT comparable**. Each model uses a different confidence formulation:

| Model | What score is | Calibrated? | Typical good range | Server threshold |
|---|---|---|---|---|
| **GroundingDINO** `IDEA-Research/grounding-dino-base` | sigmoid of per-token contrastive logit (box score = max token similarity) | ❌ ranker only | 0.35–0.5 | `threshold=0.25, text_threshold=0.20` |
| **OWLv2** `google/owlv2-base-patch16-ensemble` | sigmoid of CLIP cosine × learned temperature; multi-label per class | ❌ systematically low | 0.2–0.4 | `threshold=0.10` |
| **SAM3** `facebook/sam3` | sigmoid of dedicated **presence-head** logit | ✅ genuinely probabilistic | 0.5–0.9 | `threshold=0.50` |
| **Falcon-Perception** `tiiuae/Falcon-Perception` | **no score field** — autoregressive, TII-acknowledged calibration gap | — | hardcoded 1.0 (correct) | n/a |

### Known quirks
- **GDINO scores drift with prompt length** (padded tokens compete for attention) — argues for short, simple prompts (reinforces Decision #1).
- **OWLv2 `-ensemble` variant** scores lower than `-finetuned` for the same image.
- **Falcon workaround if ever needed**: extract token logprobs from `BatchInferenceEngine.generate` as proxy confidence. Not recommended unless a real need arises.

### Accuracy ranking (separate from calibration)
- **Detection accuracy**: SAM3 ≈ Falcon (top) > GDINO > OWLv2
- **Score calibration confidence**: SAM3 >> GDINO > OWLv2 > Falcon (Falcon has none at all)

These are different axes. The subagent's initial priority `[sam3, gdino, owlvit2, falcon]` was based on calibration, not accuracy — corrected to `[sam3, falcon, grounding_dino, owlvit2]` for dedup tiebreaking purposes.

Full source list in original subagent report (paper + HF model card + GitHub issue references).

---

## 4. Dedup semantics — current behavior & issues

### Current behavior ([utils.py:192-227](../utils.py#L192))

- **Per-class** (groups by `class_name` before NMS). So `forklift` vs `palletjack` bboxes overlapping at IoU=0.9 are **not** deduped here — they survive to step 6 cross-class suppression, which uses `confusion_pairs`.
- **Tiebreak by raw score only** — sorts by `score` desc and greedy-suppresses.

### Problems

1. **Falcon's score=1.0 always wins NMS tiebreaks.** If Falcon emits a loose forklift bbox and GDINO emits a tight one at IoU≥0.7, Falcon's wins, GDINO's is discarded. VLM never sees the tight version.
2. **Scores aren't comparable** so even when two non-Falcon boxes collide, the winner is unprincipled (SAM3@0.6 always beats GDINO@0.5, but SAM3 being calibrated doesn't mean its bbox is more accurate).
3. **`compute_agreement` runs AFTER dedup** ([detect.py:116](../stages/detect.py#L116)) — but dedup has already suppressed the overlapping witnesses from other models. So agreement almost always reports 0 for surviving boxes, even when multiple models originally saw the object. **Agreement as currently measured is nearly useless.**

### Example
Forklift with pallets on forks:
- GDINO: tight bbox on machine, score 0.42
- SAM3: tight bbox on machine, score 0.65
- Falcon: loose bbox including pallets, score 1.0

All three at mutual IoU ≈ 0.75. Dedup sorts by score, keeps Falcon's loose one, suppresses GDINO and SAM3. `compute_agreement` runs on Falcon-only survivor → agreement=0. Routing sees 1 candidate, agreement=0, fails `min_model_agreement=2` → sent to VLM even though 3 models actually agreed.

---

## 5. Proposed redesign — cluster-and-collapse

Replace steps 4 + 7 with a single clustering pass.

```
1. Geometric filter                  (unchanged)
2. Per-class cluster by IoU>=iou_dedup_threshold across ALL models
     → each cluster = one physical object
     → agreement = # distinct source_models in cluster
     → agreeing_models = list of those model names
3. Pick ONE representative per cluster via cascade:
     tiebreak_by: [agreement, model_priority, score]
4. Per-class cap                     (unchanged)
5. Cross-class suppression           (unchanged, uses confusion_pairs)
6. Route                             (agreement now meaningful)
```

Agreement is computed **during** clustering and attached to the survivor — `agreement=3, agreeing_models=[gdino, sam3, falcon]` carried on the single representative. `compute_agreement` as a separate step disappears.

---

## 6. Final proposed hyperparameters

### Server-side (already in `servers/serve_*.py` — keep as-is)

| Model | Current threshold | Keep |
|---|---|---|
| GDINO | `threshold=0.25, text_threshold=0.20` | ✅ |
| OWLv2 | `threshold=0.10` | ✅ |
| SAM3 | `threshold=0.50` | ✅ |
| Falcon | hardcoded `1.0` | ✅ |

Rationale: server thresholds are "mining mode" — capture broadly, filter in pipeline.

### Pipeline config (replaces current `auto_accept` / `filtering` blocks)

```yaml
auto_accept:
  tiers: [1]
  min_model_agreement: 2           # PRIMARY gate — 2+ detectors must see it
  min_score: 0.0                   # disable global gate (scores incomparable)
  per_model_score:                 # NEW — per-model auto-accept floor
    grounding_dino: 0.35
    owlvit2:        0.25
    sam3:           0.60
    falcon:         0.0            # agreement-only (no score)

filtering:
  min_area:             0.0005
  max_area:             0.95
  min_aspect_ratio:     0.1
  max_aspect_ratio:     10.0
  min_edge_distance:    0.0
  max_per_class:        30
  iou_dedup:                        # NEW — structured block, no magic strings
    threshold: 0.7
    tiebreak_by:                    # cascade — first discriminator wins
      - agreement                   # more agreeing models beats fewer
      - model_priority              # earlier in priority list beats later
      - score                       # higher raw score (last resort)
    model_priority: [sam3, falcon, grounding_dino, owlvit2]

refinement:
  class_rules:
    forklift:   {strategy: load_extension}
    palletjack: {strategy: load_extension}
  auto_accept_iou: 0.3
  reject_iou:      0.1

co_existence:
  globally_exempt: [person, head]
  confusion_pairs:
    - [forklift, palletjack]
    - [forklift, trolley]
    - [backpack, handbag]
    - [handbag, suitcase]
    - [wallet, cellphone]
    - [car, truck]
```

### Config-style note

Ordered list `tiebreak_by: [agreement, model_priority, score]` is preferred over
string flags like `"agreement_then_priority"`. Extensible (add `area`,
`aspect_compactness`, etc.), no magic strings.

---

## 7. Status — what's live vs what needs code wiring

- ✅ **Live**: detector prompts simplified; all existing `filtering.*`,
  `co_existence.*`, `refinement.*` fields.
- ⚙️ **Needs code**:
  - Extend `AutoAcceptConfig`: add `per_model_score: dict[str, float]`.
  - Extend `FilterConfig`: replace flat `iou_dedup_threshold` with nested
    `iou_dedup: {threshold, tiebreak_by, model_priority}`.
  - Rewrite `utils.dedup_by_iou` into cluster-and-collapse: produce cluster
    survivors with `agreement` and `agreeing_models` populated.
  - Remove standalone `utils.compute_agreement` call in
    [stages/detect.py:116](../stages/detect.py#L116).
  - Update `utils.route_candidates` to consult `per_model_score` per candidate.

---

## 8. Post-stage filtering gap — new `finalize` stage

### Problem

Filtering (`geometric_filter`, `dedup_by_iou`, `apply_cross_class_rules`) runs
**only** in stage 1 — [stages/detect.py:100-113](../stages/detect.py#L100).
Downstream stages mutate candidates without re-checking invariants:

- **VLM evaluate** issues `relabels: {candidate_id: new_class}`
  ([stages/evaluate.py:408](../stages/evaluate.py#L408)). A relabeled box
  may now duplicate another survivor of the new class, fall under a different
  `confusion_pairs` rule, or overflow the new class's `max_per_class` bucket.
- **SAM refine** can materially change bbox geometry (e.g. `load_extension`).
  Refined bbox may breach `max_area`, change aspect ratio, create new overlap
  with a separately-accepted candidate, or violate `min_edge_distance`.

| Change | Severity |
|---|---|
| Relabel within confusion pair (forklift→palletjack) | **Real** — two same-class bboxes can survive |
| Relabel across non-confusion classes | Minor — VLM rarely does this |
| Refine extends bbox past `max_area` | **Real** on heavily-loaded scenes |
| Refine creates overlap with separate bbox | **Real** when a nearby object was detected separately |
| Refine fails aspect-ratio / edge-distance | Minor |

### Proposed `finalize` stage (stage 4)

A dedicated stage that consolidates post-refine cleanup:

```
finalize:
  inputs  = detect.json + evaluate.json + refine.json
  steps:
    1. Build canonical candidate list:
         - apply evaluate.relabels (class_name swaps)
         - apply refine.results (use refined_bbox when accepted, else original)
         - drop evaluate.rejected candidate_ids
         - mark was_refined flag
    2. Re-run on the canonical list:
         - geometric_filter  (area / aspect / edge-distance on refined bbox)
         - dedup_by_iou      (now catches relabel collisions + refine overlaps)
         - apply_cross_class_rules  (confusion_pairs re-evaluated post-relabel)
    3. Write final outputs:
         - labels/{image_id}.txt    (YOLO)
         - traces/{image_id}.json   (with annotations[], drop reasons, dedup log)
         - review/{image_id}.json   (if any survived stage 2 but dropped here)
```

Clean properties:
- All `output_writer.write_yolo_labels` / `.write_trace` calls move out of
  individual stage workers into `finalize`. Currently split between
  `stages/detect.py::_write_auto_accepted_output` and `stages/refine.py` —
  consolidates to one place.
- Pipeline stage order becomes `detect → evaluate → refine → finalize` (and
  `STAGE_ORDER` in [checkpoint.py:28](../checkpoint.py#L28) extended).
- Finalize has no model calls, so it's CPU-cheap; can run in-process with
  refine or as a thin dedicated worker.
- Checkpoint: `finalize.json` carries the final candidate list + drop log so
  the viewer can show "dropped in finalize" as a distinct bucket.

### Contract sketch

```python
class FinalizeResult(BaseModel):
    image_id: str
    final_annotations: list[FinalAnnotation]
    dropped: list[dict]           # [{candidate_id, reason, stage: "finalize"}]
    filter_stats: dict[str, int]  # before_geometric, after_geometric, …
    stage_timing_ms: float = 0.0
```

### Open sub-questions

- Should `finalize` re-run `compute_agreement` on the post-relabel set? Probably
  not — by this point routing is done; agreement numbers are only informational.
- `max_per_class` cap at finalize: yes, relabels could push a class over the cap.
- If `refine` drops a candidate (IoU < `reject_iou`), does it fall back to the
  original bbox or get rejected entirely? Currently rejected. `finalize` should
  honour that — no resurrection.
- Viewer integration: add a "Finalize" tab between "Refine" and "Final" showing
  the drop log.

---

## 9. Refinement deep-dive — current implementation, SAM recipes, Falcon alternative

### 9.1 What `load_extension` actually does today

**Important**: `load_extension` is **config-only, not implemented**. It appears
in [configs/default.yaml:181](../configs/default.yaml#L181) and in the
`RefinementStrategy` enum ([contracts.py:38-41](../contracts.py#L38)) but
[stages/refine.py:154](../stages/refine.py#L154) hardcodes
`method="sam_point"` regardless of class. No strategy dispatch exists.

Actual refine flow:

1. **Evaluate stage** (VLM) — decides `bbox_quality=needs_expansion` for a
   candidate, emits a `RefinementInstruction` with an optional foreground
   point `(point_x, point_y)` picked from the VLM's reading of the image.
2. **Refine stage** — POSTs to SAM3 `/predict` in `refine` mode:
   ```json
   {"image_path": "…", "mode": "refine",
    "bbox": [px1,py1,px2,py2],
    "point": [point_x, point_y],
    "point_label": 1}
   ```
3. **SAM3** returns a new bbox (`pred_iou` max mask, bbox of mask pixels).
4. **Accept gate**: `iou_with_original ∈ [reject_iou=0.1, 1.0]`, else fallback
   to original bbox.

This is **Pattern 1 (box-prompted SAM + mask union-bbox)** in the taxonomy
below — the weakest pattern for "extend to attached-but-not-overlapping
load". SAM's prior is to segment the *dominant object inside the box*; it
almost never reaches outside the box corners to pull in a pallet sitting
next to the mast.

### 9.2 Standard SAM-based load-extension recipes (subagent research)

Five canonical patterns for extending a tight bbox to cover attached cargo:

| # | Pattern | Mechanics | Key failure mode |
|---|---|---|---|
| 1 | **Box-prompted + mask union-bbox** (what we do today) | Feed orig bbox → SAM → bbox of mask pixels, union with orig | SAM segments dominant object inside box; rarely extends past corners |
| 2 | **Multi-seed point prompt** | 3–9 positive seeds inside + along bottom/outside edges + 2–4 negatives on floor; pick mask by **largest area under sanity cap** (IoU-max favours the tight object) | Background leakage on cardboard-on-pallet-on-concrete |
| 3 | **Mask dilation + CC regrowth** | Box-prompt SAM → dilate mask (3–5% image diag kernel) → re-prompt as `mask_input` or re-run box → keep largest CC overlapping orig ≥ 50% | Over-growth into adjacent racks/machines |
| 4 | **Detect-both-then-union (compositional)** | Detect `forklift` AND `pallet`/`load` separately → union boxes when gap ≤ 2% image width and vertical overlap ≥ 30% | Depends on detector seeing the load (pallets under forks often missed) |
| 5 | **Class-conditional geometric extension** | SAM only refines the forklift itself; deterministically extend bbox (down+forward for palletjack, forward for counterbalance) by a calibrated ratio | Brittle to viewpoint; needs orientation estimation |

**Subagent's verdict for warehouse forklift+load:** Pattern 4 (compositional
detect + union) gated by Pattern 1 (SAM refine per component), with VLM
verification on the unioned crop as the rejector. Reason: SAM's prior doesn't
know "forks-and-pallet" is one object — forcing extension from a single box
fights the prior.

**If constrained to single-box extension** (our current architecture):
Pattern 2 with grid + bottom-edge-outside seeds, strict area-ratio cap
(≤ 3×), and VLM verification is the least-bad option.

**Rejection signals that actually work:**
- area-ratio cap (extended / original ≤ 3)
- aspect-ratio sanity (≤ 3:1 for typical forklift views)
- mask-to-box fill ratio ≥ 0.35 (low fill = SAM grabbed disconnected background)
- VLM yes/no on the crop

### 9.3 SAM3 specifics

From subagent research (confirmed against HF model card + SAM3 paper):

- SAM3's concept/text prompts are **short noun phrases only** — "forklift
  carrying pallets" is NOT its intended input surface; model segments the
  dominant noun.
- **No part-whole / associated-mask mode.** SAM3 returns per-concept masks
  independently; the "forklift with load" case pushes you toward Pattern 4
  (detect both, union) rather than true extension.
- The **presence head** helps as a sanity filter — skip union when the
  load-class query's presence < 0.5.

Net: SAM3 doesn't provide a new "extend bbox" primitive; it gives a better
open-vocab detector that makes Pattern 4 cheaper.

### 9.4 Falcon-Perception as a replacement — research verdict

**Hypothesis tested:** replace SAM-based extension with
`Falcon + descriptive prompt` like `"pallet jack with any load, boxes,
pallets, or cargo on its forks at ground level"` to natively get a single
bbox covering machine + load.

**Verdict: weakly supported. Don't replace; consider as a hybrid after a
calibration experiment.**

**Key evidence:**

1. **Framing correction.** Current `load_extension` isn't using SAM3's text
   grounding — it's box+point prompting. So the correct comparison is
   "geometric mask-extension heuristic vs Falcon descriptive prompt", not
   "SAM3 text vs Falcon text".

2. **Falcon training skews short.** 60% of its 195M training expressions are
   Level 0–1 (objects + attributes); only 40% cover spatial/relations. Long
   descriptive prompts are tail-distribution. No prompt-length ablations in
   the paper.

3. **Falcon beats SAM3 on spatial/relations** (PBench L3: 53.5 vs 31.6;
   L4: 49.1 vs 33.3) — encouraging for the spatial claim.

4. **But Falcon's calibration is worse** — MCC 0.64 vs SAM3's 0.82. TII calls
   this "the main remaining gap". For auto-annotation this is the wrong
   failure mode: Falcon with `"with any load"` will **hallucinate load
   extension on load-less machines**.

5. **No paraphrase stability data.** `"with cargo"` vs `"including cargo"`
   may flip bboxes; nobody has tested it. Prompt becomes a brittle config.

6. **GDINO isn't a stronger candidate** — documented ~11% compositional drop
   on referring expressions without REC-specific training.

**Per-scenario comparison:**

| Case | SAM3 (current) | Falcon + descriptive prompt |
|---|---|---|
| Load physically on forks | ✅ mask connectivity works | ✅ plausible |
| Load separated on floor near machine | ❌ disconnected masks | ✅ semantic reach |
| Machine with no load | ✅ no-op | ❌ calibration gap → may invent load |
| Sub-pixel tight boundary | ✅ mask-accurate | ❌ autoregressive coords, coarser |

**Sources** (all primary):
- [tiiuae/Falcon-Perception HF](https://huggingface.co/tiiuae/Falcon-Perception)
- [Falcon Perception arXiv](https://arxiv.org/abs/2603.27365)
- [TII Falcon Perception blog](https://huggingface.co/blog/tiiuae/falcon-perception)
- [facebook/sam3 HF](https://huggingface.co/facebook/sam3)
- [SAM3 paper arXiv 2511.16719](https://arxiv.org/html/2511.16719v1)
- [RefBench-PRO arXiv 2512.06276](https://arxiv.org/html/2512.06276)

### 9.5 Recommendations — concrete next steps

**Short-term (close the metadata/code gap):**
- Either implement real strategy dispatch in `refine.py` OR remove
  `RefinementStrategy.LOAD_EXTENSION` and `class_rules.*.strategy` from
  config. Don't leave dead config keys.
- Add area-ratio and aspect-ratio sanity filters to the current `sam_point`
  refine path (Pattern 1 rejection signals that work).

**Medium-term (improve load coverage):**
- Implement **Pattern 4 (detect-both-then-union)**: add `pallet`, `cargo`,
  `load` to the detection class registry; at finalize-stage union forklift +
  adjacent pallet when spatial gap ≤ 2% image width. This is the best-
  evidence path per the research.
- When SAM refine is used, adopt Pattern 2 seeds: VLM provides not just one
  `point_x, point_y` but a small set of positive + negative seeds. Requires
  extending `RefinementInstruction` schema and the SAM3 server's `refine`
  mode to accept `points: [[x,y,label], ...]`.

**Experiment-before-commit:**
- **Falcon descriptive-prompt calibration**: 200 forklift/palletjack frames,
  3 prompt paraphrases, measure IoU vs "machine+load union" GT AND the
  false-extension rate on load-less machines. Only adopt hybrid if the
  false-extension rate is low enough that VLM adjudication can catch it.

**VLM adjudication layer (cheap insurance):**
- When two candidate bboxes disagree materially (e.g. SAM-refined vs
  Falcon-descriptive vs original), pass all crops to the Qwen VLM with
  `annotation_rules` to pick the correct one. Neutralises prompt sensitivity
  and adds a final sanity gate.

### 9.6 Proposed `load_extension` flow — revised

Replaces the dead `{strategy: load_extension}` config with a real
implementation. Invoked on-demand when evaluate flags a candidate as
`needs_expansion` for a load-bearing class.

```
1. VLM (already part of evaluate stage)
     Emits: coarse bbox/region for the load + load vocab
            (e.g. "pallet", "cardboard boxes", "wooden crate")
     Do NOT ask for a single pixel — VLM point accuracy is unreliable.
     Ask for a bbox; take centroid as SAM seed if needed.

2. SAM3 point/box prompt
     Input: VLM-provided region (point or box) + positive/negative seeds
     Output: mask → axis-aligned load bbox
     Mask selection: pred_iou max, rejected if mask_area > 2.0 × orig_bbox_area

3. Presence check on LOAD bbox alone  ← KEY GATE
     Query SAM3 presence head with load vocab on the tight load crop.
     This is the presence head's trained sweet spot (focused region).
     Threshold: 0.5 (tune up to 0.6 if false-accepts appear)
     FAIL → fallback to original bbox, done.

4. Geometric merge sanity (spatial — distinct from concept check in step 3)
     gap(orig_bbox, load_bbox)   ≤ 0.02 × image_diagonal     # touching
     OR iou(orig_bbox, load_bbox) > 0                         # overlapping
     area(merged) / area(orig)   ≤ 3.0                        # hard cap
     aspect_ratio(merged)         ∈ [0.25, 4.0]                # sanity
     FAIL → fallback to original bbox, done.

5. VLM adjudication (final)
     Prompt MUST reference the annotation rule explicitly:
       "Per 'forklift bbox must include any load on the forks',
        which bbox is correct — original tight or merged?"
     Without this framing VLM drifts toward tighter/cleaner bboxes.
     - accept → refined bbox wins
     - uncertain → write to review/ queue for human review
```

### 9.7 Why presence-on-load-bbox beats presence-on-merged-crop

Earlier draft queried presence on the MERGED crop. This is strictly worse:

| Aspect | Presence on merged crop | Presence on load bbox (chosen) |
|---|---|---|
| Signal quality | Diluted (90% machine + 10% load) | Clean (load fills crop) |
| SAM3 training fit | Outside distribution | Matches presence-head training |
| Threshold tunability | Must lower to ~0.3, noisy | 0.5 works cleanly |
| Failure mode coverage | Concept AND spatial in one check | Concept only — clean |
| Early-reject on bad mask | Late (after merge compute) | Immediate (fail fast) |

Crucially: **presence and geometric merge checks answer different questions
and should remain separate.**
- Presence = "is this object actually a pallet?"
- Merge rules = "does combining it with this machine make spatial sense?"

A valid pallet on an *adjacent* forklift passes presence but fails merge
(gap too large). A garbage mask from a stray point fails presence immediately.
Keep both gates.

### 9.8 Separation of concerns (final design)

Three tools, three questions, three gates:

| Gate | Question | Tool | Cost |
|---|---|---|---|
| Concept validity | "Is this a pallet?" | SAM3 presence head on load crop | 1 SAM call |
| Spatial plausibility | "Does merging fit geometrically?" | Deterministic merge rules | ~0 |
| Semantic correctness | "Is this the right final bbox per the rule?" | VLM with rule-explicit prompt | 1 VLM call |

Per refined candidate: ~2 SAM calls + 1 extra VLM call on top of the existing
evaluate VLM call. Acceptable — refinement is a minority path.

### 9.9 Tuning notes

- **Presence query phrasing**: use exactly the VLM-emitted load vocab word —
  SAM3 is phrase-sensitive. Do not translate "cardboard boxes" to "boxes".
- **Negative seeds**: when VLM can identify a "not this" location (floor
  patch, adjacent pallet), pass to SAM as `point_label=0`. Materially
  improves mask quality per multi-seed SAM literature.
- **Logging for calibration**: when presence fails but VLM was confident
  (or vice versa), log these to a dedicated `calibration/` directory. These
  are the cases that reveal whether VLM or SAM3 is miscalibrated on this
  dataset. Use them to re-tune thresholds after the first 1k-image run.
- **Multiple loads** (stacked pallets, box on pallet): single-iteration flow
  handles only one. Either have VLM return multiple seed regions (union all
  SAM masks before merge check), or accept this limitation and route
  "multi-object load" cases to human review directly.
- **Oversized loads** (container carrier where load >> machine): the 3×
  area cap will wrongly reject. Either raise per-class (`forklift: cap=5`,
  `palletjack: cap=3`), or route high-aspect-ratio results to human review.

---

## 10. Redesigned evaluate → refine flow (class-driven refinement)

Cleaner redesign that replaces the VLM-verdict-driven refinement trigger with
class-match-driven refinement. Decided in discussion; supersedes §9.6 as the
overall architecture (§9.6's internal 5-step flow still applies inside each
refine prompt).

### 10.1 Unified verdict vocabulary

All stages that adjudicate a candidate output one of three verdicts:

| Verdict | Meaning |
|---|---|
| `accept` | Candidate is valid; use this bbox. |
| `review` | Candidate uncertain or ambiguous — queue for human review. |
| `reject` | Candidate is wrong; drop it. |

Note: renamed from earlier `fix` to `review` to match existing `human_review`
enum and avoid "fix = auto-fix" ambiguity.

### 10.2 Evaluate stage — pure classification

Evaluate's sole job is *"what is this object and is it a valid detection?"*
No spatial-expansion signaling here.

Routing cascade (replaces [evaluate.py:411-424](../stages/evaluate.py#L411)):

```
if verdict.confidence < config.evaluate.reject_below   → reject
elif verdict.confidence >= config.evaluate.accept_above → accept
else                                                    → review
```

Two thresholds become config (currently hardcoded as 0.3 / 0.5):

```yaml
evaluate:
  reject_below: 0.3
  accept_above: 0.5
```

`bbox_quality` and `object_complete` verdict fields stop driving routing.
They remain on the verdict for audit/telemetry, but spatial decisions belong
to refine.

### 10.3 Refine stage — class-driven, per-class prompts

**Trigger** (replaces current VLM-signal-driven trigger):

```
candidate enters refine if and only if:
   candidate.class_name in config.refine_rules
   AND evaluate verdict in {accept, review}
```

No `strategy` enum. No `bbox_quality == NEEDS_EXPANSION` check. Just class
membership in the `refine_rules` map + non-reject from evaluate.

**Config schema** (replaces `refinement.class_rules[*].strategy`):

```yaml
refine_rules:
  forklift:
    prompts:
      - id: load_extension
        description: |
          If cargo/pallets/boxes/drums are on the forks, propose a bbox
          covering the load. If the machine carries nothing, return
          action=skip.
        load_vocab: [pallet, cardboard box, wooden crate, drum, barrel]
        presence_threshold: 0.5
    merge_rules:
      max_area_ratio:    3.0
      max_gap_diag_frac: 0.02
      aspect_ratio_range: [0.25, 4.0]
  palletjack:
    prompts:
      - id: load_extension
        description: |
          If cargo is on the pallet jack forks, propose a bbox covering it.
          If empty, return action=skip.
        load_vocab: [pallet, box, crate]
        presence_threshold: 0.5
    merge_rules:
      max_area_ratio:    3.0
      max_gap_diag_frac: 0.02
      aspect_ratio_range: [0.25, 4.0]
```

- Multiple prompts per class: executed **sequentially**; each prompt's VLM
  sees the bbox produced by the previous prompt. Self-gated via `action:
  skip` — no SAM / merge / presence cost when VLM determines no action.
- All bbox sanity thresholds live in the refine_rules block beside the rule
  that triggers them.
- Adding a new special class = YAML edit, no code change.

**Per-prompt execution** (implements §9.6 internally):

```
1. VLM sees: image + current bbox + class + prompt.description
   Responds: {action: "skip"} OR {action: "propose", target_region, load_vocab}
2. If skip → move to next prompt (or finish).
3. If propose → SAM3 point/box prompt → load mask/bbox.
4. Presence check (SAM3 presence head) on load bbox alone with load_vocab.
     FAIL → revert to pre-prompt bbox, move on.
5. Geometric merge sanity (merge_rules on resulting union bbox).
     FAIL → revert to pre-prompt bbox, move on.
6. Merged bbox becomes the current bbox for the next prompt.
```

**Final adjudication (after all prompts complete)**:

Second VLM call framed with the class rule explicitly:
  *"Per the annotation rule '{description}', which bbox is correct — the
  original or the refined?"*

Returns: `accept | review | reject`.

### 10.4 Verdict combination table (final routing)

Verdict after refine is the combination of evaluate's verdict and refine's
adjudication:

| Evaluate | Refine adjudicate | Final verdict | Final bbox |
|---|---|---|---|
| accept | (not triggered — class not in refine_rules) | accept | original |
| accept | accept | accept | refined |
| accept | review | review | refined |
| **accept** | **reject** | **review** (NEW — was "accept original") | original |
| review | (not triggered) | review | original |
| review | accept | accept | refined |
| review | review | review | refined |
| review | reject | review | original |
| reject | (not triggered — reject short-circuits) | reject | n/a |

**Key rule change**: evaluate-accept + refine-reject → `review` (not
`accept`). Rationale: if refinement was triggered (special class) and both
the refine-VLM and the sanity checks failed to produce a valid extension,
something is inconsistent — the class's annotation rule likely applies but
couldn't be satisfied automatically. Safer to route to human review than
silently accept a tight bbox on a class the pipeline believed needed
special handling.

### 10.5 Edge cases handled explicitly

- **Tier-1 auto-accepts of special classes** — currently bypass evaluate; in
  this design they must still enter refine if their class is in
  `refine_rules`. Implementation: detect stage forwards auto-accepted
  candidates to refine directly when `class_name in config.refine_rules`,
  skipping evaluate only.
- **Relabels from evaluate** (e.g. forklift → palletjack): refine's class
  match uses the *relabeled* class, so a relabeled candidate picks up the
  new class's refine rules automatically.
- **Multi-load cases** (stacked pallets, box-on-pallet): handled via
  multiple prompts or a single prompt emitting a union region. Out-of-scope
  for v1; route persistently-multi-load cases to `review`.
- **Oversized loads** breaking `max_area_ratio: 3`: either raise cap
  per-class (e.g. forklift: 5, palletjack: 3) or let the merge-sanity
  failure revert to pre-prompt bbox → final adjudicator sees tight bbox →
  typically routes to review.
- **Post-refine dedup/filter** is still needed — §8 `finalize` stage stays
  on the roadmap independently; this redesign does not subsume it.

### 10.6 Known limitations

- **VLM-bbox / VLM-pixel accuracy** is the single weakest link. Ask for
  bboxes, not pixels. Presence + merge-sanity + final-adjudication
  compound to catch most errors, but silent misses (VLM says `skip` when
  action is needed) are unrecoverable by this pipeline. Mitigation: log all
  `skip` decisions with the crop to a calibration directory for periodic
  sample-review.
- **Deterministic prompt ordering**: sequential execution means prompt N+1
  always sees prompt N's output. If prompt ordering affects the final
  result, order in YAML matters and is not sorted alphabetically.
- **VLM adjudication bias**: VLMs tend to prefer "cleaner" tight bboxes.
  Final-adjudication prompt MUST cite the annotation rule text verbatim;
  without that framing the adjudicator drifts toward aesthetics.

### 10.7 Code changes required

1. **Rename** `needs_refine` → `needs_review` in routing; `FinalAction`
   already has `HUMAN_REVIEW` — align on a single enum across stages.
2. **Add** `EvaluateConfig` with `reject_below`, `accept_above`.
3. **Replace** `RefinementConfig.class_rules` with `RefineRulesConfig`
   (per-class `prompts` + `merge_rules`).
4. **Remove** `RefinementStrategy` enum (dead).
5. **Rewrite** `RefineWorker.process`:
   - Iterate class in `refine_rules` ∩ incoming candidates.
   - For each prompt: VLM (skip/propose) → SAM → presence → merge-sanity.
   - Final VLM adjudication per candidate.
   - Apply verdict combination table from §10.4.
6. **Update** `DetectWorker` to forward tier-1 auto-accepts of
   `class in refine_rules` into refine (bypassing evaluate only).
7. **Extend** `RefinementInstruction` schema: `action: skip|propose`,
   `target_region`, `load_vocab`, per-prompt `prompt_id`.
8. **Implement** `finalize` stage (§8) separately — still required for
   post-refine dedup/geometry recheck.

---

## 11. Open questions / TODO

- [ ] Per-image calibration on `fl_pj_sample` — measure actual score distribution
      per detector to tune `per_model_score` values empirically.
- [ ] Should `iou_dedup.threshold` drop from 0.7 → 0.5 once tiebreaker is fixed?
      Lower threshold means more aggressive collapsing of tight/loose pairs.
      Currently 0.7 is safe because buggy tiebreaker made aggressive dedup risky.
- [ ] Falcon logprob proxy — evaluate whether extracting token-level logprobs
      as a confidence signal is worth the code complexity. Probably skip unless
      forced to.
- [ ] Confusion-pair behaviour: today cross-class suppression is pairwise. Do we
      also want a VLM disambiguation step for high-IoU confusion pairs, rather
      than suppressing one? Noted in architecture doc but not implemented.
