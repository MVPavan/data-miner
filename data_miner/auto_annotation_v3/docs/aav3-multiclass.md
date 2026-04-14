# Auto Annotation V3 — Multi-Class Strategy & Aggressive Reduction

> Deep analysis of annotating 24 classes at scale.
> Validates each idea, applies Elon's 5-step reduction, proposes a simplified pipeline.

---

## 1. Your Class Landscape

Before touching the pipeline, let's understand what we're actually annotating. These 24 classes are **not equal** — they have wildly different detection characteristics, confusion risks, and annotation rules.

### Class Tiers by Difficulty

**Tier 1 — COCO-native, easy detection, low confusion** (14 classes)

```
person, bicycle, car, motorcycle, airplane, bus, train, truck,
boat, bird, cat, dog, laptop, cellphone
```

These are in COCO, ImageNet, and every foundation model's training data. GroundingDINO and Falcon will nail them. VLM review is almost always wasted compute on these — proposal agreement between two models is sufficient signal.

**Tier 2 — Moderate confusion within group** (4 classes)

```
backpack, handbag, suitcase, wallet
```

Detectors will find "bag-like objects" but may mislabel backpack as handbag or miss wallet entirely (too small). VLM needs to see a crop and classify. Wallet is especially hard — small, flat, easily confused with cellphone or just missed.

**Tier 3 — High confusion + special annotation rules** (4 classes)

```
forklift, palletjack, trolley, head
```

- **forklift vs palletjack**: Your annotation validator already proved this needs detailed visual descriptions (mast vs no-mast, height profile, handle type). Detectors will find both as "industrial vehicle."
- **forklift + load**: Needs bbox extension beyond the body. Unique annotation rule.
- **trolley**: Confusable with cart, wheelchair, hand truck. Needs definition.
- **head**: Part-of-person. When to annotate separately vs as part of person? Needs clear rules.

**Tier 4 — Rare / domain-specific** (2 classes)

```
helicopter, wallet
```

Rare in typical training data. Helicopter is easy to detect when present but rarely present. Wallet is tiny and may need specialized detection prompts.

### Why this matters for pipeline design

The current V2 pipeline treats **all classes identically** — every candidate goes through the same 6-stage gauntlet. For person/car/truck, 80%+ of VLM compute is wasted. For forklift, it's essential. The pipeline should route adaptively.

---

## 2. Validating Your Ideas

### Idea 1: VLM class descriptions for confusion resolution ✅ VALIDATED — PROVEN

Your annotation validator already proves this works. The key is the description format:

```
- FORKLIFT: Key features: vertical MAST, counterweight body, overhead guard...
  Common confusion: do NOT classify reach trucks... do NOT classify forklifts with
  lowered forks as pallet jacks — check for a mast.
```

This is your single best accuracy lever. The description tells the VLM **what to look for** (positive features) AND **what to reject** (negative differentiators). Your validator's keep/fix/discard results prove this works at scale.

**For 24 classes**, you need descriptions for confusable groups only, not every class:

| Confusion Group | Classes | Description Focus |
|-----------------|---------|-------------------|
| Industrial vehicles | forklift, palletjack, trolley | Mast presence, height profile, steering mechanism |
| Luggage/bags | backpack, handbag, suitcase, wallet | Size, carry method, shape profile, strap presence |
| Vehicles | car, bus, truck, train | Size, wheel count (usually not confused, but truck vs bus edge cases exist) |
| Person parts | person, head | When head is separately annotated (visible but body occluded?) |
| Small objects | wallet, cellphone, laptop | Size thresholds, context (on table vs in hand) |

Classes like person, bicycle, cat, dog, bird, airplane, boat, motorcycle **don't need descriptions** — VLMs identify these perfectly from a crop.

### Idea 2: Cross-class IoU handling for co-existing objects ✅ VALIDATED — NEEDS NUANCE

The current pipeline does IoU dedup **per-class only**. Your concern is correct: cross-class dedup is needed, but it must be **selective**.

**Three cases:**

**Case A — Never suppress each other (co-existing classes):**

```
person + forklift    (person rides forklift, IoU can be 0.8+)
person + car         (person next to/in car)
person + bicycle     (person on bicycle)
person + head        (head is PART OF person)
backpack + person    (backpack on person's back, IoU 0.3-0.6)
forklift + palletjack  (can genuinely be in same scene)
```

These should NEVER suppress each other regardless of IoU. A person-on-forklift with IoU 0.85 is two valid annotations.

**Case B — Confusable pairs (mutual exclusion within same physical object):**

```
forklift ↔ palletjack   (same object can't be both)
backpack ↔ handbag       (same bag can't be both)
suitcase ↔ handbag       (same object can't be both)
car ↔ truck              (edge cases: pickup trucks)
```

If forklift and palletjack candidates overlap at IoU > 0.7 on the same physical object, the VLM must decide which one it is. Don't suppress — **flag for VLM disambiguation**.

**Case C — Parent-child (one contains the other):**

```
person → head    (head is always inside person bbox)
```

Both annotations are valid. Never suppress. But head should only be annotated when the rules say so (e.g., head visible but body partially occluded).

**Implementation:**

```yaml
co_existence_rules:
  # Pairs that should NEVER suppress each other
  never_suppress:
    - [person, forklift]
    - [person, palletjack]
    - [person, car]
    - [person, bicycle]
    - [person, motorcycle]
    - [person, head]
    - [person, backpack]
    - [forklift, palletjack]  # can co-exist in same scene

  # Pairs where same-object overlap means VLM must disambiguate
  confusion_pairs:
    - [forklift, palletjack]
    - [backpack, handbag]
    - [backpack, suitcase]
    - [handbag, suitcase]
    - [car, truck]
    - [wallet, cellphone]
```

Cross-class dedup logic: if two candidates from different classes have IoU > 0.7, check the rules. If `never_suppress` → keep both. If `confusion_pairs` → send both to VLM for disambiguation. Otherwise → suppress the lower-scored one.

### Idea 3: VLM suggests load pixel → SAM dense segmentation ✅ VALIDATED — WITH CAVEATS

This is your most creative idea and it's architecturally sound. Let me validate it carefully.

**The problem**: "forklift including any load on the forks" — detectors see the forklift body and draw a tight box. The load (pallet, boxes, crate) extends beyond. You want the bbox to encompass forklift + load as one unit.

**The proposed flow**:
1. Detector proposes forklift bbox (tight around body)
2. VLM sees the crop, reasons: "load extends 200px to the right"
3. VLM outputs a pixel coordinate on the load
4. SAM3 gets: original box prompt + foreground point on load → expands mask to include load

**Why this should work**:
- SAM3's strongest mode is box + point prompts. A foreground point on an object partially outside the box causes SAM to expand the segmentation to include it. This is SAM's core design.
- The VLM doesn't need pixel-perfect accuracy. SAM just needs a point **roughly on** the load, not at the exact edge. Even 50px error tolerance is fine because SAM's mask prediction is robust to point placement.

**Caveats**:

1. **VLM pixel accuracy is mediocre.** VLMs typically achieve ~30-50px accuracy on point localization at 1024px resolution. For the "load" use case this is sufficient — the load is usually a large region (pallet, stack of boxes), so even a rough point hits it.

2. **The 1024px resize is a real problem here.** If you downsize a 1920×1080 image to 1024×576, the VLM's pixel coordinates need to be rescaled back. But the VLM doesn't know the original resolution unless you tell it. **Fix**: Include original image dimensions in the prompt AND tell the VLM to output coordinates in the original resolution space.

3. **Not all loads are visible.** If the forklift is facing away from the camera with a load behind the mast, the VLM can't see it and can't place a point. The VLM should be allowed to say "no load visible, bbox is fine as-is."

4. **Better alternative for some cases**: Instead of a single point, ask the VLM for a **direction and distance** ("load extends RIGHT by approximately 30% of forklift width"). Then programmatically extend the box prompt for SAM. This is more robust than raw pixel coordinates because it's relative, not absolute.

**Recommended implementation**:

```python
# VLM output schema for load extension
class LoadExtension(BaseModel):
    has_visible_load: bool
    direction: Literal["left", "right", "up", "down", "none"]
    estimated_extension: float  # 0.0-1.0 as fraction of current bbox dimension
    point_x: int | None  # pixel on load (original resolution)
    point_y: int | None
```

Try the point approach first. If VLM point accuracy is poor, fall back to the direction+extension approach which lets you expand the box prompt without relying on absolute coordinates.

**This should be a class-specific rule**, not applied to all classes. Only classes with annotation_rule = "include_load" trigger this flow.

### Idea 4: Image-level validation (zoom, visibility) ✅ VALIDATED — BUT WRONG PLACEMENT

You're right that image quality matters, but the question is **when** to validate.

**Option A: Pre-filter (before proposal)**

Check: is the image suitable for annotation at all? Too blurry? Too dark? Object cut off at edges?

Problem: You don't know WHAT's in the image yet. "Too zoomed in for forklift" is meaningless if the image only contains persons.

**Option B: Post-annotation validation (after finalize)**

For each accepted annotation, check:
- Is the object complete or cut off at the image edge?
- Is the object too large (>80% of frame = likely too zoomed in for context)?
- Is the annotation usable for training?

This is cheaper and more precise because you validate **specific annotations**, not the abstract image.

**Recommendation**: Do both, but differently.

- **Pre-filter (cheap, one VLM call)**: "Is this image severely blurry, too dark, or completely featureless?" Reject obvious junk. This saves proposal compute on useless images.
- **Post-annotation (per accepted annotation)**: For each annotation, check completion rules. A forklift must have visible mast + forks. A person must have visible torso. This is a class-specific quality gate.

The post-annotation check can be merged into the VLM evaluation — when the VLM reviews a candidate, it also answers "is this object fully visible and complete?" This adds zero extra VLM calls.

### Idea 5: Proposal consensus as first step ⚠️ PARTIALLY VALIDATED — RETHINK

V1 had an explicit consensus clustering stage. V2 replaced it with VLM reasoning (the VLM IS the consensus mechanism). Your `silly-growing-falcon.md` research found:

> "The real win is model consensus, not agentic reasoning" — Inter-model agreement entropy improved F1 by 42% over VLM-as-Judge baselines.

But the V2 approach has a key insight: **VLM consensus is more accurate than detector consensus for semantic classification**. Two detectors agreeing on a box doesn't tell you if it's a forklift or pallet jack. The VLM does.

**The right hybrid**: Use detector agreement for **easy classes** (auto-accept if 2+ models agree), use VLM for **hard classes** (always send to VLM for disambiguation). This is where tier-based routing pays off.

---

## 3. Applying Elon's Reduction Principle

### Step 1: Make the requirements less dumb

**Question every stage**: Does each stage produce a decision that changes the outcome?

| Current V2 Stage | What it decides | Verdict |
|---|---|---|
| Proposal | Which objects exist | **Essential** |
| Filtering | Remove junk geometry | **Essential but trivial** — 10 lines of code, not a "stage" |
| VLM Screening (Pass 1) | Batch accept/review/reject | **Essential** — this is the core value |
| VLM Detailed (Pass 2) | Per-candidate deep dive | **Conditional** — only for uncertain cases |
| VLM Refinement | Improve bbox via SAM | **Class-specific** — only for classes with special rules |
| VLM Validation | Re-evaluate after refinement | **Redundant** — refinement already produces a score |
| Finalize | Write YOLO files | **Serialization, not a stage** |

### Step 2: Delete the part

**DELETE: VLM Validation stage.** It re-runs the full two-pass VLM reasoning on refined candidates. If SAM improved a bbox from IoU 0.3 to IoU 0.8 with the original, that's measurably better. Auto-accept if the refinement improved the metric. Only re-validate if the refinement changed the bbox dramatically (IoU < 0.3 with original = likely different object = reject the refinement).

**DELETE: Filtering as a separate stage.** Inline it into proposal post-processing. Area/aspect/edge checks are 5 lines of code. IoU dedup is 20 lines. These run in <100ms. They don't need a separate checkpoint, a separate trace entry, or a separate stage context.

**DELETE: Finalize as a checkpointed stage.** It's just file serialization. If it crashes, re-run it (it's <50ms). No checkpoint needed.

### Step 3: Simplify

**MERGE VLM Screening + Detailed into one adaptive call.** The current pipeline makes a screening call, then for NEEDS_REVIEW candidates makes a second detailed call. For 24 classes, this means up to 24 screening calls + N detailed calls.

Instead: **one VLM call per class that does everything.** The prompt includes class descriptions, confusion pairs, and annotation rules. The VLM returns a full verdict including classification, bbox quality, and load extension instructions — all in one pass. For easy classes, the VLM call is fast (short response). For hard classes, the VLM naturally spends more tokens reasoning.

This eliminates the screening→detailed handoff and the "did the VLM skip a candidate?" bug.

**MERGE proposal + filtering + dedup into a single "Detect" stage.** Propose → filter junk → dedup overlaps → output clean candidates. One checkpoint, one trace entry.

### Step 4: Accelerate

**Tier-based routing to skip VLM for easy classes.** If Falcon AND GroundingDINO both detect a person with IoU > 0.5 and both score > threshold, auto-accept without VLM. This alone eliminates ~60% of VLM calls for typical images (persons and vehicles dominate most scenes).

**Class-group VLM calls instead of per-class.** Instead of one VLM call for "forklift candidates" and another for "palletjack candidates," make one call for the entire confusion group: "Here are candidates labeled as industrial vehicles. Classify each as forklift, palletjack, trolley, or reject." This halves the VLM call count for confusable groups.

### Step 5: Automate

This comes after the pipeline works — feedback loops, active learning, threshold tuning from validation results.

---

## 4. The Reduced Pipeline

### Before (V2): 7 logical steps

```
Proposal → Filter → VLM Screen → VLM Detailed → VLM Refine → VLM Validate → Finalize
```

VLM calls per image: O(C) screening + O(K) detailed + O(R) refinement + O(R) validation
where C = classes, K = uncertain candidates, R = refined candidates.

For 24 classes with 50 candidates: ~24 + 15 + 5 + 5 = **~49 VLM calls per image**.

### After (V3): 3 real stages

```
┌───────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐
│  1. DETECT        │────▶│  2. EVALUATE + CLASSIFY   │────▶│  3. REFINE (conditional) │
│  (GPU models)     │     │  (VLM, tier-routed)       │     │  (class-specific, SAM)   │
│                   │     │                            │     │                           │
│  - Propose        │     │  - Tier 1: auto-accept    │     │  - Load extension         │
│  - Filter junk    │     │    if 2+ models agree     │     │  - Bbox tightening        │
│  - Dedup          │     │  - Tier 2-4: VLM call     │     │  - Auto-accept if         │
│  - Cross-class    │     │    per confusion group    │     │    IoU improved > 0.3     │
│    routing        │     │  - Classify + quality     │     │  - Reject if IoU with     │
│                   │     │    + completeness in      │     │    original < 0.1         │
│                   │     │    ONE pass               │     │                           │
└───────────────────┘     └──────────────────────────┘     └───────────┬───────────────┘
                                                                       │
                                                                       ▼
                                                              Write YOLO + traces
```

VLM calls per image: O(G) where G = confusion groups with candidates.
For 24 classes: ~5 confusion groups × 1 call each + 0 for auto-accepted Tier 1 = **~5-8 VLM calls per image**.

**That's a 6-8× reduction in VLM calls with equal or better accuracy.**

---

## 5. Stage Details

### Stage 1: DETECT

Same as current proposal + filtering, but merged and with cross-class awareness.

```
Input: Image
Output: list[Candidate] with routing annotations

Steps:
  1. Run detection models (Falcon, GDINO, optionally SAM/OWLv2)
  2. Geometric filtering (area, aspect ratio, edge distance)
  3. Per-class IoU dedup (score-ranked)
  4. Cross-class routing:
     - For each candidate pair with IoU > 0.7 across different classes:
       - If never_suppress pair → keep both, annotate as "co-existing"
       - If confusion_pair → flag both for VLM disambiguation
       - Else → suppress lower-scored
  5. Compute agreement signals:
     - For each candidate, check if 2+ models proposed overlapping boxes
     - If yes AND Tier 1 class → mark as "auto_accept_candidate"
```

**New: Agreement-based auto-accept for Tier 1.**

```python
def compute_agreement(candidates: list[Candidate], iou_threshold: float = 0.5) -> dict[str, int]:
    """For each candidate, count how many other models proposed a similar box."""
    agreement = {}
    for i, cand in enumerate(candidates):
        count = 1  # self
        for j, other in enumerate(candidates):
            if i == j or cand.source_model == other.source_model:
                continue
            if cand.class_name == other.class_name and bbox_iou(cand.bbox, other.bbox) >= iou_threshold:
                count += 1
        agreement[cand.candidate_id] = count
    return agreement
```

If a Tier 1 candidate has agreement >= 2, auto-accept it. No VLM call. For your 24-class list, this will auto-accept the majority of person, car, truck, bicycle detections.

### Stage 2: EVALUATE + CLASSIFY

**One VLM call per confusion group**, not per class. The VLM sees all candidates in the group and evaluates everything in one pass.

**Confusion groups for your 24 classes:**

```yaml
evaluation_groups:
  # Group 1: Industrial vehicles (high confusion)
  industrial:
    classes: [forklift, palletjack, trolley]
    requires_description: true
    requires_crop: true
    annotation_rules:
      forklift: "Include any load on forks in the bbox"
      palletjack: "Include any load on forks in the bbox"

  # Group 2: Luggage (moderate confusion)
  luggage:
    classes: [backpack, handbag, suitcase, wallet]
    requires_description: true
    requires_crop: true

  # Group 3: Vehicles (low confusion, but truck/bus edge cases)
  vehicles:
    classes: [car, bus, truck, motorcycle, bicycle, train, boat, airplane]
    requires_description: false  # VLM knows these
    requires_crop: false  # annotated image is sufficient

  # Group 4: Animals
  animals:
    classes: [bird, cat, dog]
    requires_description: false
    requires_crop: false

  # Group 5: Person + parts
  person:
    classes: [person, head]
    requires_description: true  # when to annotate head separately
    requires_crop: false

  # Group 6: Electronics
  electronics:
    classes: [laptop, cellphone, wallet]  # wallet can be confused with phone
    requires_description: true
    requires_crop: true

  # Group 7: Rare
  rare:
    classes: [helicopter]
    requires_description: false
    requires_crop: false
```

**The unified VLM prompt per group:**

```
You are evaluating {N} candidate annotations for these classes: {class_list}.

{class_descriptions if requires_description}

For each candidate, return:
- correct_class: what class is this actually? (one of {class_list}, or "other")
- confidence: 0.0-1.0
- bbox_quality: "good" / "needs_expansion" / "too_loose" / "bad"
- object_complete: true/false (is the full object visible, not cut off?)
- load_extension: null OR {direction, point_x, point_y}  # only for forklift/palletjack
```

This single call replaces: screening + detailed review + class disambiguation + completeness check + load detection. Five functions, one VLM call.

### Stage 3: REFINE (Conditional)

Only fires for candidates where the VLM said `bbox_quality = "needs_expansion"` or returned a `load_extension`.

```
Input: candidates needing refinement + VLM's refinement instructions
Output: refined candidates

Logic:
  For each candidate needing refinement:
    If load_extension:
      - Use VLM's point (point_x, point_y) as foreground prompt
      - Run SAM3 with: original box + foreground point → expanded mask → new bbox
      - If new bbox IoU with original > 0.1: accept refinement
      - If IoU < 0.1: reject (SAM drifted to different object), keep original
    If bbox_quality == "too_loose":
      - Run SAM3 with box-only prompt → tighter mask → new bbox
    If bbox_quality == "needs_expansion":
      - Extend box by 10-20% in VLM-indicated direction
      - Run SAM3 with expanded box → new mask → new bbox
```

**No VLM validation after refinement.** Auto-accept if:
- Refined bbox IoU with original > 0.3 (same object, just adjusted)
- Refined bbox passes geometric filters (area, aspect ratio)
- If both fail → keep original bbox, flag for human review

This eliminates the entire VLM Validation stage.

---

## 6. VLM Call Budget Comparison

### Typical image: warehouse scene with 2 forklifts, 3 persons, 1 palletjack

**V2 (current):**
- Screening: 3 calls (person class, forklift class, palletjack class)
- Detailed: ~3 calls (uncertain candidates)
- Refinement: ~2 calls (load extension)
- Validation: ~2 calls (re-evaluate refined)
- **Total: ~10 VLM calls**

**V3 (proposed):**
- Persons: **0 calls** (Tier 1, 2+ model agreement → auto-accept)
- Industrial group: **1 call** (all forklift + palletjack candidates evaluated together)
- Refinement: 0 VLM calls (SAM-only, using VLM's point from evaluation)
- **Total: ~1 VLM call**

### Complex image: airport scene with persons, luggage, vehicles, helicopter

**V2:** ~24+ VLM calls (one screening per class with candidates)
**V3:** ~3 VLM calls (person auto-accepted, vehicles auto-accepted, luggage group, rare group)

---

## 7. Config Schema for V3

```yaml
# ─── Class Tiers ───────────────────────────────────────────────
class_tiers:
  # Tier 1: Auto-accept with 2+ model agreement, no VLM
  auto_accept:
    classes: [person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, bird, cat, dog]
    min_model_agreement: 2
    min_score: 0.3  # at least one model above this

  # Tier 2+: Always send to VLM
  vlm_required:
    classes: [backpack, handbag, suitcase, wallet, forklift, palletjack, trolley,
              head, helicopter, laptop, cellphone]

# ─── Evaluation Groups ─────────────────────────────────────────
evaluation_groups:
  industrial:
    classes: [forklift, palletjack, trolley]
    description: |
      FORKLIFT: Powered truck with vertical MAST, counterweight body, overhead guard.
      Always has visible mast even with forks lowered.
      PALLET JACK: LOW-PROFILE wheeled device, NO mast, forks slide under pallets.
      Long steering handle. Manual has pump handle, electric has motor housing.
      TROLLEY: Wheeled cart with handle, no forks. Flat platform or basket.
      KEY: If it has a MAST → forklift. If NO mast, low-profile forks → pallet jack.
      If flat platform/basket with handle → trolley.
    annotation_rules:
      forklift: "Include any load (pallets, boxes) on the forks within the bbox"
      palletjack: "Include any load on the forks within the bbox"

  luggage:
    classes: [backpack, handbag, suitcase, wallet]
    description: |
      BACKPACK: Carried on back with two shoulder straps. Typically larger.
      HANDBAG: Carried by hand or single shoulder strap. Smaller.
      SUITCASE: Rigid or semi-rigid with wheels and retractable handle. Travel-sized.
      WALLET: Small flat foldable item. Fits in pocket or hand. Very small on camera.

  person_parts:
    classes: [person, head]
    description: |
      PERSON: Full or partial body. Annotate even if partially occluded.
      HEAD: Annotate separately ONLY when body is not visible (e.g., head visible
      through window, behind counter, in vehicle). If full body visible, annotate
      only as person, NOT separately as head.

  electronics:
    classes: [laptop, cellphone, wallet]
    description: |
      LAPTOP: Open or closed portable computer. Usually on flat surface.
      CELLPHONE: Rectangular device in hand, on surface, or near ear.
      WALLET: Small flat foldable item. Distinguish from phone by thickness/texture.

# ─── Co-existence Rules ────────────────────────────────────────
co_existence:
  never_suppress:
    - [person, forklift]
    - [person, palletjack]
    - [person, car]
    - [person, bicycle]
    - [person, motorcycle]
    - [person, truck]
    - [person, bus]
    - [person, head]
    - [person, backpack]
    - [person, handbag]
    - [person, cellphone]
    - [person, laptop]

  confusion_pairs:
    - [forklift, palletjack]
    - [forklift, trolley]
    - [backpack, handbag]
    - [backpack, suitcase]
    - [handbag, suitcase]
    - [wallet, cellphone]
    - [car, truck]

# ─── Refinement Rules (class-specific) ─────────────────────────
refinement:
  enabled_for:
    forklift:
      strategy: load_extension    # VLM provides point → SAM expands
      fallback: box_extension     # if point fails, extend box 20% in direction
    palletjack:
      strategy: load_extension
      fallback: box_extension
  auto_accept_iou_threshold: 0.3  # accept refinement if IoU with original > this
  reject_iou_threshold: 0.1       # reject refinement if IoU < this (different object)
```

---

## 8. What This Means for Implementation

### Changes to current V2:

| V2 Component | V3 Action |
|---|---|
| `proposal.py` | Keep, add agreement computation |
| `filtering.py` | Merge into proposal post-processing |
| `vlm_reasoning.py` | **Rewrite**: tier routing + group-based calls + unified prompt |
| `vlm_refinement.py` | **Simplify**: remove VLM call for refinement proposal, use evaluation output directly |
| `vlm_validation.py` | **Delete entirely** |
| `finalize.py` | Keep, add relabel canonicalization fix |
| `contracts.py` | Add `ClassTier`, `EvaluationGroup`, `CoExistenceRule` models |
| `config.py` | Add tier/group/co-existence config sections |
| New: `routing.py` | Tier-based routing logic (which candidates skip VLM) |

### New evaluation prompt (replaces screening + detailed):

```python
UNIFIED_EVALUATION = """\
Evaluate these {n} candidate annotations.

{group_description}

Candidates:
{candidate_list_with_crops}

For EACH candidate, respond with JSON:
{{
  "candidate_id": "...",
  "correct_class": "{one of class_list} or other",
  "confidence": 0.0-1.0,
  "bbox_quality": "good|needs_expansion|too_loose|bad",
  "object_complete": true/false,
  "load_extension": null | {{"direction": "left|right|up|down", "point_x": int, "point_y": int}},
  "reasoning": "brief explanation"
}}
"""
```

---

## 9. Accuracy Validation

How do we know V3 is as accurate or more accurate than V2?

**For Tier 1 auto-accept (person, car, etc.):**
- Run V2's full pipeline on 100 images. Record what VLM decides for Tier 1 candidates that had 2+ model agreement.
- Expected result: 95%+ of these are accepted by VLM. The remaining 5% are edge cases (mannequin detected as person, toy car, etc.) that will also fool auto-accept.
- If VLM rejection rate for agreed-upon Tier 1 candidates is >10%, the auto-accept threshold needs tuning.

**For confusion groups (forklift/palletjack, luggage):**
- Run both V2 and V3 on 100 images from your fl_pj dataset.
- Compare: class assignment accuracy, bbox quality scores, human review rate.
- The unified prompt with detailed descriptions should match or beat the two-pass approach because the VLM gets ALL the context in one shot (descriptions + all candidates + crops) instead of making a quick screening judgment first.

**For load extension:**
- Run on 50 forklift images with visible loads.
- Measure: does the refined bbox actually include the load? Compute IoU with manually-drawn ground truth.
- Compare: VLM point-guided SAM vs current refinement flow.

---

## 10. Summary: Do We Need So Many Stages?

**No.** Applying aggressive reduction:

| V2 Stage | V3 Fate | Reason |
|---|---|---|
| Proposal | **Keep** | Essential — detectors find objects |
| Filtering | **Merge into Proposal** | 20 lines of code, not a stage |
| VLM Screening | **Merge into Evaluate** | One unified VLM call does everything |
| VLM Detailed | **Merge into Evaluate** | Same call, no two-pass handoff |
| VLM Refinement | **Simplify** | Use evaluation output directly, no separate VLM call |
| VLM Validation | **Delete** | Auto-accept/reject based on IoU improvement metric |
| Finalize | **Keep as serialization** | Not a real stage, just file I/O |

**Result: 3 real stages instead of 7. ~5-8 VLM calls instead of ~49. Same or better accuracy because the VLM gets richer context in fewer, better-structured calls.**