# Auto Annotation V3 — Complete Architecture (Revised)

> 3-stage reduced pipeline + LitServe model serving + Redis Streams orchestration + versioned prompt management + JSON checkpoints.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Model Serving Layer (LitServe)                         │
│                                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ GDINO Server │ │ Falcon Server│ │  SAM3 Server  │ │ OWLv2 Server │           │
│  │ :3001 GPU:0  │ │ :3002 GPU:1  │ │ :3003 GPU:2  │ │ :3004 GPU:0  │           │
│  │ batch=8      │ │ batch=4      │ │ batch=8      │ │ batch=8      │           │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘           │
│         │                │                │                 │                    │
│  ┌──────┴────────────────┴────────────────┴─────────────────┴──────────────────┐│
│  │                         vLLM  (Qwen 3.5 27B)                                ││
│  │                         :8955  GPU:3,4  (tensor parallel=2)                 ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │                   │
          ▼                  ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Redis Streams (Message Broker)                            │
│  stream:detect ──▶ stream:evaluate ──▶ stream:refine ──▶ stream:done            │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌──────────────────┐ ┌────────────────┐
│  Detect Workers │ │ Evaluate Workers │ │ Refine Workers │
│  (HTTP clients) │ │ (HTTP→vLLM)      │ │ (HTTP→SAM)     │
│  CPU × 4-6      │ │ CPU × 6-8        │ │ CPU × 2        │
└─────────────────┘ └──────────────────┘ └────────────────┘
```

Workers are stateless CPU processes making HTTP calls. No models loaded in workers. GPU management fully delegated to LitServe servers.

---

## 2. Model Serving: LitServe

Each detection model gets its own LitServe server with dynamic batching.

### Server Implementations

**GroundingDINO** (`servers/serve_gdino.py`):

```python
import litserve as ls
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GDINOApi(ls.LitAPI):
    def setup(self, device):
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(device).eval()
        self.device = device

    def decode_request(self, request):
        image = Image.open(request["image_path"]).convert("RGB")
        text = request["text_prompt"]  # "person . forklift . palletjack"
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        return {"inputs": inputs, "image_size": image.size, "text": text,
                "input_ids": inputs["input_ids"]}

    def predict(self, x):
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v
                  for k, v in x["inputs"].items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return {"outputs": outputs, "input_ids": x["input_ids"],
                "image_size": x["image_size"]}

    def encode_response(self, r):
        w, h = r["image_size"]
        post = self.processor.post_process_grounded_object_detection(
            r["outputs"], r["input_ids"],
            threshold=0.25, text_threshold=0.2, target_sizes=[(h, w)]
        )[0]
        return {
            "boxes": post["boxes"].cpu().tolist(),
            "scores": post["scores"].cpu().tolist(),
            "labels": post["labels"],
        }

if __name__ == "__main__":
    server = ls.LitServer(GDINOApi(), accelerator="cuda",
                          devices=1, max_batch_size=8, batch_timeout=0.05)
    server.run(port=3001)
```

**Same pattern for**: Falcon (`:3002`), SAM3 (`:3003`), OWLv2 (`:3004`). Each ~30-40 lines.

SAM3 server exposes TWO endpoints: `/predict` (text-prompted proposal) and `/refine` (box+point-prompted refinement). Same server, same GPU, two APIs.

### Serving Topology

```yaml
# serve_config.yaml
servers:
  grounding_dino:
    port: 3001
    gpu: "cuda:0"
    max_batch_size: 8
    batch_timeout_ms: 50
    model_id: "IDEA-Research/grounding-dino-base"

  falcon:
    port: 3002
    gpu: "cuda:1"
    max_batch_size: 4       # slower model, smaller batches
    batch_timeout_ms: 100
    model_id: "tiiuae/Falcon-Perception"

  sam3:
    port: 3003
    gpu: "cuda:2"
    max_batch_size: 8
    batch_timeout_ms: 50
    model_id: "facebook/sam3"

  owlvit2:
    port: 3004
    gpu: "cuda:0"           # shares GPU with GDINO (small model)
    max_batch_size: 8
    batch_timeout_ms: 50
    model_id: "google/owlv2-base-patch16-ensemble"

  vlm:
    url: "http://localhost:8955/v1"
    model: "Qwen/Qwen3.5-27B-FP8"
    # GPU:3,4 via vLLM tensor parallel
```

### Why Per-Model Servers

| Concern | Answer |
|---------|--------|
| Model crash? | 503 → worker retries. Pipeline doesn't die. |
| Scale one model? | Add replicas of that server. Others unchanged. |
| Different GPUs? | Each server gets its own `devices` param. |
| Batching? | LitServe auto-batches across concurrent requests from ALL workers. |
| Add new model? | New LitServe server + config entry. No pipeline code change. |

---

## 3. Output Structure & Checkpoints

### Directory Layout

```
output/{job_id}/
├── config.yaml                        # Frozen config for this run
│
├── checkpoints/
│   └── {image_stem}/
│       ├── proposals/                 # Per-model raw results (Stage 1 substep)
│       │   ├── grounding_dino.json    # Raw GDINO detections before any filtering
│       │   ├── falcon.json            # Raw Falcon detections
│       │   ├── sam3.json              # Raw SAM3 detections
│       │   └── owlvit2.json           # Raw OWLv2 detections
│       ├── detect.json                # Stage 1 final: filtered, deduped, routed
│       ├── evaluate.json              # Stage 2: VLM verdicts + routing decisions
│       ├── refine.json                # Stage 3: SAM refinement results (if needed)
│       └── meta.json                  # Timing, status, config_hash
│
├── labels/                            # Final YOLO (accepted only)
│   └── {image_stem}.txt
│
├── traces/                            # Human-readable full audit trail
│   └── {image_stem}.json
│
├── review/                            # Human review queue
│   └── {image_stem}.json
│
├── classes.txt                        # Class index → name
└── summary.json                       # Aggregate stats
```

### Per-Model Proposal Files

Each model server's raw output is stored **before** any filtering or dedup:

```json
// checkpoints/{stem}/proposals/grounding_dino.json
{
  "model": "grounding_dino",
  "image_id": "warehouse_042",
  "image_size": [1920, 1080],
  "latency_ms": 145.3,
  "candidates": [
    {
      "candidate_id": "gdino:person:person:1",
      "class_name": "person",
      "label": "person",
      "source_model": "grounding_dino",
      "expression": "person",
      "bbox": {"x1": 0.12, "y1": 0.34, "x2": 0.28, "y2": 0.89},
      "score": 0.87
    },
    ...
  ]
}
```

This gives you:
- **Debug**: "Why was this object missed?" → check each model's raw output
- **Re-run filtering only**: Change filter thresholds → re-process from proposal files, skip model inference
- **Model comparison**: Which model found what? Overlap analysis per-model.

### Stage Checkpoint Schemas

**`detect.json`** — After filtering + dedup + routing:

```json
{
  "image_id": "warehouse_042",
  "image_path": "/data/images/warehouse_042.jpg",
  "image_size": [1920, 1080],
  "models_used": ["grounding_dino", "falcon", "sam3", "owlvit2"],

  "candidates": [
    {
      "candidate_id": "gdino:person:person:1",
      "class_name": "person",
      "bbox": {"x1": 0.12, "y1": 0.34, "x2": 0.28, "y2": 0.89},
      "score": 0.87,
      "source_model": "grounding_dino",
      "agreement": 3,
      "agreeing_models": ["grounding_dino", "falcon", "owlvit2"]
    }
  ],

  "routing": {
    "auto_accepted": ["gdino:person:person:1", "falcon:car:car:2"],
    "needs_evaluation": ["falcon:forklift:forklift:1", "gdino:palletjack:palletjack:1"],
    "confusion_flags": [
      {"cand_a": "falcon:forklift:forklift:1", "cand_b": "gdino:palletjack:palletjack:1", "iou": 0.82}
    ]
  },

  "filter_stats": {
    "total_proposed": 47,
    "after_geometric_filter": 38,
    "after_iou_dedup": 24,
    "after_per_class_cap": 24,
    "auto_accepted": 18,
    "sent_to_vlm": 6
  },

  "stage_timing_ms": 4230.5
}
```

**`evaluate.json`** — VLM evaluation results:

```json
{
  "image_id": "warehouse_042",
  "vlm_calls": 2,
  "vlm_total_tokens": 3400,

  "prompts_used": [
    {"group": "industrial", "prompt_id": "classify_industrial", "version": "2", "hash": "a3f8b2c1d4e5"}
  ],

  "verdicts": [
    {
      "candidate_id": "falcon:forklift:forklift:1",
      "correct_class": "forklift",
      "confidence": 0.92,
      "bbox_quality": "needs_expansion",
      "object_complete": true,
      "reasoning": "Clear forklift with mast visible. Load on forks extends beyond bbox."
    },
    {
      "candidate_id": "gdino:palletjack:palletjack:1",
      "correct_class": "palletjack",
      "confidence": 0.88,
      "bbox_quality": "good",
      "object_complete": true,
      "reasoning": "Low-profile pallet jack, no mast, steering handle visible."
    }
  ],

  "refinement_needed": [
    {
      "candidate_id": "falcon:forklift:forklift:1",
      "reason": "bbox_needs_expansion",
      "class_rule": "include_load"
    }
  ],

  "accepted": ["gdino:palletjack:palletjack:1"],
  "rejected": [],
  "relabels": {},
  "stage_timing_ms": 5120.0
}
```

**`refine.json`** — Refinement results (only exists when refinement fired):

```json
{
  "image_id": "warehouse_042",

  "refinement_instructions": [
    {
      "candidate_id": "falcon:forklift:forklift:1",
      "strategy": "load_extension",
      "direction": "right",
      "point_x": 1450,
      "point_y": 620,
      "vlm_reasoning": "Pallet load extends to the right of the forklift body"
    }
  ],

  "results": [
    {
      "candidate_id": "falcon:forklift:forklift:1",
      "original_bbox": {"x1": 0.34, "y1": 0.21, "x2": 0.58, "y2": 0.67},
      "refined_bbox": {"x1": 0.34, "y1": 0.21, "x2": 0.72, "y2": 0.67},
      "iou_with_original": 0.71,
      "accepted": true,
      "method": "sam_point"
    }
  ],

  "vlm_calls": 1,
  "sam_calls": 1,
  "prompt_used": {"prompt_id": "refine_spatial", "version": "2", "hash": "f7e2a1b3c8d9"},
  "stage_timing_ms": 3800.0
}
```

**`meta.json`**:

```json
{
  "image_id": "warehouse_042",
  "config_hash": "a3f8b2c1...",
  "prompt_version": "v2",
  "status": "complete",
  "stages_completed": ["detect", "evaluate", "refine"],
  "total_timing_ms": 13150.5,
  "final_counts": {"accepted": 19, "rejected": 3, "human_review": 2}
}
```

### Resume Logic

```python
def should_run_stage(image_id, stage, checkpoint_dir, config_hash):
    meta_path = checkpoint_dir / image_id / "meta.json"
    if not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text())
    if meta["config_hash"] != config_hash:
        clear_downstream(image_id, stage)
        return True
    return stage not in meta["stages_completed"]
```

Resume granularity:
- Per-model proposal resume (GDINO done, Falcon crashed → only re-run Falcon)
- Per-stage resume (detect done, evaluate crashed → re-run from evaluate)
- Config-aware (change filter params → re-run from detect, skip model proposals if unchanged)
- Prompt-aware (change prompt version → re-run evaluate + refine, skip detect)

---

## 4. The Three Pipeline Stages

### Stage 1: DETECT

```
┌─────────────────────────────────────────────────────────────┐
│  DETECT WORKER                                               │
│                                                              │
│  1. Call model servers IN PARALLEL (async HTTP)              │
│     ├── HTTP → GDINO:3001                                   │
│     ├── HTTP → Falcon:3002                                  │
│     ├── HTTP → SAM3:3003                                    │
│     └── HTTP → OWLv2:3004                                   │
│     (all concurrent, worker waits for slowest)              │
│                                                              │
│  2. Store per-model raw results                             │
│     └── checkpoints/{stem}/proposals/{model}.json           │
│                                                              │
│  3. Merge all into unified candidate list                   │
│                                                              │
│  4. Geometric filtering (area, aspect, edge)                │
│                                                              │
│  5. Per-class IoU dedup (score-ranked, source-aware)        │
│                                                              │
│  6. Cross-class routing                                     │
│     ├── person + anything → never suppress                  │
│     ├── head + anything → never suppress                    │
│     ├── forklift + palletjack IoU>0.7 → confusion flag      │
│     ├── backpack + handbag IoU>0.7 → confusion flag          │
│     └── default high IoU cross-class → suppress lower       │
│                                                              │
│  7. Agreement computation                                   │
│     ├── Tier 1 + agreement≥2 → auto_accept                 │
│     └── Tier 2-4 OR agreement<2 → needs_evaluation          │
│                                                              │
│  8. Write detect.json + forward                             │
└─────────────────────────────────────────────────────────────┘
```

**On the "all models synced" question**: The detect worker calls all 4 model servers concurrently via `asyncio.gather`. Yes, it waits for the slowest model (Falcon, ~5-8s). But this is the right approach because:

1. **Dedup requires all results.** You can't IoU-dedup GDINO vs Falcon candidates if Falcon hasn't returned yet.
2. **Agreement requires all results.** "2+ models agree" needs all models to have voted.
3. **Workers are cheap.** While Worker-1 waits for Falcon, Workers 2-6 are sending their own requests. The model servers batch across ALL concurrent requests from ALL workers. GDINO's server doesn't idle — it's processing images from other workers.
4. **The real bottleneck is the MODEL, not the worker.** Falcon takes 5-8s regardless of whether the worker is blocking or not. Splitting into per-model workers doesn't make Falcon faster.

The scaling lever is **number of detect workers** (more images in flight simultaneously) and **number of model server replicas** (if Falcon is the bottleneck, add a second Falcon server on another GPU).

### Stage 2: EVALUATE

**Two focused VLM calls, not one packed call.**

```
┌─────────────────────────────────────────────────────────────┐
│  EVALUATE WORKER                                             │
│                                                              │
│  1. Read detect.json                                        │
│                                                              │
│  2. Group candidates by evaluation_group                    │
│     ├── industrial: [forklift_1, palletjack_1]              │
│     ├── luggage: [backpack_1, suitcase_1]                   │
│     └── etc.                                                │
│                                                              │
│  3. VLM CALL 1 — Classification + Quality (per group)       │
│     ├── Prompt: class descriptions + annotated image + crops │
│     ├── Response: class, confidence, bbox_quality, complete │
│     ├── One call per group with candidates                  │
│     └── Groups can run in parallel                          │
│                                                              │
│  4. Resolve: accepted / rejected / needs_refinement         │
│     ├── Canonicalize any relabels via alias map             │
│     └── Identify candidates needing refinement              │
│                                                              │
│  5. If refinement needed:                                   │
│     VLM CALL 2 — Spatial refinement (per candidate)         │
│     ├── Prompt: crop of candidate, focused question          │
│     ├── "Where does the load extend? Give pixel coordinate" │
│     └── One call per candidate needing refinement           │
│                                                              │
│  6. Write evaluate.json + forward                           │
└─────────────────────────────────────────────────────────────┘
```

**Why two calls, not one:**

A 27B parameter model handles **focused tasks** well and **multi-objective tasks** poorly. Classification ("is this a forklift or pallet jack?") is semantic reasoning. Point localization ("where exactly is the load?") is spatial reasoning. These use different internal representations and compete for attention in a single prompt.

Benchmarking from your own `silly-growing-falcon.md` research confirms this: simple focused VLM calls outperform packed multi-objective ones. The ComfyUI author found that "VLM hallucination can be amplified" in complex prompts.

**Call 1 — Classification + Quality** (per group, all candidates):

```
You are evaluating candidate annotations for: forklift, palletjack, trolley.

CLASS DESCRIPTIONS:
- FORKLIFT: Powered truck with vertical MAST, counterweight body...
  Key: if it has a MAST → forklift, even with forks lowered.
- PALLET JACK: LOW-PROFILE, NO mast, forks at ground level...
  Key: if NO mast and low-profile forks → pallet jack.
- TROLLEY: Wheeled cart with handle, no forks...

[Annotated image with numbered boxes]
[Crops of each candidate]

For EACH candidate, return JSON:
{
  "candidate_id": "...",
  "correct_class": "forklift | palletjack | trolley | other",
  "confidence": 0.0-1.0,
  "bbox_quality": "good | needs_expansion | too_loose | bad",
  "object_complete": true/false,
  "reasoning": "brief"
}
```

This prompt does ONE thing: semantic evaluation. The model classifies, assesses quality, and checks completeness. No spatial coordinates, no multi-step reasoning.

**Call 2 — Spatial refinement** (per candidate, only when needed):

```
This forklift's bounding box doesn't include its load.
[Crop of forklift with current bbox drawn]

The load (pallets, boxes, etc.) extends beyond the current bbox.
Image dimensions: 1920x1080 pixels.

Where is the center of the load that's outside the box?
Return JSON: {"point_x": int, "point_y": int, "direction": "left|right|up|down"}
```

Simple, focused, one object. The model only needs to look at one crop and point at one thing.

**Call budget per image:**

| Scenario | Call 1 (classify) | Call 2 (spatial) | Total |
|----------|-------------------|------------------|-------|
| Easy image (persons + cars only) | 0 (all auto-accepted) | 0 | **0** |
| Mixed (persons + 2 forklifts) | 1 (industrial group) | 1 (load extension) | **2** |
| Complex (all groups present) | 3-4 (one per group) | 1-2 (refinements) | **5-6** |

### Stage 3: REFINE

```
┌─────────────────────────────────────────────────────────────┐
│  REFINE WORKER                                               │
│                                                              │
│  1. Read detect.json + evaluate.json                        │
│                                                              │
│  2. For each candidate needing refinement:                  │
│     ├── Read VLM's point coordinates from evaluate.json     │
│     ├── HTTP → SAM3:3003/refine                             │
│     │   (box prompt + foreground point)                     │
│     ├── Get refined bbox from SAM mask                      │
│     ├── Compute IoU with original bbox                      │
│     ├── IoU > 0.3 → accept refinement                      │
│     ├── IoU < 0.1 → reject (SAM drifted), keep original    │
│     └── Store result                                        │
│                                                              │
│  3. Write refine.json                                       │
│                                                              │
│  4. Write final outputs:                                    │
│     ├── labels/{stem}.txt  (YOLO)                           │
│     ├── traces/{stem}.json (audit trail)                    │
│     └── review/{stem}.json (if any human_review)            │
│                                                              │
│  5. Forward to stream:done                                  │
└─────────────────────────────────────────────────────────────┘
```

No VLM validation after refinement. Auto-accept/reject based on IoU metric. This eliminates the entire V2 validation stage.

---

## 5. Evaluation Groups & Class Config

```yaml
classes:
  # ─── Tier 1: Auto-accept with model agreement ────────────
  - {id: 0,  name: person,     tier: 1, prompt: "person"}
  - {id: 1,  name: bicycle,    tier: 1, prompt: "bicycle"}
  - {id: 2,  name: car,        tier: 1, prompt: "car"}
  - {id: 3,  name: motorcycle,  tier: 1, prompt: "motorcycle"}
  - {id: 4,  name: airplane,   tier: 1, prompt: "airplane"}
  - {id: 5,  name: bus,        tier: 1, prompt: "bus"}
  - {id: 6,  name: train,      tier: 1, prompt: "train"}
  - {id: 7,  name: truck,      tier: 1, prompt: "truck"}
  - {id: 8,  name: boat,       tier: 1, prompt: "boat"}
  - {id: 14, name: bird,       tier: 1, prompt: "bird"}
  - {id: 15, name: cat,        tier: 1, prompt: "cat"}
  - {id: 16, name: dog,        tier: 1, prompt: "dog"}

  # ─── Tier 2+: Always VLM evaluated ───────────────────────
  - {id: 24, name: backpack,   tier: 2, prompt: "backpack"}
  - {id: 26, name: handbag,    tier: 2, prompt: "handbag"}
  - {id: 28, name: suitcase,   tier: 2, prompt: "suitcase"}
  - {id: 30, name: trolley,    tier: 3, prompt: "trolley"}
  - {id: 31, name: helicopter,  tier: 2, prompt: "helicopter"}
  - {id: 32, name: wallet,     tier: 2, prompt: "wallet"}
  - {id: 33, name: forklift,   tier: 3, prompt: "forklift including any load"}
  - {id: 34, name: palletjack,  tier: 3, prompt: "pallet jack including any load"}
  - {id: 35, name: head,       tier: 2, prompt: "human head"}
  - {id: 63, name: laptop,     tier: 2, prompt: "laptop"}
  - {id: 67, name: cellphone,   tier: 2, prompt: "cellphone"}

auto_accept:
  min_model_agreement: 2
  min_score: 0.3
  applies_to: tier_1_only

evaluation_groups:
  industrial:
    classes: [forklift, palletjack, trolley]
    requires_crops: true
    description: |
      FORKLIFT: Powered truck with vertical MAST, counterweight body, overhead guard.
      Always has visible mast even with forks lowered.
      PALLET JACK: LOW-PROFILE, NO mast, forks slide under pallets at ground level.
      Steering handle/tiller. Manual has pump handle, electric has motor housing.
      TROLLEY: Wheeled cart with handle, no forks, flat platform or basket.
      KEY: MAST → forklift. No mast + low forks → pallet jack. Platform + handle → trolley.
    annotation_rules:
      forklift: "Include any load (pallets, boxes) on the forks in bbox"
      palletjack: "Include any load on the forks in bbox"

  luggage:
    classes: [backpack, handbag, suitcase, wallet]
    requires_crops: true
    description: |
      BACKPACK: On back, two shoulder straps. Typically larger.
      HANDBAG: In hand or single shoulder strap. Smaller.
      SUITCASE: Rigid/semi-rigid, wheels, retractable handle.
      WALLET: Small, flat, foldable. Fits in pocket. Very small on camera.

  person_parts:
    classes: [person, head]
    requires_crops: false
    description: |
      HEAD: Annotate separately ONLY when body is NOT visible
      (head visible through window, behind counter, in vehicle).
      If full body visible → annotate as person only, NOT separately as head.

  electronics:
    classes: [laptop, cellphone, wallet]
    requires_crops: true
    description: |
      LAPTOP: Open or closed portable computer on flat surface.
      CELLPHONE: Rectangular device in hand, on surface, or near ear.
      WALLET: Small flat foldable. Distinguish from phone by thickness/texture.

  vehicles:
    classes: [car, bus, truck, train, boat, airplane, motorcycle, bicycle, helicopter]
    requires_crops: false
    description: null   # VLM knows these, no descriptions needed

  animals:
    classes: [bird, cat, dog]
    requires_crops: false
    description: null

co_existence:
  globally_exempt: [person, head]    # never suppressed against anything

  confusion_pairs:                   # same object can't be both → VLM disambiguates
    - [forklift, palletjack]
    - [forklift, trolley]
    - [backpack, handbag]
    - [handbag, suitcase]
    - [wallet, cellphone]
    - [car, truck]

refinement:
  class_rules:
    forklift: {strategy: load_extension}
    palletjack: {strategy: load_extension}
  auto_accept_iou: 0.3
  reject_iou: 0.1
```

---

## 6. Prompt Management

Prompts are versioned artifacts, not hardcoded strings. Each VLM call in the pipeline loads its prompt from a versioned template file. The rendered prompt hash is stored in every checkpoint, so you always know exactly which prompt produced which result.

### Directory Layout

```
prompts/
├── v1/
│   ├── classify_industrial.yaml
│   ├── classify_luggage.yaml
│   ├── classify_person_parts.yaml
│   ├── classify_electronics.yaml
│   ├── classify_vehicles.yaml
│   ├── classify_animals.yaml
│   ├── refine_spatial.yaml
│   └── manifest.yaml              # version metadata
├── v2/
│   ├── classify_industrial.yaml   # improved forklift/palletjack differentiation
│   ├── refine_spatial.yaml        # better point localization instructions
│   └── manifest.yaml
└── active -> v2/                  # symlink to current version
```

Prompts not overridden in `v2/` fall back to `v1/` (inheritance). Only changed prompts need to exist in the new version directory.

### Prompt Template Format

```yaml
# prompts/v2/classify_industrial.yaml
id: classify_industrial
version: "2"
stage: evaluate
group: industrial
changelog: "Added reach truck to forklift negatives, clarified mast as primary differentiator"

system: |
  You are evaluating candidate annotations for: {class_list}.

  CLASS DESCRIPTIONS:
  {class_descriptions}

  ANNOTATION RULES:
  {annotation_rules}

  [Annotated image with numbered boxes follows]
  [Individual crops of uncertain candidates follow]

  For EACH candidate, return a JSON array. Each element:
  {{
    "candidate_id": "string",
    "correct_class": "{class_list} or other",
    "confidence": 0.0-1.0,
    "bbox_quality": "good | needs_expansion | too_loose | bad",
    "object_complete": true/false,
    "reasoning": "brief explanation"
  }}

variables:
  - class_list           # injected from evaluation_group config
  - class_descriptions   # injected from evaluation_group config
  - annotation_rules     # injected from class-specific rules

model_params:
  temperature: 0.0
  max_tokens: 2048
```

```yaml
# prompts/v2/refine_spatial.yaml
id: refine_spatial
version: "2"
stage: refine
group: null  # per-candidate, not per-group
changelog: "Added explicit instruction to use original image pixel coordinates"

system: |
  This {class_name}'s bounding box doesn't fully cover the object and its load.

  Image dimensions: {image_width}x{image_height} pixels.
  Current bbox: ({x1}, {y1}) to ({x2}, {y2}) in pixels.

  Look at the crop. Identify where the load/cargo extends BEYOND the current box.
  Return the approximate pixel coordinate of the CENTER of the uncovered load area,
  in the ORIGINAL image coordinate space (not the crop).

  JSON only:
  {{
    "point_x": int,
    "point_y": int,
    "direction": "left | right | up | down",
    "reasoning": "brief"
  }}

variables:
  - class_name
  - image_width
  - image_height
  - x1
  - y1
  - x2
  - y2

model_params:
  temperature: 0.0
  max_tokens: 256
```

### Manifest

```yaml
# prompts/v2/manifest.yaml
version: "2"
parent: "v1"                       # inherits unchanged prompts from v1
created: "2026-04-13"
author: "pavan"
notes: "Improved industrial class differentiation after 100-image eval"
changed_files:
  - classify_industrial.yaml
  - refine_spatial.yaml
```

### Loading & Hashing

```python
from pathlib import Path
import hashlib
import yaml

PROMPTS_DIR = Path("prompts")

class PromptTemplate:
    def __init__(self, data: dict):
        self.id = data["id"]
        self.version = data["version"]
        self.system = data["system"]
        self.variables = data.get("variables", [])
        self.model_params = data.get("model_params", {})

    def render(self, **kwargs) -> str:
        return self.system.format(**kwargs)

    def render_and_hash(self, **kwargs) -> tuple[str, str]:
        rendered = self.render(**kwargs)
        prompt_hash = hashlib.sha256(rendered.encode()).hexdigest()[:12]
        return rendered, prompt_hash

def load_prompt(prompt_id: str, version_dir: Path = None) -> PromptTemplate:
    """Load prompt with fallback to parent version."""
    if version_dir is None:
        version_dir = PROMPTS_DIR / "active"
    version_dir = version_dir.resolve()  # resolve symlink

    # Try current version
    path = version_dir / f"{prompt_id}.yaml"
    if path.exists():
        return PromptTemplate(yaml.safe_load(path.read_text()))

    # Fall back to parent
    manifest = yaml.safe_load((version_dir / "manifest.yaml").read_text())
    parent = manifest.get("parent")
    if parent:
        return load_prompt(prompt_id, PROMPTS_DIR / parent)

    raise FileNotFoundError(f"Prompt '{prompt_id}' not found in {version_dir} or parents")
```

### Integration with Pipeline

The evaluate worker loads prompts and records what was used:

```python
class EvaluateWorker(StageWorker):
    async def _classify_group(self, session, msg, detect, group_name, candidates):
        # Load prompt template for this group
        template = load_prompt(f"classify_{group_name}")
        group_cfg = self.config.evaluation_groups[group_name]

        # Render with runtime variables
        rendered, prompt_hash = template.render_and_hash(
            class_list=", ".join(group_cfg.classes),
            class_descriptions=group_cfg.description or "",
            annotation_rules=self._format_rules(group_cfg),
        )

        # Call VLM
        response = await self._call_vlm(session, rendered, images, template.model_params)

        # Record in checkpoint
        return {
            "group": group_name,
            "prompt_id": template.id,
            "prompt_version": template.version,
            "prompt_hash": prompt_hash,
            "verdicts": parse_verdicts(response),
        }
```

The evaluate checkpoint records prompt provenance:

```json
{
  "vlm_calls": [
    {
      "group": "industrial",
      "prompt_id": "classify_industrial",
      "prompt_version": "2",
      "prompt_hash": "a3f8b2c1d4e5",
      "candidates_evaluated": 3,
      "latency_ms": 2340
    }
  ]
}
```

### Versioning Workflow

**Iterate on a prompt:**

```bash
# 1. Copy current version
cp -r prompts/v2 prompts/v3
# 2. Edit the prompt you want to change
vim prompts/v3/classify_industrial.yaml
# 3. Update manifest
vim prompts/v3/manifest.yaml  # set parent: v2, note changes
# 4. Switch active
ln -sfn v3 prompts/active
# 5. Re-run pipeline (only evaluate stage re-runs due to prompt hash change)
python -m aa_v3.submit --image-dir /data/images --config config.yaml
# 6. Compare results
python -m aa_v3.compare --job-a run_v2 --job-b run_v3
```

The compare script reads traces from both jobs and outputs:

```
Prompt v2 → v3 comparison (100 images):
  industrial group:
    accept rate:  87% → 93%  (+6%)
    reject rate:  8%  → 5%   (-3%)
    review rate:  5%  → 2%   (-3%)
    forklift↔palletjack confusion: 4 → 1  (-75%)
  luggage group:
    accept rate:  unchanged (91%)
```

No external eval framework needed — the traces already contain everything. The compare script is ~50 lines reading JSON files.

### Config-Hash Interaction

The pipeline config hash includes the active prompt version directory. If you change `prompts/active` symlink, the config hash changes, which triggers re-evaluation (but NOT re-detection — model proposals are prompt-independent).

```python
def compute_config_hash(config, prompts_dir):
    """Hash that captures everything affecting pipeline behavior."""
    h = hashlib.sha256()
    # Pipeline config
    h.update(yaml.dump(config.dict(), sort_keys=True).encode())
    # Active prompt version
    active = (prompts_dir / "active").resolve()
    for f in sorted(active.glob("*.yaml")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
```

Changing a prompt re-runs evaluate + refine. Changing a filter threshold re-runs detect + evaluate + refine. Changing a model server config re-runs everything. Each level of change triggers only the minimum re-work.

---

## 7. Scaling Analysis

### Why Sync-All-Models Works

```
Timeline with 4 detect workers, each calling 4 model servers:

Worker 1: ──[GDINO 1s]──[wait]──[wait]──[Falcon 8s done]──[filter]──▶
Worker 2:    ──[GDINO 1s]──[wait]──[wait]──[Falcon 8s done]──[filter]──▶
Worker 3:       ──[GDINO 1s]──[wait]──[wait]──[Falcon 8s done]──[filter]──▶
Worker 4:          ──[GDINO 1s]──[wait]──[Falcon 8s done]──[filter]──▶

GDINO server:  [W1][W2][W3][W4] ← batches from 4 workers, fully utilized
Falcon server: [W1][W2][W3][W4] ← same, processes batch of 4 images
```

Model servers stay saturated because multiple workers send requests concurrently. The "wait for slowest model" gap in each worker is irrelevant — it's just CPU idle time, costs nothing. The GPU throughput is what matters, and it's fully utilized.

**When to split detect into per-model stages**: Only if you have 10+ GPUs and need independent model scaling ratios (3 Falcon replicas, 1 GDINO). For ≤4-6 GPUs, the sync approach with multiple detect workers is simpler and equally efficient.

### Throughput Estimates (4 GPUs)

| Component | Per-image | With workers |
|-----------|-----------|-------------|
| Detection (4 models parallel) | ~8s (Falcon bottleneck) | 4 workers → ~2 img/s effective |
| VLM evaluate (1-3 calls) | ~3-8s | 6 workers → ~1-2 img/s effective |
| SAM refine (conditional, ~30% of images) | ~2-4s | 2 workers → not bottleneck |
| **Pipeline throughput** | | **~1 img/s sustained** |
| **1000 images** | | **~17 minutes** |

---

## 8. Worker Implementation

### DetectWorker

```python
class DetectWorker(StageWorker):
    stage = "detect"

    async def process(self, msg: StageMessage) -> StageMessage | None:
        image_path = msg.image_path
        image_size = get_image_size(image_path)

        # 1. Call all model servers in parallel
        async with aiohttp.ClientSession() as session:
            model_tasks = {}
            for model_name, server_cfg in self.config.servers.items():
                if model_name == "vlm":
                    continue
                model_tasks[model_name] = self._call_detector(
                    session, server_cfg, image_path, self.config.classes
                )
            model_results = {}
            for model_name, task in model_tasks.items():
                try:
                    model_results[model_name] = await task
                except Exception as e:
                    logger.warning("Model %s failed: %s", model_name, e)
                    model_results[model_name] = []

        # 2. Store per-model raw results
        proposals_dir = self.checkpoint_dir / msg.image_id / "proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        for model_name, candidates in model_results.items():
            (proposals_dir / f"{model_name}.json").write_text(
                json.dumps({"model": model_name, "candidates": [c.dict() for c in candidates]})
            )

        # 3. Merge + filter + dedup + route
        all_candidates = []
        for candidates in model_results.values():
            all_candidates.extend(candidates)

        filtered = filter_candidates(all_candidates, self.config.filtering)
        deduped = dedup_candidates(filtered, self.config)
        routed = route_candidates(deduped, self.config)

        # 4. Write detect.json
        self.save_checkpoint(msg.image_id, "detect", routed)

        # 5. Forward
        if routed.needs_evaluation:
            return msg.forward("evaluate")
        else:
            self._write_final_outputs(msg.image_id, routed)
            return msg.forward("done")
```

### EvaluateWorker

```python
class EvaluateWorker(StageWorker):
    stage = "evaluate"

    async def process(self, msg: StageMessage) -> StageMessage | None:
        detect = self.load_checkpoint(msg.image_id, "detect")
        to_eval = [c for c in detect.candidates if c.candidate_id in detect.routing.needs_evaluation]

        # 1. Group by evaluation group
        groups = group_by_eval_group(to_eval, self.config.evaluation_groups)

        # 2. VLM Call 1 — Classification + Quality (per group, parallel)
        async with aiohttp.ClientSession() as session:
            tasks = [self._classify_group(session, msg, detect, group_name, cands)
                     for group_name, cands in groups.items() if cands]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        verdicts = self._merge_verdicts(results)
        accepted, rejected, needs_refine, relabels = resolve_verdicts(
            verdicts, detect, self.config
        )

        # 3. VLM Call 2 — Spatial refinement instructions (only for candidates that need it)
        refinement_instructions = {}
        for cand_id in needs_refine:
            cand = next(c for c in detect.candidates if c.candidate_id == cand_id)
            rule = self.config.refinement.class_rules.get(cand.class_name)
            if rule:
                try:
                    instr = await self._get_refinement_point(session, msg, cand)
                    refinement_instructions[cand_id] = instr
                except Exception:
                    logger.warning("Refinement instruction failed for %s", cand_id)

        # 4. Write evaluate.json
        checkpoint = EvaluateCheckpoint(
            verdicts=verdicts,
            accepted=detect.routing.auto_accepted + accepted,
            rejected=rejected,
            refinement_needed=list(refinement_instructions.keys()),
            refinement_instructions=refinement_instructions,
            relabels=relabels,
            ...
        )
        self.save_checkpoint(msg.image_id, "evaluate", checkpoint)

        if refinement_instructions:
            return msg.forward("refine")
        else:
            self._write_final_outputs(msg.image_id, detect, checkpoint)
            return msg.forward("done")
```

---

## 9. Implementation Roadmap

### Phase 1: Model Servers (2 days)
- [ ] LitServe servers for GDINO, Falcon, SAM3, OWLv2 (see `servers/`)
- [ ] SAM3 dual endpoint: `/predict` (proposal) + `/refine` (box+point)
- [ ] Health checks, latency logging
- [ ] Create `data_miner/auto_annotation_v3/compare_litserve.py` similar to `scripts/compare_proposals.py` to validate server outputs against direct inference
- [ ] Run `data_miner/auto_annotation_v3/compare_litserve.py` on `output/sample/fl_pj_sample` — validate each LitServe server produces equivalent results to aa_v2 direct inference

### Phase 2: Prompt Management (0.5 day)
- [ ] Create `prompts/v1/` with templates for each evaluation group + refinement
- [ ] Implement `PromptTemplate` loader with parent-version fallback
- [ ] Render + hash function, integrated into config hash computation
- [ ] Manifest format with changelog tracking
- [ ] Compare script for A/B across prompt versions

### Phase 3: Checkpoint + Output Layer (1 day)
- [ ] Per-model proposal file writing
- [ ] Stage checkpoint schemas (Pydantic models → JSON)
- [ ] Config-hash-aware resume logic
- [ ] YOLO label writer with relabel canonicalization (fix V2 bug C5)

### Phase 4: Core Stages (3 days)
- [ ] Detect: HTTP client to servers, filtering (port from V2), agreement, tier routing, cross-class rules
- [ ] Evaluate: Group-based VLM Call 1 (classify), focused VLM Call 2 (spatial), verdict resolution
- [ ] Refine: SAM HTTP with VLM points, IoU auto-accept/reject

### Phase 5: Redis + Workers (1 day)
- [ ] Redis Streams messaging
- [ ] StageWorker base class
- [ ] Dead letter, retry
- [ ] Job submitter + monitor

### Phase 6: Integration Test (2 days)
- [ ] 100 images end-to-end through full V3 pipeline
- [ ] Manual spot-check: review YOLO labels + trace JSONs for correctness
- [ ] Measure VLM calls per image, accuracy per tier, latency per stage
- [ ] Tune: agreement threshold, VLM prompts, batch sizes
- [ ] Edge cases: empty images, all-person images, high-confusion scenes

### Phase 7: Robustness (1 day)
- [ ] Robust JSON parsing (port from annotation-validator: `<think>` stripping, fence removal)
- [ ] Screening skip detection (VLM returns fewer verdicts than candidates)
- [ ] Image resolution handling (tell VLM original dimensions, not downscaled)
- [ ] Summary stats + pipeline.log
