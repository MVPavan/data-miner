# Auto Annotation V2

Pydantic-validated, checkpoint-resumable annotation pipeline that combines multi-model object detection with VLM-driven reasoning. Detection models propose candidates, programmatic filters clean them, and a PydanticAI agent (Qwen 3.5 / Gemma 4 via vLLM) evaluates, refines, and validates each annotation — all orchestrated through typed contracts with zero loose strings.

## Pipeline

```
┌───────────┐    ┌───────────┐    ┌──────────────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────┐
│  Proposal │───▶│  Filter   │───▶│  VLM Reasoning       │───▶│ VLM Refine    │───▶│ VLM Validate │───▶│ Finalize │
│  (models) │    │  (code)   │    │  (PydanticAI agent)  │    │ (agent+models)│    │  (agent)     │    │  (code)  │
└───────────┘    └───────────┘    └──────────────────────┘    └───────────────┘    └──────────────┘    └──────────┘
   Falcon           Area            Pass 1: Batch screen       VLM proposes         Re-evaluate         YOLO labels
   DINO             Aspect ratio    Pass 2: Detailed review    SAM/Falcon runs      refined candidates  JSON traces
   SAM3             IoU dedup       per uncertain candidate    with points/text     same two-pass       Review queue
```

**Each stage saves a JSON checkpoint.** If the pipeline crashes, it resumes from the last completed stage per image.

### Stage Details

| # | Stage | Engine | What It Does |
|---|-------|--------|-------------|
| 1 | **Proposal** | Falcon, GroundingDINO, SAM3 | Run configured detection models across all class/expression combos. Returns raw `Candidate` list |
| 2 | **Filter** | Pure code | Apply min/max area, aspect ratio, edge proximity, IoU dedup, per-class cap. Marks rejects as `FILTERED_OUT` |
| 3 | **VLM Reasoning** | PydanticAI Agent | **Pass 1 (Screening):** One VLM call per class with annotated image showing all candidates numbered. Quick accept/review/reject. **Pass 2 (Detailed):** One call per uncertain candidate with original + annotated + crop. Deep analysis: semantic match, bbox quality, relabel suggestion |
| 4 | **VLM Refinement** | PydanticAI Agent + detection models | VLM proposes refinement strategy per candidate (SAM point prompts, corrected text, etc). Detection model executes the strategy |
| 5 | **VLM Validation** | PydanticAI Agent | Re-evaluate refined candidates through the same two-pass reasoning |
| 6 | **Finalize** | Pure code | Resolve all verdicts into final accept/reject/human_review. Export YOLO labels, full trace JSON, review queue |

### KV-Cache Optimization

Pass 1 uses **O(C)** VLM calls (C = number of classes) instead of O(N) (N = candidates), because all candidates of one class are sent in a single annotated image. Pass 2 shares the system prompt + image prefix across candidates of the same class, so vLLM's automatic prefix caching reuses the KV cache. Total cost: **O(C + K)** where K = uncertain candidates.

## Quick Start

### Prerequisites

- Python 3.11+
- Detection models: Falcon Perception, GroundingDINO, SAM3 (via HuggingFace)
- VLM endpoint: Qwen 3.5 or Gemma 4 served via vLLM (OpenAI-compatible API)

### Install

```bash
cd /path/to/data_miner
uv sync  # installs pydantic-ai and all dependencies
```

### Run

```bash
# Single image
uv run python -m data_miner.auto_annotation_v2.cli \
  --config data_miner/auto_annotation_v2/fl_pj.yaml \
  --image path/to/image.jpg \
  --output-dir output/v2_results

# Batch (directory of images)
uv run python -m data_miner.auto_annotation_v2.cli \
  --config data_miner/auto_annotation_v2/fl_pj.yaml \
  --image-dir path/to/images/ \
  --output-dir output/v2_results

# With config overrides (OmegaConf dotlist syntax)
uv run python -m data_miner.auto_annotation_v2.cli \
  --config data_miner/auto_annotation_v2/fl_pj.yaml \
  --image-dir path/to/images/ \
  --output-dir output/v2_results \
  vlm.temperature=0.1 \
  filtering.min_area=0.001 \
  reasoning.max_concurrent_calls=8

# Change VLM model
uv run python -m data_miner.auto_annotation_v2.cli \
  --config data_miner/auto_annotation_v2/fl_pj.yaml \
  --image-dir images/ \
  vlm.model_name=google/gemma-4-27b \
  vlm.base_url=http://localhost:8000/v1

# Skip stages (e.g. run only detection + filtering)
uv run python -m data_miner.auto_annotation_v2.cli \
  --config config.yaml \
  --image-dir images/ \
  stages.vlm_reasoning=false \
  stages.vlm_refinement=false \
  stages.vlm_validation=false
```

### Resume After Crash

Just run the same command again. The pipeline checks `{output_dir}/.checkpoints/{image_stem}/{stage}.json` and skips completed stages.

```bash
# First run — crashes during VLM reasoning on image_042
uv run python -m data_miner.auto_annotation_v2.cli --config fl_pj.yaml --image-dir images/

# Second run — resumes image_042 from vlm_reasoning, skips all earlier images
uv run python -m data_miner.auto_annotation_v2.cli --config fl_pj.yaml --image-dir images/
```

## Configuration

Config is loaded through a 3-layer merge:

```
default.yaml  ←  custom.yaml  ←  CLI overrides
(built-in)        (--config)       (dotlist args)
```

All values are Pydantic-validated after merge. Missing fields fall back to `default.yaml`. Invalid values raise immediately with a clear error.

### Full Config Reference

```yaml
# ─── Classes ────────────────────────────────────────────────────
classes:
  - name: forklift                     # Canonical class name
    synonyms: [fork-lift, forklift truck]  # Alternative names VLM accepts
    negatives: [pallet jack, cart]     # Hard negatives VLM must reject
    prompt_variants:                   # Text prompts for detection models
      - forklift
      - forklift truck
      - yellow forklift

  - name: person
    synonyms: [worker, operator]
    negatives: [mannequin, statue]
    prompt_variants: [person, worker]

# ─── Detection Models ──────────────────────────────────────────
detection_models:
  falcon:
    kind: falcon                       # falcon | grounding_dino | sam
    enabled: true
    model_id: tiiuae/Falcon-Perception
    device: auto                       # auto | cuda | cuda:0 | cpu
    params:
      task: segmentation               # detection | segmentation
      dtype: bfloat16
      min_dimension: 256
      max_dimension: 512
      max_length: 4096
      max_new_tokens: 2048
      seed: 42

  grounding_dino:
    kind: grounding_dino
    enabled: true
    model_id: IDEA-Research/grounding-dino-base
    device: auto
    params:
      box_threshold: 0.25             # Detection confidence cutoff
      text_threshold: 0.2             # Text-image alignment cutoff

  sam:
    kind: sam
    enabled: true
    model_id: facebook/sam3
    device: auto
    params:
      threshold: 0.5                  # Mask confidence threshold

# ─── VLM (Qwen 3.5 / Gemma 4 via vLLM) ────────────────────────
vlm:
  provider: openai-compatible          # Only supported provider
  model_name: Qwen/Qwen3.5-27B-FP8   # Model served by vLLM
  base_url: http://localhost:8955/v1   # vLLM endpoint
  api_key: dummy                       # API key (dummy for local)
  temperature: 0.0                     # 0 = deterministic
  max_tokens: 4096                     # Max response tokens
  max_retries: 2                       # Retry on API errors
  timeout: 120.0                       # Request timeout (seconds)

# ─── Proposal Stage ────────────────────────────────────────────
proposal:
  models: [falcon, grounding_dino, sam]  # Which detection_models to run

# ─── Filtering Stage ───────────────────────────────────────────
filtering:
  min_area: 0.0005                    # Reject if bbox area < this (normalized)
  max_area: 0.95                      # Reject if bbox area > this
  min_aspect_ratio: 0.1              # Reject if width/height < this
  max_aspect_ratio: 10.0             # Reject if width/height > this
  min_edge_distance: 0.0             # Min distance from image edge (0 = allow)
  iou_dedup_threshold: 0.7           # Remove overlaps above this IoU
  max_candidates_per_class: 30       # Cap after all other filters

# ─── VLM Reasoning ─────────────────────────────────────────────
reasoning:
  screening:                          # Pass 1: Batch evaluation
    accept_confidence_threshold: 0.75 # ≥ this → ACCEPT
    reject_confidence_threshold: 0.4  # < this → REJECT, else NEEDS_REVIEW
    image_inputs: [annotated]         # What images the VLM sees
  detailed:                           # Pass 2: Per-candidate deep review
    accept_confidence_threshold: 0.75
    reject_confidence_threshold: 0.4
    image_inputs: [original, annotated, crop]
  max_concurrent_calls: 4            # Parallel detailed reviews

# ─── VLM Refinement ────────────────────────────────────────────
refinement:
  enabled: true                      # Set false to skip entirely
  refinement_models: [sam]           # Models available for refinement
  max_refinement_rounds: 1          # How many refine→validate iterations

# ─── Stage Toggles ─────────────────────────────────────────────
stages:
  proposal: true
  filtering: true
  vlm_reasoning: true
  vlm_refinement: true
  vlm_validation: true
  finalize: true

# ─── Output ────────────────────────────────────────────────────
output:
  save_labels: true                  # YOLO .txt files
  save_traces: true                  # Full audit JSON per image
  save_review_queue: true            # Human review JSON
  label_dirname: labels
  trace_dirname: traces
  review_dirname: review
```

### Minimal Custom Config

You only need to override what differs from `default.yaml`:

```yaml
# my_project.yaml — only specify what changes
classes:
  - name: forklift
    negatives: [pallet jack, cart, trolley]
    prompt_variants: [forklift, fork-lift, forklift truck]
  - name: person
    negatives: [mannequin]
    prompt_variants: [person, worker, operator]

vlm:
  model_name: Qwen/Qwen3.5-27B-FP8
  base_url: http://gpu-server:8955/v1

detection_models:
  sam:
    enabled: false                   # Skip SAM in proposal
```

## Output Structure

```
output/v2_results/
├── .checkpoints/                    # Resume state (can delete after completion)
│   ├── image_001/
│   │   ├── proposal.json
│   │   ├── filtering.json
│   │   ├── vlm_reasoning.json
│   │   ├── vlm_refinement.json
│   │   ├── vlm_validation.json
│   │   └── finalize.json
│   └── image_002/
│       └── proposal.json           # Pipeline crashed here — will resume
│
├── labels/                          # YOLO format (only accepted annotations)
│   ├── image_001.txt               # "0 0.512 0.489 0.234 0.312\n1 ..."
│   └── image_002.txt
│
├── traces/                          # Full audit trail per image
│   ├── image_001.json              # Complete ImageTrace (all stages, all data)
│   └── image_002.json
│
└── review/                          # Candidates needing human review
    └── image_001.json              # { image_path, candidate_ids, ... }
```

### Trace JSON Structure

Each trace file contains the complete history of every annotation through the pipeline:

```jsonc
{
  "image_path": "images/image_001.jpg",
  "stages": [
    {"stage": "proposal", "started_at": "...", "completed_at": "...", "candidate_count_in": 0, "candidate_count_out": 47},
    {"stage": "filtering", "candidate_count_in": 47, "candidate_count_out": 23},
    // ... all stages
  ],
  "failures": [],                          // Any errors encountered
  "proposal_candidates": [ /* 47 raw detections */ ],
  "filtered_candidates": [ /* 23 after filtering */ ],
  "screening_results": [ /* per-candidate batch VLM verdicts */ ],
  "detailed_verdicts": [ /* per-candidate deep VLM analysis */ ],
  "refinement_proposals": [ /* VLM-proposed refinement actions */ ],
  "refined_candidates": [ /* candidates after SAM/Falcon re-run */ ],
  "validation_verdicts": [ /* re-evaluation of refined candidates */ ],
  "final_annotations": [
    {
      "candidate_id": "falcon:forklift:forklift:0",
      "class_name": "forklift",
      "bbox": {"x1": 0.34, "y1": 0.21, "x2": 0.67, "y2": 0.58},
      "action": "accept",
      "confidence": 0.92,
      "source_model": "falcon",
      "reasoning_trace": ["screening: accept (0.92)", "Pass 1 accepted"],
      "was_refined": false
    }
    // ...
  ]
}
```

## Architecture

### Design Principles

1. **Pydantic everywhere** — Every data structure is a validated Pydantic model. No raw dicts, no loose strings. Config, candidates, VLM outputs, checkpoints — all typed and validated.

2. **OmegaConf + YAML** — Three-layer merge (default → custom → CLI overrides). All values validated through Pydantic after merge.

3. **PydanticAI for VLM only** — Detection stages are pure model inference (no LLM). VLM stages use PydanticAI agents with `output_type` for structured output enforcement — no regex JSON parsing.

4. **Checkpoint everything** — Atomic JSON writes per stage per image. Crash at any point, resume with zero rework.

5. **Full traceability** — Every annotation carries its complete history: which model proposed it, how filtering scored it, what the VLM said at screening, what it said in detailed review, whether it was refined, and the final decision with reasoning.

### Module Layout

```
auto_annotation_v2/
├── __init__.py              # Public exports
├── config.py                # OmegaConf loader + Pydantic config models
├── contracts.py             # All data models (Candidate, Verdict, Trace, etc.)
├── checkpoint.py            # Atomic per-stage-per-image checkpoint manager
├── utils.py                 # Bbox math, image drawing, YOLO export, class alias maps
├── log_utils.py             # Structured logging setup
├── default.yaml             # Built-in default config
├── pipeline.py              # Pipeline runner with checkpoint resume
├── cli.py                   # CLI entry point
├── agents/
│   ├── reasoning.py         # PydanticAI screening + detailed review agents
│   └── refinement.py        # PydanticAI refinement proposal agent
└── stages/
    ├── proposal.py          # Detection model inference (Falcon, DINO, SAM3)
    ├── filtering.py         # Programmatic bbox filtering
    ├── vlm_reasoning.py     # Two-pass VLM evaluation orchestration
    ├── vlm_refinement.py    # VLM-guided re-detection execution
    ├── vlm_validation.py    # Re-evaluation of refined candidates
    └── finalize.py          # YOLO export + trace compilation
```

### Data Flow

```
Image
  │
  ▼
Proposal ─────────▶ list[Candidate]     (status=PROPOSED, from each model)
  │
  ▼
Filter ───────────▶ list[Candidate]     (some marked FILTERED_OUT, rest pass through)
  │
  ▼
VLM Screening ────▶ list[ScreeningVerdict]  (ACCEPT / NEEDS_REVIEW / REJECT per candidate)
  │
  ▼ (only NEEDS_REVIEW candidates)
VLM Detailed ─────▶ list[DetailedVerdict]   (semantic_match, bbox_quality, decision, relabel)
  │
  ▼ (still uncertain → refinement)
VLM Refine ───────▶ list[RefinementAction]  (SAM points, text re-prompt)
  │                  ▶ list[Candidate]       (status=REFINED)
  ▼
VLM Validation ───▶ list[DetailedVerdict]   (re-evaluate refined candidates)
  │
  ▼
Finalize ─────────▶ list[FinalAnnotation]   (ACCEPT / REJECT / HUMAN_REVIEW)
                    ▶ YOLO labels (.txt)
                    ▶ Full trace (.json)
                    ▶ Review queue (.json)
```

## Programmatic Usage

```python
import asyncio
from pathlib import Path
from data_miner.auto_annotation_v2 import AutoAnnotationPipelineV2, load_config

config = load_config("my_config.yaml", overrides=["vlm.temperature=0.1"])
pipeline = AutoAnnotationPipelineV2(config, output_dir=Path("output/results"))

# Async
result = asyncio.run(pipeline.run_image("path/to/image.jpg"))
print(f"Accepted: {len(result.accepted)}, Rejected: {len(result.rejected)}")
print(f"YOLO lines: {len(result.yolo_lines)}")

# Sync wrapper
result = pipeline.run_image_sync("path/to/image.jpg")

# Batch
from pathlib import Path
images = sorted(Path("images/").glob("*.jpg"))
results = pipeline.run_batch_sync(images)
```

## Tips

### Tuning Detection Quality

- **Low recall (missing objects):** Add more `prompt_variants`, enable all three detection models, lower `filtering.min_area`
- **Low precision (too many false positives):** Raise `filtering.min_area`, tighten `reasoning.screening.accept_confidence_threshold`, add hard `negatives` to class packs
- **Boxes too loose:** Enable refinement with SAM (`refinement.enabled: true`), lower `refinement.max_refinement_rounds` to 1

### Tuning VLM Behavior

- **VLM too strict (rejecting valid detections):** Lower `reasoning.screening.accept_confidence_threshold` (e.g. 0.6), add more `synonyms`
- **VLM too lenient:** Raise thresholds, add hard `negatives`, set `vlm.temperature: 0.0`
- **Slow VLM stage:** Increase `reasoning.max_concurrent_calls`, reduce `reasoning.detailed.image_inputs` to `[crop]` only

### Reducing Compute Cost

- Skip stages you don't need: `stages.vlm_refinement: false`
- Disable slow models: `detection_models.falcon.enabled: false`
- Reduce candidate cap: `filtering.max_candidates_per_class: 15`
- For fast iteration on VLM prompting, skip proposal: run once to checkpoint, then re-run with `stages.proposal: false` and `stages.filtering: false`


### Visualization
`uv run python -m data_miner.auto_annotation_v2.viewer --image-dir output/sample/fl_pj_sample --port 8956`

`uv run python -m data_miner.auto_annotation_v2.viewer --image-dir /media/data_2/datasets/datasets_pavan/fl_pj/frames_dedup_v1_cls_0.85 --port 8956`

`uv run python -m data_miner.auto_annotation_v2.viewer_fast --image-dir /media/data_2/datasets/datasets_pavan/fl_pj/frames_dedup_v1_cls_0.85 --port 8957 --output-dir output/auto_annotation_v2/fl_pj/frames_dedup_v1_cls_0.85/v1`