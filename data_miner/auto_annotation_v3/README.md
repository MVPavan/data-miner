# Auto Annotation V3

3-stage annotation pipeline with LitServe model serving, Redis Streams orchestration, versioned prompt management, and JSON checkpoints.

## Architecture

```
                        Model Serving Layer (LitServe)
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ GDINO :3001  │ │ Falcon :3002 │ │  SAM3 :3003  │ │ OWLv2 :3004  │
  │ GPU:0 bs=8   │ │ GPU:1 bs=4   │ │ GPU:2 bs=8   │ │ GPU:0 bs=8   │
  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                │                │                 │
  ┌──────┴────────────────┴────────────────┴─────────────────┴──────┐
  │                   vLLM  (Qwen 3.5 27B)  :8955                  │
  └─────────────────────────────────────────────────────────────────┘
         │                │                │                 │
  ┌─────────────────────────────────────────────────────────────────┐
  │              Redis Streams (Message Broker)                     │
  │  stream:detect ──▶ stream:evaluate ──▶ stream:refine ──▶ done  │
  └─────────────────────────────────────────────────────────────────┘
         │                │                │
  ┌──────┴──────┐  ┌──────┴───────┐  ┌────┴──────┐
  │Detect ×4-6  │  │Evaluate ×6-8 │  │Refine ×2  │
  │HTTP clients │  │HTTP → vLLM   │  │HTTP → SAM │
  └─────────────┘  └──────────────┘  └───────────┘
```

Workers are stateless CPU processes making HTTP calls. No models loaded in workers. GPU management is fully delegated to LitServe servers.

## Quick Start

```bash
# 1. Launch all model servers (requires GPUs)
python -m data_miner.auto_annotation_v3.servers.launch_all

# 2. Start Redis (if not already running)
redis-server &

# 3. Run pipeline
python -m data_miner.auto_annotation_v3 --image-dir /data/images --config custom.yaml

# 4. Compare prompt versions
python -m data_miner.auto_annotation_v3.compare --job-a output/job_1 --job-b output/job_2
```

## Pipeline Stages

### Stage 1: DETECT

Calls all 4 model servers in parallel via async HTTP, then:
1. Stores raw per-model proposals (`proposals/{model}.json`)
2. Geometric filtering (area, aspect ratio, edge distance)
3. Per-class IoU dedup (score-ranked, source-aware)
4. Cross-class routing (confusion pairs, co-existence rules)
5. Agreement computation (how many models agree per candidate)
6. Tier routing: Tier-1 + agreement >= 2 + score >= 0.3 -> auto-accept, else -> VLM evaluation

### Stage 2: EVALUATE

Two focused VLM calls per image (not one packed call):
1. **Call 1 - Classification + Quality** (per evaluation group, parallel): class, confidence, bbox quality, completeness
2. **Call 2 - Spatial Refinement** (per candidate needing it): pixel coordinate for SAM point prompt
3. Verdict resolution: accepted / rejected / needs refinement / relabeled

### Stage 3: REFINE

SAM3-based bbox refinement using VLM-provided point coordinates:
1. Call SAM3 `/predict` in refine mode (box + foreground point)
2. IoU-based auto-accept/reject (no VLM validation loop)
3. Write final YOLO labels, audit traces, and human review queue

## Project Structure

```
auto_annotation_v3/
├── __init__.py                 # Public API exports
├── __main__.py                 # python -m entry point
├── cli.py                      # CLI argument parsing
├── pipeline.py                 # Pipeline orchestrator (launches workers)
├── config.py                   # Pydantic config + YAML loading
├── default.yaml                # Default configuration (23 classes, 6 eval groups)
├── contracts.py                # Pydantic data models (Candidate, BoundingBox, etc.)
├── utils.py                    # Bbox math, filtering, dedup, image ops, YOLO export
├── checkpoint.py               # CheckpointManager (atomic writes, resume logic)
├── output.py                   # OutputWriter (YOLO labels, traces, review queue)
├── prompt_manager.py           # Versioned prompt templates with inheritance
├── compare.py                  # A/B comparison across pipeline runs
├── compare_litserve.py         # LitServe vs direct inference validation
│
├── servers/                    # LitServe model servers (one per model)
│   ├── serve_gdino.py          # GroundingDINO  :3001
│   ├── serve_falcon.py         # Falcon-Perception :3002
│   ├── serve_sam3.py           # SAM3 (dual mode: proposal + refine) :3003
│   ├── serve_owlvit2.py        # OWLv2 :3004
│   ├── serve_config.yaml       # Server topology config
│   └── launch_all.py           # Launch + health check + watchdog
│
├── workers/                    # Redis Streams worker infrastructure
│   ├── messaging.py            # RedisMessageBroker (async Redis Streams)
│   ├── base.py                 # StageWorker ABC (read/process/forward loop)
│   ├── submitter.py            # JobSubmitter (submit images to detect stream)
│   └── monitor.py              # PipelineMonitor (progress tracking)
│
├── stages/                     # Pipeline stage implementations
│   ├── detect.py               # DetectWorker (4 models parallel, filter, route)
│   ├── evaluate.py             # EvaluateWorker (VLM classify + spatial)
│   └── refine.py               # RefineWorker (SAM refinement, final output)
│
└── prompts/                    # Versioned prompt templates
    ├── active -> v1/           # Symlink to current version
    └── v1/
        ├── manifest.yaml
        ├── classify_industrial.yaml
        ├── classify_luggage.yaml
        ├── classify_person_parts.yaml
        ├── classify_electronics.yaml
        ├── classify_vehicles.yaml
        ├── classify_animals.yaml
        └── refine_spatial.yaml
```

## Output Structure

```
output/{job_id}/
├── config.yaml                     # Frozen config for this run
├── checkpoints/{image_stem}/
│   ├── proposals/                  # Per-model raw results
│   │   ├── grounding_dino.json
│   │   ├── falcon.json
│   │   ├── sam3.json
│   │   └── owlvit2.json
│   ├── detect.json                 # Stage 1: filtered, deduped, routed
│   ├── evaluate.json               # Stage 2: VLM verdicts + routing
│   ├── refine.json                 # Stage 3: SAM refinement results
│   └── meta.json                   # Timing, status, config_hash
├── labels/{image_stem}.txt         # Final YOLO labels (accepted only)
├── traces/{image_stem}.json        # Full audit trail
├── review/{image_stem}.json        # Human review queue
├── classes.txt                     # Class index -> name
└── summary.json                    # Aggregate stats
```

## Configuration

Default config in `default.yaml`. Override with custom YAML:

```bash
python -m data_miner.auto_annotation_v3 --image-dir /data/images --config my_config.yaml
```

### Key config sections

- **servers**: LitServe server ports, GPUs, batch sizes
- **classes**: 23 classes across 4 tiers (Tier 1 = auto-accept eligible)
- **evaluation_groups**: 6 groups (industrial, luggage, person_parts, electronics, vehicles, animals)
- **auto_accept**: min_model_agreement=2, min_score=0.3, tier_1_only
- **filtering**: geometric filters, IoU dedup threshold, per-class cap
- **refinement**: SAM refinement IoU thresholds per class
- **redis**: connection settings, stream names
- **workers**: detect=4, evaluate=6, refine=2

## Versioned Prompts

Prompts are YAML files with Python format-string templates. Each version directory inherits unchanged prompts from its parent:

```bash
# Iterate on a prompt
cp -r prompts/v1 prompts/v2
vim prompts/v2/classify_industrial.yaml
vim prompts/v2/manifest.yaml   # set parent: v1
ln -sfn v2 prompts/active

# Re-run — only evaluate+refine re-runs (detect is prompt-independent)
python -m data_miner.auto_annotation_v3 --image-dir /data/images

# Compare
python -m data_miner.auto_annotation_v3.compare --job-a output/job_v1 --job-b output/job_v2
```

## Resume Logic

The pipeline resumes at the finest granularity possible:

| What changed | Re-runs from |
|---|---|
| Nothing (re-run same job) | Skips everything (all checkpoints valid) |
| Filter thresholds | detect (reuses per-model proposals) |
| Prompt version | evaluate + refine (skips detect) |
| Model server config | Everything (full re-run) |
| Single model crashed mid-job | Only that model's proposals (other models cached) |

## Validating LitServe Servers

Compare LitServe HTTP outputs against aa_v2 direct inference to ensure model serving doesn't alter results:

```bash
# Single image
python -m data_miner.auto_annotation_v3.compare_litserve \
    --image output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg

# Full sample directory
python -m data_miner.auto_annotation_v3.compare_litserve \
    --image-dir output/sample/fl_pj_sample

# Specific servers only
python -m data_miner.auto_annotation_v3.compare_litserve \
    --image output/sample/fl_pj_sample/-pFdYqqFUl8_003840.jpg \
    --servers grounding_dino falcon

# LitServe only (skip direct inference, just check servers are responding)
python -m data_miner.auto_annotation_v3.compare_litserve \
    --image-dir output/sample/fl_pj_sample --litserve-only

# Save results
python -m data_miner.auto_annotation_v3.compare_litserve \
    --image-dir output/sample/fl_pj_sample --save-results results.json
```

### What compare_litserve checks

For each model server (GDINO, Falcon, SAM3, OWLv2):

| Metric | Description |
|---|---|
| Detection count | Same number of boxes from both paths |
| IoU matching | Each LitServe box matches a direct-inference box at IoU >= 0.5 |
| Score difference | Confidence scores within tolerance (< 0.05 for EQUIVALENT) |
| Label match | Same class labels assigned |

Verdicts:
- **EQUIVALENT**: Same count, all matched, labels agree, score diff < 0.05
- **CLOSE**: Minor differences (1 extra/missing box, or slight score variance) -- expected from batching or numerical precision
- **DIVERGED**: Significant differences -- investigate model loading or preprocessing

## Key Differences from V2

| Aspect | V2 | V3 |
|---|---|---|
| Model loading | In-process, per-worker GPU memory | LitServe servers, shared across workers |
| Orchestration | Sequential per-image | Redis Streams, parallel workers |
| VLM strategy | 2-pass (screening + detailed) per class | 2-call (classify per group + spatial per candidate) |
| Refinement validation | VLM validates SAM output | IoU-based auto-accept/reject (no VLM loop) |
| Pipeline stages | 6 (proposal, filter, reasoning, refinement, validation, finalize) | 3 (detect, evaluate, refine) |
| Prompts | Hardcoded in agents | Versioned YAML templates with inheritance |
| Resume | Per-stage checkpoints | Per-stage + per-model proposal caching |
| Scaling | Single process | Add workers per stage independently |

## Dependencies

```
pydantic>=2.0
pyyaml
aiohttp
redis[hiredis]
litserve
torch
transformers
Pillow
numpy
```
