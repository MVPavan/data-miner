# Auto Annotation

`data_miner.auto_annotation` is a fresh, standalone auto-annotation package for image datasets.

It is built around a stage-based pipeline where:

- configuration is defined in YAML
- configuration is loaded with OmegaConf
- configuration is validated with Pydantic
- stage order is config-defined
- model choice at each stage is config-defined

The package is intentionally isolated from the older local annotation pipeline. It is meant to be a future-proof foundation where proposal, consensus, refinement, verification, and escalation can evolve independently.

## What It Does

The default pipeline implements this flow:

1. `proposal`
   Falcon and GroundingDINO propose candidate detections for each configured class.

2. `consensus`
   Candidates are clustered by IoU and scored for quality and uncertainty.

3. `refinement`
   SAM refines flagged candidates.

4. `verification`
   Qwen verifies ambiguous candidates with structured JSON output.

5. `escalation`
   Unresolved candidates are sent to a human-review queue.

The exact stages, order, and models are all configurable.

## Current Adapters

The package currently includes these adapters:

- `falcon`
  Primary semantic proposer using Falcon Perception.

- `grounding_dino`
  Secondary semantic proposer and cross-check.

- `sam`
  Box-based refinement adapter.

- `qwen`
  Verifier using an OpenAI-compatible chat-completions endpoint.

## Current Files

- `config.py`
  Pydantic schemas and OmegaConf loading.

- `contracts.py`
  Typed pipeline contracts for candidates, clusters, reviews, and results.

- `pipeline.py`
  Stage runner and retry orchestration.

- `cli.py`
  Command-line entry point.

- `adapters/`
  Model-specific adapters.

- `stages/`
  Pipeline stage implementations.

- `config/default.yaml`
  Default pipeline definition.

## Requirements

This package assumes the runtime already has the model dependencies needed by the configured adapters.

In the default config, that means:

- Falcon Perception must be importable as `falcon_perception`
- GroundingDINO support must be available through `transformers`
- SAM 3 support must be available through `transformers`
- Qwen verification must be available behind an OpenAI-compatible HTTP endpoint

For Qwen verification, the package currently calls:

- `POST {base_url}/chat/completions`

The default base URL is:

- `http://localhost:8005/v1`

## How Configuration Works

The pipeline is driven by one config object with five main sections:

### `classes`

Defines the class packs.

Each class pack supports:

- `name`
- `synonyms`
- `negatives`
- `prompt_variants`

### `models`

Defines all available model instances.

Each model has:

- `kind`
- `enabled`
- `model_id`
- `device`
- `params`

The `kind` maps to an adapter implementation.

### `stages`

Defines the pipeline itself.

Each stage has:

- `name`
- `kind`
- `enabled`
- `implementation`
- `models`
- `params`

This is where stage choice is fully config-defined. You can reorder, disable, or swap stage models without changing code.

### `limits`

Defines retry and routing thresholds such as:

- max candidates
- retry rounds
- auto-accept thresholds
- reject thresholds

### `output`

Defines what gets written to disk:

- YOLO labels
- sidecar JSON files
- human review queue files

## Default Config

The default config lives at:

- `data_miner/auto_annotation/config/default.yaml`

It currently sets:

- Falcon + GroundingDINO for proposal
- consensus clustering
- SAM for refinement
- Qwen for verification
- escalation for unresolved cases

## How To Run

Run a single image:

```bash
python -m data_miner.auto_annotation.cli \
  --image /path/to/image.jpg \
  --output-dir /path/to/output
```

Run a folder:

```bash
python -m data_miner.auto_annotation.cli \
  --image-dir /path/to/images \
  --output-dir /path/to/output
```

Run with a custom config file:

```bash
python -m data_miner.auto_annotation.cli \
  --config /path/to/auto_annotation.yaml \
  --image-dir /path/to/images \
  --output-dir /path/to/output
```

Run with OmegaConf overrides:

```bash
python -m data_miner.auto_annotation.cli \
  --image-dir /path/to/images \
  --output-dir /path/to/output \
  stages[0].models=[falcon] \
  limits.max_retry_rounds=0 \
  output.save_review_queue=false
```

## Example Custom Config

```yaml
classes:
  - name: forklift
    synonyms: [lift truck]
    negatives: [pallet jack]
    prompt_variants:
      - forklift
      - industrial forklift
  - name: pallet jack
    synonyms: [pallet truck]
    negatives: [forklift]
    prompt_variants:
      - pallet jack
      - low-profile pallet jack

models:
  falcon:
    kind: falcon
    enabled: true
    model_id: tiiuae/Falcon-Perception
    device: auto
    params:
      task: segmentation
  grounding_dino:
    kind: grounding_dino
    enabled: true
    model_id: IDEA-Research/grounding-dino-base
    device: auto
    params:
      box_threshold: 0.25
      text_threshold: 0.2
  sam:
    kind: sam
    enabled: true
    model_id: facebook/sam3
    device: auto
    params:
      threshold: 0.5
  qwen:
    kind: qwen
    enabled: true
    model_id: Qwen/Qwen3.5-27B-FP8
    device: auto
    params:
      base_url: http://localhost:8005/v1
      api_key: dummy

stages:
  - name: proposal
    kind: proposal
    models: [falcon, grounding_dino]
    params: {}
  - name: consensus
    kind: consensus
    models: []
    params:
      iou_threshold: 0.5
      min_agreement: 2
  - name: refinement
    kind: refinement
    models: [sam]
    params:
      targets: [flagged]
      threshold: 0.5
  - name: verification
    kind: verification
    models: [qwen]
    params:
      padding: 0.08
  - name: escalation
    kind: escalation
    models: []
    params: {}
```

## Outputs

Depending on config, the pipeline writes:

- `labels/`
  YOLO-style accepted labels.

- `sidecars/`
  Full JSON pipeline results per image.

- `review/`
  Human-review queue entries for unresolved candidates.

## How The Retry Loop Works

The pipeline runs all stages up to verification once.

If verification returns `recommended_action = refine`, the pipeline will:

1. increment the retry round
2. run the refinement stage again
3. run verification again

This is bounded by `limits.max_retry_rounds`.

## Important Notes

1. This is a first-pass implementation scaffold, not a tuned production pipeline.
2. The default thresholds are sensible starting points, not final values.
3. The Qwen verifier expects a working OpenAI-compatible server.
4. Proposal quality and refinement behavior should be tuned on a real validation set before large runs.

## Recommended First Use

1. Start with one class and a small image folder.
2. Confirm that proposal outputs are reasonable.
3. Inspect the `sidecars/` output to understand the routing behavior.
4. Tune class packs and thresholds before scaling up.

## Minimal Smoke Test

You can at least confirm that config loading and pipeline construction work with:

```bash
python -c "from data_miner.auto_annotation import load_config, AutoAnnotationPipeline; cfg = load_config(); pipe = AutoAnnotationPipeline(cfg); print(type(pipe).__name__, len(pipe.stages), sorted(pipe.adapters))"
```

Expected output shape:

```text
AutoAnnotationPipeline 5 ['falcon', 'grounding_dino', 'qwen', 'sam']
```