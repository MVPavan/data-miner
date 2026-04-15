# Auto Annotation V3

4-stage annotation pipeline: LitServe model servers, Redis Streams orchestration,
versioned prompts, JSON checkpoints, class-driven SAM refinement.

## Architecture

```
                    Model serving (LitServe)
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ GDINO :3001  │ │ Falcon :3002 │ │  SAM3 :3003  │ │ OWLv2 :3004  │
  │ GPU:0 bs=8   │ │ GPU:0 bs=4   │ │ GPU:1 bs=8   │ │ GPU:1 bs=8   │
  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                    vLLM (Qwen 3.5 27B) :8955
                    Redis Streams (broker)
                            │
   detect ──▶ evaluate ──▶ refine ──▶ finalize ──▶ done
   (×4)        (×6)        (×2)        (×2)
```

Workers are stateless CPU processes; all GPU work lives in the LitServe servers.

## Quick start

```bash
# 1. Launch model servers (one-time per session — health-checks then blocks)
python -m data_miner.auto_annotation_v3.servers.launch_all \
    --log-dir /tmp/aav3_logs

# 2. Redis + vLLM should already be running (check)
python -c "import redis; print(redis.Redis().ping())"
curl -s http://localhost:8955/v1/models | head -c 80

# 3. Run the pipeline
python -m data_miner.auto_annotation_v3 \
    runtime.image_dir=/path/to/images \
    runtime.job_id=my_job \
    'detect_classes=[forklift,palletjack,person]'

# 4. Browse results
python -m data_miner.auto_annotation_v3.viewer \
    --job-dir output/auto_annotation_v3/my_job --port 8998
```

CLI overrides use OmegaConf dotlist syntax (override any config key).

## How-to recipes

### Run on one image
```bash
python -m data_miner.auto_annotation_v3 \
    runtime.image_paths='[/path/img.jpg]' runtime.job_id=single
```

### Restrict / extend classes
```bash
'detect_classes=[forklift,palletjack,person]'   # subset
'detect_classes=[]'                              # all classes in registry
```

### Iterate on prompts (skips detect — proposals are cached)
```bash
cp -r prompts/v1 prompts/v2
$EDITOR prompts/v2/classify_industrial.yaml
ln -sfn v2 prompts/active
python -m data_miner.auto_annotation_v3 runtime.image_dir=/path runtime.job_id=v2_run
```

### Compare two runs
```bash
python -m data_miner.auto_annotation_v3.compare \
    --job-a output/auto_annotation_v3/v1_run \
    --job-b output/auto_annotation_v3/v2_run
```

### Validate model servers (LitServe vs direct inference)
```bash
python -m data_miner.auto_annotation_v3.tests.compare_litserve \
    --image-dir output/sample/fl_pj_sample
```

### Multi-class proposal sanity (joint vs per-class for each model)
```bash
python -m data_miner.auto_annotation_v3.tests.test_multiclass_proposal \
    --classes person forklift palletjack --limit 3
```

### Batch-vs-sequential parity test (LitServe batching invariant)
```bash
python -m data_miner.auto_annotation_v3.tests.test_batch_accuracy
```

### Stop everything
```bash
pkill -TERM -f data_miner.auto_annotation_v3
pkill -TERM -f auto_annotation_v3/servers/serve_
```

## Pipeline stages

1. **DETECT** — per-class HTTP fan-out to all 4 servers (joint multi-class
   prompts catastrophically degrade SAM3 + Falcon, see §10/§11 of
   [docs/aav3_filtering_scores_discussion.md](docs/aav3_filtering_scores_discussion.md)).
   Geometric filter → cluster-and-collapse dedup (with `agreement` attached) →
   cross-class suppression → tier/score routing.
2. **EVALUATE** — one VLM call per evaluation group; pure confidence-based
   three-way verdict (`accept | review | reject`). Bbox-quality is recorded
   for telemetry only — spatial decisions belong to refine.
3. **REFINE** — class-driven (triggers iff class ∈ `refine_rules` and verdict
   ≠ reject). Per-prompt loop: VLM(skip|propose) → SAM3 segment → SAM3
   presence on load bbox → geometric merge sanity → next prompt. Final VLM
   adjudication, then §10.4 verdict combination table.
4. **FINALIZE** — sole owner of YOLO/trace/review writes. Rebuilds canonical
   list (relabels + refined bboxes) and re-runs `geometric_filter +
   cluster_and_collapse + cross_class + per_class_cap` to catch relabel
   collisions and refine-induced overlaps.

## Output structure

```
output/auto_annotation_v3/{job_id}/
├── config.yaml                  # frozen Pydantic config
├── classes.txt
├── summary.json
├── checkpoints/{image_id}/
│   ├── proposals/{model}.json   # per-model raw output
│   ├── detect.json
│   ├── evaluate.json
│   ├── refine.json
│   ├── finalize.json            # canonical list + drop log
│   └── meta.json
├── labels/{image_id}.txt        # YOLO (written by finalize only)
├── traces/{image_id}.json       # full audit trail
└── review/{image_id}.json       # human-review queue
```

## Viewer

Tabs: Proposals → Detect → Evaluate → Refine → Finalize → Final → Meta.
Per-tab toggles (routing buckets, verdicts, etc.) AND with two **global**
filters always shown in the toolbar:

- **Class** chips — show overlays only for selected classes
- **Model** chips — show overlays only for selected `source_model`s

Both have an `all on/off` button. Filters persist across tab switches.

## Configuration

Edit [configs/default.yaml](configs/default.yaml) or override at the CLI.
Key blocks:

| Block | Purpose | Notes |
|---|---|---|
| `servers` | model server ports/GPUs/batch sizes | Mirror in `servers/serve_config.yaml` for the launcher. |
| `class_registry` | full class catalog (id, name, tier, prompt) | `detect_classes: []` runs everything. |
| `auto_accept` | tier + agreement + `per_model_score` floor | Per-model floors because detector scores are NOT comparable. |
| `evaluate` | `reject_below`, `accept_above` confidence thresholds | Three-way routing. |
| `co_existence` | `globally_exempt` + `confusion_pairs` | Cross-class suppression rules. |
| `refine_rules` | per-class `prompts` + `merge_rules` | Class-match trigger; no `strategy` enum. |
| `filtering` | geom + `iou_dedup{threshold,tiebreak_by,model_priority}` | Tiebreak cascade: `[agreement, model_priority, score]`. |
| `redis` | host/port + per-stage stream keys | Default streams are `stream:<stage>`. |
| `workers` | `detect/evaluate/refine/finalize_count` | Tune per box. |

## Resume logic

| What changed | Re-runs from |
|---|---|
| Re-run with same config | nothing — every stage's checkpoint is reused |
| Filter / dedup thresholds | detect (per-model proposals are cached) |
| Prompt version | evaluate + refine (detect skipped) |
| Server config or model_id | full re-run |

`config_hash` invalidates a stage's checkpoint when the relevant slice of
config changes.

## Layout

```
auto_annotation_v3/
├── __main__.py · cli.py · pipeline.py     # entry + orchestrator
├── config.py · contracts.py · utils.py    # config / models / helpers
├── checkpoint.py · output.py
├── prompt_manager.py · compare.py
├── configs/default.yaml
├── servers/                               # LitServe (4 detectors)
│   ├── serve_{gdino,falcon,sam3,owlvit2}.py
│   ├── serve_config.yaml · launch_all.py
├── workers/                               # Redis Streams worker base
│   ├── messaging.py · base.py · submitter.py · monitor.py
├── stages/
│   ├── detect.py · evaluate.py · refine.py · finalize.py
├── prompts/
│   ├── active -> v1/
│   └── v1/{classify_*,refine_prompt,refine_adjudicate}.yaml
├── tests/
│   ├── compare_litserve.py
│   ├── test_batch_accuracy.py
│   └── test_multiclass_proposal.py
├── viewer/                                # FastAPI single-page viewer
└── docs/aav3_filtering_scores_discussion.md   # design log
```

## Dependencies

`pydantic>=2`, `omegaconf`, `pyyaml`, `aiohttp`, `redis[hiredis]`, `litserve`,
`torch`, `transformers`, `Pillow`, `numpy`, `requests`, `fastapi`, `uvicorn`.
