# Auto Annotation V4

5-stage annotation pipeline. **SQLite-backed** (no Redis, no file checkpoints),
per-job `pipeline.db` in WAL mode, LitServe model servers, typed config
(StrEnum + Pydantic + OmegaConf), granular stage/model control, atomic
save-and-forward, config-hash invalidation.

```
                        Model serving (LitServe)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ GDINO :3001  │  │ SAM3  :3003  │  │ DART  :3013  │  │ OmDet :3005  │
  │ GPU:0 bs=8   │  │ GPU:1 bs=8   │  │ GPU:1 bs=8   │  │ GPU:0 bs=8   │
  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                    vLLM (Qwen 3.5 27B) :8955
                    SQLite pipeline.db  (WAL, per job)
                            │
  detect ──▶ filter ──▶ evaluate ──▶ refine ──▶ finalize ──▶ done
  (per-model             (POST_DETECT    (VLM 3-way       (class-driven
   + merge)               filters)        verdict)         SAM refine)
```

Every worker is a stateless async task in one process; GPU work lives only
in the LitServe servers. Work distribution is an atomic SQL claim
(`UPDATE ... WHERE rowid = (SELECT ... LIMIT 1) RETURNING`) against
`work_queue`.

---

## Table of contents

- [Install](#install)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Scenario cookbook](#scenario-cookbook)
  - [1. Run end-to-end on a folder](#1-run-end-to-end-on-a-folder)
  - [2. Run on an explicit image list](#2-run-on-an-explicit-image-list)
  - [3. Restrict or extend classes](#3-restrict-or-extend-classes)
  - [4. Run only detect (proposals)](#4-run-only-detect-proposals)
  - [5. Run only detect+filter (supervised review)](#5-run-only-detectfilter-supervised-review)
  - [6. Add filter/evaluate/refine on top of a proposal-only job](#6-add-filterevaluaterefine-on-top-of-a-proposal-only-job)
  - [7. Re-run a single stage (force_stages)](#7-re-run-a-single-stage-force_stages)
  - [8. Re-run a single detector (force_detect_models)](#8-re-run-a-single-detector-force_detect_models)
  - [9. Select a subset of detectors](#9-select-a-subset-of-detectors)
  - [10. Iterate on prompts (keeps detect cached)](#10-iterate-on-prompts-keeps-detect-cached)
  - [11. Change a filtering threshold and re-run](#11-change-a-filtering-threshold-and-re-run)
  - [12. Resume after a crash](#12-resume-after-a-crash)
  - [13. Run two jobs concurrently](#13-run-two-jobs-concurrently)
  - [14. Nuclear re-run](#14-nuclear-re-run)
  - [15. Browse results in the viewer](#15-browse-results-in-the-viewer)
  - [16. Online DB backup + retention](#16-online-db-backup--retention)
- [Resume / force / skip — cheat sheet](#resume--force--skip--cheat-sheet)
- [Output layout](#output-layout)
- [Operations](#operations)
  - [Starting / stopping servers](#starting--stopping-servers)
  - [Monitoring a live run](#monitoring-a-live-run)
  - [Tuning](#tuning)
  - [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Design references](#design-references)

---

## Install

```bash
pip install -e .
# aiohttp, aiosqlite, pydantic>=2, omegaconf, pyyaml, tenacity, litserve,
# torch, transformers, Pillow, numpy, fastapi, uvicorn
```

GPUs used in examples: GDINO on `cuda:0`, SAM3-DART on `cuda:1`.
Targets 48 GB cards; shrink `max_batch_size` for 24 GB.

---

## Quick start

```bash
# 1. Start a vLLM server (one-time, for evaluate/refine stages)
#    -- whatever you already use for Qwen 3.5 27B at http://localhost:8955

# 2. Start the detector model servers (one-time per session)
python -m data_miner.auto_annotation_v4.model_servers.serve \
    --config data_miner/auto_annotation_v4/configs/servers.yaml --all

# 3. Run the pipeline on a folder
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=/path/to/images \
    runtime.job_id=my_job \
    'detect_classes=[forklift,palletjack,person]'

# 4. Browse the run
python -m data_miner.auto_annotation_v4.viewer \
    --job-dir output/auto_annotation_v4/my_job --port 8998
```

Every CLI override uses OmegaConf dotlist syntax and deep-merges over
the packaged YAMLs.

---

## CLI reference

```bash
python -m data_miner.auto_annotation_v4 \
    [--config my_job.yaml] \
    [key.path=value ...]
```

- `--config / -c` — user YAML. Deep-merged over the packaged base
  (`default.yaml` + `servers.yaml` + `class_config.yaml` + `database.yaml` +
  `runtime.yaml`).
- positional dotlist overrides — applied on top of any `--config`.

Minimum requirements: either `--config` or at least one dotlist override
that sets input (`runtime.image_dir=...` or `runtime.image_paths=[...]`).

---

## Configuration

All YAMLs live under [configs/](configs/) and are merged at load time.

| File | Owns | Override in user YAML? |
|---|---|---|
| [default.yaml](configs/default.yaml) | `detect_classes`, `auto_accept`, `evaluate`, `filtering`, `workers`, `output`, `prompts_dir` | yes |
| [servers.yaml](configs/servers.yaml) | per-detector `enabled / port / gpu / max_batch_size / model_id / options`, `vlm.*` | yes |
| [class_config.yaml](configs/class_config.yaml) | `class_registry`, `evaluation_groups`, `co_existence`, `refine_rules` | yes |
| [database.yaml](configs/database.yaml) | `database.filename / lock_ttl / max_retries` | yes |
| [runtime.yaml](configs/runtime.yaml) | `runtime.*` — image inputs, job_id, log_level, stages, force_* | yes |

Key runtime knobs (from [configs/runtime.yaml](configs/runtime.yaml)):

```yaml
runtime:
  image_dir: null                   # folder to scan
  image_paths: []                   # explicit list (alternative to image_dir)
  job_id: null                      # derived from image_dir if null
  log_level: INFO
  log_file: null                    # logs to stdout when null
  stages: [detect, filter, evaluate, refine, finalize]
  force_stages: []                  # ignore caches for these stages
  force_rerun: false                # nuclear wipe before run
  detect_models: []                 # subset of enabled detectors; [] = all
  force_detect_models: []           # delete proposals + re-run these models
```

Detectors are enabled/disabled in [configs/servers.yaml](configs/servers.yaml):

```yaml
servers:
  detectors:
    grounding_dino: { enabled: true,  port: 3001, gpu: "cuda:0", ... }
    sam3_dart:      { enabled: true,  port: 3013, gpu: "cuda:1", ... }
    falcon:         { enabled: false, ... }
    omdet_turbo:    { enabled: false, ... }
    owlvit2:        { enabled: false, ... }
```

---

## Scenario cookbook

All examples assume model servers and vLLM are already running.
Environment variable shorthand used below:

```bash
export IMG_DIR=output/projects/delivery_pov_v1/frames_filtered_v2_dedup
```

### 1. Run end-to-end on a folder

```bash
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=$IMG_DIR \
    runtime.job_id=my_job
```

Runs `detect → filter → evaluate → refine → finalize` for every image
in the folder. Writes YOLO labels under
`output/auto_annotation_v4/my_job/labels/`.

### 2. Run on an explicit image list

```bash
python -m data_miner.auto_annotation_v4 \
    'runtime.image_paths=[/data/a.jpg,/data/b.jpg]' \
    runtime.job_id=adhoc
```

### 3. Restrict or extend classes

```bash
# subset
'detect_classes=[forklift,palletjack,person]'

# all classes in the registry
'detect_classes=[]'
```

### 4. Run only detect (proposals)

Raw per-model proposals, no filtering, no VLM. Useful for dataset
inspection, debugging detectors, or building a review queue.

```bash
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=$IMG_DIR \
    runtime.job_id=proposals_only \
    'runtime.stages=[detect]'
```

Proposals land in `proposals` (one row per `image_id × model`) and
`stages.stage='detect'` (raw merge, no filter drops applied).

### 5. Run only detect+filter (supervised review)

Apply the POST_DETECT filter chain (geometric → per-model score → dedup →
per-class cap → cross-class) and stop. YOLO labels are written for every
auto-accepted candidate; review queue lists everything needing VLM.

```bash
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=$IMG_DIR \
    runtime.job_id=reviewed \
    'runtime.stages=[detect,filter]'
```

### 6. Add filter/evaluate/refine on top of a proposal-only job

Re-run the same `job_id` with more stages — detect is cached, the new
stages enqueue automatically.

```bash
# After scenario 4 above:
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=$IMG_DIR \
    runtime.job_id=proposals_only \
    'runtime.stages=[detect,filter]'
```

Detect proposals are kept; filter runs on every image in the job. Works
for any extension (`...,filter,evaluate]`, `...,filter,evaluate,refine]`,
full pipeline).

### 7. Re-run a single stage (force_stages)

Invalidates that stage + everything downstream for every image.
Upstream stages stay cached.

```bash
# tweak filtering thresholds and re-run just filter → finalize
python -m data_miner.auto_annotation_v4 \
    --config my_job.yaml \
    'runtime.force_stages=[filter]' \
    filtering.per_model_score.grounding_dino=0.40
```

The submitter routes each image to the earliest force-stage that has
inputs ready — detect cache is preserved.

### 8. Re-run a single detector (force_detect_models)

Deletes that detector's proposals + the merged detect row + everything
downstream. Other detectors' proposals are kept; the detect barrier
reassembles once the forced detector finishes.

```bash
python -m data_miner.auto_annotation_v4 \
    --config my_job.yaml \
    'runtime.force_detect_models=[grounding_dino]'
```

### 9. Select a subset of detectors

Only run the listed enabled detectors; others are ignored for this run
(their proposals, if any, stay in the DB but don't participate in the
merge barrier).

```bash
'runtime.detect_models=[grounding_dino,sam3_dart]'
```

### 10. Iterate on prompts (keeps detect cached)

Prompts hash into the config_hash; stages that depend on prompts
(`evaluate`, `refine`) get invalidated automatically, `detect` stays
cached.

```bash
cp -r data_miner/auto_annotation_v4/prompts/v1 data_miner/auto_annotation_v4/prompts/v2
$EDITOR data_miner/auto_annotation_v4/prompts/v2/classify_industrial.yaml
ln -sfn v2 data_miner/auto_annotation_v4/prompts/active

python -m data_miner.auto_annotation_v4 --config my_job.yaml
# pipeline prints: "config changed since last run — continue? [y/N]"
```

### 11. Change a filtering threshold and re-run

```bash
python -m data_miner.auto_annotation_v4 --config my_job.yaml \
    filtering.iou_dedup.threshold=0.6
```

The scoped `compute_config_hash` (see [configs/loader.py](configs/loader.py))
invalidates only stages whose relevant slice changed — here `filter`
onward.

### 12. Resume after a crash

Just run the same command again. What happens:

1. `pipeline._check_config_continuity()` — config unchanged → no warning.
2. Submitter scans images: `all_stages_complete()` → skips done images.
3. For incomplete: `INSERT OR IGNORE INTO work_queue` — no-op for items
   already queued (they survived the crash).
4. Monitor's `recover_stale()` finds `status=PROCESSING` older than
   `database.lock_ttl` → resets to `PENDING`.
5. Workers claim and reprocess.
6. `save_and_forward` is atomic — no "half-done" image state.

### 13. Run two jobs concurrently

Each job gets its own `job_dir` and `pipeline.db`; a `pidfile.lock`
inside the job_dir prevents two pipelines sharing the same directory.
Shared model servers handle the combined load.

```bash
# shell 1
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=/data/job_a runtime.job_id=a &

# shell 2 — different job_id → different job_dir → different DB → OK
python -m data_miner.auto_annotation_v4 \
    runtime.image_dir=/data/job_b runtime.job_id=b &
```

If two pipelines launch with the same `job_id`, the second exits
immediately with a clear "job already in use" error.

### 14. Nuclear re-run

Wipes every row for every image (proposals, stages, work_queue, meta,
failures) before re-queuing. Use when the schema or detector identity
changed beyond what `force_*` covers.

```bash
'runtime.force_rerun=true'
```

### 15. Browse results in the viewer

```bash
python -m data_miner.auto_annotation_v4.viewer \
    --job-dir output/auto_annotation_v4/my_job \
    --port 8998
```

Reads `pipeline.db` directly (WAL allows concurrent reads while the
pipeline writes). Tabs: Proposals → Detect → Filter → Evaluate → Refine
→ Finalize → Final → Meta. Global class/model chip filters persist
across tab switches.

### 16. Online DB backup + retention

`sqlite3 .backup` under WAL is crash-consistent and does not block
writers. The bundled script keeps every backup <24h plus one snapshot
per day for the last 7 days.

```bash
data_miner/auto_annotation_v4/scripts/backup_pipeline_db.sh \
    output/auto_annotation_v4/my_job

# Cron (hourly):
0 * * * * /abs/path/backup_pipeline_db.sh /abs/path/to/job_dir \
    >> /abs/path/to/job_dir/backups/backup.log 2>&1
```

Backups land in `{job_dir}/backups/pipeline.db.YYYYMMDD-HHMMSS.bak`.

---

## Resume / force / skip — cheat sheet

| You want to... | Command |
|---|---|
| Resume after any crash | Same command, nothing else needed |
| Skip an early stage (input from elsewhere) | `'runtime.stages=[evaluate,refine,finalize]'` |
| Only run detect | `'runtime.stages=[detect]'` |
| Detect + filter only (review workflow) | `'runtime.stages=[detect,filter]'` |
| Re-run filter onward with new thresholds | `'runtime.force_stages=[filter]' filtering.*.=...` |
| Re-run one detector, reuse others | `'runtime.force_detect_models=[grounding_dino]'` |
| Use a detector subset this run | `'runtime.detect_models=[grounding_dino,sam3_dart]'` |
| Wipe everything, start over | `'runtime.force_rerun=true'` |
| Add stages to an earlier proposal-only run | Re-run same `job_id` with more `runtime.stages` |

`config_hash` invalidation is **per stage** — changing `evaluate.accept_above`
will not invalidate `detect` or `filter`, only `evaluate` onward.

---

## Output layout

```
output/auto_annotation_v4/{job_id}/
├── pipeline.db                    # SQLite (WAL: pipeline.db-wal, pipeline.db-shm)
├── pidfile.lock                   # fcntl.flock — one pipeline per job_dir
├── config.yaml                    # frozen merged config written at startup
├── classes.txt                    # YOLO class index
├── labels/{image_id}.txt          # YOLO boxes (finalize, or filter in stop-at-filter)
├── traces/{image_id}.json         # per-image audit trail
├── review/{image_id}.json         # human-review queue
└── backups/                       # populated by backup_pipeline_db.sh
    ├── pipeline.db.20260417-120000.bak
    └── backup.log
```

SQLite tables (see [checkpoint.py](checkpoint.py)):

| Table | Purpose |
|---|---|
| `job_info` | one-row: job_id, image_dir, config_hash, prompt_version, status |
| `image_meta` | per-image status, stages_completed, timing, hashes |
| `proposals` | per-model raw output (keyed by `image_id × model`) |
| `stages` | per-stage results (keyed by `image_id × stage`) |
| `work_queue` | work distribution — atomic claim replaces Redis Streams |
| `failures` | dead-letter — rows that exceed `max_retries` |

---

## Operations

### Starting / stopping servers

```bash
# Start all enabled detectors (reads servers.yaml)
python -m data_miner.auto_annotation_v4.model_servers.serve \
    --config data_miner/auto_annotation_v4/configs/servers.yaml --all

# Start a specific one
python -m data_miner.auto_annotation_v4.model_servers.serve \
    --model grounding_dino --port 3001 --gpu 0 --max-batch-size 4

# Stop everything
pkill -TERM -f data_miner.auto_annotation_v4.model_servers.serve
pkill -TERM -f data_miner.auto_annotation_v4        # the pipeline process
```

The `--gpu` arg accepts `"0"`, `"cuda:0"`, `"4"`, or `"0,1"` for
multi-GPU — `_parse_gpu_arg()` normalizes to the int list LitServe
expects.

### Monitoring a live run

```bash
# DB snapshot
sqlite3 output/auto_annotation_v4/my_job/pipeline.db \
    "SELECT stage, status, COUNT(*) FROM work_queue GROUP BY stage, status;"

# Tail WAL size — should stay under ~10 MB with wal_autocheckpoint=1000
watch -n5 'ls -la output/auto_annotation_v4/my_job/pipeline.db*'

# Failed items
sqlite3 output/auto_annotation_v4/my_job/pipeline.db \
    "SELECT stage, COUNT(*), MAX(last_attempt_at) FROM failures GROUP BY stage;"
```

The built-in `PipelineMonitor` logs aggregate progress every few seconds
(stage counts, stale claims recovered, WAL checkpoint summary).

### Tuning

Defaults in [configs/default.yaml](configs/default.yaml) `workers:` target
a 2-GPU box with 48 GB cards:

```yaml
workers:
  detect_per_model: 2   # async workers per enabled detector
  detect_merge: 2
  filter_count: 4
  evaluate_count: 6
  refine_count: 2
  finalize_count: 2
```

For 24 GB GPUs, drop `servers.detectors.grounding_dino.max_batch_size`
to `2–4` and `sam3_dart.max_batch_size` to `4`. The bench sweeper
[tests/bench_batch_sizes.py](tests/bench_batch_sizes.py) finds the
memory/throughput sweet spot per GPU before a scale run.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "job already in use" on startup | pidfile.lock held by running (or crashed) pipeline | `ps aux | grep auto_annotation_v4` — kill stale, or `rm pidfile.lock` if no process owns it |
| Images stuck in `PROCESSING` after crash | `lock_ttl` not yet elapsed | Wait `database.lock_ttl` seconds; monitor auto-resets to `PENDING` |
| `database is locked` in logs | Contention on WAL checkpoint | Already mitigated — checkpoint uses a separate short-lived connection with 5s busy_timeout |
| Work never advances past detect | Merge barrier not satisfied — one of the enabled detectors has no proposal row | Check `servers.detectors.*.enabled` matches what ran; use `runtime.detect_models` if intentionally subsetting |
| "config changed" warning on every run | Something mutable is hashed | Check [configs/loader.py](configs/loader.py)::`_HASH_FIELDS` — only the listed slices affect the hash |
| GDINO OOM on 24 GB | Batch too large | Lower `servers.detectors.grounding_dino.max_batch_size` |
| HTTP 503 storm from a server | Tenacity already retries 3× with exp backoff (1→16s); 4xx does not retry | Check server logs; the pipeline writes to `failures` only after 3 transient attempts |

---

## Testing

All tests run under pytest, no GPU required for Tier 1.

```bash
# Tier 1 — fast unit tests (~7s total)
pytest data_miner/auto_annotation_v4/tests/ -x

# Lifecycle scenarios (S2/S3/S5/S6/S7/S9 — simulated, no GPU)
pytest data_miner/auto_annotation_v4/tests/test_lifecycle_scenarios.py -v

# GPU-backed batching parity (requires model servers)
pytest data_miner/auto_annotation_v4/tests/test_gdino_v4_batching.py
pytest data_miner/auto_annotation_v4/tests/test_sam3_dart_v4_batching.py
```

See [docs/test_plan.md](docs/test_plan.md) for the 4-tier plan and
[docs/test_progress.md](docs/test_progress.md) for status of each test
and lifecycle scenario.

---

## Design references

- [docs/test_plan.md](docs/test_plan.md) — test tiers + coverage matrix
- [docs/test_progress.md](docs/test_progress.md) — per-test status, live run log
- [docs/discussion_1.md](docs/discussion_1.md) — filtering scores + routing rationale (carried forward from v3)
- [checkpoint.py](checkpoint.py) — SQL schema, atomic claim, save_and_forward
- [configs/enums.py](configs/enums.py) — every StrEnum in the system (single source of truth for routing keys, statuses, drop reasons)
- [configs/loader.py](configs/loader.py) — YAML merge order + scoped config hash
- [workers/http_retry.py](workers/http_retry.py) — shared tenacity policy used by detect/evaluate/refine
