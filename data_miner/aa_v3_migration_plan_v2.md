# aa_v4 — Clean-Room Pipeline Rewrite

## Context

v3 pipeline works but has structural issues: Redis Streams for work
distribution (every bug this year was in this layer), file-based checkpoints
(800k files at scale), free strings everywhere, model code tangled with
LitServe glue, configs scattered. v4 is a clean-room rewrite: same pipeline
logic, better architecture.

**Core changes:**
1. SQLite replaces both Redis Streams AND file checkpoints
2. models/ (pure inference) separated from model_servers/ (LitAPI wrappers)
3. All types enforced via StrEnum + Pydantic — zero free strings
4. configs/ folder with YAML + OmegaConf merge for all settings and contracts
5. Config change detection warns user before invalidating cached work

---

## Directory Structure

```
auto_annotation_v4/
├── __main__.py                       # CLI entry
├── cli.py                            # Arg parsing
├── pipeline.py                       # Orchestrator: CheckpointDB, no broker
├── checkpoint.py                     # SQLite: proposals + stages + work_queue + meta
├── output.py                         # YOLO labels, traces, review writer
├── prompt_manager.py                 # Versioned prompt loading
├── utils.py                          # Geometry, VLM parsing, image helpers
│
├── configs/                          # ALL typed definitions + YAML
│   ├── __init__.py                   # re-exports load_config, all classes
│   ├── enums.py                      # ALL StrEnums (Stage, WorkStatus, DetectorName, ...)
│   ├── settings.py                   # Pydantic config models (pipeline, filter, evaluate, ...)
│   ├── contracts.py                  # Pipeline data models (Candidate, DetectResult, ...)
│   ├── wire.py                       # HTTP wire contracts (DetectorRequest/Response, SAM3...)
│   ├── loader.py                     # load_config(), _load_base_config(), compute_config_hash()
│   ├── default.yaml                  # Pipeline: auto_accept, evaluate, filtering, workers, output
│   ├── database.yaml                 # SQLite: filename, lock_ttl, max_retries
│   ├── servers.yaml                  # Detector + VLM server endpoints
│   ├── class_config.yaml             # Class registry, eval groups, co-existence, refine rules
│   └── runtime.yaml                  # Defaults for runtime section (image_dir, stages, force_*)
│
├── models/                           # Pure inference — zero LitServe imports
│   ├── __init__.py
│   ├── base.py                       # Shared: clamp, normalize, device/dtype helpers
│   ├── grounding_dino.py             # GDINO: load, prepare, infer, postprocess
│   ├── falcon.py                     # Falcon: load, prepare, infer, postprocess
│   ├── sam3_dart.py                  # SAM3-DART: proposal + refine modes
│   ├── sam3_dart_batch.py            # DART batch predictor (from dart_batch.py)
│   ├── owlvit2.py                    # OWLv2: load, prepare, infer, postprocess
│   ├── omdet_turbo.py                # OmDet-Turbo: load, prepare, infer, postprocess
│   └── omdet_turbo_batch.py          # OmDet batch predictor (from omdet_batch.py)
│
├── model_servers/                    # LitAPI wrappers — thin routing only
│   ├── __init__.py
│   ├── base.py                       # DetectorServerBase(LitAPI)
│   ├── grounding_dino.py             # GDINOApi — calls models.grounding_dino
│   ├── falcon.py                     # FalconApi — calls models.falcon
│   ├── sam3_dart.py                  # SAM3DartApi — dual mode dispatch
│   ├── owlvit2.py                    # OWLv2Api — calls models.owlvit2
│   ├── omdet_turbo.py                # OmDetTurboApi — calls models.omdet_turbo
│   └── serve.py                      # Unified entry: --model NAME --port N --gpu DEV
│
├── workers/
│   ├── __init__.py
│   ├── base.py                       # StageWorker: claim → process → save_and_forward
│   ├── submitter.py                  # JobSubmitter: register images, queue work
│   └── monitor.py                    # PipelineMonitor: SQL progress, stale recovery
│
├── stages/
│   ├── detect.py                     # DetectWorker (Phase 1) / DetectMergeWorker (Phase 2)
│   ├── detect_model.py               # Phase 2: DetectModelWorker (per-model)
│   ├── evaluate.py                   # EvaluateWorker
│   ├── refine.py                     # RefineWorker
│   └── finalize.py                   # FinalizeWorker
│
├── viewer/
│   ├── app.py                        # FastAPI — reads SQLite (not files)
│   └── static/
│
├── prompts/                          # Versioned prompt templates (shared with v3)
│   ├── active -> v1/
│   └── v1/
│
└── tests/
    ├── test_checkpoint_db.py         # Unit: SQLite operations, atomicity, concurrency
    ├── test_enums.py                 # Unit: all enums serialize/deserialize correctly
    └── test_pipeline_e2e.py          # E2E: full pipeline on sample data
```

---

## StrEnum Registry — Zero Free Strings

Every string that appears as a field value, method parameter, DB column
value, or routing key MUST be a StrEnum member. Coding agents should never
write a bare string literal for any of these.

All enums live in `configs/enums.py`:

```python
from enum import StrEnum

# ── Pipeline stages ───────────────────────────────────────────────────
class Stage(StrEnum):
    """Pipeline stage identifiers. Used in work_queue, stages table, routing."""
    DETECT = "detect"
    EVALUATE = "evaluate"
    REFINE = "refine"
    FINALIZE = "finalize"
    DONE = "done"                      # terminal — no work_queue entry

STAGE_ORDER: list[Stage] = [Stage.DETECT, Stage.EVALUATE, Stage.REFINE, Stage.FINALIZE]

# ── Work queue status ─────────────────────────────────────────────────
class WorkStatus(StrEnum):
    """work_queue.status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"

# ── Image-level tracking ─────────────────────────────────────────────
class ImageStatus(StrEnum):
    """image_meta.status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

# ── Detector identity ────────────────────────────────────────────────
class DetectorName(StrEnum):
    """Canonical detector identifiers. Keys in servers.yaml detectors dict."""
    GROUNDING_DINO = "grounding_dino"
    FALCON = "falcon"
    SAM3 = "sam3"
    SAM3_DART = "sam3_dart"
    OWLVIT2 = "owlvit2"
    OMDET_TURBO = "omdet_turbo"

    @property
    def is_sam3_family(self) -> bool:
        return self in (DetectorName.SAM3, DetectorName.SAM3_DART)

# ── Candidate lifecycle ──────────────────────────────────────────────
class CandidateStatus(StrEnum):
    """Candidate.status values through pipeline."""
    PROPOSED = "proposed"
    FILTERED_OUT = "filtered_out"
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_EVALUATION = "needs_evaluation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REFINED = "refined"

# ── Bbox quality (VLM verdict) ───────────────────────────────────────
class BboxQuality(StrEnum):
    GOOD = "good"
    NEEDS_EXPANSION = "needs_expansion"
    TOO_LOOSE = "too_loose"
    BAD = "bad"

# ── Final disposition ────────────────────────────────────────────────
class FinalAction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"

# ── Refine stage ─────────────────────────────────────────────────────
class RefineAction(StrEnum):
    """VLM refine prompt response: skip (no extension) or propose."""
    SKIP = "skip"
    PROPOSE = "propose"

class RefineOutcome(StrEnum):
    """Per-prompt step outcome in refine inner loop."""
    SKIPPED = "skipped"
    SAM_NO_MASK = "sam_no_mask"
    PRESENCE_FAILED = "presence_failed"
    MERGE_FAILED = "merge_failed"
    MERGED = "merged"
    VLM_ERROR = "vlm_error"
    SAM_ERROR = "sam_error"

class Verdict(StrEnum):
    """Tri-state verdict used in evaluate and refine adjudication."""
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"

class BboxSource(StrEnum):
    """Which bbox to use after refinement."""
    REFINED = "refined"
    ORIGINAL = "original"

# ── Finalize drop reasons ────────────────────────────────────────────
class DropReason(StrEnum):
    GEOMETRIC_FILTER = "geometric_filter"
    DEDUP = "dedup"
    CROSS_CLASS = "cross_class"
    PER_CLASS_CAP = "per_class_cap"
    REJECTED_UPSTREAM = "rejected_upstream"

# ── Class tier ───────────────────────────────────────────────────────
class ClassTier(int, enum.Enum):
    """Class tier levels. Tier 1 = auto-acceptable, higher = always VLM."""
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
```

**Usage rule**: Every Pydantic model field, method parameter, DB query value,
and dict key that represents one of these concepts MUST use the enum type,
not `str` or `Literal[...]`. Example:

```python
# WRONG
class PromptStepResult(BaseModel):
    action: Literal["skip", "propose"]
    outcome: Literal["skipped", "sam_no_mask", ...]

# RIGHT
class PromptStepResult(BaseModel):
    action: RefineAction
    outcome: RefineOutcome
```

---

## SQLite Schema

One database per job at `{job_dir}/pipeline.db`. Per-job scoping eliminates
cross-job contamination by construction — no `job_id` column needed on work
tables.

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

-- ── Job-level metadata (single row) ──────────────────────────────────
CREATE TABLE job_info (
    job_id          TEXT NOT NULL,
    image_dir       TEXT,
    config_hash     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    created_at      REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running'  -- ImageStatus enum values
);

-- ── Per-image tracking (replaces meta.json) ──────────────────────────
CREATE TABLE image_meta (
    image_id         TEXT PRIMARY KEY,
    image_path       TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',  -- ImageStatus
    stages_completed TEXT NOT NULL DEFAULT '[]',       -- JSON list of Stage values
    config_hash      TEXT NOT NULL DEFAULT '',
    prompt_version   TEXT NOT NULL DEFAULT '',
    total_timing_ms  REAL NOT NULL DEFAULT 0.0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);

-- ── Per-model raw proposals ──────────────────────────────────────────
CREATE TABLE proposals (
    image_id    TEXT NOT NULL,
    model       TEXT NOT NULL,             -- DetectorName enum value
    data        TEXT NOT NULL,             -- JSON: ProposalResult
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, model)
);
CREATE INDEX idx_proposals_model ON proposals(model);

-- ── Per-stage results ────────────────────────────────────────────────
CREATE TABLE stages (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,             -- Stage enum value
    data        TEXT NOT NULL,             -- JSON: DetectResult | EvaluateResult | ...
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX idx_stages_stage ON stages(stage);

-- ── Work distribution (replaces Redis Streams) ──────────────────────
CREATE TABLE work_queue (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,             -- Stage enum value
    status      TEXT NOT NULL DEFAULT 'pending',  -- WorkStatus
    worker_id   TEXT,
    score       REAL NOT NULL,             -- submission timestamp (FIFO)
    claimed_at  REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX idx_wq_claim ON work_queue(stage, status, score);
CREATE INDEX idx_wq_stale ON work_queue(status, claimed_at);

-- ── Failed items (replaces dead-letter stream) ──────────────────────
CREATE TABLE failures (
    image_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,         -- Stage enum value
    attempts        INTEGER NOT NULL DEFAULT 1,
    last_error      TEXT,
    last_attempt_at REAL,
    PRIMARY KEY (image_id, stage)
);
```

---

## Config Change Warning

When a user re-runs a job and the `config_hash` differs from `job_info`,
the pipeline MUST warn and ask for confirmation before proceeding:

```python
# pipeline.py — inside run(), after db.connect()
async def _check_config_continuity(self):
    """Warn user if config changed since last run on this job_dir."""
    stored = await self.db.get_job_info()
    if stored is None:
        return  # fresh job, no warning needed

    current_hash = compute_config_hash(self.config, self.config.prompts_dir)
    if stored.config_hash == current_hash:
        return  # same config, no warning

    logger.warning(
        "Config has changed since last run on this job.\n"
        "  Stored hash: %s\n"
        "  Current hash: %s\n"
        "Affected images will have stale stages invalidated "
        "and re-processed from the first changed stage.",
        stored.config_hash, current_hash,
    )

    if sys.stdin.isatty():
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            logger.info("Aborted by user.")
            sys.exit(0)
    else:
        logger.info("Non-interactive mode — proceeding with invalidation.")
```

The per-image `should_run_stage()` logic then handles the actual
invalidation (same as v3 — compare config_hash, clear_downstream if
mismatch).

---

## Phase 1 — Core SQLite Migration + Clean Architecture

**Goal**: Same pipeline behavior as v3, new storage backend, clean project
structure. Redis removed, file checkpoints replaced by SQLite. All types
enforced via StrEnum.

### 1.1 configs/ — Typed definitions + YAML

**`configs/enums.py`** — All StrEnums (see registry above). Single source
of truth for every enumerated value in the project.

**`configs/settings.py`** — Pydantic config models. Moved from v3 `config.py`:
- `DetectorConfig`, `VLMConfig`, `ServersConfig`
- `ClassConfig` (dict-keyed, tier as ClassTier enum)
- `AutoAcceptConfig`, `EvaluateConfig`, `FilterConfig`, `IouDedupConfig`
- `EvaluationGroupConfig`, `CoExistenceConfig`
- `RefineRulesConfig`, `RefineRuleConfig`, `RefinePromptConfig`, `MergeRulesConfig`
- `DatabaseConfig` (NEW: filename, lock_ttl, max_retries)
- `WorkersConfig`, `OutputConfig`, `RuntimeConfig`
- `AutoAnnotationV4Config` (top-level, references all above)

Key changes from v3:
- `RedisConfig` deleted
- `DatabaseConfig` added
- `RuntimeConfig` adds: `stages`, `force_stages`, `force_rerun` (Phase 2 fields, defined now with defaults)
- All `str` fields that represent enums → use enum types
- `class_registry` remains dict-keyed (from v3 YAML redesign)

**`configs/contracts.py`** — Pipeline data models. Moved from v3 `contracts.py`:
- `BoundingBox`, `Candidate`, `StageMessage` (simplified, no forward())
- `ProposalResult`, `DetectResult`, `DetectRouting`
- `VLMVerdict`, `EvaluateResult`, `RefinementInstruction`, `RefinementResult`, `RefineResult`
- `FinalAnnotation`, `FinalizeDrop`, `FinalizeResult`
- `MetaCheckpoint` (status: ImageStatus, stages_completed: list[Stage])
- All `Literal[...]` fields replaced with StrEnum types

**`configs/wire.py`** — HTTP wire contracts (model server ↔ pipeline):
- `DetectorRequest`, `DetectorResponse`
- `SAM3RefineRequest`, `SAM3RefineResponse`
- `PreparedInput`, `RawPrediction` (internal server handoff, frozen)

**`configs/loader.py`** — Config loading + hashing:
- `_load_base_config()` — merges YAML files via OmegaConf
- `load_config(user_config, overrides)` — base → user → CLI dotlist → validate
- `compute_config_hash()` — SHA256 of config + prompt files

**YAML split:**

| File | Contents |
|---|---|
| `default.yaml` | detect_classes, auto_accept, evaluate, filtering, workers, output, prompts_dir |
| `database.yaml` | filename, lock_ttl, max_retries |
| `servers.yaml` | detectors (per-model: enabled, port, gpu, batch, script), vlm |
| `class_config.yaml` | class_registry, evaluation_groups, co_existence, refine_rules |
| `runtime.yaml` | image_dir, image_paths, job_id, log_level, stages, force_stages, force_rerun |

All merged by `_load_base_config()` in deterministic order.

### 1.2 models/ — Pure inference

Each model file exports a class with a standard interface:

```python
class BaseDetectorModel(ABC):
    """Pure inference — no LitServe, no HTTP, no checkpoint awareness.

    Subclasses implement model-specific loading, preprocessing, forward
    pass, and postprocessing. All methods are synchronous (GPU-bound).
    """

    @abstractmethod
    def load(self, device: str, model_id: str, **options) -> None:
        """One-time model + processor initialization."""

    @abstractmethod
    def prepare(self, image: Image.Image, prompts: list[str],
                threshold: float | None) -> PreparedInput:
        """Preprocess image + prompts for inference."""

    @abstractmethod
    def infer(self, prepared: PreparedInput) -> RawPrediction:
        """Run model forward pass. Returns raw tensors."""

    @abstractmethod
    def postprocess(self, raw: RawPrediction) -> DetectorResponse:
        """Convert raw tensors to normalized boxes/scores/labels."""
```

| File | Model class | Notes |
|---|---|---|
| `grounding_dino.py` | `GDINOModel` | Per-prompt loop (multi-class degrades) |
| `falcon.py` | `FalconModel` | Native multi-class, RLE mask decode |
| `sam3_dart.py` | `SAM3DartModel` | Dual mode: proposal + refine. Uses DART batch predictor |
| `sam3_dart_batch.py` | `Sam3MultiClassPredictorBatch` | Moved from `dart_batch.py` |
| `owlvit2.py` | `OWLv2Model` | Native multi-class |
| `omdet_turbo.py` | `OmDetTurboModel` | Uses batch predictor |
| `omdet_turbo_batch.py` | `OmDetTurboBatchPredictor` | Moved from `omdet_batch.py` |
| `base.py` | `BaseDetectorModel` | ABC + shared helpers (clamp, normalize) |

### 1.3 model_servers/ — LitAPI wrappers

Thin routing layer. Each server file imports its model class and wires
it to LitServe hooks:

```python
class GDINOApi(DetectorServerBase):
    """LitAPI wrapper for GroundingDINO. Delegates all inference to GDINOModel.

    setup() → model.load()
    decode_request() → model.prepare()
    predict() → model.infer()
    encode_response() → model.postprocess()
    """

    def setup(self, device: str) -> None:
        self.model = GDINOModel()
        self.model.load(device, self.model_id)

    # ... thin delegation only, zero inference logic
```

**`serve.py` — Unified server entry point:**

```python
"""Launch any detector model server from CLI.

Usage:
    # Single model
    python -m data_miner.auto_annotation_v4.model_servers.serve \\
        --model grounding_dino --port 3001 --gpu cuda:0

    # All enabled models from config
    python -m data_miner.auto_annotation_v4.model_servers.serve \\
        --config configs/servers.yaml --all

    # Specific models from config
    python -m data_miner.auto_annotation_v4.model_servers.serve \\
        --config configs/servers.yaml --models grounding_dino sam3_dart
"""

# Registry mapping DetectorName → LitAPI class
_SERVER_REGISTRY: dict[DetectorName, type[DetectorServerBase]] = {
    DetectorName.GROUNDING_DINO: GDINOApi,
    DetectorName.FALCON: FalconApi,
    DetectorName.SAM3_DART: SAM3DartApi,
    DetectorName.OWLVIT2: OWLv2Api,
    DetectorName.OMDET_TURBO: OmDetTurboApi,
}

def launch_server(name: DetectorName, port: int, gpu: str, **kwargs):
    """Launch a single model server as a LitServe instance."""
    api_cls = _SERVER_REGISTRY[name]
    api = api_cls(model_id=kwargs.get("model_id"), **kwargs)
    server = ls.LitServer(api, devices=gpu, ...)
    server.run(port=port)
```

### 1.4 checkpoint.py — SQLite (replaces files + messaging)

`CheckpointDB` class. All method parameters and return values use StrEnums.

**Key methods (all async):**

```python
class CheckpointDB:
    """SQLite-backed checkpoint + work distribution.

    One instance per pipeline run. All state for a single job in one file.
    Replaces both CheckpointManager (file-based) and RedisMessageBroker.

    Args:
        db_path: Path to SQLite database file (created if missing).
        lock_ttl: Seconds before stale processing claims are recovered.
        max_retries: Attempts before an image/stage moves to failures table.
    """

    async def connect(self) -> None
    async def close(self) -> None

    # ── Job info ──────────────────────────────────────────────────────
    async def save_job_info(self, job_id: str, image_dir: str | None,
                            config_hash: str, prompt_version: str) -> None
    async def get_job_info(self) -> JobInfo | None

    # ── Image registration ────────────────────────────────────────────
    async def register_image(self, image_id: str, image_path: str) -> None
        """INSERT OR IGNORE into image_meta. Called by submitter."""

    async def resolve_image_path(self, image_id: str) -> str
        """SELECT image_path FROM image_meta. Called by base worker."""

    # ── Stage checkpoints ─────────────────────────────────────────────
    async def save_stage(self, image_id: str, stage: Stage,
                         data: BaseModel, config_hash: str) -> None
    async def load_stage(self, image_id: str, stage: Stage,
                         model_class: type[T]) -> T | None
    async def stage_exists(self, image_id: str, stage: Stage) -> bool

    # ── Proposals ─────────────────────────────────────────────────────
    async def save_proposal(self, image_id: str, model: DetectorName,
                            data: BaseModel) -> None
    async def load_proposal(self, image_id: str, model: DetectorName,
                            model_class: type[T]) -> T | None
    async def proposal_exists(self, image_id: str, model: DetectorName) -> bool

    # ── Resume logic ──────────────────────────────────────────────────
    async def should_run_stage(self, image_id: str, stage: Stage,
                               config_hash: str) -> bool
        """Check if stage needs to run. Invalidates downstream on hash mismatch."""

    async def clear_downstream(self, image_id: str, from_stage: Stage) -> None
        """Delete from_stage + all subsequent from stages, work_queue, image_meta."""

    async def all_stages_complete(self, image_id: str) -> bool
        """True if image_meta.status == ImageStatus.COMPLETE."""

    # ── Work distribution ─────────────────────────────────────────────
    async def add_work(self, stage: Stage | str, image_id: str,
                       score: float | None = None) -> None
        """INSERT OR IGNORE into work_queue. score defaults to time.time()."""

    async def claim_work(self, stage: Stage | str,
                         worker_id: str) -> str | None
        """Atomic claim: UPDATE...WHERE rowid=(SELECT...LIMIT 1) RETURNING.
        Returns image_id or None if queue empty."""

    async def release_work(self, image_id: str, stage: Stage | str) -> None
        """Reset to pending (on CancelledError / graceful shutdown)."""

    # ── Atomic save + forward (the key correctness win) ───────────────
    async def save_and_forward(
        self, image_id: str, stage: Stage, data: BaseModel,
        config_hash: str, next_stage: Stage | None, timing_ms: float,
    ) -> None:
        """One atomic commit: save result + update meta + complete work + queue next.
        If process crashes between any step, nothing commits."""

    async def fail_work(self, image_id: str, stage: Stage | str,
                        error: str) -> None
        """Increment attempts. Move to failures table if max_retries exceeded."""

    # ── Monitoring ────────────────────────────────────────────────────
    async def recover_stale(self, lock_ttl: int, max_retries: int) -> int
        """Reset processing claims older than lock_ttl. Returns count recovered."""

    async def progress_summary(self) -> dict
        """Aggregate counts across all tables for monitor display."""

    async def queue_counts(self) -> dict[str, dict[WorkStatus, int]]
        """Per-stage counts by status."""
```

### 1.5 workers/base.py — StageWorker rewrite

```python
class StageWorker(ABC):
    """Base class for all pipeline stage workers.

    Subclasses implement:
      - process(msg) → stage result (Pydantic model)
      - _resolve_next_stage(result) → Stage enum for routing

    The base class handles: work claiming, checkpoint skip/resume,
    atomic save+forward, retry/failure, graceful shutdown.
    """
    stage: Stage                       # MUST override with Stage enum member
    max_retries: int = 3

    def __init__(self, config: AutoAnnotationV4Config, db: CheckpointDB,
                 worker_id: str | None = None, job_id: str | None = None):
        ...

    @abstractmethod
    async def process(self, msg: StageMessage) -> BaseModel:
        """Process one image. Return stage result (DetectResult, etc.)."""

    @abstractmethod
    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        """Determine next stage based on result. Called by base after process()."""

    async def run(self) -> None:
        """Main loop: claim → skip-check → process → save_and_forward."""
        ...
```

**Routing is per-stage** — each subclass implements `_resolve_next_stage()`:

```python
# stages/detect.py
class DetectWorker(StageWorker):
    stage = Stage.DETECT

    def _resolve_next_stage(self, result: DetectResult) -> Stage:
        """Route based on detection results.
        - Has candidates needing VLM eval → EVALUATE
        - All auto-accepted, some refinable → REFINE
        - All auto-accepted, none refinable → FINALIZE
        """
        if result.routing.needs_evaluation:
            return Stage.EVALUATE
        if any(c.class_name in self._refine_classes
               for c in result.candidates
               if c.candidate_id in result.routing.auto_accepted):
            return Stage.REFINE
        return Stage.FINALIZE

# stages/evaluate.py
class EvaluateWorker(StageWorker):
    stage = Stage.EVALUATE

    def _resolve_next_stage(self, result: EvaluateResult) -> Stage:
        """Route: any accepted/review candidate refinable → REFINE, else FINALIZE."""
        refinable = set(self.config.refine_rules.classes.keys())
        if any(cid for cid in (result.accepted + result.review)
               if self._post_relabel_class(cid) in refinable):
            return Stage.REFINE
        return Stage.FINALIZE

# stages/refine.py — always FINALIZE
# stages/finalize.py — always DONE
```

### 1.6 workers/submitter.py, monitor.py, pipeline.py

Same as previously planned. Key changes:
- Submitter: `db.register_image()` + `db.add_work(Stage.DETECT, ...)`
- Monitor: `db.progress_summary()` + `db.recover_stale()`
- Pipeline: `CheckpointDB` replaces broker. Config change warning on startup.
  No Redis imports. `_create_workers()` passes `db` to all workers.

### 1.7 viewer/app.py — SQLite reads

Replace all `_load_json(ckpt_dir / image_id / "detect.json")` with
`SELECT data FROM stages WHERE image_id=? AND stage=?`. Use sync `sqlite3`
(read-only, WAL allows concurrent reads while pipeline writes).

### 1.8 Cleanup

- `workers/messaging.py` — NOT copied to v4
- `docker-compose-vllm.yml` — remove Redis service
- `v3/servers/` — replaced by `v4/models/` + `v4/model_servers/`
- Do NOT remove `redis` from `pyproject.toml` (other modules may use it)
- Add `aiosqlite>=0.19.0` to dependencies

### Phase 1 implementation order

| Step | Files | Depends on | Parallelizable |
|---|---|---|---|
| 1 | `configs/enums.py` | — | Yes |
| 2 | `configs/wire.py` | Step 1 | Yes |
| 3 | `configs/contracts.py` | Step 1 | Yes |
| 4 | `configs/settings.py` + YAMLs | Step 1 | Yes |
| 5 | `configs/loader.py` + `configs/__init__.py` | Steps 1-4 | No |
| 6 | `checkpoint.py` | Steps 1, 3 | No |
| 7 | `models/base.py` + all model files | Step 2 | Yes (per model) |
| 8 | `model_servers/base.py` + all servers + `serve.py` | Steps 2, 7 | Yes (per server) |
| 9 | `workers/base.py` | Steps 1, 3, 6 | No |
| 10 | `workers/submitter.py` | Step 6 | Yes |
| 11 | `workers/monitor.py` | Step 6 | Yes |
| 12 | `stages/detect.py` | Steps 6, 9 | Yes |
| 13 | `stages/evaluate.py` | Steps 6, 9 | Yes |
| 14 | `stages/refine.py` | Steps 6, 9 | Yes |
| 15 | `stages/finalize.py` | Steps 6, 9 | Yes |
| 16 | `pipeline.py` | Steps 6, 9-15 | No |
| 17 | `viewer/app.py` | Step 6 | Yes |
| 18 | `output.py`, `utils.py`, `prompt_manager.py` | — | Copy + enum updates |
| 19 | `cli.py`, `__main__.py` | Step 16 | No |
| 20 | Tests | Steps 1-19 | Last |

---

## Phase 2 — Per-Model Detect + Granular Control

**Goal**: Split monolithic detect into per-model workers + merge. Add
runtime config for stage/model selection.

### New runtime config fields

```yaml
# runtime.yaml additions
runtime:
  stages: [detect, evaluate, refine, finalize]  # which stages to run
  force_stages: []           # ignore checkpoints for these stages
  detect_models: []          # empty = all enabled; subset to target
  force_detect_models: []    # delete proposals + re-run these models
  force_rerun: false         # nuclear: clear all checkpoints
```

### DetectModelWorker (per-model)

```python
class DetectModelWorker(StageWorker):
    """Per-model detect worker. Claims from work_queue stage='detect:{model}'.

    Calls one model server, saves proposal, checks barrier.
    When all enabled models have proposals → queues detect:merge.
    """
    # stage is dynamic: set per-instance as f"detect:{model_name}"

    async def process(self, msg: StageMessage) -> ProposalResult:
        """Call single model server, save proposal, check barrier."""
        ...

    async def _check_barrier(self, image_id: str) -> bool:
        """All enabled models have proposals? → queue merge."""
        return await self.db.barrier_ready(image_id, self._enabled_models)
```

### DetectMergeWorker

```python
class DetectMergeWorker(StageWorker):
    """Merge worker. Claims from work_queue stage='detect:merge'.

    Loads all per-model proposals, runs full filtering pipeline,
    saves detect.json, routes to next stage.
    """
    stage = "detect:merge"  # special compound stage

    async def process(self, msg: StageMessage) -> DetectResult:
        """Load proposals, filter, route."""
        ...
```

### Smart submitter

Checks what's cached before queuing. Only queues work that's actually needed.
See detailed flow in "How Resume / Force / Skip Works" section below.

### Worker count config

```yaml
workers:
  detect_per_model: 2     # workers per enabled detector
  detect_merge: 2         # merge workers
  evaluate_count: 6
  refine_count: 2
  finalize_count: 2
```

### Phase 2 files

| File | Action |
|---|---|
| `stages/detect_model.py` | NEW |
| `stages/detect.py` | REWRITE → DetectMergeWorker |
| `workers/submitter.py` | MODIFY — smart submission |
| `configs/settings.py` | MODIFY — RuntimeConfig + WorkersConfig fields |
| `configs/runtime.yaml` | MODIFY — new fields |
| `pipeline.py` | MODIFY — per-model workers, conditional stage launch |
| `checkpoint.py` | MODIFY — barrier_ready, delete_proposal, delete_stage |

---

## Phase 3 — Asyncio Improvements

**Goal**: Better error handling, connection management, concurrency control.

| Improvement | File | Detail |
|---|---|---|
| Session-per-worker | `workers/base.py` | `aiohttp.ClientSession` in `run()` scope, shared across iterations |
| Model semaphores | `pipeline.py` + `stages/detect.py` | `asyncio.Semaphore(max_batch_size)` per model, prevents server overload |
| `asyncio.timeout` | All stages | 60s detect, 120s VLM evaluate, 60s VLM refine, 30s SAM refine |
| Two-phase shutdown | `pipeline.py` | `w.stop()` (cooperative, 30s) → `task.cancel()` (forced) |
| `run_in_executor` | `stages/detect.py` | CPU-bound filtering offloaded when >100 candidates |
| CancelledError handling | `workers/base.py` | Separated from Exception, always re-raised, releases claimed work |

---

## How Resume / Force / Skip Works

### Normal crash recovery

Pipeline crashes. On restart, same command:
1. `pipeline._check_config_continuity()` — config unchanged → no warning
2. Submitter scans images: `all_stages_complete()` → skips done images
3. For incomplete: `INSERT OR IGNORE INTO work_queue` — no-op for items
   already queued (they survived in DB from crashed run)
4. Monitor's `recover_stale()` finds `status=PROCESSING` older than
   `lock_ttl` → resets to `PENDING`
5. Workers claim recovered items and reprocess
6. `save_and_forward` is atomic — no "half-done" state possible

### Skip stages (run evaluate onward)

```bash
python -m ... runtime.stages="[evaluate,refine,finalize]"
```
Submitter skips detect. For each image, checks `stage_exists(image_id, Stage.DETECT)`.
If no detect result → warns and skips image. Otherwise queues for evaluate.
No detect workers launched.

### Force re-run specific stages

```bash
python -m ... runtime.force_stages="[evaluate]"
```
Submitter calls `clear_downstream(image_id, Stage.EVALUATE)` which:
```sql
DELETE FROM stages WHERE image_id=? AND stage IN ('evaluate','refine','finalize');
DELETE FROM work_queue WHERE image_id=? AND stage IN ('evaluate','refine','finalize');
UPDATE image_meta SET stages_completed='["detect"]', status='running';
```
Then queues for evaluate. Detect results cached.

### Force re-run one detect model

```bash
python -m ... runtime.force_detect_models="[grounding_dino]"
```
Deletes GDINO proposals + detect.json + downstream. Re-queues GDINO only.
SAM3-DART/Falcon proposals cached. After GDINO re-runs, barrier triggers merge.

### Config change between runs

`pipeline._check_config_continuity()` detects hash mismatch → warns user →
asks confirmation. If confirmed, per-image `should_run_stage()` handles
invalidation (compares per-stage config_hash, clears downstream on mismatch).

### Nuclear re-run

```bash
python -m ... runtime.force_rerun=true
```
Submitter calls `clear_image()` for every image. All tables wiped per-image.

---

## Verification

### Phase 1

| Test | What it proves |
|---|---|
| Unit: all enums round-trip through JSON + SQLite | StrEnum serialization correct |
| Unit: CheckpointDB save/load/claim/forward | SQL correctness, atomic commits |
| Unit: claim_work with 5 concurrent coroutines | No duplicate claims |
| Unit: save_and_forward crash simulation | Rollback on partial commit |
| Unit: should_run_stage with hash mismatch | Downstream invalidation works |
| Unit: config change warning triggers | _check_config_continuity logic |
| Integration: model load + infer (per model) | models/ separation correct |
| Integration: serve.py launches server | model_servers/ wiring correct |
| E2E: full pipeline on fl_pj_sample | Same output as v3 |
| E2E: kill mid-detect, restart | Resume from checkpoints |
| E2E: run same config twice | All cached, zero work |
| Viewer: all endpoints return data | SQLite reads work |

### Phase 2

| Test | What it proves |
|---|---|
| `stages=[detect]` only | Only detect workers run |
| `force_detect_models=[grounding_dino]` | Only GDINO re-runs, others cached |
| `stages=[evaluate,refine,finalize]` | Detect cached, rest runs |
| 3 model workers + barrier | Merge triggers correctly |
| `force_rerun=true` | All state cleared, full re-run |

### Phase 3

| Test | What it proves |
|---|---|
| Model server timeout → retry | Timeout handling works |
| Ctrl+C during processing | Graceful shutdown, no orphaned claims |
| 10 workers, 2 servers | Semaphore prevents overload |

---

## Code Style Rules for Implementation

1. **Zero free strings** — every field value, routing key, DB value is a StrEnum member
2. **Pydantic everywhere** — all data crossing module boundaries is a typed model
3. **Docstrings on every public method** — describe what it does, args, returns,
   and any side effects. Coding agents reading the docstring should understand
   the method without reading the body.
4. **No speculative abstractions** — build what's needed, not what might be needed
5. **Comments explain WHY, not WHAT** — the code shows what, comments explain
   non-obvious decisions
6. **Imports from configs/** — never import enums/models from deep paths;
   import from `configs` package (`from .configs import Stage, DetectResult, ...`)

## Out of Scope

- Web UI for multi-job management
- Distributed multi-machine (SQLite is single-node)
- v3 checkpoint migration script (v3 jobs finish on v3)
- Priority queues
- Auto-scaling workers
- Removing `redis` from project-wide `pyproject.toml`
