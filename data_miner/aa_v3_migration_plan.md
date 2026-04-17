# aa_v3 — Migration Plan: Redis Streams + File Checkpoints → SQLite

## Current state (what exists today)

```
auto_annotation_v3/
├── __main__.py                    # CLI entry point
├── cli.py                         # Argument parsing
├── pipeline.py                    # Orchestrator: launches workers, Redis broker, monitor
├── config.py                      # AutoAnnotationV3Config + RedisConfig + compute_config_hash
├── contracts.py                   # StageMessage, DetectResult, EvaluateResult, etc.
├── checkpoint.py                  # File-based: {base_dir}/{image_id}/{stage}.json
├── output.py                      # YOLO labels, traces, review queue writer
├── prompt_manager.py              # Versioned prompt loading
├── compare.py                     # A/B comparison across jobs
├── utils.py                       # Geometry, VLM parsing, image helpers
├── configs/default.yaml           # Default pipeline config
├── servers/                       # LitServe model servers (GDINO, Falcon, SAM3, OWLv2)
│   ├── serve_gdino.py
│   ├── serve_falcon.py
│   ├── serve_sam3.py
│   ├── serve_owlvit2.py
│   ├── serve_config.yaml
│   └── launch_all.py
├── workers/
│   ├── messaging.py               # RedisMessageBroker (Streams + consumer groups)
│   ├── base.py                    # StageWorker: reads from Streams, checkpoint skip, retry, dead-letter
│   ├── submitter.py               # JobSubmitter: writes StageMessages to detect stream
│   └── monitor.py                 # PipelineMonitor: polls stream lengths + pending counts
├── stages/
│   ├── detect.py                  # DetectWorker: HTTP to model servers, filter, route
│   ├── evaluate.py                # EvaluateWorker: VLM classification + quality
│   ├── refine.py                  # RefineWorker: SAM extension + VLM adjudication
│   └── finalize.py                # FinalizeWorker: canonical list, re-check invariants, outputs
└── prompts/
    ├── active -> v1/
    └── v1/
```

### Current data flow

```
submitter → broker.submit("detect", StageMessage)
                ↓
StageWorker.run():
  msg = broker.read(stage, consumer)     ← Redis XREADGROUP
  if checkpoint.exists(image_id, stage):
      skip, forward to next stage
  result = self.process(msg)             ← stage-specific logic
  checkpoint.save(image_id, stage, data) ← writes {image_id}/{stage}.json
  broker.submit(next_stage, msg.forward) ← Redis XADD
  broker.ack(stage, msg_id)              ← Redis XACK
```

### What goes wrong at scale

- **Redis Streams complexity**: consumer groups, XREADGROUP, dead-letter,
  SETID stomping, stale messages, cross-job contamination — every bug
  fixed this year was in this layer.
- **File-based checkpoints**: 100k images × 8 files = 800k files + 100k
  directories. Slow `find`, slow `rsync`, inode pressure.
- **Split-brain**: checkpoint truth in filesystem, work queue truth in Redis.
  Save to checkpoint then crash before XADD → image stranded.
- **Redis dependency**: extra service in supervisord, extra connection,
  extra failure mode.

---

## Target state

**One SQLite database** replaces both Redis Streams (work distribution) and
the file-based checkpoint system (state storage). All state in one file,
one source of truth, atomic transitions.

```
auto_annotation_v3/
├── __main__.py                    # unchanged
├── cli.py                         # unchanged
├── pipeline.py                    # REWRITE: remove broker, use CheckpointDB
├── config.py                      # MODIFY: remove RedisConfig, add db_path
├── contracts.py                   # MODIFY: simplify StageMessage
├── checkpoint.py                  # REWRITE: SQLite tables (proposals, stages, work_queue, etc.)
├── output.py                      # unchanged
├── prompt_manager.py              # unchanged
├── compare.py                     # unchanged
├── utils.py                       # unchanged
├── configs/default.yaml           # MODIFY: remove redis section
├── servers/                       # unchanged (entire directory)
├── workers/
│   ├── messaging.py               # DELETE
│   ├── base.py                    # REWRITE: claim from work_queue, no broker
│   ├── submitter.py               # REWRITE: INSERT into work_queue, no broker
│   └── monitor.py                 # REWRITE: SQL queries, no Redis introspection
├── stages/
│   ├── detect.py                  # MODIFY: minor — process() signature unchanged,
│   ├── evaluate.py                #   but save_checkpoint + forward is now one
│   ├── refine.py                  #   atomic transaction instead of two systems
│   └── finalize.py                #
└── prompts/                       # unchanged
```

---

## File-by-file change spec

### DELETE: `workers/messaging.py`

The entire `RedisMessageBroker`, `StreamConfig`, and all Redis Streams
logic. 250+ lines removed. No replacement file — its responsibilities
are absorbed into `checkpoint.py`.

**What it did → where it goes:**

| `messaging.py` method | New home |
|----------------------|----------|
| `submit(stage, message)` | `CheckpointDB.add_work(stage, image_id)` |
| `read(stage, consumer, count, block_ms)` | `CheckpointDB.claim_work(stage, worker_id)` |
| `ack(stage, msg_id)` | Implicit — `claim_work` already atomically marks processing |
| `stream_length(stage)` | `CheckpointDB.queue_counts()` |
| `pending_count(stage)` | `CheckpointDB.queue_counts()` |
| `count_by_job(stage, job_id)` | Not needed — DB is per-job |
| `from_config(config)` | Not needed — DB path from config |
| `connect()` / `close()` | `CheckpointDB.connect()` / `close()` |

---

### REWRITE: `checkpoint.py`

**Current**: `CheckpointManager` with file-based storage.

| Current method | Change |
|---------------|--------|
| `save(image_id, stage, data)` | `INSERT OR REPLACE INTO stages` |
| `load(image_id, stage, model_class)` | `SELECT data FROM stages` → `json.loads` → `model_class.model_validate` |
| `exists(image_id, stage)` | `SELECT 1 FROM stages WHERE ...` |
| `should_run_stage(image_id, stage, config_hash)` | `SELECT config_hash FROM stages WHERE ...` + compare |
| `clear_stage_and_downstream(image_id, stage)` | `DELETE FROM stages WHERE stage IN (...)` |
| `save_proposals(image_id, model, data)` | `INSERT OR REPLACE INTO proposals` |
| `load_proposals(image_id, model)` | `SELECT data FROM proposals WHERE ...` |
| `proposal_exists(image_id, model)` | `SELECT 1 FROM proposals WHERE ...` |
| `_atomic_write(path, text)` | Gone — SQLite transactions replace tempfile+rename |
| `_image_dir(image_id)` | Gone — no per-image directories |
| `_stage_path(image_id, stage)` | Gone — no file paths |
| `_meta_path(image_id)` | `meta` is now a row in the `stages` table |

**New methods (work distribution, replaces messaging.py):**

| New method | SQL |
|-----------|-----|
| `add_work(stage, image_id, score)` | `INSERT OR IGNORE INTO work_queue` |
| `add_work_batch(stage, image_ids)` | `executemany INSERT` |
| `claim_work(stage, worker_id)` | `UPDATE ... WHERE rowid = (SELECT ... LIMIT 1) RETURNING` |
| `complete_work(image_id, stage)` | `UPDATE work_queue SET status='done'` |
| `forward_work(image_id, from_stage, to_stage)` | `complete` + `add_work` in one commit |
| `recover_stale(lock_ttl, max_retries)` | `UPDATE ... WHERE claimed_at < ? AND attempts < ?` |
| `queue_counts()` | `SELECT stage, status, COUNT(*) GROUP BY ...` |
| `progress_summary()` | Aggregate across proposals + stages + work_queue + failures |
| `barrier_ready(image_id, models)` | `SELECT COUNT(DISTINCT model) FROM proposals` |
| `flush_job()` | `DELETE FROM` all tables + `VACUUM` |

**Schema** (5 tables):

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE proposals (
    image_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, model)
);
CREATE INDEX idx_proposals_model ON proposals(model);

CREATE TABLE stages (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX idx_stages_stage ON stages(stage);

CREATE TABLE work_queue (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    worker_id   TEXT,
    score       REAL NOT NULL,
    claimed_at  REAL,
    attempts    INTEGER DEFAULT 0,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX idx_wq_claim ON work_queue(stage, status, score);
CREATE INDEX idx_wq_stale ON work_queue(status, claimed_at);

CREATE TABLE masks (
    image_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    data        BLOB NOT NULL,
    PRIMARY KEY (image_id, model)
);

CREATE TABLE failures (
    image_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,
    attempts        INTEGER DEFAULT 1,
    last_error      TEXT,
    last_attempt_at REAL,
    PRIMARY KEY (image_id, stage)
);
```

**Atomic save + forward** (the key correctness win):

```python
async def save_and_forward(self, image_id, stage, data, config_hash,
                           from_stage, to_stage):
    """Save checkpoint + complete current work + queue next — one commit."""
    payload = _serialize(data)
    now = time.time()
    await self._db.execute(
        "INSERT OR REPLACE INTO stages VALUES (?,?,?,?,?)",
        (image_id, stage, payload, config_hash, now))
    await self._db.execute(
        "UPDATE work_queue SET status='done' WHERE image_id=? AND stage=?",
        (image_id, from_stage))
    if to_stage and to_stage != "done":
        await self._db.execute(
            "INSERT OR IGNORE INTO work_queue VALUES (?,?,'pending',NULL,?,NULL,0)",
            (image_id, to_stage, now))
    await self._db.commit()
```

If the process crashes between any of these lines, nothing commits.
No stranded images. No split-brain.

---

### REWRITE: `workers/base.py`

**Current `StageWorker.run()` loop** (simplified):

```python
async def run(self):
    self._running = True
    while self._running:
        messages = await self.broker.read(self.stage, self.worker_id)
        if not messages:
            continue
        for msg_id, raw in messages:
            msg = StageMessage(**raw)
            if self.checkpoint.exists(msg.image_id, self.stage):
                if not self.checkpoint.should_run_stage(...):
                    await self._forward_and_ack(msg, msg_id)
                    continue
            try:
                result = await self.process(msg)
                self.checkpoint.update_meta(...)
                if result:
                    await self._forward_and_ack(result, msg_id)
            except Exception:
                attempt = msg.attempt + 1
                if attempt >= self.max_retries:
                    await self.broker.dead_letter(self.stage, msg_id)
                else:
                    await self.broker.submit(self.stage, msg.copy(attempt=attempt))
                    await self.broker.ack(self.stage, msg_id)
```

**New `StageWorker.run()` loop:**

```python
async def run(self):
    self._running = True
    while self._running:
        # claim replaces broker.read — atomic ZPOPMIN-equivalent
        claimed = await self.db.claim_work(self.stage, self.worker_id)
        if claimed is None:
            await asyncio.sleep(1)
            continue

        image_id, image_path = claimed

        try:
            # Checkpoint skip (same logic as today)
            if await self.db.stage_exists(image_id, self.stage):
                if not await self.db.should_run_stage(image_id, self.stage, self._config_hash):
                    next_stage = self._next_stage()
                    await self.db.forward_work(image_id, self.stage, next_stage)
                    continue

            # Build StageMessage from claimed data (no Redis envelope)
            msg = StageMessage(
                image_id=image_id,
                image_path=image_path,
                job_id=self.job_id,
                stage=self.stage,
            )

            result = await self.process(msg)

            # Atomic: save checkpoint + advance work queue
            next_stage = self._resolve_next_stage(result)
            await self.db.save_and_forward(
                image_id, self.stage, result,
                self._config_hash, self.stage, next_stage
            )

        except Exception as exc:
            # Retry: increment attempts via recover_stale or explicit re-queue
            logger.exception("Failed %s/%s", image_id, self.stage)
            await self.db.fail_work(image_id, self.stage, str(exc))
```

**What changes in the signature:**

| Current | New |
|---------|-----|
| `__init__(config, broker, checkpoint_mgr, ...)` | `__init__(config, db, ...)` |
| `self.broker` (RedisMessageBroker) | Gone |
| `self.checkpoint` (CheckpointManager) | `self.db` (CheckpointDB) |
| `self.save_checkpoint(image_id, stage, data)` | `self.db.save_stage(image_id, stage, data)` |
| `self.load_checkpoint(image_id, stage, cls)` | `self.db.load_stage(image_id, stage, cls)` |
| `self._forward_and_ack(msg, msg_id)` | `self.db.forward_work(image_id, from, to)` |
| `self.broker.dead_letter(stage, msg_id)` | `self.db.fail_work(image_id, stage, error)` |

---

### REWRITE: `workers/submitter.py`

**Current**: `JobSubmitter.submit_images()` calls `broker.submit("detect", msg.model_dump())` per image.

**New**: `JobSubmitter.submit_images()` does:

```python
async def submit_images(self, image_paths, job_id):
    submitted = 0
    for path in image_paths:
        image_id = Path(path).stem

        # Skip already-complete images (same as today)
        if await self.db.all_stages_complete(image_id):
            continue

        # Force-rerun handling (same logic, new API)
        if self.config.runtime.force_rerun:
            await self.db.clear_all(image_id)
        for stage in self.config.runtime.force_stages:
            await self.db.clear_downstream(image_id, stage)

        # Queue detect work
        await self.db.add_work("detect", image_id, score=time.time(),
                               metadata={"image_path": str(path)})
        submitted += 1

    await self.db.commit()
    return submitted, len(image_paths)
```

The `broker` parameter is gone from `__init__`. Replaced by `db: CheckpointDB`.

---

### REWRITE: `workers/monitor.py`

**Current**: `PipelineMonitor` polls Redis stream lengths via `broker.stream_length()` and `broker.pending_count()`.

**New**: All monitoring is SQL queries against the same SQLite DB:

```python
class PipelineMonitor:
    def __init__(self, db: CheckpointDB):
        self.db = db

    async def get_status(self):
        return await self.db.progress_summary()
        # Returns:
        # {
        #   "proposals": {"falcon": 850, "gdino": 820, "sam3": 810},
        #   "stages": {"detect": 800, "evaluate": 650, "refine": 400, "finalize": 350},
        #   "queue": {"detect": 50, "evaluate": 150, "refine": 50, "finalize": 50},
        #   "processing": {"detect": 5, "evaluate": 3, "refine": 2, "finalize": 1},
        #   "failed": 3,
        # }

    async def recover_stale(self, lock_ttl=300, max_retries=3):
        """Periodic sweep replacing Redis TTL auto-expiry."""
        recovered = await self.db.recover_stale(lock_ttl, max_retries)
        if recovered:
            logger.info("Recovered %d stale claims", recovered)

    async def wait_for_completion(self, total_images, poll_interval=5.0):
        while True:
            status = await self.get_status()
            done = status["stages"].get("finalize", 0)
            failed = status["failed"]
            if done + failed >= total_images:
                return True
            self._print_progress(status, total_images)
            # Stale recovery on every poll
            await self.recover_stale()
            await asyncio.sleep(poll_interval)
```

---

### REWRITE: `pipeline.py`

**Current**: Creates `RedisMessageBroker`, connects to Redis, launches workers with broker.

**New**: Creates `CheckpointDB`, connects SQLite, launches workers with db.

```python
class AutoAnnotationPipelineV3:
    def __init__(self, config, job_id=None):
        self.config = config
        self.job_id = job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.job_dir = Path(config.output.job_dir) / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # ONE database for everything
        self.db = CheckpointDB(self.job_dir / "pipeline.db")
        self.output_writer = OutputWriter(self.job_dir)
        self.submitter = JobSubmitter(config, self.db)
        self.monitor = PipelineMonitor(self.db)

    async def run(self):
        self.preflight()
        await self.db.connect()

        submitted, total = await self.submitter.submit_images(...)

        workers = self._create_workers()
        tasks = [asyncio.create_task(w.run()) for w in workers]

        # Monitor in background (includes stale recovery)
        monitor_task = asyncio.create_task(
            self.monitor.wait_for_completion(total)
        )

        await monitor_task
        for w in workers:
            w.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.db.close()

    def _create_workers(self):
        workers = []
        cfg = self.config
        for i in range(cfg.workers.detect_count):
            workers.append(DetectWorker(cfg, self.db, self.output_writer,
                                        worker_id=f"detect-{i}", job_id=self.job_id))
        for i in range(cfg.workers.evaluate_count):
            workers.append(EvaluateWorker(cfg, self.db, self.output_writer,
                                          worker_id=f"evaluate-{i}", job_id=self.job_id))
        for i in range(cfg.workers.refine_count):
            workers.append(RefineWorker(cfg, self.db, self.output_writer,
                                        worker_id=f"refine-{i}", job_id=self.job_id))
        for i in range(cfg.workers.finalize_count):
            workers.append(FinalizeWorker(cfg, self.db, self.output_writer,
                                          worker_id=f"finalize-{i}", job_id=self.job_id))
        return workers
```

**What's removed**: `self.broker`, `RedisMessageBroker.from_config()`, `broker.connect()`, `broker.close()`. All replaced by `self.db`.

---

### MODIFY: `config.py`

**Remove:**

```python
class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    consumer_group: str = "aa_v3"
    streams: dict[str, str] = {}
```

**Add:**

```python
class DatabaseConfig(BaseModel):
    # path is relative to job_dir, auto-created
    filename: str = "pipeline.db"
    lock_ttl: int = 300       # seconds before stale claims are recovered
    max_retries: int = 3      # attempts before moving to failures table
```

**In `AutoAnnotationV3Config`:**

```python
# Before
redis: RedisConfig = RedisConfig()

# After
database: DatabaseConfig = DatabaseConfig()
```

---

### MODIFY: `configs/default.yaml`

**Remove:**

```yaml
redis:
  host: localhost
  port: 6379
  db: 0
  consumer_group: aa_v3
  streams:
    detect: "stream:detect"
    evaluate: "stream:evaluate"
    refine: "stream:refine"
    finalize: "stream:finalize"
    done: "stream:done"
    dead_letter: "stream:dead_letter"
```

**Add:**

```yaml
database:
  filename: pipeline.db
  lock_ttl: 300
  max_retries: 3
```

---

### MODIFY: `contracts.py`

`StageMessage` simplifies — no longer a Redis envelope:

```python
# Before
class StageMessage(BaseModel):
    image_id: str
    image_path: str
    job_id: str
    stage: str
    attempt: int = 0                    # ← tracked in work_queue now
    metadata: dict[str, Any] = {}

    def forward(self, next_stage):       # ← done by db.forward_work now
        return StageMessage(...)

# After
class StageMessage(BaseModel):
    """Lightweight message passed to process(). Built from work_queue claim."""
    image_id: str
    image_path: str
    job_id: str
    stage: str
```

`forward()` method removed — forwarding is handled by `db.forward_work()` in
the base worker. `attempt` is tracked in `work_queue.attempts`, not in the
message.

All data models (`DetectResult`, `EvaluateResult`, `RefineResult`,
`FinalizeResult`, `Candidate`, `VLMVerdict`, etc.) are **unchanged**. They
serialize to JSON the same way — the only difference is the JSON goes into
a SQLite column instead of a file.

---

### MODIFY: `stages/detect.py`, `evaluate.py`, `refine.py`, `finalize.py`

The `process()` method signatures and logic are **unchanged**. These files
change minimally:

1. `__init__` receives `db: CheckpointDB` instead of `broker: RedisMessageBroker` + `checkpoint_mgr: CheckpointManager`
2. `self.save_checkpoint(image_id, stage, data)` → `await self.db.save_stage(image_id, stage, data)` (already a thin wrapper in base.py)
3. `self.load_checkpoint(image_id, stage, cls)` → `await self.db.load_stage(image_id, stage, cls)`
4. No changes to detection logic, VLM calls, filtering, SAM calls, or output writing

**Detect-specific change**: The detect stage currently saves per-model
proposals via `self.checkpoint.save_proposals(image_id, model, data)`. This
becomes `await self.db.save_proposal(image_id, model, data)`. The barrier
check (are all models done?) becomes:

```python
# Before (filesystem scan):
all_done = all(
    self.checkpoint.proposal_exists(image_id, m) for m in enabled_models
)

# After (single SQL query):
all_done = await self.db.barrier_ready(image_id, enabled_models)
```

---

### UNCHANGED files

| File | Why unchanged |
|------|---------------|
| `servers/*` | LitServe servers are independent HTTP services. No Redis or checkpoint dependency. |
| `output.py` | Writes YOLO labels and trace JSONs to the output directory. Doesn't touch checkpoints or work queues. |
| `prompt_manager.py` | Loads YAML prompt templates. No infrastructure dependency. |
| `compare.py` | Reads finished trace JSONs from output directory. No checkpoint dependency. |
| `utils.py` | Geometry math, VLM JSON parsing, image drawing. Pure functions. |
| `prompts/*` | YAML template files. |
| `viewer/*` | FastAPI viewer. Reads output files. |

---

## Dependencies change

**Remove:**

```
redis[hiredis]
```

**Add:**

```
aiosqlite
```

Net: one dependency removed, one lighter dependency added. Redis server
no longer needed in supervisord.

---

## Implementation order

| Step | Files | Depends on | Scope |
|------|-------|------------|-------|
| 1 | `checkpoint.py` | — | Full rewrite: SQLite schema + `CheckpointDB` class with all methods |
| 2 | `workers/base.py` | Step 1 | Rewrite run loop: `claim_work` → `process` → `save_and_forward` |
| 3 | `workers/submitter.py` | Step 1 | Rewrite: `db.add_work()` instead of `broker.submit()` |
| 4 | `workers/monitor.py` | Step 1 | Rewrite: SQL queries instead of Redis stream introspection |
| 5 | `contracts.py` | — | Simplify `StageMessage`, remove `forward()` |
| 6 | `config.py` + `default.yaml` | — | Remove `RedisConfig`, add `DatabaseConfig` |
| 7 | `pipeline.py` | Steps 1-6 | Rewrite: `CheckpointDB` instead of `RedisMessageBroker` |
| 8 | `stages/detect.py` | Steps 1-2 | Modify: `db.save_proposal`, `db.barrier_ready` |
| 9 | `stages/evaluate.py` | Steps 1-2 | Modify: constructor signature |
| 10 | `stages/refine.py` | Steps 1-2 | Modify: constructor signature |
| 11 | `stages/finalize.py` | Steps 1-2 | Modify: constructor signature |
| 12 | DELETE `workers/messaging.py` | Steps 7-11 | Remove Redis Streams broker |
| 13 | Remove `redis` from dependencies | Step 12 | `pyproject.toml` / `requirements.txt` |

Steps 1-4 are the core work. Steps 5-6 are config cleanup. Step 7 wires
everything together. Steps 8-11 are mechanical. Steps 12-13 are cleanup.

---

## Verification

| Test | What it proves |
|------|---------------|
| Submit 10 images, run full pipeline | End-to-end: submitter → detect → evaluate → refine → finalize |
| Kill mid-detect, restart | Resume: claimed items recovered by stale sweep, no re-processing |
| `runtime.force_stages=[evaluate]` | Force rerun: evaluate + downstream cleared, detect cached |
| Run with same config twice | Skip: all stages cached, zero work done |
| Change prompt version, re-run | Selective rerun: detect cached, evaluate + refine re-run |
| 5 detect workers claiming concurrently | No duplicate processing: each image claimed exactly once |
| Image causes OOM in model server | Retry: attempts increment, eventually moves to failures table |
| Monitor reports progress correctly | `progress_summary()` matches actual state |
| `sqlite3 pipeline.db "SELECT ..."` | Queryable: all state inspectable from command line |
| Full run on fl_pj_sample (100 images) | Regression: output matches current Redis-based pipeline |

---

## Supervisord update

**Before:**

```ini
[program:redis]
command=redis-server --port 6379 --save ""
priority=1
autorestart=true

[program:pipeline]
command=python -m data_miner.auto_annotation_v3 --config my.yaml
priority=50
```

**After:**

```ini
# No redis program

[program:pipeline]
command=python -m data_miner.auto_annotation_v3 --config my.yaml
priority=50
```

One fewer service. The pipeline.db is created automatically by `CheckpointDB.connect()`.

---

## Concurrency model

### How parallelism works

All workers are asyncio coroutines in **one Python process, one thread,
one event loop**. There is no `multiprocessing`, no threading (except
aiosqlite's internal background thread).

```
pipeline process (PID 1234)
└── asyncio event loop (single thread)
    ├── detect-0.run()      ← coroutine, loops: claim → HTTP → save
    ├── detect-1.run()      ← coroutine, loops: claim → HTTP → save
    ├── detect-2.run()      ← coroutine, loops: claim → HTTP → save
    ├── evaluate-0.run()    ← coroutine, loops: claim → HTTP → save
    ├── refine-0.run()      ← coroutine, loops: claim → HTTP → save
    ├── finalize-0.run()    ← coroutine, loops: claim → save
    └── monitor.run()       ← coroutine, loops: query → sleep
```

Only one coroutine executes Python code at any instant. When a coroutine
hits `await` (HTTP response, SQLite write, sleep), it suspends and another
coroutine runs. Parallelism comes from overlapping I/O waits — 3 HTTP
requests to 3 model servers are in-flight simultaneously even though one
thread issued them all.

**Why this works**: Workers spend 99% of their time at `await` — waiting
for model server HTTP responses (500ms-5s each). The Python code between
awaits (claim from SQLite, serialize JSON, check barrier) takes <1ms.
Adding OS processes would just multiply overhead without throughput gain.

**What controls throughput**: The bottleneck is always model servers, not
the pipeline process. If Falcon takes 2s/image and has 2 workers, you get
1 image/s through detect-falcon regardless of how many pipeline coroutines
you run. The config knob that matters is worker count — set it to match the
model server's effective concurrency (`workers_per_device × device_count`).

### Worker lifecycle

Workers are launched as `create_task()` (not `TaskGroup`) because they run
indefinitely and stop on an external signal:

```python
# pipeline.py
async def run(self):
    await self.db.connect()
    await self.submitter.submit_images(...)

    # Launch long-running workers as independent tasks
    worker_tasks = [
        asyncio.create_task(w.run(), name=w.worker_id)
        for w in self._create_workers()
    ]
    monitor_task = asyncio.create_task(
        self.monitor.wait_for_completion(total)
    )

    # Wait for completion or shutdown signal
    await monitor_task

    # Cooperative shutdown
    for w in self._workers:
        w.stop()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    await self.db.close()
```

`TaskGroup` is wrong here — it exits when all tasks finish, but workers
never finish on their own. `create_task` + manual `stop()` is the
correct pattern for long-running worker loops.

---

## Asyncio best practices

These are the specific patterns used in the pipeline implementation.
Not general asyncio theory — rules applied to every file in this migration.

### 1. TaskGroup for bounded concurrent work inside stages

When a stage needs to fire N HTTP requests and wait for all results, use
`asyncio.TaskGroup` (Python 3.11+). This replaces `asyncio.gather()` with
proper cancellation and error propagation.

```python
# stages/detect.py — call 3 model servers concurrently for one image
async def _detect_all_models(self, image_id, image_path):
    async with asyncio.TaskGroup() as tg:
        tasks = {
            model: tg.create_task(
                self._call_model(model, image_path)
            )
            for model in self.enabled_models
        }
    # All done — if any failed, TaskGroup cancelled the rest
    # and raised ExceptionGroup
    return {model: t.result() for model, t in tasks.items()}


# stages/evaluate.py — concurrent VLM calls for multiple candidates
async def _evaluate_candidates(self, candidates, image):
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(self._call_vlm(cand, image))
            for cand in candidates
        ]
    return [t.result() for t in tasks]
```

**Rule**: `TaskGroup` for bounded fan-out (known number of tasks, need all
results). `create_task` + manual stop for unbounded worker loops.

### 2. Worker loop error handling

The worker loop must separate `CancelledError` from business exceptions.
Swallowing `CancelledError` breaks shutdown, timeouts, and TaskGroup.

```python
# workers/base.py
async def run(self):
    self._running = True
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as self._session:
        try:
            while self._running:
                claimed = await self.db.claim_work(
                    self.stage, self.worker_id
                )
                if claimed is None:
                    await asyncio.sleep(1)
                    continue

                image_id, image_path = claimed

                try:
                    result = await self.process(
                        StageMessage(
                            image_id=image_id,
                            image_path=image_path,
                            job_id=self.job_id,
                            stage=self.stage,
                        )
                    )
                    next_stage = self._resolve_next_stage(result)
                    await self.db.save_and_forward(
                        image_id, self.stage, result,
                        self._config_hash, self.stage, next_stage,
                    )

                except asyncio.CancelledError:
                    # Shutdown — release work so next run can claim it
                    await self.db.release_work(image_id, self.stage)
                    raise  # ALWAYS re-raise

                except Exception as exc:
                    self.logger.exception(
                        "Failed %s/%s", image_id, self.stage
                    )
                    await self.db.fail_work(
                        image_id, self.stage, str(exc)
                    )

        except asyncio.CancelledError:
            self.logger.info("%s shutting down", self.worker_id)
```

**Rules**:
- `CancelledError` catch is always separate from `Exception` catch
- Always `raise` after catching `CancelledError` — never suppress it
- Release resources (claimed work items) in the cancellation path

### 3. One aiohttp session per worker, not per request

Creating a `ClientSession` per HTTP call destroys and recreates the TCP
connection pool every time. Create it once in `run()` and reuse:

```python
# WRONG: new session per request — kills connection reuse
async def _call_model(self, url, payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

# RIGHT: session created in run(), shared across all iterations
async def run(self):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as self._session:
        while self._running:
            # ... self._session is reused for every HTTP call
```

The session is created with `async with` in `run()` so it's cleaned up
when the worker exits (including on cancellation — `async with` guarantees
`__aexit__` runs).

### 4. Semaphores for model server concurrency limits

Model servers have finite GPU capacity. Without admission control, N
concurrent detect workers can fire N simultaneous requests to a LitServe
server that can only handle 2. Use `asyncio.Semaphore`:

```python
# Shared across all detect workers via config or pipeline-level object
class DetectWorker(StageWorker):
    def __init__(self, config, db, model_semaphores, **kwargs):
        super().__init__(config, db, **kwargs)
        self._sema = model_semaphores  # dict[str, asyncio.Semaphore]

    async def _call_model(self, model_name, payload):
        async with self._sema[model_name]:
            async with self._session.post(url, json=payload) as resp:
                return await resp.json()
```

```python
# pipeline.py — create semaphores once, shared by all workers
model_semaphores = {
    "falcon": asyncio.Semaphore(
        config.servers.falcon.workers_per_device
        * len(config.servers.falcon.devices)
    ),
    "gdino": asyncio.Semaphore(4),
    "sam3": asyncio.Semaphore(4),
}
```

The semaphore limits in-flight requests to the server's actual capacity.
Excess coroutines wait at `async with self._sema[...]` instead of
overwhelming the server with HTTP 503s.

### 5. asyncio.timeout for every external call

Model servers can hang. Without a timeout, the worker holds the claimed
work item forever (until stale sweep recovers it after `lock_ttl` seconds):

```python
async def _call_model(self, url, payload):
    try:
        async with asyncio.timeout(60):
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
    except TimeoutError:
        raise RuntimeError(f"Model server {url} timed out after 60s")
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"Model server {url}: {exc}")
```

`asyncio.timeout` (Python 3.11+) cancels the inner coroutine cleanly.
Don't use `asyncio.wait_for` — it has edge cases with double-cancellation.

**Timeout guidance per stage:**

| Call | Timeout | Rationale |
|------|---------|-----------|
| Detect model HTTP (Falcon/GDINO/SAM3) | 60s | Large images can be slow |
| VLM HTTP (evaluate) | 120s | Qwen 27B can be slow on complex scenes |
| VLM HTTP (refine adjudicate) | 60s | Single candidate, simpler prompt |
| SAM HTTP (refine) | 30s | Point-prompt refinement is fast |
| SQLite operations | 10s | Should be <100ms; 10s catches deadlocks |

### 6. Graceful shutdown with signal handlers

When supervisord sends SIGTERM (or user presses Ctrl+C), the pipeline
should finish in-flight images rather than crash mid-write:

```python
# pipeline.py
async def run(self):
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _on_signal():
        self.logger.info("Shutdown signal received")
        shutdown_event.set()
        for w in self._workers:
            w.stop()  # sets _running = False — cooperative

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    # Launch workers
    worker_tasks = [asyncio.create_task(w.run()) for w in self._workers]
    monitor_task = asyncio.create_task(
        self.monitor.wait_for_completion(total)
    )

    # Wait for natural completion OR shutdown signal
    done, pending = await asyncio.wait(
        [monitor_task, asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()

    # Phase 1: cooperative — workers finish current image
    try:
        async with asyncio.timeout(30):
            await asyncio.gather(*worker_tasks, return_exceptions=True)
    except TimeoutError:
        # Phase 2: forced — cancel workers that didn't stop
        self.logger.warning("Workers didn't stop in 30s, cancelling")
        for t in worker_tasks:
            t.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    await self.db.close()
```

Two-phase shutdown: first `w.stop()` (cooperative — workers finish current
item then exit their loop), then after 30s timeout, `task.cancel()` (forced
— raises `CancelledError` in the worker's current `await`).

### 7. run_in_executor for CPU-bound filtering

Detect-merge runs CPU-bound filtering (geometric, cluster_and_collapse,
per_class_cap). In an asyncio loop, sync CPU work blocks the event loop.
For most images (<50 candidates), filtering takes <10ms — not worth
offloading. For edge cases (200+ candidates), offload to avoid blocking
other workers:

```python
# stages/detect.py — inside DetectWorker.process()
all_candidates = merge_proposals(proposals)

if len(all_candidates) > 100:
    loop = asyncio.get_running_loop()
    filtered = await loop.run_in_executor(
        None,  # default ThreadPoolExecutor
        filter_candidates, all_candidates, self.config.filtering,
    )
else:
    filtered = filter_candidates(all_candidates, self.config.filtering)
```

**Rule of thumb**: sync operations under 10ms — call directly. Over 50ms —
offload with `run_in_executor`. Between 10-50ms — measure before deciding.

### 8. Mutable state safety between awaits

Asyncio is single-threaded so Python data structures don't need locks.
But coroutines interleave at every `await`. Read-await-write on shared
state is a logic bug:

```python
# DANGEROUS: another coroutine can modify stats during the await
count = self.stats["processed"]
await self.db.save_stage(...)        # ← yields here
self.stats["processed"] = count + 1  # ← may overwrite another update

# SAFE: atomic update, no await in between
await self.db.save_stage(...)
self.stats["processed"] += 1
```

In the pipeline, the main shared mutable state is the `CheckpointDB`
connection. Since all operations go through `aiosqlite` (which serializes
through its background thread), this is safe by construction. But be
careful with any in-memory caches or counters shared between workers.

### 9. Pattern summary by file

| File | Pattern | Why |
|------|---------|-----|
| `pipeline.py` | `create_task` + signal handlers + two-phase shutdown | Workers are long-running, need graceful stop |
| `workers/base.py` | Worker loop with separated `CancelledError` handling | Correctness: shutdown vs business errors |
| `workers/base.py` | `aiohttp.ClientSession` in `run()` scope | Connection reuse across all iterations |
| `workers/base.py` | `asyncio.timeout` on every `_call_model` | Prevent hangs from unresponsive model servers |
| `workers/monitor.py` | `asyncio.sleep` + periodic sweep | Replaces Redis TTL auto-expiry |
| `stages/detect.py` | `TaskGroup` for concurrent model calls | Fan-out to 3 model servers per image |
| `stages/detect.py` | `run_in_executor` for heavy filtering | CPU-bound work shouldn't block event loop |
| `stages/detect.py` | `Semaphore` per model server | Don't overwhelm LitServe capacity |
| `stages/evaluate.py` | `TaskGroup` for concurrent VLM calls | Multiple candidates evaluated in parallel |
| `stages/evaluate.py` | `asyncio.timeout(120)` on VLM calls | VLM can be slow, need upper bound |
| `stages/refine.py` | `asyncio.timeout(30)` on SAM calls | SAM refinement is fast, tight timeout |
| `stages/finalize.py` | Direct sync calls (no executor) | CPU-only, always <10ms per image |
