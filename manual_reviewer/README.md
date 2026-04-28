# manual_reviewer

Human-in-the-loop layer that sits **after** `auto_annotation_v4`. It pushes
finalize annotations into Label Studio, lets reviewers correct, deletes,
relabels and adds boxes, then writes the corrected truth back to
`pipeline.db` as a new `Stage.HUMAN_REVIEW` row alongside the original
audit trail.

```
pipeline.db  ──▶ build_tasks.py ──▶ Label Studio  ──▶ export_to_aa_v4.py ──▶ pipeline.db
                  (LS predictions)    (humans edit)        (Stage.HUMAN_REVIEW)
                                          ▲
                                          │ smart=true draws
                                          │
                                  ML backend (port 9090)
                                          │ HTTP
                                          ▼
                              SAM 3.1 LitServe (port 3014)
```

The ML backend never loads weights — it's a pure protocol adapter that
turns LS smart-tool drafts into SAM 3.1 HTTP calls and a sqlite SELECT
against `proposals`. SAM 3.1 covers click→mask, text→detect, refine, and
the optional video tracker.

For design rationale see [docs/review_system.md](docs/review_system.md);
for which detector serves which role see [docs/detectors.md](docs/detectors.md);
for the phased build log see [docs/next_phases.md](docs/next_phases.md).

---

## Layout

```
manual_reviewer/
├── configs/labeling_config.xml      LS XML (RectangleLabels + smart KeyPoint + smart TextArea)
├── docker-compose.review.yml        LS + Postgres + ML backend
├── ml_backend/                      LabelStudioMLBase adapter (no torch)
│   ├── server.py                    Entry: `python -m manual_reviewer.ml_backend.server`
│   ├── routes.py                    smart_click / smart_text / batch_proposals dispatch
│   ├── aav4_client.py               Sam3OneHttpClient builder + cached-proposals reader
│   └── ls_payload.py                LS region builders + predictions envelope
├── pipeline_io/
│   ├── db_reader.py                 read-only sqlite (PRAGMA query_only)
│   ├── db_writer.py                 write Stage.HUMAN_REVIEW / Stage.RECONCILE / dedup
│   ├── task_builder.py              per-image LS task assembly
│   └── ls_export_parser.py          LS completion → HumanReviewResult
├── reconcile/                       cross-frame static-object propagation
│   ├── grouping.py                  clip_id / all / per_image strategies
│   ├── clustering.py                greedy IoU same-class clustering
│   ├── propagate.py                 reconcile_group(...) orchestrator
│   └── sam3_client.py               Sam3OneHttpClient (refine/click/text/track)
├── scripts/
│   ├── build_tasks.py               pipeline.db → LS REST  (or JSON file)
│   ├── export_to_aa_v4.py           LS export → pipeline.db
│   ├── run_reconcile.py             cross-frame reconcile → Stage.RECONCILE
│   └── mark_dedup.py                apply external dedup manifest
└── tests/                           169 tests (round-trip + ML backend + reconcile)
```

---

## Prerequisites

- An aav4 job that produced a `pipeline.db` with at least the `finalize`
  stage. (Detect-only jobs work too, but reviewers will see no
  pre-annotations because finalize is what `build_tasks.py` reads.)
- `docker compose` for Label Studio + Postgres.
- A GPU host running `data_miner.auto_annotation_v4.model_servers.serve
  --model sam3_1 --port 3014` if you want the ML smart tools.

`pipeline.db` is opened read-only by both the ML backend and `build_tasks.py`,
so it is safe to point the review stack at a DB the auto-pipeline is still
appending to.

---

## Workflow A — local single-job review (the common case)

This is the workflow you run on a single laptop or a single GPU box for
one aav4 job at a time. ~10 min from `pipeline.db` to a reviewer typing.

### A1. Start Label Studio + Postgres + ML backend

```sh
# Where the job lives
export JOB_DIR=/abs/path/to/output/auto_annotation_v4/<job_id>

# Where its images live (build_tasks defaults to /data/local-files/?d=<absolute_path>,
# so the LS container needs that absolute path mounted at /label-studio/data/images)
export IMAGE_HOST_DIR=/abs/path/to/your/image_dir

# Path to the DB the ML backend reads (read-only bind mount)
export AAV4_PIPELINE_DB_HOST=$JOB_DIR/pipeline.db

# Where the host SAM 3.1 LitServe is listening
export SAM3_1_URL=http://host.docker.internal:3014/predict

docker compose -f manual_reviewer/docker-compose.review.yml up -d
```

LS comes up at <http://localhost:8080> (defaults `admin@example.com` /
`changeme`; override via `LS_USERNAME` / `LS_PASSWORD` env vars before
`up`).

### A2. Create the LS project + paste the labeling config

In LS UI:
1. **Create project** → name it whatever you want.
2. **Labeling Setup → Browse Templates → Custom template** → paste
   [configs/labeling_config.xml](configs/labeling_config.xml). Save.
3. **Settings → Cloud Storage** is **not** needed — local-files serving is
   already enabled in compose.
4. **Account & Settings → Access Token** → copy the token.
5. **Settings → Machine Learning → Add Model** → URL
   `http://ml_backend:9090` (or `http://host.docker.internal:9090` if
   running the backend outside compose). Tick *"Use for interactive
   preannotations"*.

Note the project ID from the URL (`/projects/<N>/data`).

### A3. Push tasks from pipeline.db

```sh
.venv/bin/python -m manual_reviewer.scripts.build_tasks \
    --db $JOB_DIR/pipeline.db \
    --traces-dir $JOB_DIR/traces \
    --ls-url http://localhost:8080 \
    --ls-token <token-from-A2.4> \
    --ls-project <project-id-from-A2> \
    --skip-existing \
    --limit 50            # start small to validate; remove for full job
```

Each task carries:
- `data.image` URL (`/data/local-files/?d=<absolute_path>`),
- `data.image_id`, `data.image_size`, `data.cluster_id`,
- `data.proposal_summary` (per-detector raw),
- `data.vlm_summary` (filter / VLM verdicts),
- `data.ghost_drops` (boxes the pipeline rejected — visible in LS as a
  separate group, hidden by default),
- `predictions[]` — RectangleLabels seeded from `FinalizeResult.final_annotations`.

Re-running with `--skip-existing` is safe; it pre-flights LS for already-imported
image_ids.

### A4. Review

Reviewers open tasks one at a time. Available tools:

| Tool | When | What happens |
|---|---|---|
| Drag a **Rectangle** with one of the class labels selected | Add a missed box | Goes back as `source="added"` |
| Click a seeded box → change class via dropdown | Relabel | `source="relabeled"`, `original_class` recorded |
| Drag a seeded box's edge | Resize | `source="edited"`, `original_bbox` recorded |
| Right-click a seeded box → Delete | Reject pipeline output | Recorded in `deletions[]` |
| Click a **KeyPoint** with `positive`/`negative` selected | Smart click → mask | ML backend calls SAM 3.1 `click_mask`, returns a sized box; reviewer accepts/edits |
| Type into the **Text query** box | Smart text → detect | ML backend calls SAM 3.1 `text_detect`, seeds matching boxes |
| Toggle a `ghost_drop` box's "kept" flag | Restore a rejected proposal | `source="kept_dropped"` |
| Pick `frame_state` (clean / needs_more_review / ambiguous_skip) | Triage | Stored at the image level |
| Type into the per-region `track_id` field | Track linking | Carried through to `corrections[].track_id` |
| Notes textarea | Free-form | Carried through to `notes` |

Submit moves to the next task.

### A5. Pull the corrections back

```sh
.venv/bin/python -m manual_reviewer.scripts.export_to_aa_v4 \
    --db $JOB_DIR/pipeline.db \
    --traces-dir $JOB_DIR/traces \
    --labels-dir $JOB_DIR/labels \
    --classes-file $JOB_DIR/classes.txt \
    --rewrite-yolo \
    --ls-url http://localhost:8080 \
    --ls-token <token> \
    --ls-project <project-id> \
    --since $(date -d 'yesterday' +%s)   # optional: only fetch new
```

For each completed task this:
1. Builds a `HumanReviewResult` (Pydantic; see [contracts.py](../data_miner/auto_annotation_v4/configs/contracts.py)).
2. Writes it via `CheckpointDB.save_stage(image_id, Stage.HUMAN_REVIEW, ...)`.
   This is `INSERT OR REPLACE` on `(image_id, stage)`, so re-exports are idempotent.
3. Appends a `human_review` block to `traces/<image_id>.json`.
4. With `--rewrite-yolo`, replaces `labels/<image_id>.txt` with the
   reviewer's truth so downstream YOLO consumers reflect human edits.

Verify:

```sh
.venv/bin/python -c "
import sqlite3, json, sys
db = sys.argv[1]
con = sqlite3.connect(db); con.row_factory = sqlite3.Row
n = con.execute(\"SELECT COUNT(*) FROM stages WHERE stage='human_review'\").fetchone()[0]
print('human_review rows:', n)
" $JOB_DIR/pipeline.db
```

### A6. (Optional) Inspect in the aav4 viewer

```sh
.venv/bin/python -m data_miner.auto_annotation_v4.viewer.app \
    --db $JOB_DIR/pipeline.db --port 8994
```

Open <http://localhost:8994>. Each image now shows a `human_review` block
alongside `finalize` so you can audit reviewer behaviour against the
pipeline.

---

## Workflow B — air-gapped / offline review

If the GPU box and the review host can't talk to each other, swap the LS
REST integration for files:

```sh
# On GPU box: build the task JSON
.venv/bin/python -m manual_reviewer.scripts.build_tasks \
    --db $JOB_DIR/pipeline.db \
    --traces-dir $JOB_DIR/traces \
    --out-file /tmp/tasks.json
# scp /tmp/tasks.json to the review host
```

On the review host, **Project → Import → Upload Files → tasks.json**.
Reviewers work in LS as in A4. When done, **Project → Export → JSON**
gives `export.json`.

```sh
# Back on the GPU box
.venv/bin/python -m manual_reviewer.scripts.export_to_aa_v4 \
    --db $JOB_DIR/pipeline.db \
    --traces-dir $JOB_DIR/traces \
    --in-file /tmp/export.json \
    --rewrite-yolo --labels-dir $JOB_DIR/labels --classes-file $JOB_DIR/classes.txt
```

The ML smart tools won't fire in this mode (no network to SAM 3.1) — LS
falls back to RectangleLabels-only.

---

## Workflow C — cross-frame reconciliation (static cameras)

For batches where the same scene appears in many frames (CCTV, fixed
mounts) the auto-pipeline can find an instance in some frames and miss
it in others. `run_reconcile.py` clusters cross-frame detections by IoU
per class, asks SAM 3.1 to confirm the cluster on missing frames, and
writes a new `Stage.RECONCILE` row.

`build_tasks.py` will pick these up as additional pre-annotations on the
next push.

```sh
# Group by clip_id (filename prefix before _f<frame>); other strategies: all, per_image
.venv/bin/python -m manual_reviewer.scripts.run_reconcile \
    --db $JOB_DIR/pipeline.db \
    --backend sam3_1 \
    --grouping clip_id \
    --cluster-iou 0.5 \
    --min-positive-frames 2 \
    --accept-score 0.5 \
    --accept-iou 0.7

# Then re-push to LS so reviewers see the reconciled boxes
.venv/bin/python -m manual_reviewer.scripts.build_tasks \
    --db $JOB_DIR/pipeline.db --ls-url ... --ls-token ... --ls-project ... --skip-existing
```

`run_reconcile.py --help` documents `--grouping`, `--clip-regex`,
`--cluster-iou`, `--min-positive-frames`, `--accept-score`, `--accept-iou`,
`--refine-threshold`, `--limit-images`, `--keep-empty`.

Don't enable this on moving-camera footage without a homography step —
pixel coordinates won't align across frames and you'll get duplicates.

---

## Workflow D — apply external dedup before review

The auto-pipeline's `data_miner/modules/deduplicator.py` writes
`dedup_status` / `dedup_cluster_id` directly into `image_meta`, so review
just works against survivors.

If your dedup ran outside aav4 and produced a manifest, drop it in via:

```sh
.venv/bin/python -m manual_reviewer.scripts.mark_dedup \
    --db $JOB_DIR/pipeline.db \
    --manifest-clusters /path/to/clusters.json
```

`build_tasks.py` filters `WHERE dedup_status='survivor'` automatically. On
pre-migration DBs the column is missing; `db_reader` falls back to
"every image is a survivor" so old jobs still flow.

---

## Configuration knobs

`build_tasks.py` flags worth knowing:

- `--limit N` — cap pushed tasks for smoke testing.
- `--image-url-template` — override the default `/data/local-files/?d={path}`
  if your images are served via S3/HTTPS.
- `--skip-existing` — pre-flight LS to skip tasks whose `image_id` is
  already imported. Safe but adds ~1 HTTP roundtrip per task; omit when
  pushing to a clean project.
- `--out-file foo.json` — write tasks to disk instead of (or in addition
  to) POSTing.

`export_to_aa_v4.py`:

- `--rewrite-yolo` + `--labels-dir` + `--classes-file` — rewrite YOLO
  label files from corrected boxes (otherwise only `pipeline.db` and the
  trace are updated).
- `--config-hash` — recorded on the `stages` row so you can join human
  reviews back to a particular pipeline config.
- `--since <unix_ts>` — only pull completions touched at/after this time.

ML backend env vars: see [ml_backend/README.md](ml_backend/README.md).

---

## Tests

```sh
.venv/bin/python -m pytest manual_reviewer/tests/ -q
```

169 tests cover: DB round-trip on a fixture pipeline.db, task_builder JSON
shape, ls_export_parser correction classification, ML backend route
dispatch + LS payload helpers, reconcile clustering / grouping /
propagation, and the SAM 3.1 click_mask wire.

---

## Quick reference

| What | Where |
|---|---|
| Schema additions | `Stage.HUMAN_REVIEW`, `Stage.RECONCILE` in [enums.py](../data_miner/auto_annotation_v4/configs/enums.py); `dedup_status` / `dedup_cluster_id` columns on `image_meta` |
| Pydantic contracts | `HumanCorrection`, `HumanReviewResult`, `ReconcileResult`, `ReconciledDetection` in [contracts.py](../data_miner/auto_annotation_v4/configs/contracts.py) |
| LS labeling XML | [configs/labeling_config.xml](configs/labeling_config.xml) |
| LS predictions envelope | `manual_reviewer.ml_backend.ls_payload.predictions_envelope` |
| Default SAM 3.1 endpoint | `http://localhost:3014/predict` (refine + click_mask + text_detect + track all on one port; mode picked by request shape) |
| Review trace block | last entry of `traces/<image_id>.json`, `stage="human_review"` |
