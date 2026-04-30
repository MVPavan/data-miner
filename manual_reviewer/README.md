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
                                          │ smart="true" drafts (Ctrl+V/K/T/G)
                                          │
                                  ML backend (port 9090)
                                          │ HTTP
                                          ▼
                              SAM 3.1 LitServe (port 3014)
```

The ML backend never loads weights — it's a pure protocol adapter that
turns LS smart-tool drafts into SAM 3.1 HTTP calls and a sqlite SELECT
against `proposals`. SAM 3.1 covers click→mask, text→detect,
visual-prompt (exemplar) detection, and video-track for cross-frame
static-object propagation.

Annotation events are also captured to disk (`.ls_backup/`) on a 5-minute
cron via `sync_ls_to_disk.py` — a lossless audit trail independent of
`pipeline.db`.

For design rationale see [docs/review_system.md](docs/review_system.md);
for which detector serves which role see [docs/detectors.md](docs/detectors.md);
for the phased build log see [docs/next_phases.md](docs/next_phases.md).

---

## Layout

```
manual_reviewer/
├── configs/
│   ├── labeling_config.xml          GENERATED from classes.txt — do not hand-edit <Label> palette
│   └── build_labeling_config.py     classes.txt → labeling_config.xml renderer
├── docker-compose.review.yml        LS + Postgres + ML backend (alternative to manage_stack.sh)
├── ml_backend/                      LabelStudioMLBase adapter (no torch)
│   ├── server.py                    Entry: `python -m manual_reviewer.ml_backend.server`
│   ├── routes.py                    smart_click / smart_text / visual_prompt / smart_track / batch_proposals
│   ├── smart_track_lib.py           SAM 3.1 video-tracker propagation across sibling frames
│   ├── ls_rest.py                   LS REST writer for cross-task prediction posts
│   ├── lswebhook.py                 LS annotation webhook → .ls_backup/ (push-side, idle today)
│   ├── aav4_client.py               Sam3OneHttpClient builder + cached-proposals reader
│   └── ls_payload.py                LS region builders + predictions envelope
├── pipeline_io/
│   ├── clip_id.py                   image_id → clip prefix (strip _f<digits>)
│   ├── db_reader.py                 read-only sqlite (PRAGMA query_only)
│   ├── db_writer.py                 write Stage.HUMAN_REVIEW / Stage.RECONCILE / dedup
│   ├── task_builder.py              per-image LS task assembly
│   └── ls_export_parser.py          LS completion → HumanReviewResult
├── reconcile/                       cross-frame static-object propagation
│   ├── grouping.py                  clip_id / all / per_image strategies
│   ├── clustering.py                greedy IoU same-class clustering
│   ├── propagate.py                 reconcile_group(...) orchestrator
│   ├── propagate_static.py          DINOv3 single-seed propagation library (not wired to UI)
│   └── sam3_client.py               Sam3OneHttpClient (refine/click/text/track)
├── scripts/
│   ├── manage_stack.sh              start / stop / restart / status / logs the 3 services
│   ├── create_ls_project.py         classes.txt → LS project + storage + ml_backend connect
│   ├── build_tasks.py               pipeline.db → LS REST  (or JSON file)
│   ├── sync_ls_to_disk.py           cron-driven LS → .ls_backup/ pull
│   ├── export_to_aa_v4.py           LS export → pipeline.db
│   ├── run_reconcile.py             cross-frame reconcile → Stage.RECONCILE
│   └── mark_dedup.py                apply external dedup manifest
└── tests/                           403 tests (round-trip + ML backend + reconcile + sync + smart_track)
```

---

## Prerequisites

- An aav4 job that produced a `pipeline.db` with at least the `finalize`
  stage. (Detect-only jobs work too, but reviewers will see no
  pre-annotations because finalize is what `build_tasks.py` reads.)
- A GPU host. SAM 3.1 LitServe runs on `cuda:0` by default.
- The `.venv` set up by the project's standard install (`pip install -e .`
  from the repo root).

`pipeline.db` is opened read-only by both the ML backend and `build_tasks.py`,
so it is safe to point the review stack at a DB the auto-pipeline is still
appending to.

---

## Workflow A — local single-job review (the common case)

Run on a single laptop or GPU box for one aav4 job at a time. ~5 min from
`pipeline.db` to a reviewer typing.

### A1. Bring up the stack

```sh
./manual_reviewer/scripts/manage_stack.sh start
```

This launches three services in order, with port-up probes between each:

| Service    | Port | What it does                                              |
|------------|-----:|-----------------------------------------------------------|
| sam3_1     | 3014 | SAM 3.1 LitServe on `cuda:0` (PYTHONPATH=scratchpad/DART) |
| ls         | 8080 | Label Studio with `LOCAL_FILES_SERVING_ENABLED=true`     |
| ml_backend | 9090 | manual_reviewer ML adapter (LSML SDK + lswebhook)         |

Override defaults via env vars (see the script's header docstring for the
full list — `LS_TOKEN`, `LS_PORT`, `SAM3_GPU`, `AAV4_PIPELINE_DB`,
`LS_BACKUP_DIR`, etc.). Pidfiles live in `/tmp/datatang_review/pids/`,
logs append to `/tmp/datatang_review/<service>.log`.

Other commands:

```sh
./manual_reviewer/scripts/manage_stack.sh status            # PIDs + reachability
./manual_reviewer/scripts/manage_stack.sh restart ml_backend # one service
./manual_reviewer/scripts/manage_stack.sh logs ls            # tail -f
./manual_reviewer/scripts/manage_stack.sh stop               # tear down (reverse order)
```

LS comes up at <http://localhost:8080> with admin token
`datatang-demo-token-1234567890abcdef` (override via `LS_TOKEN`).

### A2. Create the LS project (classes from the dataset)

The class palette is **rendered from the dataset's `classes.txt`** — never
hand-edited. One command creates the project + attaches Local Files
storage + connects the ML backend:

```sh
LS_TOKEN=$YOUR_TOKEN \
.venv/bin/python -m manual_reviewer.scripts.create_ls_project \
    --classes /path/to/job/classes.txt \
    --dataset-path /path/to/images \
    --title "my_review_v1" \
    --ls-url http://localhost:8080 \
    --ml-backend-url http://127.0.0.1:9090 \
    --write-xml-to manual_reviewer/configs/labeling_config.xml
# → prints: {"project_id": <N>, "ls_url": "..."}
```

`--per-class hotkey` follows a fixed map (`1-0`, `q-w-e-r-t-y-u-i-o-p`,
`a-s-d` — 23 keys; `v` and `f` are reserved). To re-render after the
dataset's `classes.txt` changes:

```sh
.venv/bin/python -m manual_reviewer.configs.build_labeling_config \
    --classes /path/to/classes.txt \
    --out manual_reviewer/configs/labeling_config.xml
# Then PATCH the live project: see create_ls_project.py --project-id <N>.
```

### A3. Push tasks from pipeline.db

Prefer `LS_TOKEN=…` in the environment over `--ls-token <…>` on the
command line — token strings on argv are visible to anyone with `ps`
access.

```sh
LS_TOKEN=$YOUR_TOKEN \
.venv/bin/python -m manual_reviewer.scripts.build_tasks \
    --db $JOB_DIR/pipeline.db \
    --traces-dir $JOB_DIR/traces \
    --ls-url http://localhost:8080 \
    --ls-project <project-id-from-A2> \
    --skip-existing \
    --limit 40 \
    --per-clip-limit 5     # multi-clip diversity: 8 clips × 5 frames = 40
```

`--per-clip-limit` is optional. Without it, tasks are pulled in DB
insertion order (often clusters into a single clip). With it, the script
groups survivors by `_f<digits>` clip-prefix, sorts clips by descending
frame count (real videos beat one-image entries), and round-robins to
fill `--limit`.

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

Open <http://localhost:8080/projects/<N>/data> and click into a task. Each
task opens with the seeded finalize boxes already on canvas as yellow
drafts; reviewers accept, reject, edit, or ignore them, and add new boxes
where the pipeline missed.

Standard editing tools:

| Action | Source recorded |
|---|---|
| Drag a Rectangle with a class label selected | `source="added"` |
| Click a seeded box → change class via dropdown | `source="relabeled"`, `original_class` recorded |
| Drag a seeded box's edge to resize | `source="edited"`, `original_bbox` recorded |
| Right-click a seeded box → Delete (or **Backspace**) | Recorded in `deletions[]` |
| Toggle a `ghost_drop` box's "kept" flag | `source="kept_dropped"` |
| Pick `frame_state` (clean / needs_more_review / ambiguous_skip) | Stored at the image level |
| Type into the per-region `track_id` field | Carried through to `corrections[].track_id` |
| Notes textarea | Carried through to `notes` |

For the **smart tools** (click→mask, text→detect, V-tool exemplar) see
the next section. Submit to commit the annotation.

### A5. Pull the corrections back

```sh
LS_TOKEN=$YOUR_TOKEN \
.venv/bin/python -m manual_reviewer.scripts.export_to_aa_v4 \
    --db $JOB_DIR/pipeline.db \
    --traces-dir $JOB_DIR/traces \
    --labels-dir $JOB_DIR/labels \
    --classes-file $JOB_DIR/classes.txt \
    --rewrite-yolo \
    --ls-url http://localhost:8080 \
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
import sqlite3, sys
db = sys.argv[1]
con = sqlite3.connect(db)
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

## AI features (smart tools in the labeling UI)

The labeling page has a top-bar **Auto:** toggle that selects which AI
tool is active. Exactly one label palette is visible at a time, paired
with the active tool. Hotkeys mirror the toggle so you don't need the
mouse:

| Auto: choice | Hotkey  | What's shown                                              |
|--------------|---------|-----------------------------------------------------------|
| `none`       | `Esc`   | Plain Rectangle tool with the bbox class palette          |
| `smart_click`| `Ctrl+K`| KeyPoint tool — click on an object → SAM 3.1 click→mask   |
| `smart_text` | `Ctrl+T`| TextArea — type a prompt → SAM 3.1 text→detect            |
| `visual_prompt` (V-tool) | `Ctrl+V`| Rectangle exemplar — draw a box → SAM 3.1 visual prompting on similar instances |
| `smart_track`| `Ctrl+G`| Rectangle seed — draw a box on a static object → SAM 3.1 video tracker propagates it across sibling clip frames |

Class hotkeys (active class for whichever tool is selected): `1` `2` `3`
`4` `5` `6` `7` `8` `9` `0` `q` `w` `e` `r` `t` `y` `u` `i` `o` `p` `a`
`s` `d`. The reserved keys are `v` (Auto: toggle) and `f` (kept free for
future use).

### smart_click — click → mask

Pick a class hotkey → press `Ctrl+K` → click on the object you want
boxed. The ML backend runs SAM 3.1's `click_mask` route at the clicked
pixel, returns the highest-scoring region, and LS shows it as a draft
inheriting the picked class.

**Refine semantics:** clicking on a yellow seeded prediction (or any
already-on-canvas box) intentionally produces a fresh region — the
reviewer can then keep whichever they prefer or delete the duplicate.
The dedup is opt-out for this route on purpose; it's the
"give me a region here" tool.

### smart_text — text prompt → detect

Press `Ctrl+T` → type a class name (`forklift`, `palletjack`, ...) into
the prompt box → submit. The ML backend runs SAM 3.1's `text_detect`
and seeds one RectangleLabels region per match.

**Dedup:** SAM-returned boxes that overlap an existing canvas rectangle
at IoU > 0.7 (class-aware) are dropped server-side. The pool includes
seeded yellow predictions, so re-firing `forklift` won't stack 16 fresh
duplicates on top of finalize boxes the pipeline already produced.

### visual_prompt (V-tool) — exemplar → similar

Press `Ctrl+V` → draw a tight rectangle around one good example of the
target class. SAM 3.1's visual-prompting head runs box-prompt grounding
on the same image and returns every matching instance. The exemplar
**persists** as a regular RectangleLabels annotation (so a freshly drawn
exemplar is preserved as ground truth in addition to whatever SAM
propagates).

**Dedup:** two layers run on the SAM output —
- internal NMS at IoU > 0.85 (class-aware) collapses near-duplicate
  matches the model emitted on the same instance,
- external dedup at IoU > 0.7 against every rectangle on the canvas
  (accepted boxes, seeded predictions, and the exemplar itself).

The exemplar is a seed, not a duplicate, so the SAM-returned box at the
exemplar location gets dropped automatically.

### smart_track — propagate a static object across sibling frames

Press `Ctrl+G` → draw a tight rectangle around a **static** object
(parked car, sign, fixed equipment, building corner, etc.). The ML
backend:

1. Computes the clip prefix from the current task's `image_id` by
   stripping the trailing `_f<frame_index>` (so
   `Caifu_Center_Fewer_2_f00516` and `Caifu_Center_Fewer_2_f00645`
   group together).
2. Queries LS for every other task in the project with the same
   prefix — the "siblings".
3. Builds a JPEG-folder of seed + siblings, calls SAM 3.1's video
   `/track` mode forward from the seed frame, gets per-sibling bbox +
   score back.
4. **Filters** for static-only: keeps siblings where the propagated
   bbox center moved ≤ 0.05 (normalized) from the seed AND the score
   is ≥ 0.5. A moving object's tracker output drifts spatially or
   drops in confidence; either way it's rejected. SAM 3.1's temporal
   disambiguation can leak unrelated detections into the response —
   the motion threshold doubles as a same-object filter.
5. POSTs surviving propagations as predictions to the matching
   sibling tasks via `POST /api/predictions/`. The reviewer sees them
   the next time they open one of those tasks.

The route doesn't add anything to the *current* task's canvas — the
seed rectangle the reviewer drew stays as their draft and they can
keep it or discard it. The propagation lands on siblings.

**What gets propagated:** the meta carries `outcome=propagated`,
`from_image=<seed_image_id>`, `from_task=<seed_task_id>`, and
`source=smart_track` so a downstream consumer can tell automatic
propagations apart from human edits.

**Tunables (env vars on the ML backend):**
- `ENABLE_SMART_TRACK` — falsy disables the route entirely
- The motion / score thresholds and per-call sibling cap (default
  100) live in `manual_reviewer/ml_backend/smart_track_lib.py` —
  v1 keeps them as code constants. Tighten if you see false-positive
  propagations on near-static-but-moving objects.

### Auto-seeded finalize predictions (yellow drafts)

When a task opens, LS shows the finalize boxes from `pipeline.db` as
yellow draft suggestions (LS calls these "predictions"). The reviewer
clicks each yellow box to accept it, edits to refine, deletes if wrong,
or just lets them ride if they're correct — submitting the annotation
captures whatever ended up on canvas.

These are NOT live ML calls — they're cached `FinalizeResult` boxes the
pipeline already produced, embedded in the LS task `predictions[]` at
import time. No SAM 3.1 round-trip on task open (fast).

The route `batch_proposals` would expose **per-detector raw proposals**
(SAM 3.1, Rex-Omni, etc) on top of finalize. It's gated off by default
(`ENABLE_BATCH_PROPOSALS=false`) because firing all 30+ raw boxes per
task overwhelmed the canvas in live testing. Set
`ENABLE_BATCH_PROPOSALS=true` and restart `ml_backend` to enable.

### Cross-frame reconciliation (offline, before review)

For static-camera batches, run the SAM 3.1-driven reconcile pass on
`pipeline.db` — see [Workflow C](#workflow-c--cross-frame-reconciliation-static-cameras)
below. It writes `Stage.RECONCILE` rows, which `build_tasks.py` then
seeds as additional pre-annotations alongside finalize.

---

## Annotation backup

LS Community has no built-in scheduled backup. We do it via cron pull:

```sh
*/5 * * * * cd /media/data_2/vlm/code/data_miner && \
  LS_TOKEN="$LS_TOKEN" .venv/bin/python -m manual_reviewer.scripts.sync_ls_to_disk \
    --ls-url http://localhost:8080 --project <N> \
    >> /tmp/ls_sync.log 2>&1
```

On each run the script GETs every annotation on the project, diffs
against `manual_reviewer/.ls_backup/project_<N>/annotations/`, and feeds
synthesized events through the same pipeline the (currently idle)
webhook handler uses:

```
.ls_backup/project_<N>/
├── events.jsonl                     append-only audit (every change)
├── annotations/
│   ├── <ann_id>.json                latest live snapshot
│   └── deleted/<ann_id>-<ts>.json   preserved on delete (never lost)
└── tasks/<task_id>.json             live aggregate per task
```

Worst-case data loss is the cron interval (5 min for a single reviewer
is fine). Re-runs are idempotent: `unchanged=N` runs append no events
and overwrite no snapshots. Body diffs are volatile-aware — LS server-
side `updated_at` refreshes that don't change `result` don't trigger
fake updates.

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

`build_tasks.py`:

- `--limit N` — cap pushed tasks for smoke testing.
- `--per-clip-limit N` — round-robin pick N frames per clip (largest
  clips first, ties alphabetic) so the reviewer sees frames from many
  videos instead of all-from-one.
- `--image-url-template` — override the default `/data/local-files/?d={path}`
  if your images are served via S3/HTTPS.
- `--skip-existing` — pre-flight LS to skip tasks whose `image_id` is
  already imported. Safe but adds ~1 HTTP roundtrip per task; omit when
  pushing to a clean project.
- `--out-file foo.json` — write tasks to disk instead of (or in addition
  to) POSTing.

`create_ls_project.py`:

- `--classes` (required) — dataset's `classes.txt`.
- `--dataset-path` (required) — filesystem dir for LS Local Files storage.
- `--ml-backend-url` — connect ML backend (omit to skip).
- `--project-id N` — PATCH an existing project's XML (skip create) so
  class palette refreshes after dataset changes.
- `--skip-storage` / `--skip-ml-backend` — re-runs friendliness.
- `--enable-annotation-webhook` — register an LS webhook for the
  push-side `lswebhook` handler. **Currently flaky** in our LS Community
  1.23 install (user-webhooks don't fire reliably alongside the
  auto-registered ML-backend webhook); prefer the cron pull above.

`export_to_aa_v4.py`:

- `--rewrite-yolo` + `--labels-dir` + `--classes-file` — rewrite YOLO
  label files from corrected boxes (otherwise only `pipeline.db` and the
  trace are updated).
- `--config-hash` — recorded on the `stages` row so you can join human
  reviews back to a particular pipeline config.
- `--since <unix_ts>` — only pull completions touched at/after this time.

ML backend env vars:

- `SAM3_1_URL` (default `http://localhost:3014/predict`).
- `AAV4_PIPELINE_DB` — required for the (default-off) batch_proposals route.
- `LS_BACKUP_DIR` — propagated to the lswebhook handler. Default is
  `manual_reviewer/.ls_backup/`.
- `ENABLE_BATCH_PROPOSALS=true|false` (default `false`) and analogous
  per-route gates (`ENABLE_SMART_CLICK`, `ENABLE_SMART_TEXT`,
  `ENABLE_VISUAL_PROMPT`, `ENABLE_PROPAGATE_STATIC`).

See [ml_backend/README.md](ml_backend/README.md) for the full env list.

---

## Tests

```sh
.venv/bin/python -m pytest manual_reviewer/tests/ -q
```

377 tests cover: DB round-trip on a fixture pipeline.db, task_builder
JSON shape, ls_export_parser correction classification, ML backend route
dispatch (smart_click / smart_text / visual_prompt) + dedup pool
semantics + LS payload helpers, reconcile clustering / grouping /
propagation, the SAM 3.1 click_mask wire, the lswebhook on-disk schema,
and the cron-driven `sync_ls_to_disk` diff.

---

## Quick reference

| What | Where |
|---|---|
| Stack lifecycle | [`scripts/manage_stack.sh`](scripts/manage_stack.sh) — start/stop/restart/status/logs |
| New project bootstrap | [`scripts/create_ls_project.py`](scripts/create_ls_project.py) — XML + storage + ML in one shot |
| Class palette source | dataset's `classes.txt` (renderer: [`configs/build_labeling_config.py`](configs/build_labeling_config.py)) |
| Auto: toggle hotkeys | `Esc` (none), `Ctrl+K` (smart_click), `Ctrl+T` (smart_text), `Ctrl+V` (visual_prompt), `Ctrl+G` (smart_track) |
| Annotation backup | [`scripts/sync_ls_to_disk.py`](scripts/sync_ls_to_disk.py) (cron, 5 min) → `.ls_backup/project_<N>/` |
| Schema additions | `Stage.HUMAN_REVIEW`, `Stage.RECONCILE` in [enums.py](../data_miner/auto_annotation_v4/configs/enums.py); `dedup_status` / `dedup_cluster_id` columns on `image_meta` |
| Pydantic contracts | `HumanCorrection`, `HumanReviewResult`, `ReconcileResult`, `ReconciledDetection` in [contracts.py](../data_miner/auto_annotation_v4/configs/contracts.py) |
| LS labeling XML | [configs/labeling_config.xml](configs/labeling_config.xml) — generated, do not hand-edit |
| LS predictions envelope | `manual_reviewer.ml_backend.ls_payload.predictions_envelope` |
| Default SAM 3.1 endpoint | `http://localhost:3014/predict` (refine + click_mask + text_detect + visual_prompt + track all on one port; mode picked by request shape) |
| Review trace block | last entry of `traces/<image_id>.json`, `stage="human_review"` |
