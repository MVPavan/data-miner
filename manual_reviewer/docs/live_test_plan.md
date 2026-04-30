# Manual Reviewer — Live Test Plan (post-fix-pass, 2026-04-29)

This document is the **handoff runbook** for end-to-end validation against a live
Label Studio instance. Every unit test (351 passing) is green; the only
remaining verification is the LS contract surface that unit tests cannot reach.

---

## 0 · Handoff context (read first)

**What just happened (2026-04-29 fix passes):**
- Round 1: 17 items from `docs/codebase_review_2026-04-29.md` fixed (288 → 288 passing).
- Round 2: Multi-agent re-review surfaced 11 P0-P1 critical + ~30 high/medium issues; 4 parallel fix-agents + 1 test-hardening agent landed all of them (288 → 351 passing).

**Working tree (uncommitted) at handoff:**
- 8 production files in `manual_reviewer/` (`pipeline_io/`, `ml_backend/`, `reconcile/`, `scripts/`)
- 4 cross-cutting (`__init__.py`, `docker-compose.review.yml`, `README.md`, `data_miner/auto_annotation_v4/{configs/contracts.py, viewer/app.py}`)
- 9 test files modified, 2 new (`test_boundary.py`, `test_build_tasks.py`)
- Nothing committed — operator should commit before live testing or run on the dirty tree.

**Critical fixes that the live test must validate (each maps to a workflow below):**
| # | Fix | Validating workflow |
|---|---|---|
| 1 | V-tool XML → `<RectangleLabels smartOnly="true">` | W5 (V-tool) |
| 2 | `original_width/height` on smart routes | W4, W5, W6 |
| 3 | LS POST chunking + retry + strict skip-existing | W2 |
| 4 | Per-step try/except in export | W7, W11 |
| 5 | Trace dedup fallback when `ls_completion_id` is null | W8 |
| 6 | `LS_TOKEN` env + token redaction | W2, W7 (also W12) |
| 7 | Reconcile anchor-greedy determinism | W9 |
| 8 | `seed_iou` against best member | W9 |
| 9 | Viewer regex + `_BUILDERS` + `PIPELINE_STAGES` extended | W10 |
| 10 | WAL pragma + connection closing | W7 (concurrent) |
| 11 | aa_v4 reader round-trip | W7, W9 (validation step) |

---

## 1 · Pre-flight checklist

```bash
cd /media/data_2/vlm/code/data_miner

# 1. Sanity-check tree state
git status --short        # expect 27 M + 2 ??
python -m pytest manual_reviewer/tests/ -q | tail -3
# expect: 351 passed in ~5s

# 2. Confirm docker is reachable
docker version
docker compose version

# 3. Confirm aa_v4 LitServe servers are reachable (or be ready to start them)
curl -fsS http://localhost:3014/predict -X POST -H 'content-type: application/json' \
  -d '{"image_path":"/dev/null","mode":"text","text":"x"}' || echo "sam3_1 not up"
curl -fsS http://localhost:3013/predict ... || echo "sam3_dart not up"
# rex_omni at :3015 if Phase 6 is finished — optional

# 4. Decide on a fixture pipeline.db
# Option A: real pipeline run on a small clip (preferred — exercises real proposals/finalize)
# Option B: synthetic — use manual_reviewer/tests/conftest.py fixtures as starting point
# Pick A if any aa_v4 dataset is available; otherwise B
```

**Outputs of pre-flight:**
- A reachable `pipeline.db` path (note: `$PIPELINE_DB`)
- A directory of `traces/` JSONs alongside it (note: `$TRACES_DIR`)
- A directory of source images served by LS (note: `$IMAGE_ROOT`)

---

## 2 · Bring up the LS stack

```bash
cd /media/data_2/vlm/code/data_miner/manual_reviewer

# Set required env (do NOT commit these to shell history persistently)
export LS_PASSWORD='<strong-password>'
export LS_PG_PASSWORD='<strong-pg-password>'
export LS_BIND_HOST='127.0.0.1'   # default after fix; override only if intentional
export LS_PORT=8080
export AAV4_PIPELINE_DB="$PIPELINE_DB"

docker compose -f docker-compose.review.yml up -d
docker compose -f docker-compose.review.yml ps
# wait for healthcheck on label-studio (curl localhost:8080/health)

# Verify ML backend is reachable
curl -fsS http://localhost:9090/health
```

**Acceptance:** LS UI loads at `http://127.0.0.1:8080`, login with `LS_PASSWORD`,
no exposed Postgres on host.

**Failure modes to watch for:**
- LS binds to `0.0.0.0` (regression on docker-compose fix #7)
- ML backend container fails to start (likely missing AAV4_PIPELINE_DB mount)
- Healthcheck never goes green (LS migration error → check logs)

---

## 3 · Project + labeling config setup

```bash
# Create LS project
export LS_TOKEN='<reviewer-token-from-LS-Account-Settings>'
curl -fsS -X POST http://127.0.0.1:8080/api/projects/ \
  -H "Authorization: Token $LS_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"title":"manual_reviewer_smoke","label_config":"@manual_reviewer/configs/labeling_config.xml"}'
# OR upload via UI: Project → Settings → Labeling Interface → paste configs/labeling_config.xml

# Confirm XML uploaded cleanly
# UI sanity: 24-class palette appears for both <RectangleLabels name="bbox"> and
# the visual_prompt smart tool. KeyPointLabels palette also has 24 classes.
```

**Acceptance:**
- No XML parse error in LS UI.
- `<RectangleLabels name="visual_prompt" smart="true" smartOnly="true">` present (post-fix).
- 24-class palette visible on every label-bearing control.

---

## 4 · Workflow tests

### W1 — Dedup + survivor filter (mark_dedup)

```bash
python -m manual_reviewer.scripts.mark_dedup \
  --db "$PIPELINE_DB" \
  --manifest "$DEDUP_MANIFEST"   # or --auto if a flag exists for cosine clustering
```

**Acceptance:**
- `argparse` rejects calls with neither flag (`required=True` group).
- `image_meta.dedup_status` and `dedup_cluster_id` populated.
- `SELECT COUNT(*) FROM image_meta WHERE dedup_status='dropped'` > 0.

**Validation:**
```sql
sqlite3 "$PIPELINE_DB" "SELECT dedup_status, COUNT(*) FROM image_meta GROUP BY dedup_status;"
```

---

### W2 — Build LS tasks (happy path + chunking + idempotency)

```bash
# Happy path with env-token (NOT --ls-token to validate fix #6)
python -m manual_reviewer.scripts.build_tasks \
  --db "$PIPELINE_DB" \
  --image-url-template '/data/local-files/?d={path}' \
  --ls-url http://127.0.0.1:8080 \
  --ls-project 1 \
  --ls-batch-size 10 \
  --skip-existing \
  --limit 50

# Re-run — must be idempotent
python -m manual_reviewer.scripts.build_tasks ... --skip-existing
# Expect: 0 new tasks created, exit 0
```

**Acceptance:**
- First run creates exactly 50 tasks in LS.
- Second run reports "skipped X already imported" and creates 0 duplicates.
- `LS_TOKEN` env var works without `--ls-token`.
- `ps auxf` during run does NOT show the token in command line.
- Logs show chunking ("posting batch 1/5 …").

**Failure-path test:**
```bash
# Stop LS mid-batch then restart
docker compose stop label-studio
# In another terminal: kick off build_tasks with batch-size=2 against 10 images
# Expected: 5xx retry × 3 with backoff, then RuntimeError with chunk image_ids logged
docker compose start label-studio
# Re-run with --skip-existing: must succeed and not duplicate the chunk that did land
```

**Validation:**
```bash
curl -s -H "Authorization: Token $LS_TOKEN" \
  'http://127.0.0.1:8080/api/projects/1/tasks?page_size=200' | jq 'length'
```

---

### W3 — Task open: pre-annotations render correctly

In the LS UI, open a task. Validate visually:

| Element | Expected |
|---|---|
| Image | Loads (LS resolves `/data/local-files/?d=…`) |
| Pre-annotations (`predictions[0]`) | Rectangles match finalize boxes |
| Region color | Yellow tint on `review_items` |
| Ghost-drops control | Toggleable group, off by default, holds drop reasons |
| Class label dropdown | All 24 classes selectable |
| `original_width/height` | Regions remain at correct positions when zooming |

**Acceptance:**
- Boxes render at the correct pixel locations (NOT in the top-left 1% — that
  would be the missing-percent-scaling regression of fix #2/#10 review).
- No console errors in browser dev tools.
- `data.image_size` and `data.cluster_id` accessible via `Data` panel.

---

### W4 — Smart-click → SAM 3.1 mask

In the LS UI, on an open task:
1. Select KeyPointLabels palette → click on a "person" hotkey then on the canvas.
2. Wait for ML backend response.

**Acceptance:**
- A new RectangleLabels region appears within ~1s.
- Region's class is the **clicked keypoint label** (validates fix item 7 — triggering keypoint label preferred).
- Region has `original_width/height` set (validates fix item 2).
- Region's `score` is a real SAM confidence (NOT constant 1.0).

**Validation (server-side):**
```bash
docker compose logs ml_backend | grep -i 'click\|mask' | tail -20
# Expect one POST to sam3_1 :3014, no retry warnings.
```

**Failure-path test:**
- Stop sam3_1 server, click again → reviewer sees no smart suggestion within
  timeout; ML backend logs ONE retry-once warning then a clean error; reviewer
  can still proceed manually.

---

### W5 — Visual prompt (V-tool)

In the LS UI:
1. Press **V** → V-tool selected.
2. Class hotkey for "forklift" → draw exemplar rectangle.
3. ML backend should propose more forklifts on the same canvas.

**Acceptance:**
- V-tool draws a `<RectangleLabels>`-typed rectangle (not a labelless `<Rectangle>`).
- Proposed regions carry the class label `"forklift"` — NOT `"other"` (this is
  the validation for **CRITICAL fix #1**).
- Exemplar is `smartOnly="true"` and disappears after acceptance (does NOT
  persist in the saved annotation).
- Proposed regions don't include the exemplar's own bbox (dedup filter from fix item 8).

---

### W6 — Smart-text → SAM 3.1 text-detect

In the LS UI:
1. Type "forklift" in the smart `text_query` TextArea → submit.

**Acceptance:**
- Proposed regions appear within ~2s.
- Regions are labeled `"forklift"` (palette-snapped).
- If you type a non-palette word like `"thingamajig"`, regions are labeled
  `"other"` (palette snap to default — validates fix item 4).
- Free-text prompt is preserved on `meta.prompt` (visible in Region details).
- Typing in the **non-smart** `notes` TextArea does NOT trigger predict()
  (validates that smart-only gating works).

---

### W7 — Submit annotation → export → DB round-trip

1. Edit one box's class (e.g. `person` → `forklift_operator`).
2. Add one new box.
3. Delete one ghost-drop... wait, ghost drops are `data`-only, not predictions.
   Instead: delete one of the seeded boxes.
4. Set `frame_state` → `"clean"`.
5. Submit annotation.

```bash
# Pull and write
python -m manual_reviewer.scripts.export_to_aa_v4 \
  --db "$PIPELINE_DB" \
  --ls-url http://127.0.0.1:8080 \
  --ls-project 1 \
  --since "$(date -u -d '1 hour ago' +%s)"
```

**Acceptance:**
- Exit code 0.
- DB row added: `SELECT data FROM stages WHERE stage='human_review' AND image_id='<id>';`
  → JSON contains `corrections`, `deletions`, `frame_state="clean"`, `ml_modes_used`.
- Trace appended: `traces/<image_id>.json` last entry has `human_review` block
  with `ls_completion_id` matching the LS annotation id.
- YOLO label rewritten: `labels/<image_id>.txt` reflects edits + additions, not
  deletions.

**Round-trip validation (CRITICAL fix #11):**
```python
from data_miner.auto_annotation_v4.checkpoint import CheckpointDB
from data_miner.auto_annotation_v4.configs.enums import Stage
from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult
db = CheckpointDB("$PIPELINE_DB")
result = await db.load_stage("<image_id>", Stage.HUMAN_REVIEW, HumanReviewResult)
assert result.frame_state == "clean"
assert any(c.source == "relabeled" for c in result.corrections)
```

**Failure-path tests:**
- Re-run export with same `--since`. Idempotency: NO duplicate stage rows, NO
  duplicate trace entries (validates fix #5 even though `ls_completion_id` is
  set — the dedup logic should still no-op).
- Submit annotation with malformed class (e.g. inject via API) → export should
  log the error, skip that completion, and continue with the rest (validates
  fix #4 — per-step try/except).
- Run with `--rewrite-yolo` but without `--classes-file` → exit 2.
- Run with empty `classes.txt` → fail-fast `ValueError`.

---

### W8 — Trace dedup with null `ls_completion_id`

This path is exercised by legacy/disk-export imports (fix #5).

```bash
# Manually craft an export.json with one completion lacking "id"
cat > /tmp/legacy_export.json <<'EOF'
[{"task":{"id":1,"data":{"image_id":"abc"}},"annotations":[{"id":null,"result":[],"lead_time":5}]}]
EOF

python -m manual_reviewer.scripts.export_to_aa_v4 \
  --db "$PIPELINE_DB" \
  --in-file /tmp/legacy_export.json
# Run twice. Validate trace appended ONCE (composite-key fallback works).
```

**Acceptance:**
- Trace file has exactly ONE `human_review` block for `image_id=abc`.
- Logger warning visible if both `ls_completion_id` and `reviewed_at` are 0.

**Failure-path test:**
- Combine `--in-file` with `--since` → argparse rejects (mutually exclusive).

---

### W9 — Cross-frame reconcile

```bash
python -m manual_reviewer.scripts.run_reconcile \
  --db "$PIPELINE_DB" \
  --grouping clip-regex \
  --clip-regex '(.+?)_\d+\.[A-Za-z]+$' \
  --backend sam3_1 \
  --sam3-url http://localhost:3014/predict \
  --max-consecutive-transport-errors 5
```

**Acceptance:**
- Per-group flush message after each group (validates fix item 4 — durable progress).
- For singleton or skipped groups, an empty `ReconcileResult` row is written
  (validates fix item 6).
- `SELECT COUNT(*) FROM stages WHERE stage='reconcile';` matches input image
  count (or the survivor subset).
- For each `ReconciledDetection`, the new structured `reject_reason` field is
  one of `{"below_score","below_iou","no_mask","transport_error", null}`.

**Round-trip validation:**
```python
from data_miner.auto_annotation_v4.configs.contracts import ReconcileResult
result = await db.load_stage("<image_id>", Stage.RECONCILE, ReconcileResult)
assert all(d.reject_reason in (None, "below_score", "below_iou", "no_mask", "transport_error") for d in result.rejected)
```

**Failure-path tests:**
- Stop sam3_1, run reconcile against a 5-cluster group:
  - Should bail after 5 consecutive transport errors with a clear `RuntimeError`.
  - Per-group flush means earlier groups' results are durably written.
  - Exit code is 3 on any failed group (per existing convention).
  - `KeyboardInterrupt` (Ctrl-C) mid-run flushes the in-flight buffer.
- Re-run reconcile on the same DB with sam3_1 back up:
  - Determinism: same input order produces identical cluster_ids and detections
    (validates fix #7 — confidence rounded to 4 decimals).
- Run on a DB with no finalize stages → exit 1 (NOT 0; validates fix item 9).

---

### W10 — Viewer integration

```bash
# Bring up the aa_v4 viewer (separate from LS)
python -m data_miner.auto_annotation_v4.viewer.app --db "$PIPELINE_DB" --port 8994 &
sleep 2

# Validate viewer renders human_review + reconcile stages
curl -s 'http://127.0.0.1:8994/api/data/<image_id>' | jq '.human_review,.reconcile'
# Expect both keys populated for an image that has both stages.

# Validate search regex now accepts new stages (CRITICAL fix #9 + boundary-test bug fix)
curl -s 'http://127.0.0.1:8994/api/search?stage=human_review' | jq '.image_ids | length'
curl -s 'http://127.0.0.1:8994/api/search?stage=reconcile' | jq '.image_ids | length'
# Both must return 200 (NOT 422 KeyError as the regex-only fix would have).
```

**Acceptance:**
- Viewer page for an image shows both `finalize` and `human_review` blocks
  side-by-side.
- Filter dropdown lists `human_review` and `reconcile` as selectable stages.
- Search by either stage returns matching image_ids without 500.

---

### W11 — Operational + edge-case scenarios

| Scenario | How to trigger | Expected |
|---|---|---|
| Reviewer cancels annotation (`was_cancelled=true`) | Mark in LS UI before submit | Export skips with no DB write (validates Tier 1 #2 fix from prior pass) |
| `frame_state=ambiguous_skip` | Set in LS UI submit | Export writes DB+trace but does NOT rewrite YOLO label |
| Path traversal in image_path | Inject `../../etc/passwd` into `image_meta.image_path` | `build_task` returns `None` with warning, task skipped (fix item 5) |
| Bbox out of `[0,1]` | Inject a normalized bbox like `[1.5, 0.2, 2.0, 0.8]` | `build_task` clamps + warns (fix item 2) |
| Non-numeric LS id | Patch a completion's `id` to `"abc"` | Export skips with safe fallback (fix item 4 in pipeline_io) |
| Concurrent write+read | Run `write_human_review` in a thread while another thread iterates survivors | No DB lock errors, no corruption (validates WAL fix #10) |
| `--dry-run` on export | Run with `--dry-run` flag | No DB writes, no trace writes, no YOLO writes; logs intended actions |

---

### W12 — Security / token handling

```bash
# 1. ps audit during build_tasks run
LS_TOKEN='secret-token-xyz' python -m manual_reviewer.scripts.build_tasks ... &
ps auxf | grep build_tasks
# Expect: token NOT visible in command line

# 2. Force a 4xx error and inspect the log
curl ... # cause LS to return 401
docker compose logs ml_backend | grep -i 'token\|secret'
# Expect: token replaced with <redacted>

# 3. Docker port binding
ss -tlnp | grep 8080
# Expect: bound to 127.0.0.1, NOT 0.0.0.0

# 4. Default password rejection — N/A (we chose doc-only fix; verify env override works)
docker compose config | grep -i password
# Verify env vars are interpolated correctly
```

---

## 5 · Final acceptance summary

After running W1–W12, the operator should be able to fill in this table:

| Workflow | Pass / Fail | Notes |
|---|---|---|
| W1 dedup | | |
| W2 build_tasks | | |
| W3 task open | | |
| W4 smart_click | | |
| W5 visual_prompt | | |
| W6 smart_text | | |
| W7 export round-trip | | |
| W8 trace null-id | | |
| W9 reconcile | | |
| W10 viewer | | |
| W11 ops scenarios | | |
| W12 security | | |

**Stop-the-line failures (must fix before declaring "live-validated"):**
- W3 mis-positioned boxes (would mean fix #2 regressed)
- W5 visual_prompt labels collapse to "other" (would mean fix #1 regressed)
- W7 round-trip failure (would mean Pydantic boundary regressed)
- W9 non-deterministic cluster ids across runs (would mean fix #7 regressed)
- W10 search returns 500 (would mean viewer builders missing — fix landed in
  test_boundary.py agent's last edit)

---

## 6 · Cleanup

```bash
docker compose -f docker-compose.review.yml down -v   # destroy postgres volume too
unset LS_TOKEN LS_PASSWORD LS_PG_PASSWORD AAV4_PIPELINE_DB
```

If anything in this plan needed code changes beyond what was already landed,
record them in a new `docs/codebase_review_$(date +%F)_live.md` for the next
fix-pass.

---

## 7 · Quick-start command for the next agent

After compaction, the resuming agent should:

1. Read this file in full.
2. Re-run pytest to confirm baseline (`python -m pytest manual_reviewer/tests/ -q` → expect 351 passed).
3. Confirm working tree is still dirty with the same 27 modified + 2 new files (`git status --short`).
4. Begin at section 1 (pre-flight) and proceed sequentially.
5. Report any deviation from "Expected" in each workflow back to the user.
