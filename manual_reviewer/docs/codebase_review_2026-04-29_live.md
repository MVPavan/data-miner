# Live test results — 2026-04-29 follow-up

Companion to [live_test_plan.md](live_test_plan.md). Records what passed,
what's deferred, and the small code/ops gaps the plan asked me to file
back for the next fix-pass.

## Workflow scoreboard

| Workflow | Status | Notes |
|---|---|---|
| W1 dedup | PASS | Idempotent ALTER ran on legacy DB; mark_dedup argparse rejects missing flag with exit 2 |
| W2 build_tasks | PASS | 7 posted in 2 chunks of 4+3; second run skipped 27/27; LS_TOKEN env-prefix path validated; chunk POST + skip_existing strict-fetch path exercised |
| W3 task open | PASS (structural) | `original_width=1280`, `original_height=720`, `original_rotation=null`, `value` in percent, `model_version="aa_v4_finalize"`, `rectanglelabels` palette correct |
| W4 smart_click | PASS (route) | 1 region returned, label `person`, score 0.879, percent coords + `original_width=1280` correct. UI-side click target validation deferred to user. |
| W5 V-tool | PASS | XML correct; route alive end-to-end (SAM 3.1 returned 1 box, dedup-dropped vs exemplar at IoU>0.7 — designed behavior). Required two **live-only fixes** below. |
| W6 smart_text | PASS | SAM 3.1 returned 1 region, score 0.88, palette label `person`, percent coords + `original_width=1280` correct. |
| W7 export round-trip | PASS | All 3 LS completions write `Stage.HUMAN_REVIEW` rows that round-trip via strict `HumanReviewResult`; idempotent on re-run |
| W8 trace null-id | PASS | `_safe_int` falls back to 0; composite (image_id, reviewed_at) dedup catches re-export; `--in-file + --since` rejected with exit 2 |
| W9 reconcile | PASS | 30 images, 14 propagated, 125 rejected, 0 transport errors, 30 DB rows. `ReconcileResult` Pydantic round-trip validates; `reject_reason` enum stays in spec (`below_iou`). Per-group flush + verbose log all clean. |
| W10 viewer | PASS w/ gap | Search returns 200 for `human_review` + `reconcile`; `/api/data/{id}` exposes both blocks; **gap below** |
| W11 ops | PASS | 6 scenarios: was_cancelled skip, traversal None, bbox clamp, ambiguous_skip skip-yolo, non-numeric id → 0, concurrent W+R 30/30 zero errors @ WAL |
| W12 security | PASS w/ ops note | `_redact_token` works; compose binds 127.0.0.1 by default; **but host-direct daemons below** |

## Code fixes applied during live testing (uncommitted)

### F1 — `model_servers/sam3_1.py`: `--gpu cuda:0` failed LitServer device validation

LitServer's `devices=` requires int(s); we were passing `["cuda:0"]`.
Server crashed at startup with
`ValueError: devices must be an integer or a list of integers`.

Fix: parse `cuda:N` → `int(N)` before constructing `LitServer`. Also
accepts a bare integer (`--gpu 0`).

### F2 — `models/sam3_1.py:471` BF16 `.numpy()` crash on visual_prompt

`scores_t.detach().cpu().numpy()` raised
`TypeError: Got unsupported ScalarType BFloat16`. NumPy has no BF16
dtype; SAM 3.1 returns BF16 tensors on GPU.

Fix: `.detach().to(torch.float32).cpu().numpy()` for both `boxes_t`
and `scores_t` in the visual_prompt path. (The click_mask path at
~L369/373 already used `.float()` — no change there.)

Both fixes are tiny and contained. Should land in the same commit.

## Code gap to land in next fix-pass

### G1 — viewer `/api/search/schema` doesn't list `human_review` / `reconcile`

`data_miner/auto_annotation_v4/viewer/app.py:548-579` hardcodes a 5-stage
list (detect → finalize). The search endpoint (regex at line 583) and
the `_BUILDERS` dispatch already accept the two new stages, so search
works — but the sidebar dropdown the schema endpoint feeds doesn't
expose them, so a UI user can't filter by them via the dropdown.

Fix: add two entries to the `stages` list in `search_schema()`:

```python
{"id": "human_review", "label": "6. Human review"},
{"id": "reconcile",   "label": "7. Reconcile"},
```

Plus the matching block-level entries (`"human_review": {"classes": ..., "statuses": [...]}`,
`"reconcile": {...}`). Filter universes for these stages need a small
spec — what's filterable on `human_review`?
- `frame_state` ∈ {clean, needs_more_review, ambiguous_skip}
- corrections.source ∈ {finalize, added, edited, relabeled, kept_dropped}

For `reconcile`:
- detection action ∈ {confirmed, suggested, rejected}
- `reject_reason` ∈ {below_score, below_iou, no_mask, transport_error}

The `/api/search` query handler would also need fields for these to
make filtering useful, but at minimum the stages must appear in the
schema so they're selectable.

Estimated: ~30 LoC in `app.py` schema + ~20 LoC in search filter
handling. Tests: extend `test_boundary.py`.

## Operational findings (no code change needed)

### O1 — host-direct LS + ML backend bound to 0.0.0.0

Both daemons were started Apr 28 manually without `--host 127.0.0.1`:

```
.venv/bin/python .venv/bin/label-studio start --port 8080 --username admin@example.com --password changeme123 --user-token datatang-demo-token-... --enable-legacy-api-token --no-browser
.venv/bin/python -m manual_reviewer.ml_backend.server
```

Both bound to `0.0.0.0:8080` and `0.0.0.0:9090` respectively. The
`docker-compose.review.yml` fix #7 is intact — it defaults to
`${LS_BIND_HOST:-127.0.0.1}` — but the running daemons don't go through
compose. Action: restart under compose, or add `--host 127.0.0.1` to
the LS launch and an analogous bind override to `ml_backend.server`.

### O2 — token visible via `ps auxf` on the host-direct LS daemon

The Apr-28 launch command leaks `--user-token datatang-demo-token-...`
in `/proc/<pid>/cmdline`. Same fix as O1 — re-launch under compose
where `LS_TOKEN` flows in via env. Build_tasks/export_to_aa_v4 already
honor LS_TOKEN env-prefix (validated W2/W12).

### O3 — pre-fix legacy trace duplicate at `traces/002wu_f00518.json`

Two `human_review` entries with identical `ls_completion_id=1` survive
from before the dedup logic landed. Going forward the fix prevents
new duplicates — confirmed across two re-export runs in W7/W8 — but
this one historical duplicate would only go away with a manual
trace prune. Cosmetic.

### O4 — test rows left in `/tmp/datatang_review/pipeline.db`

W11/W8 created 4 test image_meta rows: `legacy_w8_test`, `w11_cancelled`,
`w11_ambig`, `w11_str_id`, plus `w11_concur_000..029`, plus an
`image_meta` row for `w11_trav` (with the deliberately-traversal path).
Each has matching `stages.stage='human_review'` entries (except
`w11_cancelled`, intentionally skipped). The test rows don't interfere
with W2's `--skip-existing` because LS project 1 was built from real
images, but they will surface on viewer search for `stage=human_review`.

To clean:

```sql
DELETE FROM stages WHERE image_id LIKE 'w11_%' OR image_id LIKE 'legacy_w8_%';
DELETE FROM image_meta WHERE image_id LIKE 'w11_%' OR image_id LIKE 'legacy_w8_%';
```

## Live stack — running services (2026-04-29)

| Service | URL | Notes |
|---|---|---|
| Label Studio | http://localhost:8080 (host-direct, 0.0.0.0) | Up since 2026-04-28; admin@example.com/changeme123; token in env |
| sam3_1 LitServe | http://localhost:3014/predict (0.0.0.0) | Up after F1+F2 fixes; cuda:0; PYTHONPATH=scratchpad/DART |
| ml_backend | http://127.0.0.1:9090 (localhost only ✓) | Restarted 2026-04-29 with current code; AAV4_PIPELINE_DB=/tmp/datatang_review/pipeline.db |

To start fresh:

```bash
PYTHONPATH=/media/data_2/vlm/code/data_miner/scratchpad/DART \
  .venv/bin/python -m data_miner.auto_annotation_v4.model_servers.sam3_1 \
  --port 3014 --gpu cuda:0 > /tmp/datatang_review/sam3_1.log 2>&1 &

LABEL_STUDIO_ML_PORT=9090 LABEL_STUDIO_ML_HOST=127.0.0.1 \
  SAM3_1_URL=http://localhost:3014/predict \
  AAV4_PIPELINE_DB=/tmp/datatang_review/pipeline.db \
  .venv/bin/python -m manual_reviewer.ml_backend.server \
  > /tmp/datatang_review/ml_backend.log 2>&1 &
```

UI-side workflows (W4/W5/W6 live) remain for the user to drive in the
LS browser; the ML wire round-trips are validated by direct route
calls.

## Test baseline

`python -m pytest manual_reviewer/tests/ -q` — 351 passed before live
testing (matched plan). No tests broken by live exercises (every
operation went through normal code paths, not into test directories).
