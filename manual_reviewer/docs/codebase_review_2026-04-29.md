# manual_reviewer codebase review — 2026-04-29

Independent multi-agent review covering ML backend / pipeline.db round-trip /
reconcile after Phase A/B/C(headless)/D-toggles + uniform dedup policy ship.
Test baseline (post-fixes): **288 passed** (`pytest manual_reviewer/tests/ -q`),
up from 281 pre-fix.

This file is a fix-ordered worklist. Each item is actionable: severity,
file:line, why it matters, fix sketch. Use it directly to drive cleanup work
post-compact.

Severity legend: 🔴 critical (could crash, corrupt data, or silently produce
wrong output) / 🟡 medium (incorrect on edge case, missing log/handling) /
🔵 minor (style, cosmetic, marginal).

**Status legend**: ✅ fixed in this pass / ⏳ pending / ⛔ deferred.

## Fix log — 2026-04-29

All Tier 1 (3) + Tier 2 (13) + Tier 3 (3) + Tier 4 #23 closed in this pass.
New tests: V-tool dispatch routing (×2), `keypointlabels` hint accepted,
NaN-score NMS, per-region track_id round-trip, global notes vs per-region
disambiguation, transport-error audit row, gate-off mixed context. The XML
wire mismatch (#1) was resolved by switching to `<RectangleLabels>` +
`<KeyPointLabels>` (option a in the original recommendation) — the live-LS
smoke test is still on the to-do list and remains the only thing that can
prove end-to-end LS integration without further code changes.

---

## Tier 1 — Critical (fix before any reviewer touches the system)

### ✅ #1 — XML / predictions wire mismatch
- **Severity**: 🔴 → ✅ FIXED 2026-04-29 (option a; live-LS smoke still pending)
- **Where**: [configs/labeling_config.xml:25-52](../configs/labeling_config.xml#L25), [pipeline_io/task_builder.py:226-247](../pipeline_io/task_builder.py#L226), [pipeline_io/ls_export_parser.py:78](../pipeline_io/ls_export_parser.py#L78)
- **What**: XML defined split `<Labels name="label">` + `<Rectangle name="bbox">`. `task_builder` seeded predictions as a single `type="rectanglelabels"` with `from_name="bbox"`. `ls_export_parser` only accepted `type=="rectanglelabels"`. LS may silently drop seeded predictions whose `from_name` doesn't match the loaded XML, or fail to commit reviewer accepts.
- **Fix applied**: XML element rename — `<Labels>` → `<RectangleLabels name="bbox">` (consuming the `<Rectangle name="bbox">`), and `<KeyPoint>` → `<KeyPointLabels name="click">` mirroring the same 24 classes. `_picked_label_from_context`, `_region_label`, and `_exemplars_from_context` extended to accept `keypointlabels` alongside `labels`/`rectanglelabels`. New test: `test_smart_click_accepts_keypointlabels_hint` confirms the keypoint draft class round-trips. The wire is now consistent with the XML.
- **Still TODO**: a live-LS smoke test against the pinned LS image (#19) to verify task seeding + reviewer commit + export end-to-end. Unit tests can't simulate the LS XML→shape contract.

### ✅ #2 — Operator-precedence bug in export annotation filter
- **Severity**: 🔴 → ✅ FIXED 2026-04-29
- **Where**: [scripts/export_to_aa_v4.py:128](../scripts/export_to_aa_v4.py#L128)
- **Fix applied**: Replaced compound condition with explicit `if ann.get("was_cancelled"): continue`. The original `… ground_truth is False and result is None` clause was both buggy (operator precedence) and semantically empty (LS rarely sends `ground_truth=False`).

### ✅ #3 — Vacuous `batch_proposals` dedup-exemption test
- **Severity**: 🔴 (test quality) → ✅ FIXED 2026-04-29
- **Where**: [tests/test_ml_backend.py:1265-1291](../tests/test_ml_backend.py#L1265)
- **Fix applied**: Replaced the tautology with a real with-vs-without comparison: build the same task twice, once with a 0..1×0..1 same-class accepted box covering every cached proposal and once clean, and assert the lengths match. A future regression that adds dedup to `batch_proposals` would cause the with-canvas count to drop and the test to fail.

---

## Tier 2 — Should-fix (real edge-case bugs)

### ✅ #4 — V-tool gate-off short-circuits mixed-context drafts
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [ml_backend/server.py:170-175](../ml_backend/server.py#L170)
- **Fix applied**: Replaced the V-tool gate-off `return []` with fall-through. The for-loop now `continue`s on `from_name="visual_prompt"` regions so a mixed (V-tool + keypoint) draft with V-tool gated off still reaches `smart_click`. New test: `test_server_v_tool_gate_off_falls_through_to_keypoint`.

### ✅ #5 — `nms_regions` undefined sort behavior on NaN scores
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [ml_backend/dedup.py:113-119, 223](../ml_backend/dedup.py#L113)
- **Fix applied**: `_region_score` now `math.isnan` checks the float and returns `0.0` if NaN. New test: `test_nms_regions_nan_score_treated_as_zero`.

### ✅ #6 — `text_query` `maxSubmissions="1"` blocks iterative prompting
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [configs/labeling_config.xml:79-84](../configs/labeling_config.xml#L79)
- **Fix applied**: Removed `maxSubmissions` from `text_query` (kept on `notes`).

### ✅ #7 — Two parallel dispatch implementations drift
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [ml_backend/routes.py:537-558](../ml_backend/routes.py#L537) vs [ml_backend/server.py:148-208](../ml_backend/server.py#L148)
- **Fix applied**: Aligned `dispatch()` with `_predict_one` — V-tool drafts route to `visual_prompt`; non-matching drafts return `[]` rather than falling through to `batch_proposals`. New tests: `test_dispatch_routes_v_tool_to_visual_prompt`, `test_dispatch_v_tool_does_not_fall_to_batch`.

### ✅ #8 — `image_path` URL not quoted
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [pipeline_io/task_builder.py:58](../pipeline_io/task_builder.py#L58)
- **Fix applied**: `urllib.parse.quote(image_path, safe="/")` before formatting into the LS local-files URL template.

### ✅ #9 — `ambiguous_skip` frame_state not filtered, persists with corrections
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [scripts/export_to_aa_v4.py:81-86, 234-265](../scripts/export_to_aa_v4.py#L81)
- **Fix applied**: Skip the YOLO rewrite when `result.frame_state == "ambiguous_skip"`; the partial corrections still get written to the DB (audit) but never overwrite previously-correct YOLO labels.

### ✅ #10 — YOLO label rewrite non-atomic, no dry-run
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [scripts/export_to_aa_v4.py:234-265](../scripts/export_to_aa_v4.py#L234)
- **Fix applied**: Stage to `<target>.txt.tmp` then `os.replace` for atomicity. New `--dry-run` flag that logs intent without touching disk.

### ✅ #11 — YOLO unknown class names silently → id 0
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [scripts/export_to_aa_v4.py:255-257](../scripts/export_to_aa_v4.py#L255)
- **Fix applied**: Startup bails out (exit 2) if `--rewrite-yolo` is set without `--classes-file`. Inside `_rewrite_yolo_label`, an unknown class name now raises `ValueError` (caught by the per-image try/except) instead of silently collapsing to id 0.

### ✅ #12 — Trace append non-atomic, grows unbounded
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [scripts/export_to_aa_v4.py:210-231](../scripts/export_to_aa_v4.py#L210)
- **Fix applied**: Dedup on `ls_completion_id` before appending (idempotent re-runs); stage to tmp file + `os.replace` for atomicity. fcntl lock deferred — single-writer assumption documented in the docstring.

### ✅ #13 — `track_id` per-region textarea silently dropped on export
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [pipeline_io/ls_export_parser.py:83-132](../pipeline_io/ls_export_parser.py#L83)
- **Fix applied**: New `_extract_track_ids` walks the raw result list once collecting `parentID → text` for `from_name="track_id"`, then the rectangle pass attaches each region's text to its `HumanCorrection.track_id`. Also fixed `_extract_textarea` to skip per-region entries (parentID present) so the global `notes` field never picks up a per-region track_id by accident. New tests: `test_parse_picks_up_per_region_track_id_textarea`, `test_parse_global_notes_textarea_does_not_become_track_id`.

### ✅ #14 — `seeded_predictions` takes `predictions[0]` regardless of source
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [scripts/export_to_aa_v4.py:190-197](../scripts/export_to_aa_v4.py#L190)
- **Fix applied**: `_extract_seeded` now scans for `model_version == "aa_v4_finalize"` (a new module constant `FINALIZE_MODEL_VERSION` matching `task_builder.build_task`'s default) and only falls back to `predictions[0]` with a WARNING log if no tagged finalize prediction is found.

### ✅ #15 — `propagate.py` network failures silently demoted, no audit row
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [reconcile/propagate.py:135-147](../reconcile/propagate.py#L135)
- **Fix applied**: Network/server exceptions now emit a `logger.warning` and append a synthetic `ReconciledDetection` to `rejected` with `bbox==seed_bbox`, `mask_score=0`, `seed_iou=0`, and a `#transport_error` suffix on the candidate_id so the audit row is discoverable. The contract has `extra="forbid"` so a true `transport_error` flag wasn't an option — the suffix is the wire-compatible signal. Test renamed: `test_reconcile_records_transport_error_audit_row`.

### ✅ #16 — `run_reconcile.py` per-group failure kills entire CLI
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [scripts/run_reconcile.py:209-236](../scripts/run_reconcile.py#L209)
- **Fix applied**: Wrapped `reconcile_group` in try/except; failed groups log+count, partial output for clean groups is preserved. CLI now exits 3 (not 0) on partial failure so cron jobs / CI can detect it. The summary line includes a "N group(s) failed" tally.

---

## Tier 3 — Document (contract gaps, not bugs)

### ✅ #17 — Multi-seed `propagate_static` merge contract undocumented
- **Severity**: 🟡 (documentation gap) → ✅ FIXED 2026-04-29
- **Where**: [reconcile/propagate_static.py:336-434](../reconcile/propagate_static.py#L336)
- **Fix applied**: Added explicit "Single-seed by design" paragraph to `propagate_static`'s docstring stating that multi-seed callers must NMS the union of per-image verdicts before writing to LS. Built-in multi-seed parked as a Phase C v2 ticket.

### ✅ #18 — Legacy DB fallback silent for `dedup_status` defaults
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [pipeline_io/db_reader.py:52-57](../pipeline_io/db_reader.py#L52)
- **Fix applied**: Once-per-DB `logger.warning` when the dedup-column fallback path is taken. The warning identifies the DB path and tells operators to run `mark_dedup.py` or re-open via `CheckpointDB.connect` to clear the fallback. `--strict-dedup` flag deferred — the log signal is sufficient for v1.

### ✅ #19 — LS version pin missing
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [docker-compose.review.yml](../docker-compose.review.yml)
- **Fix applied**: Pinned to `heartexlabs/label-studio:${LS_VERSION:-1.13.1}` with an inline comment explaining why (smart-tool draft from_name discriminator, pagination, predictions[0] ordering all depend on LS internals). Override via `LS_VERSION` env at compose time.

---

## Tier 4 — Defer (minor / cosmetic)

### #20 — `read_cached_proposals` lacks `OperationalError` retry/log
- **Severity**: 🟡
- **Where**: [ml_backend/aav4_client.py:52-104](../ml_backend/aav4_client.py#L52)
- **What**: Concurrent aav4 writes to `proposals` while LS predict() reads → `OperationalError: database is locked` → caught by outer swallow → reviewer sees "no boxes" with generic warning. No specific log line, no retry.
- **Fix**: One-retry on `OperationalError` with 50ms backoff. Specific log line for the retry case.

### #21 — `norm_box_to_ls_region` 1e-6 degenerate fallback invisible
- **Severity**: 🟡
- **Where**: [ml_backend/ls_payload.py:84-88](../ml_backend/ls_payload.py#L84)
- **What**: Comment claims "emit them as zero-area at the click anyway so the reviewer can see *something*" — but a 0.002-pixel region is effectively invisible.
- **Fix**: Either drop the region (return None / let caller skip) or emit a real minimum size (≥2% of dimension) so it's clickable.

### #22 — `_picked_label_from_context` correctness depends on LS shape
- **Severity**: 🔵 (latent)
- **Where**: [ml_backend/routes.py:122-141](../ml_backend/routes.py#L122)
- **What**: Function works because LS emits a separate `from_name="label"` region the iterator picks up. Inline docstring claims the label rides on the draft itself ("rides on every draft as `value.labels`"). Future refactor that filters by `from_name` would silently break label preservation.
- **Fix**: Update docstring to note both shapes are accepted. Add an end-to-end test through `predict()` (not just the route) for the standalone `from_name="label"` shape.

### ✅ #23 — `visual_prompt` not exported from `ml_backend/__init__.py`
- **Severity**: 🟡 → ✅ FIXED 2026-04-29
- **Where**: [ml_backend/__init__.py:13-25](../ml_backend/__init__.py#L13)
- **Fix applied**: Added `visual_prompt` to `from manual_reviewer.ml_backend.routes import …` and to `__all__`.

### #24 — `mark_dedup` doesn't clear stale dedup state on re-run
- **Severity**: 🟡
- **Where**: [scripts/mark_dedup.py:82](../scripts/mark_dedup.py#L82), [pipeline_io/db_writer.py:140-171](../pipeline_io/db_writer.py#L140)
- **What**: Run #1 marks frame X as `survivor`; run #2 with new clusters that don't include X → X remains `survivor` in the DB. Stale state survives.
- **Fix**: Add `--reset-first` flag that bulk-resets `dedup_status=NULL, dedup_cluster_id=NULL` before applying. Document the gotcha.

### #25 — `--limit` applied before `require_finalize` filter
- **Severity**: 🟡
- **Where**: [pipeline_io/db_reader.py:60-79](../pipeline_io/db_reader.py#L60)
- **What**: `LIMIT ?` in SQL, then Python loop filters on `'finalize' in stages_completed`. `--limit 50` can yield <50 tasks. Operator misdiagnoses as "build_tasks broken".
- **Fix**: Move finalize filter into SQL (`stages_completed LIKE '%"finalize"%'`) or oversample in SQL and trim in Python.

### #26 — `build_tasks.py` doesn't detect partial LS import success
- **Severity**: 🟡
- **Where**: [scripts/build_tasks.py:138-142](../scripts/build_tasks.py#L138)
- **What**: LS import endpoint can return 200/201 with per-task validation errors in the body. Code logs "posted N tasks" without parsing the body — user gets misleading success message.
- **Fix**: Parse response body, log per-task failures, exit non-zero on any failure.

### #27 — Stale `dispatch()` test coverage; missing concurrent / NaN tests
- **Severity**: 🔵
- **Where**: [tests/test_ml_backend.py](../tests/test_ml_backend.py), [tests/test_dedup.py](../tests/test_dedup.py)
- **What**: No test for: NaN scores in `nms_regions` (#5), concurrent `predict()` (#20), `dispatch()` correctly NOT routing V-tool drafts (#7), `from_name="text_query"` filter, multiline-string prompt splitting in `ls_textarea_value_to_prompts`, multi-seed `propagate_static` re-entry (#17).
- **Fix**: Add as part of Tier 1/2 fixes.

### #28 — IoU helper drift remains: `clustering.iou` separate from `dedup.iou_xyxy`
- **Severity**: 🔵 (plan §C-OQ3 already flagged)
- **Where**: [reconcile/clustering.py:78-90](../reconcile/clustering.py#L78) vs [ml_backend/dedup.py:58-72](../ml_backend/dedup.py#L58)
- **What**: Two implementations, mathematically identical, signature-incompatible (`BoundingBox` vs tuple). Drift risk if one is later "fixed".
- **Fix**: `clustering.iou` delegates to `dedup.iou_xyxy(a.as_tuple(), b.as_tuple())`.

---

## Cross-cutting concerns

1. **Test suite cannot validate XML interplay.** Every test fakes the LS dict shape. Tier 1 #1 is invisible to 281 passing tests. **Next priority should be one live-LS smoke test.** The plan calls this out as deferred; deferring it any longer hides real bugs.

2. **Idempotency gradient.** stages-table writes idempotent; trace appends not (#12); YOLO rewrites overwrite cleanly per file but don't detect "already exported"; `mark_dedup` doesn't clear stale state (#24). Pick a uniform contract.

3. **Silent failure modes.** Several routes/scripts swallow exceptions and return empty with logs that don't distinguish "model says no" from "infrastructure failed" (#15, #20). Operators debugging "reviewer clicked, nothing happened" have no signal.

4. **LS version sensitivity.** Smart-tool `from_name` discriminator, task pagination, and predictions[0] ordering all depend on LS internals (#19).

---

## Suggested fix sequence

1. **Day 1**: #1 verify with live LS, fix XML or wire. #2 one-line filter fix. #3 replace tautology with real assertion. Add a Tier-2 live smoke test.
2. **Day 2-3**: Tier 2 cluster — gate-off short-circuit (#4), NaN sort (#5), `maxSubmissions` (#6), URL quoting (#8), `ambiguous_skip` (#9), YOLO atomic+dry-run (#10), unknown classes (#11), trace dedup (#12), track_id round-trip (#13), `predictions[0]` filter (#14), reconcile audit row (#15), per-group isolation (#16).
3. **Day 4**: Documentation — multi-seed contract (#17), legacy fallback warning (#18), LS version pin (#19).
4. **Backlog**: Tier 4 as time permits.

Once Tier 1 + Tier 2 are clean, re-run the multi-agent critique to confirm.

---

## 2026-04-29 fix-pass outcome

- **Status**: Tier 1 (3) ✅, Tier 2 (13) ✅, Tier 3 (3) ✅, Tier 4 #23 ✅.
- **Test baseline post-fix**: 288 passed (`pytest manual_reviewer/tests/ -q`),
  up from 281 with 7 new tests covering the new behaviours (V-tool dispatch
  routing, `keypointlabels` hint, NaN-score NMS, per-region track_id, global
  notes disambiguation, transport-error audit row, gate-off mixed context).
- **Outstanding**: live-LS smoke test for #1 (only thing that can prove the
  XML/wire alignment end-to-end without further code changes); Tier 4 items
  #20–#22 + #24–#28 deferred pending operator-driven need.
