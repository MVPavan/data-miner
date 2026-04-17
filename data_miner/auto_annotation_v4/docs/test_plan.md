# auto_annotation_v4 — Test Plan

Tiered plan covering unit, integration, lifecycle, and scale validation for
the v4 pipeline. Order reflects priority: Tier 1 is cheap + catches the most
regressions, Tier 4 is the last gate before a real 1M-image run.

Status column legend: ✅ exists, ❌ not written, ⚠️ partial.

---

## Tier 1 — Fast unit tests (no GPU, < 30s total)

Runnable on every commit. Uses tmp SQLite DBs; no network, no GPU.

| Test | Validates | File | Status |
|---|---|---|---|
| `test_checkpoint_db.py::test_register_claim_forward` | Full `register → add_work_batch → claim → save_and_forward` transaction atomicity | NEW | ❌ |
| `test_checkpoint_db.py::test_concurrent_claim` | 10 concurrent `claim_work()` calls, each returns a unique image (no double-claim) | NEW | ❌ |
| `test_checkpoint_db.py::test_stale_recovery` | `recover_stale()` resets PROCESSING rows older than `lock_ttl`, moves to failures after max_retries | NEW | ❌ |
| `test_checkpoint_db.py::test_clear_downstream` | `clear_downstream(Stage.FILTER)` walks STAGE_ORDER, deletes stages + work_queue rows correctly | NEW | ❌ |
| `test_checkpoint_db.py::test_wal_checkpoint` | `wal_checkpoint(TRUNCATE)` runs without error, WAL file size drops | NEW | ❌ |
| `test_checkpoint_db.py::test_config_hash_invalidation` | `should_run_stage()` on hash mismatch calls `clear_downstream` | NEW | ❌ |
| `test_checkpoint_db.py::test_compound_stage_routing` | `save_and_forward` with `work_stage='detect:merge'` closes right queue row, writes `stages.stage='detect'` | NEW | ❌ |
| `test_enums.py::test_stage_order_has_filter` | `STAGE_ORDER == [detect, filter, evaluate, refine, finalize]` | NEW | ❌ |
| `test_filters.py::test_post_detect_runs_all_5` | `FilterPipeline.run(candidates, POST_DETECT)` applies geometric + score + dedup + cap + cross-class | NEW | ❌ |
| `test_filters.py::test_post_review_runs_subset` | `FilterPipeline.run(candidates, POST_REVIEW)` applies only cross_class + per_class_cap | NEW | ❌ |
| `test_filters.py::test_drops_are_context_tagged` | Every returned `FilterDrop` has the passed `context` + correct `reason` | NEW | ❌ |
| `test_pidfile_lock.py` | Second pipeline on same `job_dir` fails fast with clear error; first completes; lock released on exit/crash | NEW | ❌ |

**Effort:** ~300 lines total, one subagent pass.

---

## Tier 2 — GPU-backed integration (5–15 min per run)

Validates model-side correctness. Existing tests still hold post-refactor;
three new ones cover the filter split and HTTP retry.

| Test | Validates | File | Status |
|---|---|---|---|
| `test_gdino_v4_batching.py` | Multi-image batching equivalent to per-image inference (7 tests, MAX_N=16) | `tests/test_gdino_v4_batching.py` | ✅ |
| `test_sam3_dart_v4_batching.py` | SAM3-DART batched ≈ per-item via IoU-pairing comparator | `tests/test_sam3_dart_v4_batching.py` | ✅ |
| `test_bbox_iou_match.py` | GDINO 100% / SAM3 99.37% batched-vs-sequential IoU match | `tests/test_bbox_iou_match.py` | ✅ |
| `bench_batch_sizes.py` | Sweep B×N, find memory/throughput sweet spot per GPU | `tests/bench_batch_sizes.py` | ✅ (must run on target 48 GB GPUs) |
| `test_detect_only_saves_raw.py` | Running `stages=[detect]` produces `DetectResult.candidates` with ZERO filter drops applied | NEW | ❌ |
| `test_filter_stage_applies_post_detect.py` | Running `stages=[detect,filter]` produces `FilterResult.drops` non-empty when inputs warrant it; YOLO written for auto-accepted | NEW | ❌ |
| `test_http_retry.py` | Mock LitServe server returns 503 twice, then 200 → `DetectModelWorker` succeeds on attempt 3 | NEW | ❌ |

---

## Tier 3 — Lifecycle / concurrency (small GPU, scenario-based)

Nine end-to-end scenarios, each ~50 images on 1 GPU, ~5 min per scenario.
Orchestrated by a single runner script with assertion fixtures.

| # | Scenario | Command sketch | What it proves |
|---|---|---|---|
| S1 | Baseline proposal-only, fresh job | `runtime.stages=[detect,filter]` | Detect + filter complete; YOLO for auto-accepted written at filter stage |
| S2 | Mid-run SIGKILL, same `job_id` restart | Kill at ~30%, re-run same config | Complete images skipped, PROCESSING rows recovered after `lock_ttl`, zero duplicates in final output |
| S3 | Mid-run SIGINT, same `job_id` restart | Ctrl-C during work | Graceful claim release; no orphaned PROCESSING rows |
| S4 | Two pipelines same `job_id` (race) | Start twice concurrently | Pidfile lock fails second process immediately with clear error |
| S5 | `force_detect_models=[grounding_dino]` | Re-run after S1 with flag | GDINO proposals + filter + downstream re-run; SAM3-DART proposals cached |
| S6 | `force_stages=[filter]` on completed job | Change filter config, re-run | Filter + downstream re-run against cached detect |
| S7 | `stages=[evaluate,refine,finalize]` after proposal-only | Add downstream stages | Detect+filter cached, downstream runs only on images with filter checkpoint |
| S8 | 4 concurrent pipelines, 4 different `job_dirs` | Four configs, shared model servers | DB isolation; shared servers don't saturate/crash |
| S9 | Config-hash change mid-job | Change threshold, re-run | User prompted; `should_run_stage()` invalidates downstream correctly per-image |

**Effort:** one runner `tests/lifecycle/run_scenarios.py` + 9 assertion fixtures.
~400 lines, one subagent pass.

---

## Tier 4 — Scale validation (GPU, hours)

Only runs before a real 1M-image job. Progressive sizing.

| Test | Size | Runtime | Validates |
|---|---|---|---|
| 1k-image smoke | 1k | ~20 min | End-to-end functionality at non-trivial size; baseline throughput |
| 10k-image stress | 10k | ~2h | WAL stability (file size bounded); backup script runs cleanly; monitor doesn't lag; worker counts tuned |
| 100k-image soak | 100k | ~20h | Extrapolates to 1M: disk growth, memory stability, retry budgets |
| 1M-image run | 1M | ~1 week | The real thing |

**What to watch during scale runs:**

- `pipeline.db-wal` file size — should stay under ~10 MB with `wal_autocheckpoint=1000`
- `progress_summary()` latency — should stay < 100 ms even at 10M+ rows
- `failures` table size — indicator of HTTP retry budget adequacy
- Backup script exit code + retention math
- `image_meta.total_timing_ms` distribution — outliers → hung server?

---

## Out of scope

- Postgres-migration tests — SQLite comfortably handles the 5M-per-job envelope
- `filter_drops` observability view tests — observability table deferred
- Label Studio integration tests — human-review workflow not yet chosen

---

## Suggested build order

1. **Tier 1 unit tests** — no GPU, catches 80% of correctness regressions
2. **Run `bench_batch_sizes.py` on target 48 GB GPUs** — pins real B values before any scale run
3. **Tier 2 new tests** (`test_detect_only_saves_raw`, `test_filter_stage_applies_post_detect`, `test_http_retry`) — validates the filter split + HTTP retry end-to-end
4. **Tier 3 lifecycle scenarios** — one runner, overnight run, confidence gate before 1M
5. **Tier 4 10k-image stress** — last checkpoint before the real run
