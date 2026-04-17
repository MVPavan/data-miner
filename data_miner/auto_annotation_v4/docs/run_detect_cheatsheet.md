# aa_v4 detect runs — cheatsheet

Copy-paste commands for running the two detect-only jobs and asking Claude
to oversee them.

**Paths assume cwd = `/media/data_2/vlm/code/data_miner`**

---

## 0. One-time: tail-friendly log locations

All scripts already write to log files. You don't need to set anything up.

| What | Where |
|---|---|
| SAM3-DART server | `logs/aav4_detect_servers/sam3_dart.log` + `.pid` |
| GDINO server | `logs/aav4_detect_servers/grounding_dino.log` + `.pid` |
| DataTang pipeline | `logs/aav4_detect_pipeline/datatang_val_detect.log` |
| LOCO pipeline | `logs/aav4_detect_pipeline/loco_unannotated_full_sam_filtered_detect.log` |
| DataTang DB | `output/auto_annotation_v4/datatang_val_detect/pipeline.db` |
| LOCO DB | `output/auto_annotation_v4/loco_unannotated_full_sam_filtered_detect/pipeline.db` |

---

## 1. Start detector servers

Launches SAM3-DART on GPUs 4,5 (port 3013) and GDINO on GPUs 6,7 (port 3001).
Blocks until `/health` returns on both, then exits leaving the servers
running in the background.

```bash
data_miner/auto_annotation_v4/scripts/start_detect_servers.sh
```

Verify manually (optional):

```bash
curl -s http://localhost:3001/health && echo       # GDINO
curl -s http://localhost:3013/health && echo       # SAM3-DART
```

---

## 2. Run detect on DataTang_val

```bash
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh
```

Runs `detect` stage only (no filter/evaluate/refine/finalize) on
`/media/data_2/datasets/datasets_pavan/DataTang_val` using both detectors.
Output lands in `output/auto_annotation_v4/datatang_val_detect/`.

---

## 3. Run detect on LOCO (after DataTang finishes — same servers)

```bash
data_miner/auto_annotation_v4/scripts/run_detect_loco.sh
```

Runs the same thing on
`/media/data_2/datasets/datasets_pavan/loco_unannotated_full_sam_filtered`.

### Alternative: run both concurrently

Shared servers will handle combined load (lower per-job throughput, same
total time). Different `job_id`s → different DBs → safe.

```bash
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh &
data_miner/auto_annotation_v4/scripts/run_detect_loco.sh &
wait
```

---

## 4. Stop servers when both jobs are done

```bash
data_miner/auto_annotation_v4/scripts/stop_detect_servers.sh
```

---

## 5. Quick manual health checks (while jobs run)

```bash
# queue state per stage
.venv/bin/python -c "import sqlite3; \
  conn=sqlite3.connect('output/auto_annotation_v4/datatang_val_detect/pipeline.db'); \
  print(list(conn.execute('SELECT stage,status,COUNT(*) FROM work_queue GROUP BY stage,status')))"

# failed rows (anything here = investigate)
.venv/bin/python -c "import sqlite3; \
  conn=sqlite3.connect('output/auto_annotation_v4/datatang_val_detect/pipeline.db'); \
  print(list(conn.execute('SELECT stage,COUNT(*) FROM failures GROUP BY stage')))"

# WAL size — should stay bounded (<10 MB)
ls -la output/auto_annotation_v4/datatang_val_detect/pipeline.db*
```

---

## 6. How to ask Claude to oversee

Paste one of these prompts. Claude will poll logs + DB and flag
errors/hangs/failed rows. No need to sit in front of it.

### Start-of-run check

```
Both jobs started. Please poll every ~2 minutes:
- logs/aav4_detect_servers/sam3_dart.log (tail last 40 lines, flag errors)
- logs/aav4_detect_servers/grounding_dino.log (same)
- logs/aav4_detect_pipeline/datatang_val_detect.log (same)
- logs/aav4_detect_pipeline/loco_unannotated_full_sam_filtered_detect.log (same)
- pipeline.db work_queue + failures counts for both jobs
Stop + alert me if: failures > 10, WAL > 50MB, or pipeline log has no
new lines for 5 minutes.
```

### During a run — quick status

```
Give me a status snapshot: for both jobs, queue counts by stage/status,
failure count, latest pipeline log line, and WAL size.
```

### After a run — verification

```
DataTang detect finished. Verify:
- image_meta rows count matches folder (folder is
  /media/data_2/datasets/datasets_pavan/DataTang_val)
- all images reached stages.stage='detect'
- proposals table has rows for both grounding_dino and sam3_dart
- failures table empty (or report what's in it)
- class distribution of detected candidates (top 10 by count)
```

### If something looks wrong

```
I see errors in logs/aav4_detect_pipeline/datatang_val_detect.log.
Diagnose: read the tail, check server logs for the same timestamp,
check failures table, propose a fix before touching anything.
```

---

## 7. Runtime knobs (if you need to tune mid-run)

Pass as extra args to the run script (forwarded to the pipeline):

```bash
# more parallelism per detector
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh \
    workers.detect_per_model=4

# force re-run one detector, keep the other cached
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh \
    'runtime.force_detect_models=[grounding_dino]'

# smaller batch if GPUs OOM (also edit start_detect_servers.sh first)
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh \
    servers.detectors.grounding_dino.max_batch_size=4

# debug-level logs to the pipeline log file
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh \
    runtime.log_level=DEBUG
```

---

## 8. Safe to Ctrl+C

Both pipeline scripts handle SIGINT gracefully — current claims are
released, DB state stays consistent, re-running the same script resumes
from where it stopped. Servers keep running.

```bash
# re-run after a Ctrl+C
data_miner/auto_annotation_v4/scripts/run_detect_datatang.sh
```

`all_stages_complete()` skips images already done; `recover_stale()`
inside `PipelineMonitor` resets any rows stuck in `PROCESSING` after
`database.lock_ttl=300s`.
