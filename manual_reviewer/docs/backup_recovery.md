# Backup and recovery

The manual_reviewer stack runs Label Studio against a single sqlite file
plus the upstream `pipeline.db`. A 5-minute cron driven by
`manage_stack.sh backup` keeps the worst-case data loss bounded by the
cron interval.

## What gets backed up (three layers)

| Layer | What it covers | Tool | Where |
|-------|----------------|------|-------|
| 1. LS sqlite snapshot | All LS state: projects, users, tokens, tasks, annotations, ML backend wiring | `sqlite3 .backup` | `$LS_BACKUP_DIR/db_snapshots/label_studio.<ts>.sqlite3` |
| 2. pipeline.db snapshot | aa_v4 stage truth: proposals, filter, evaluate, refine, finalize, human_review | `sqlite3 .backup` | `$LS_BACKUP_DIR/db_snapshots/pipeline.<ts>.sqlite3` |
| 3. Annotation event log | Per-annotation create/update/delete events as JSONL — fine-grained replay | `sync_ls_to_disk.py` | `$LS_BACKUP_DIR/project_<id>/events.jsonl` (+ `annotations/<id>.json` snapshots) |

Layer 1 is what you'd restore after disk corruption. Layer 2 is the
canonical source the pipeline can rebuild LS tasks from. Layer 3 is the
fine-grained "what did the reviewer just change?" trail — needed when
you lose annotations to a bad merge or accidental deletion but the rest
of LS is healthy.

`sqlite3 .backup` is online-safe; `cp` of a live `.sqlite3` is not.

## Install the cron

```sh
# Inspect the suggested line.
manual_reviewer/scripts/manage_stack.sh install-cron

# Then add it to your crontab.
crontab -e
```

The line embeds the current shell's `LS_TOKEN`, `LS_BACKUP_PROJECTS`,
`LS_DATA_DIR`, `AAV4_PIPELINE_DB`, and `LS_BACKUP_DIR`. Set those env
vars before running `install-cron` so the printed crontab line carries
the right values.

```sh
export LS_TOKEN=<your admin token>
export LS_BACKUP_PROJECTS="7,8"   # every LS project ID you want sync'd
export LS_DATA_DIR=/tmp/datatang_review/.ls_data
export AAV4_PIPELINE_DB=/tmp/datatang_review/pipeline.db
export LS_BACKUP_DIR=/media/data_2/.../.ls_backup
manual_reviewer/scripts/manage_stack.sh install-cron
```

Verify with a dry run before relying on the cron:

```sh
manual_reviewer/scripts/manage_stack.sh backup
ls -lt $LS_BACKUP_DIR/db_snapshots | head
```

`BACKUP_KEEP_SNAPSHOTS=288` (the default) keeps the last 24 h at 5-min
cadence. Bump it for a longer window.

## Recovery procedures

### Full disk restore (LS sqlite corrupted or lost)

```sh
manual_reviewer/scripts/manage_stack.sh stop ls

# Pick the most recent snapshot.
SNAP=$(ls -1t $LS_BACKUP_DIR/db_snapshots/label_studio.*.sqlite3 | head -1)
cp "$SNAP" "$LS_DATA_DIR/label_studio.sqlite3"

manual_reviewer/scripts/manage_stack.sh start
```

After restart, log into LS as the admin and confirm projects, users,
and the ML backend URL are intact. Worst case you lose <5 min of edits
between the last cron run and the failure.

### Pipeline.db restore

Same pattern; replace the path:

```sh
SNAP=$(ls -1t $LS_BACKUP_DIR/db_snapshots/pipeline.*.sqlite3 | head -1)
cp "$SNAP" "$AAV4_PIPELINE_DB"
```

If reviewers had finished tasks that hadn't yet been exported to
`pipeline.db` via `scripts/export_to_aa_v4.py`, re-run that script
after the restore to sync `Stage.HUMAN_REVIEW` rows back in.

### Annotation-only replay

When LS is healthy but a recent change is gone (project rebuild,
accidental delete, bad import), you can rebuild the annotation set
from the per-project event log. The events are append-only; the
trailing snapshot under `annotations/<id>.json` is always the latest
known live state for that annotation.

```sh
ls $LS_BACKUP_DIR/project_7/annotations/    # live snapshots
ls $LS_BACKUP_DIR/project_7/annotations/deleted/  # preserved deletes
tail $LS_BACKUP_DIR/project_7/events.jsonl   # change-by-change history
```

Re-importing into LS is currently a manual `POST /api/annotations/`
exercise — there's no automated replay-from-jsonl tool yet. For now,
the snapshots are the recovery medium; pull the JSON, edit IDs as
needed, and POST back.

## Verification checklist

- `manage_stack.sh backup` exits 0 and creates two `.sqlite3` files
  per run under `$LS_BACKUP_DIR/db_snapshots/`.
- `crontab -l | grep manage_stack.sh` shows the 5-min line.
- `tail $LOG_DIR/backup.cron.log` shows recent cron runs.
- `sqlite3 $SNAP "PRAGMA integrity_check;"` returns `ok` for the
  most recent snapshot of each kind.

## Why not Postgres + WAL archiving?

Worth doing if you outgrow one host or want point-in-time recovery
finer than 5 min. For 5 reviewers on a single host, sqlite + this
cron is fine and avoids the operational cost of a separate database
server. To migrate later: set `DJANGO_DB=postgresql://…` in
`manage_stack.sh start_ls` env block, run `label-studio init` once,
copy projects/tasks via `pg_dump`/import, and switch the backup
script to `pg_dump` instead of `sqlite3 .backup`.
