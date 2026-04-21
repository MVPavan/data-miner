#!/usr/bin/env bash
#
# backup_pipeline_db.sh
#
# Online backup + retention for the aa_v4 pipeline.db (SQLite, WAL mode).
# Uses `sqlite3 .backup` so writers are not blocked. Crash-consistent.
#
# Retention:
#   - Keep every backup younger than 24h (hourly window).
#   - Plus the latest one backup per day for the previous 7 days (daily snapshots).
#   - Delete everything else matching the backup naming pattern.
#
# Usage:
#   backup_pipeline_db.sh [--loop [INTERVAL_SECONDS]] <JOB_DIR>
#
# Flags:
#   --loop [N]   Keep running, backing up every N seconds (default 3600).
#                Without this flag, runs one backup and exits (cron-friendly).
#

set -euo pipefail

# ---------- args ----------
LOOP_MODE=0
LOOP_INTERVAL=3600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loop)
            LOOP_MODE=1
            shift
            if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                LOOP_INTERVAL="$1"
                shift
            fi
            ;;
        -h|--help)
            echo "usage: $0 [--loop [INTERVAL_SECONDS]] <JOB_DIR>" >&2
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "[backup] ERROR: unknown flag: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -ne 1 ]]; then
    echo "usage: $0 [--loop [INTERVAL_SECONDS]] <JOB_DIR>" >&2
    exit 2
fi

JOB_DIR="$1"
DB_PATH="${JOB_DIR}/pipeline.db"
BACKUP_DIR="${JOB_DIR}/backups"
LOG_FILE="${BACKUP_DIR}/backup.log"
PATTERN_PREFIX="pipeline.db."
PATTERN_SUFFIX=".bak"

# ---------- preflight ----------
if [[ ! -d "$JOB_DIR" ]]; then
    echo "[backup] ERROR: job_dir does not exist: $JOB_DIR" >&2
    exit 3
fi
if [[ ! -f "$DB_PATH" ]]; then
    echo "[backup] ERROR: pipeline.db not found at $DB_PATH" >&2
    exit 4
fi
if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "[backup] ERROR: sqlite3 binary not found in PATH" >&2
    exit 5
fi

mkdir -p "$BACKUP_DIR"

# ---------- backup iteration ----------
do_backup() {
TS="$(date +%Y%m%d-%H%M%S)"
NEW_NAME="${PATTERN_PREFIX}${TS}${PATTERN_SUFFIX}"
NEW_PATH="${BACKUP_DIR}/${NEW_NAME}"

# `.backup` is online and crash-consistent under WAL.
sqlite3 "$DB_PATH" ".backup '${NEW_PATH}'"

# ---------- retention ----------
# Build the keep-set: hourly (<24h old) + 1 newest per day for last 7 days.
NOW_EPOCH="$(date +%s)"
HOURLY_CUTOFF=$(( NOW_EPOCH - 24 * 3600 ))
DAILY_CUTOFF=$(( NOW_EPOCH - 7 * 24 * 3600 ))

# All backup files, oldest first. Use -print0 + xargs-free loop for safety.
mapfile -t ALL_BACKUPS < <(
    find "$BACKUP_DIR" -maxdepth 1 -type f \
        -name "${PATTERN_PREFIX}*${PATTERN_SUFFIX}" \
        -printf '%T@ %f\n' 2>/dev/null | sort -n | awk '{print $2}'
)

declare -A KEEP=()
declare -A SEEN_DAY=()

# Pass 1: keep everything <24h old.
for f in "${ALL_BACKUPS[@]}"; do
    mtime=$(stat -c %Y "${BACKUP_DIR}/${f}")
    if (( mtime >= HOURLY_CUTOFF )); then
        KEEP["$f"]=1
    fi
done

# Pass 2: walk newest -> oldest, keep one per day within last 7 days.
for (( i=${#ALL_BACKUPS[@]}-1; i>=0; i-- )); do
    f="${ALL_BACKUPS[$i]}"
    mtime=$(stat -c %Y "${BACKUP_DIR}/${f}")
    if (( mtime < DAILY_CUTOFF )); then
        continue
    fi
    day="$(date -d "@${mtime}" +%Y%m%d)"
    if [[ -z "${SEEN_DAY[$day]:-}" ]]; then
        SEEN_DAY["$day"]=1
        KEEP["$f"]=1
    fi
done

# Always keep the brand-new backup.
KEEP["$NEW_NAME"]=1

# ---------- delete the rest ----------
deleted=0
for f in "${ALL_BACKUPS[@]}"; do
    if [[ -z "${KEEP[$f]:-}" ]]; then
        # Defensive: only delete files that match the pattern and live in BACKUP_DIR.
        case "$f" in
            ${PATTERN_PREFIX}*${PATTERN_SUFFIX})
                rm -f -- "${BACKUP_DIR}/${f}"
                deleted=$(( deleted + 1 ))
                ;;
        esac
    fi
done

kept=${#KEEP[@]}

# ---------- logging ----------
SUMMARY="[backup] job_dir=${JOB_DIR} new=${NEW_NAME} kept=${kept} deleted=${deleted}"
echo "$SUMMARY"
printf '%s %s\n' "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$SUMMARY" >> "$LOG_FILE"
}

# ---------- dispatch ----------
if (( LOOP_MODE )); then
    echo "[backup] loop mode: interval=${LOOP_INTERVAL}s job_dir=${JOB_DIR} pid=$$"
    trap 'echo "[backup] stopping (signal received)"; exit 0' INT TERM
    while true; do
        if ! do_backup; then
            ts="$(date +'%Y-%m-%dT%H:%M:%S%z')"
            echo "[backup] WARN: iteration failed, continuing"
            printf '%s [backup] WARN: iteration failed rc=%s\n' "$ts" "$?" >> "$LOG_FILE" || true
        fi
        sleep "$LOOP_INTERVAL"
    done
else
    do_backup
fi

exit 0

# ---------------------------------------------------------------------------
# One-shot (cron-friendly, run every hour):
#   0 * * * * /path/to/backup_pipeline_db.sh /path/to/job_dir >> /path/to/cron.log 2>&1
#
# Loop mode (no cron needed, useful inside a Docker container):
#   nohup /path/to/backup_pipeline_db.sh --loop 3600 /path/to/job_dir \
#       >> /path/to/loop.log 2>&1 &
#   echo $! > /tmp/backup_loop.pid
# Stop:
#   kill "$(cat /tmp/backup_loop.pid)"
# ---------------------------------------------------------------------------
