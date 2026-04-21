#!/usr/bin/env bash
#
# check_good3_status.sh — one-shot status report for the laion_good3 run.
#
# Prints GPU utilization, master/shard health, labels written, rate estimate,
# ETA, and class distribution from a sample of recently-written labels.
#
# Invoked by the cron. Safe to run manually any time.

set -uo pipefail

JOB_OUT="${JOB_OUT:-/mnt/data_2/pavan/data_miner/output/auto_annotation_v4/laion_good3}"
OUT_DIR="${OUT_DIR:-$JOB_OUT/labels_standalone}"
LOG_DIR="${LOG_DIR:-$JOB_OUT/logs}"
TOTAL_IMAGES="${TOTAL_IMAGES:-2049292}"
MASTER_PID_FILE="${MASTER_PID_FILE:-/tmp/sam3_dart_good3.pid}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=============================================================="
echo " [$(ts)]  laion_good3 status"
echo "=============================================================="

# ---------- Master / shard health ----------
MASTER_PID=$(cat "$MASTER_PID_FILE" 2>/dev/null || echo "")
if [[ -n "$MASTER_PID" ]] && ps -p "$MASTER_PID" >/dev/null 2>&1; then
    MASTER_STATE="ALIVE (pid=$MASTER_PID)"
else
    MASTER_STATE="DEAD"
fi
# Match only the 8 shard main processes (--shard-index is unique to them;
# DataLoader worker subprocesses don't carry it in their cmdline).
N_SHARDS=$(pgrep -cf "run_sam3_dart_standalone.*--shard-index" 2>/dev/null || echo 0)
echo "master : $MASTER_STATE"
echo "shards : $N_SHARDS/8 alive"

# ---------- GPU utilization ----------
echo
echo "--- GPU utilization ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits | \
    awk -F, '{printf "  gpu%s : mem %5d/%d MiB  util %3d%%  mem-util %3d%%\n", $1, $2, $3, $4, $5}'

# ---------- Labels written ----------
echo
echo "--- progress ---"
if [[ ! -d "$OUT_DIR" ]]; then
    echo "  output dir does not exist yet: $OUT_DIR"
else
    # Use find+wc for speed on large dirs; matches *.txt
    N_LABELS=$(find "$OUT_DIR" -maxdepth 1 -name "*.txt" -type f | wc -l)
    PCT=$(awk -v n=$N_LABELS -v t=$TOTAL_IMAGES 'BEGIN{printf "%.2f", (n/t)*100}')
    echo "  labels written : $N_LABELS / $TOTAL_IMAGES  ($PCT%)"
fi

# ---------- Recent shard log lines (each shard's last progress window) ----------
echo
echo "--- per-shard last progress line ---"
for i in 0 1 2 3 4 5 6 7; do
    F=$(ls -t "$LOG_DIR"/sam3_dart_shard${i}-*.log 2>/dev/null | head -1)
    if [[ -z "$F" ]]; then
        echo "  shard $i : no log"
        continue
    fi
    LINE=$(grep -E "\[[0-9]+/[0-9]+\]" "$F" 2>/dev/null | tail -1)
    if [[ -z "$LINE" ]]; then
        LAST=$(tail -1 "$F")
        echo "  shard $i : (no progress yet) last=\"${LAST:0:100}\""
    else
        # Strip timestamp + logger prefix, keep progress fragment
        COMPACT=$(echo "$LINE" | sed -E 's/^[^ ]+ [^ ]+ [^ ]+ [^ ]+: //')
        echo "  shard $i : $COMPACT"
    fi
done

# ---------- Aggregate throughput + ETA ----------
echo
echo "--- aggregate throughput (parsed from shard logs) ---"
RATE_SUM=$(for i in 0 1 2 3 4 5 6 7; do
    F=$(ls -t "$LOG_DIR"/sam3_dart_shard${i}-*.log 2>/dev/null | head -1)
    [[ -z "$F" ]] && continue
    grep -oE 'overall=[0-9]+\.[0-9]+ img/s' "$F" 2>/dev/null | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+'
done | awk '{s+=$1} END{print s+0}')
if [[ "${RATE_SUM:-0}" != "0" && -n "${N_LABELS:-}" ]]; then
    REMAINING=$((TOTAL_IMAGES - N_LABELS))
    ETA_MIN=$(awk -v r=$REMAINING -v rate=$RATE_SUM 'BEGIN{printf "%.1f", r/rate/60}')
    ETA_H=$(awk -v r=$REMAINING -v rate=$RATE_SUM 'BEGIN{printf "%.2f", r/rate/3600}')
    echo "  sum of per-shard 'overall img/s' : $RATE_SUM"
    echo "  remaining                        : $REMAINING"
    echo "  ETA                              : ${ETA_MIN} min (${ETA_H} h)"
else
    echo "  rate not yet available"
fi

# ---------- Quick class distribution from a sample of recent labels ----------
echo
echo "--- class distribution (sample of 2000 most-recent .txt) ---"
if [[ -d "$OUT_DIR" ]]; then
    # ls -t is slow on huge dirs; use find+sort-by-mtime which is fastest here
    # Take 2000 most-recent, sum per-class line counts
    find "$OUT_DIR" -maxdepth 1 -name "*.txt" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -2000 | awk '{print $2}' \
        | xargs awk 'NF {print $1}' 2>/dev/null \
        | sort -n | uniq -c | sort -rn | head -15 \
        | awk '{printf "  class %2s : %6d\n", $2, $1}'
fi

echo
echo "--- end ---"
