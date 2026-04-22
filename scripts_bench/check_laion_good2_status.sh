#!/usr/bin/env bash
# check_laion_good2_status.sh
# One-shot progress snapshot for the detached SAM3-DART 8-GPU run on Good2.
# Designed to be grep-friendly for a /loop supervisor.
set -u

JOB_DIR="/media/data_2/vlm/code/data_miner/output/auto_annotation_v4/laion_good2"
LABELS_DIR="${JOB_DIR}/labels_standalone"
LOGS_DIR="${JOB_DIR}/logs"
TOTAL=5400018
MASTER_PGID=2115938

echo "=== timestamp ==="
date -Is

echo
echo "=== MASTER ==="
if ps -p "${MASTER_PGID}" -o pid,stat,etime,cmd --no-headers >/dev/null 2>&1; then
    echo "master: ALIVE"
    ps -p "${MASTER_PGID}" -o pid=,stat=,etime=,cmd= | awk '{
        printf "pid=%s stat=%s etime=%s\n", $1, $2, $3
    }'
else
    echo "master: DEAD"
fi

echo
echo "=== SHARDS ==="
# Count only MAIN shard processes — their parent is the multi-gpu bash
# (MASTER_PGID). DataLoader worker subprocesses have the main shard as parent
# and inherit the same cmdline, so a naive pgrep -cf over-counts by num_workers+1.
SHARD_COUNT=$(pgrep -P "${MASTER_PGID}" -f run_sam3_dart_standalone 2>/dev/null | wc -l)
echo "shard_count: ${SHARD_COUNT}/8"
pgrep -af run_sam3_dart_standalone | \
    awk '{
        for(i=1;i<=NF;i++) if($i=="--shard-index"){ idx=$(i+1); break }
        printf "  pid=%s shard=%s\n", $1, idx
    }' | sort -k2 -t= -n | head -8

echo
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
    --format=csv,noheader,nounits | \
    awk -F',' '{gsub(/ /,""); printf "gpu%s util=%s%% mem_used=%sMiB free=%sMiB\n", $1, $2, $3, $4}'

echo
echo "=== PROGRESS ==="
COUNT=$(ls -U "${LABELS_DIR}" 2>/dev/null | wc -l)
PCT=$(awk -v c="${COUNT}" -v t="${TOTAL}" 'BEGIN{printf "%.3f", c/t*100}')
echo "labels_written: ${COUNT} / ${TOTAL} (${PCT}%)"

START_EPOCH=$(ps -p "${MASTER_PGID}" -o lstart= 2>/dev/null | \
              xargs -I{} date -d "{}" +%s 2>/dev/null || echo "")
NOW_EPOCH=$(date +%s)
if [[ -n "${START_EPOCH}" ]]; then
    ELAPSED=$((NOW_EPOCH - START_EPOCH))
    RATE=$(awk -v c="${COUNT}" -v e="${ELAPSED}" \
           'BEGIN{if(e>0) printf "%.2f", c/e; else print "0"}')
    REMAIN=$((TOTAL - COUNT))
    if awk -v r="${RATE}" 'BEGIN{exit !(r>0)}'; then
        ETA_HR=$(awk -v r="${REMAIN}" -v rate="${RATE}" \
                 'BEGIN{printf "%.1f", r/rate/3600}')
    else
        ETA_HR="-"
    fi
    ELAPSED_HR=$(awk -v e="${ELAPSED}" 'BEGIN{printf "%.1f", e/3600}')
    echo "elapsed_hours: ${ELAPSED_HR}"
    echo "rate_img_s: ${RATE}"
    echo "eta_hours: ${ETA_HR}"
fi

echo
echo "=== PER-SHARD WINDOW img/s (latest in each log) ==="
for f in "${LOGS_DIR}"/sam3_dart_shard*.log; do
    [[ -f "$f" ]] || continue
    idx=$(basename "$f" | sed -E 's/sam3_dart_shard([0-9]+).*/\1/')
    last_win=$(grep -oE 'win=[0-9.]+ img/s' "$f" | tail -1 | grep -oE '[0-9.]+')
    last_overall=$(grep -oE 'overall=[0-9.]+ img/s' "$f" | tail -1 | grep -oE '[0-9.]+')
    last_batch=$(grep -oE '\[[0-9]+/[0-9]+\]' "$f" | tail -1)
    oom=$(grep -c "OutOfMemoryError" "$f" 2>/dev/null); oom=${oom:-0}
    echo "shard${idx}: win=${last_win:-NA} overall=${last_overall:-NA} progress=${last_batch:-NA} oom=${oom}"
done | sort

echo
echo "=== CLASS DISTRIBUTION (500-file sample) ==="
if [[ "${COUNT}" -gt 0 ]]; then
    ls -U "${LABELS_DIR}" 2>/dev/null | head -500 | while read -r f; do
        cat "${LABELS_DIR}/${f}" 2>/dev/null
    done | awk 'NF {print $1}' | sort | uniq -c | sort -rn | awk '{
        id=$2; cnt=$1;
        name=(id==16?"dog":(id==30?"shopping_cart":(id==33?"forklift":(id==34?"palletjack":"id"id))));
        printf "  %-14s %d\n", name, cnt
    }'
    EMPTY=$(ls -U "${LABELS_DIR}" 2>/dev/null | head -500 | while read -r f; do
        [[ -s "${LABELS_DIR}/${f}" ]] || echo 1
    done | wc -l)
    echo "  empty_labels   ${EMPTY}/500"
fi
