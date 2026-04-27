#!/usr/bin/env bash
# run_embed_4gpu.sh
#
# Launch 4 SigLIP2 embedding shards, one per GPU. Each shard pins to a
# single GPU via CUDA_VISIBLE_DEVICES so `cuda:0` inside the process
# maps to the right physical device. Sharding is a stable blake2s hash
# of the image_id, so resume is idempotent even if the on-disk file set
# changes between runs.
#
# A single scan pass runs in the parent up front and is written to
# <LANCE_URI>/_items_snapshot.tsv; the 4 shards then read that file
# instead of each re-scanning 5.4M paths.
#
# Usage:
#   ./run_embed_4gpu.sh
#   IMG_DIR=/path LANCE_URI=/path MODEL=siglip2-giant ./run_embed_4gpu.sh
#   FORCE_RESCAN=1 ./run_embed_4gpu.sh
#
# Resume is on by default — re-run after a crash; shards skip rows
# already in the LanceDB table.

set -euo pipefail

IMG_DIR="${IMG_DIR:-/mnt/data/deepak/DATASET/Laion/Good2/images}"
LANCE_URI="${LANCE_URI:-/mnt/data/data_miner_lance/laion_good2}"
TABLE="${TABLE:-embeddings}"
MODEL="${MODEL:-siglip2-giant}"   # 1536-dim; 'siglip2-so400m' = 1152-dim
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
BUFFER_ROWS="${BUFFER_ROWS:-1024}"
N="${N:-4}"
STAGGER_S="${STAGGER_S:-3}"
GPU_BUSY_THRESHOLD_MB="${GPU_BUSY_THRESHOLD_MB:-2000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE_RESCAN="${FORCE_RESCAN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/embed_logs}"
ITEMS_FILE="${ITEMS_FILE:-${LANCE_URI}/_items_snapshot.tsv}"

mkdir -p "${LOG_DIR}" "${LANCE_URI}"

echo "image-dir : ${IMG_DIR}"
echo "lance-uri : ${LANCE_URI}"
echo "table     : ${TABLE}"
echo "model     : ${MODEL}"
echo "shards    : ${N}  (hash-sharded by blake2s(image_id) % N)"
echo "items     : ${ITEMS_FILE}"
echo "logs      : ${LOG_DIR}/embed_gpu<i>.log  (append-mode)"
echo

# ---- 1. GPU busy preflight -------------------------------------------------
echo "GPU preflight..."
for i in $(seq 0 $((N-1))); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${i}" 2>/dev/null || echo 999999)
    used=${used// /}
    if [[ -z "${used}" || "${used}" -ge "${GPU_BUSY_THRESHOLD_MB}" ]]; then
        echo "  GPU ${i}: ${used:-unknown} MB used — BUSY (threshold ${GPU_BUSY_THRESHOLD_MB}). Aborting."
        echo "  Set GPU_BUSY_THRESHOLD_MB to override (not recommended)."
        exit 2
    fi
    echo "  GPU ${i}: ${used} MB used — ok"
done
echo

# ---- 2. One-time scan (skip if fresh snapshot exists) ----------------------
# Clean up any stray .partial from a previous SIGKILLed scan.
rm -f "${ITEMS_FILE}.partial"

if [[ "${FORCE_RESCAN}" == "1" || ! -s "${ITEMS_FILE}" ]]; then
    echo "scanning ${IMG_DIR} -> ${ITEMS_FILE} (one-time) ..."
    python - <<PYEOF
import time
from pathlib import Path
from data_miner.embeddings.run_siglip2_embed_standalone import scan_images, items_write_file
t0 = time.perf_counter()
items = scan_images(Path("${IMG_DIR}"))
items_write_file(Path("${ITEMS_FILE}"), items)
print(f"  {len(items)} items written in {time.perf_counter()-t0:.1f}s")
PYEOF
else
    echo "reusing existing snapshot: ${ITEMS_FILE} ($(wc -l <"${ITEMS_FILE}") items)"
    echo "  (set FORCE_RESCAN=1 to rebuild)"
fi
echo

# ---- 3. Launch shards ------------------------------------------------------
pids=()

cleanup() {
    local sig="${1:-TERM}"
    echo
    echo "caught signal; sending SIG${sig} to shards for graceful flush..."
    if [[ "${#pids[@]}" -gt 0 ]]; then
        kill "-${sig}" "${pids[@]}" 2>/dev/null || true
        # give shards up to 30s to flush + exit on SIGTERM
        for _ in $(seq 1 30); do
            alive=0
            for pid in "${pids[@]}"; do
                kill -0 "${pid}" 2>/dev/null && alive=$((alive+1)) || true
            done
            [[ "${alive}" -eq 0 ]] && break
            sleep 1
        done
        # anything still alive gets SIGKILL
        for pid in "${pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
        done
    fi
    exit 130
}
trap 'cleanup TERM' INT TERM HUP

for i in $(seq 0 $((N-1))); do
    log_file="${LOG_DIR}/embed_gpu${i}.log"
    pid_file="${LOG_DIR}/embed_gpu${i}.pid"
    echo "launching shard ${i}/${N} on GPU ${i} -> ${log_file}"

    # Append to logs so crash evidence from previous runs survives.
    {
        echo
        echo "=== shard ${i}/${N} start $(date --iso-8601=seconds) ==="
    } >> "${log_file}"

    # Deliberately NOT using nohup — we want shards to receive our SIGTERM
    # if the user Ctrl-Cs this launcher, so their signal handlers flush.
    CUDA_VISIBLE_DEVICES="${i}" \
    python -m data_miner.embeddings.run_siglip2_embed_standalone \
        --items-file "${ITEMS_FILE}" \
        --lance-uri "${LANCE_URI}" \
        --table "${TABLE}" \
        --model "${MODEL}" \
        --device cuda:0 \
        --shard-index "${i}" --shard-count "${N}" \
        --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}" --buffer-rows "${BUFFER_ROWS}" \
        --resume \
        ${EXTRA_ARGS} \
        >> "${log_file}" 2>&1 &
    pid=$!
    echo "${pid}" > "${pid_file}"
    pids+=("${pid}")

    # Stagger startup so the 4 initial lance table opens and first flushes
    # don't arrive in lockstep.
    if (( i < N - 1 )) && (( STAGGER_S > 0 )); then
        sleep "${STAGGER_S}"
    fi
done

echo
echo "all ${N} shards launched. PIDs: ${pids[*]}"
echo "tail logs:   tail -F ${LOG_DIR}/embed_gpu*.log"
echo "safe stop:   kill -TERM ${pids[*]}    (sends SIGTERM for graceful flush)"
echo

echo "waiting for all shards to finish..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        rc=$?
        echo "shard pid ${pid} exited non-zero (rc=${rc})"
        fail=1
    fi
done
trap - INT TERM HUP

if (( fail )); then
    echo
    echo "at least one shard failed. Check logs in ${LOG_DIR}/"
    exit 1
fi

echo
echo "all shards done. Next: run compaction + index build:"
echo "  python -m data_miner.embeddings.compact_and_index \\"
echo "    --lance-uri '${LANCE_URI}' --table '${TABLE}' --accelerator cuda"
