#!/usr/bin/env bash
#
# run_sam3_dart_multi_gpu.sh
#
# Fan out the standalone sam3_dart runner across N GPUs. Each shard:
#   - processes items[shard_idx :: shard_count] of the scanned image list
#   - runs on cuda:0 (seen via CUDA_VISIBLE_DEVICES=${GPU})
#   - writes YOLO .txt files to the SHARED out dir (no filename collisions
#     since each image_id is a distinct file)
#
# Env overrides:
#   IMG_DIR, JOB_ID, JOB_OUT
#   SHARDS (default 8), GPUS (default "0,1,2,3,4,5,6,7")
#   BATCH_SIZE (default 8), NUM_WORKERS (default 6)
#   CONFIDENCE (default 0.5)
#   OUT_DIR (default "<JOB_OUT>/labels_standalone")
#   PYTHON (default .venv/bin/python if present, else python)
#
# Usage:
#   scripts/run_sam3_dart_multi_gpu.sh [EXTRA_ARGS_TO_PY_RUNNER...]
#
# Examples:
#   # Full 8-GPU run, all classes
#   scripts/run_sam3_dart_multi_gpu.sh --resume
#
#   # 4 GPUs, 2-class filter
#   GPUS="0,1,2,3" SHARDS=4 BATCH_SIZE=16 \
#     scripts/run_sam3_dart_multi_gpu.sh --classes forklift,palletjack --resume

set -euo pipefail

IMG_DIR="${IMG_DIR:-/mnt/data/deepak/DATASET/Laion/Good1/good1}"
JOB_ID="${JOB_ID:-laion_good1}"
JOB_OUT="${JOB_OUT:-output/auto_annotation_v4/${JOB_ID}}"
SHARDS="${SHARDS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
CONFIDENCE="${CONFIDENCE:-0.5}"
OUT_DIR="${OUT_DIR:-${JOB_OUT}/labels_standalone}"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
    else PYTHON="python"; fi
fi

LOG_DIR="${JOB_OUT}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR" "$OUT_DIR"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
if (( ${#GPU_LIST[@]} < SHARDS )); then
    echo "[multi-gpu] ERROR: GPUS has ${#GPU_LIST[@]} entries but SHARDS=${SHARDS}" >&2
    exit 2
fi

echo "[multi-gpu] job=${JOB_ID} python=${PYTHON}"
echo "[multi-gpu] shards=${SHARDS}  gpus=${GPUS}  batch=${BATCH_SIZE}  workers=${NUM_WORKERS}"
echo "[multi-gpu] out_dir=${OUT_DIR}"

PIDS=()
for (( i=0; i<SHARDS; i++ )); do
    GPU="${GPU_LIST[$i]}"
    LOG_FILE="${LOG_DIR}/sam3_dart_shard${i}-${TS}.log"
    echo "[multi-gpu] shard ${i} -> GPU ${GPU}  log=${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" -u -m data_miner.auto_annotation_v4.models.run_sam3_dart_standalone \
        --image-dir "${IMG_DIR}" \
        --job-dir "${JOB_OUT}" \
        --out-dir "${OUT_DIR}" \
        --shard-index "${i}" \
        --shard-count "${SHARDS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device "cuda:0" \
        --confidence "${CONFIDENCE}" \
        "$@" \
        >"${LOG_FILE}" 2>&1 &
    PIDS+=($!)
done

echo "[multi-gpu] launched ${#PIDS[@]} shards: ${PIDS[*]}"
echo "[multi-gpu] tail a shard log:  tail -f ${LOG_DIR}/sam3_dart_shard0-${TS}.log"

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        echo "[multi-gpu] shard pid=${PID} FAILED"
        FAIL=1
    fi
done

if (( FAIL )); then
    echo "[multi-gpu] one or more shards failed"
    exit 1
fi

echo "[multi-gpu] done."
