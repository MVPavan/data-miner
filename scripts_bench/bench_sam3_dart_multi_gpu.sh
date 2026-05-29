#!/usr/bin/env bash
# Benchmark SAM3-DART on all 8 GPUs in the real shard layout across a grid of
# (batch, workers). Measures aggregate throughput so the recommendation
# reflects the actual 8-shard CPU contention, not a single-GPU best case.
set -euo pipefail

REPO="/media/data_2/vlm/code/data_miner"
PY="${REPO}/.venv/bin/python"
IMG_DIR="/media/data_1/deepak_cr/LaionFiltered/Good2/Subset115"
JOB_DIR="${REPO}/output/auto_annotation_v4/bench_laion_good2"
BENCH_ROOT="${JOB_DIR}/bench_runs"
CLASSES="forklift,palletjack,shopping cart,dog"
LIMIT_PER_SHARD="${LIMIT:-600}"
LOG_EVERY="${LOG_EVERY:-5}"
SHARDS=8
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

mkdir -p "${BENCH_ROOT}"
SUMMARY="${JOB_DIR}/bench_multi_summary-$(date +%Y%m%d-%H%M%S).tsv"
printf "batch\tworkers\ttotal_imgs\twall_s\tagg_img_s\tper_shard_mean\tper_shard_min\tper_shard_max\tgpu_mem_peak_mib\n" >"${SUMMARY}"

IFS=',' read -r -a GPU_LIST <<<"${GPUS}"

COMBOS=(
  "8 6"
  "16 8"
  "16 12"
  "24 12"
  "32 12"
  "32 16"
  "48 12"
)

cd "${REPO}"

for combo in "${COMBOS[@]}"; do
  read -r BS NW <<<"${combo}"
  TAG="bs${BS}_nw${NW}"
  RUN_DIR="${BENCH_ROOT}/multi_${TAG}"
  OUT_DIR="${RUN_DIR}/labels"
  rm -rf "${RUN_DIR}"
  mkdir -p "${OUT_DIR}"

  echo ">>> ${TAG}: 8 shards x limit=${LIMIT_PER_SHARD} each"

  PIDS=()
  T_START=$(date +%s.%N)
  for (( i=0; i<SHARDS; i++ )); do
    GPU="${GPU_LIST[$i]}"
    LOG_FILE="${RUN_DIR}/shard${i}.log"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PY}" -u -m data_miner.auto_annotation_v4.models.run_sam3_dart_standalone \
        --image-dir "${IMG_DIR}" \
        --job-dir   "${JOB_DIR}" \
        --out-dir   "${OUT_DIR}" \
        --classes   "${CLASSES}" \
        --shard-index "${i}" --shard-count "${SHARDS}" \
        --batch-size "${BS}" --num-workers "${NW}" --prefetch-factor 4 \
        --device "cuda:0" --confidence 0.5 \
        --limit "${LIMIT_PER_SHARD}" --log-every "${LOG_EVERY}" \
        >"${LOG_FILE}" 2>&1 &
    PIDS+=($!)
  done

  # Peak GPU memory across all 8 GPUs while the shards run
  PEAK_MEM=0
  ANY_RUNNING=1
  while (( ANY_RUNNING )); do
    ANY_RUNNING=0
    for p in "${PIDS[@]}"; do
      if kill -0 "${p}" 2>/dev/null; then ANY_RUNNING=1; break; fi
    done
    MEM_MAX=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -n | tail -1)
    if [[ -n "${MEM_MAX}" && "${MEM_MAX}" -gt "${PEAK_MEM}" ]]; then
      PEAK_MEM="${MEM_MAX}"
    fi
    sleep 2
  done

  FAIL=0
  for p in "${PIDS[@]}"; do
    if ! wait "${p}"; then FAIL=1; fi
  done
  T_END=$(date +%s.%N)
  WALL=$(python3 -c "print(f'{${T_END}-${T_START}:.1f}')")

  if (( FAIL )); then
    echo "    !! ${TAG} one or more shards failed"
  fi

  # Parse each shard's "wall time: X  (avg Y img/s)" line
  declare -a PER_SHARD_RATES
  PER_SHARD_RATES=()
  TOTAL_IMGS=0
  for (( i=0; i<SHARDS; i++ )); do
    L="${RUN_DIR}/shard${i}.log"
    R=$(grep -oE 'avg [0-9.]+ img/s' "${L}" | tail -1 | awk '{print $2}')
    N=$(grep -oE 'images processed: [0-9]+' "${L}" | tail -1 | awk '{print $3}')
    PER_SHARD_RATES+=("${R:-0}")
    TOTAL_IMGS=$((TOTAL_IMGS + ${N:-0}))
  done

  # Aggregate using the wall-clock time (true parallel throughput)
  if [[ -n "${TOTAL_IMGS}" && "${TOTAL_IMGS}" -gt 0 ]]; then
    AGG=$(python3 -c "print(f'{${TOTAL_IMGS}/${WALL}:.1f}')")
  else
    AGG="NA"
  fi

  # Mean/min/max of per-shard rates
  MEAN=$(printf "%s\n" "${PER_SHARD_RATES[@]}" | awk '{s+=$1; n++} END {printf "%.2f", s/n}')
  MIN=$(printf "%s\n" "${PER_SHARD_RATES[@]}" | sort -g | head -1)
  MAX=$(printf "%s\n" "${PER_SHARD_RATES[@]}" | sort -g | tail -1)

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${BS}" "${NW}" "${TOTAL_IMGS}" "${WALL}" "${AGG}" "${MEAN}" "${MIN}" "${MAX}" "${PEAK_MEM}" \
    | tee -a "${SUMMARY}"
done

echo
echo "=== SUMMARY (sorted by agg_img_s desc) ==="
{ head -1 "${SUMMARY}"; tail -n +2 "${SUMMARY}" | sort -k5 -gr; } | column -t -s $'\t'
echo
echo "tsv: ${SUMMARY}"
