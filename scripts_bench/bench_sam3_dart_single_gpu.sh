#!/usr/bin/env bash
# Benchmark SAM3-DART standalone runner on ONE GPU across a grid of
# (batch_size, num_workers) combos, using 4 classes on a sample from
# LaionFiltered/Good2/Subset115.
#
# Picks the best combo by the max steady-state window throughput
# (skipping the first window to exclude warmup).
set -euo pipefail

REPO="/media/data_2/vlm/code/data_miner"
PY="${REPO}/.venv/bin/python"
IMG_DIR="/media/data_1/deepak_cr/LaionFiltered/Good2/Subset115"
JOB_DIR="${REPO}/output/auto_annotation_v4/bench_laion_good2"
BENCH_ROOT="${JOB_DIR}/bench_runs"
CLASSES="forklift,palletjack,shopping cart,dog"
LIMIT="${LIMIT:-1600}"
LOG_EVERY="${LOG_EVERY:-20}"
GPU="${GPU:-0}"

mkdir -p "${BENCH_ROOT}"
SUMMARY="${JOB_DIR}/bench_summary-$(date +%Y%m%d-%H%M%S).tsv"
printf "batch\tworkers\twarmup_s\twin_best_img_s\toverall_img_s\tgpu_mem_peak_mib\n" >"${SUMMARY}"

# (batch, workers) grid — widen if bottleneck is clear
COMBOS=(
  "8 6"
  "16 8"
  "16 12"
  "24 12"
  "32 12"
  "32 16"
  "48 16"
)

for combo in "${COMBOS[@]}"; do
  read -r BS NW <<<"${combo}"
  TAG="bs${BS}_nw${NW}"
  OUT_DIR="${BENCH_ROOT}/${TAG}/labels"
  LOG_FILE="${BENCH_ROOT}/${TAG}.log"
  rm -rf "${BENCH_ROOT}/${TAG}"
  mkdir -p "${OUT_DIR}"

  echo ">>> running ${TAG} (limit=${LIMIT}) on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PY}" -u -m data_miner.auto_annotation_v4.models.run_sam3_dart_standalone \
      --image-dir "${IMG_DIR}" \
      --job-dir   "${JOB_DIR}" \
      --out-dir   "${OUT_DIR}" \
      --classes   "${CLASSES}" \
      --batch-size "${BS}" \
      --num-workers "${NW}" \
      --prefetch-factor 4 \
      --device "cuda:0" \
      --confidence 0.5 \
      --limit "${LIMIT}" \
      --log-every "${LOG_EVERY}" \
      >"${LOG_FILE}" 2>&1 &
  PID=$!

  # Track peak GPU memory on the real device we're using
  REAL_GPU="${GPU}"
  PEAK_MEM=0
  while kill -0 "${PID}" 2>/dev/null; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${REAL_GPU}" 2>/dev/null | tr -d ' ' || echo 0)
    if [[ -n "${MEM}" && "${MEM}" -gt "${PEAK_MEM}" ]]; then
      PEAK_MEM="${MEM}"
    fi
    sleep 1
  done
  wait "${PID}" || { echo "    !! ${TAG} FAILED — see ${LOG_FILE}"; continue; }

  # Extract window throughputs, skip the first (warmup); take max.
  # Line: "... | win=XX.X img/s overall=YY.Y img/s | batch_latency=... eta=..."
  mapfile -t WINS < <(grep -oE 'win=[0-9.]+ img/s' "${LOG_FILE}" | awk -F'=' '{print $2}' | awk '{print $1}')
  OVERALL=$(grep -oE 'avg [0-9.]+ img/s' "${LOG_FILE}" | tail -1 | awk '{print $2}')
  WARMUP=$(grep -oE 'starting inference' "${LOG_FILE}" >/dev/null && \
           awk '/loading SAM3DartModel/ {t1=$2} /starting inference/ {t2=$2; print} ' "${LOG_FILE}" | head -1 || true)
  # Simpler warmup: time from model-load log to first batch window = model load + scan time
  LOAD_TS=$(awk '/loading SAM3DartModel/ {print $1" "$2; exit}' "${LOG_FILE}")
  START_TS=$(awk '/starting inference/ {print $1" "$2; exit}' "${LOG_FILE}")
  if [[ -n "${LOAD_TS}" && -n "${START_TS}" ]]; then
    WARMUP_S=$(python3 -c "
from datetime import datetime
a=datetime.strptime('${LOAD_TS}','%Y-%m-%d %H:%M:%S,%f')
b=datetime.strptime('${START_TS}','%Y-%m-%d %H:%M:%S,%f')
print(f'{(b-a).total_seconds():.1f}')")
  else
    WARMUP_S="NA"
  fi

  # Best steady-state: max of windows 2..N (skip warmup window 1)
  if (( ${#WINS[@]} >= 2 )); then
    BEST=$(printf "%s\n" "${WINS[@]:1}" | sort -g | tail -1)
  elif (( ${#WINS[@]} == 1 )); then
    BEST="${WINS[0]}"
  else
    BEST="NA"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${BS}" "${NW}" "${WARMUP_S}" "${BEST}" "${OVERALL}" "${PEAK_MEM}" \
    | tee -a "${SUMMARY}"
done

echo
echo "=== SUMMARY (sorted by win_best_img_s) ==="
column -t -s $'\t' "${SUMMARY}" | (head -1; tail -n +2 | sort -k4 -gr)
echo
echo "full tsv: ${SUMMARY}"
