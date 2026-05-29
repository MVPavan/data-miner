#!/usr/bin/env bash
#
# run_detect_datatang.sh
#
# Run the detect stage only (no filter/evaluate/refine/finalize) on
# Laion Good1 using sam3_dart.
#
# Assumes scripts/start_detect_servers.sh has already launched the
# servers on their default ports (3013 SAM3-DART).
#
# Usage:
#   scripts/run_detect_datatang.sh [OVERRIDES...]
#
# Example: pass extra OmegaConf overrides on top of the defaults:
#   scripts/run_detect_datatang.sh workers.detect_per_model=4

set -euo pipefail

IMG_DIR=/mnt/data/deepak/DATASET/Laion/Good1/good1
JOB_ID=laion_good1
JOB_OUT="${JOB_OUT:-output/auto_annotation_v4/${JOB_ID}}"
LOG_DIR="${JOB_OUT}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/${JOB_ID}-${TS}.log"
mkdir -p "$LOG_DIR"

echo "[run] job=${JOB_ID} log=${LOG_FILE}"

python -u -m data_miner.auto_annotation_v4 \
    runtime.image_dir="${IMG_DIR}" \
    runtime.job_id="${JOB_ID}" \
    'runtime.stages=[detect]' \
    'runtime.detect_models=[sam3_dart]' \
    "$@" 2>&1 | tee -a "$LOG_FILE"
