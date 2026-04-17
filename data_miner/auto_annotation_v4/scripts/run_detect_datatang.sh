#!/usr/bin/env bash
#
# run_detect_datatang.sh
#
# Run the detect stage only (no filter/evaluate/refine/finalize) on
# DataTang_val using grounding_dino + sam3_dart.
#
# Assumes scripts/start_detect_servers.sh has already launched the
# servers on their default ports (3001 GDINO, 3013 SAM3-DART).
#
# Usage:
#   scripts/run_detect_datatang.sh [OVERRIDES...]
#
# Example: pass extra OmegaConf overrides on top of the defaults:
#   scripts/run_detect_datatang.sh workers.detect_per_model=4

set -euo pipefail

IMG_DIR=/media/data_2/datasets/datasets_pavan/DataTang_val
JOB_ID=datatang_val_detect
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
    'runtime.detect_models=[grounding_dino,sam3_dart]' \
    "$@" 2>&1 | tee -a "$LOG_FILE"
