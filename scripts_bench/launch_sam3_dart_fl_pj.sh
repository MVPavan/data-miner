#!/usr/bin/env bash
# launch_sam3_dart_fl_pj.sh
#
# Full-class (all 23 classes, 36 prompts) SAM3-DART 8-GPU run on
#   /media/data_2/datasets/datasets_pavan/fl_pj/frames_dedup_v1_cls_0.85
# Detached via nohup+setsid — survives terminal close, SSH drop, parent
# shell exit. Only machine-level failures can stop it.
#
# Env overrides (optional):
#   BATCH_SIZE   default 8   (drop to 4 if OOM — full class uses 7x more prompts
#                             than the 4-class Good2 bench, so decoder memory is
#                             much higher)
#   NUM_WORKERS  default 8
#   CONFIDENCE   default 0.25
#   NMS          default 0.7

set -euo pipefail

REPO="/media/data_2/vlm/code/data_miner"
cd "${REPO}"

IMG_DIR="/media/data_2/datasets/datasets_pavan/fl_pj/frames_dedup_v1_cls_0.85"
JOB_ID="fl_pj_frames_dedup_v1_cls_0_85"
JOB_DIR="output/auto_annotation_v4/${JOB_ID}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CONFIDENCE="${CONFIDENCE:-0.25}"
NMS="${NMS:-0.7}"

# --- sanity checks ---
if [[ ! -d "${IMG_DIR}" ]]; then
    echo "ERROR: image dir not found: ${IMG_DIR}" >&2
    exit 1
fi

N_IMAGES=$(find "${IMG_DIR}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)
if [[ "${N_IMAGES}" -eq 0 ]]; then
    echo "ERROR: no images found under ${IMG_DIR}" >&2
    exit 1
fi

# --- seed job dir ---
mkdir -p "${JOB_DIR}/logs"
if [[ ! -f "${JOB_DIR}/config.yaml" ]]; then
    echo "seeding config.yaml from s69_base (full 23-class registry)"
    cp output/auto_annotation_v4/s69_base/config.yaml "${JOB_DIR}/config.yaml"
fi

TS=$(date +%Y%m%d-%H%M%S)
OUTER_LOG="${JOB_DIR}/logs/outer-${TS}.log"

# --- launch detached ---
# nohup+setsid+disown: re-parents to init (PPID=1), no controlling terminal.
nohup setsid bash -c "
    IMG_DIR='${IMG_DIR}' \
    JOB_ID='${JOB_ID}' \
    GPUS='0,1,2,3,4,5,6,7' \
    SHARDS=8 \
    BATCH_SIZE='${BATCH_SIZE}' \
    NUM_WORKERS='${NUM_WORKERS}' \
    CONFIDENCE='${CONFIDENCE}' \
    data_miner/auto_annotation_v4/scripts/run_sam3_dart_multi_gpu.sh \
        --nms '${NMS}' \
        --resume
" </dev/null >"${OUTER_LOG}" 2>&1 &
MASTER_PID=$!
disown

sleep 2

echo "=== launched ==="
echo "  master_pid : ${MASTER_PID}"
echo "  job_id     : ${JOB_ID}"
echo "  dataset    : ${IMG_DIR}"
echo "  images     : ${N_IMAGES}"
echo "  config     : bs=${BATCH_SIZE} workers=${NUM_WORKERS} conf=${CONFIDENCE} nms=${NMS}"
echo "  outer_log  : ${OUTER_LOG}"
echo "  labels_dir : ${JOB_DIR}/labels_standalone/"
echo ""
echo "=== monitor ==="
echo "  cat ${OUTER_LOG}"
echo "  tail -f ${JOB_DIR}/logs/sam3_dart_shard0-${TS}.log   # wait ~90s for shard log"
echo "  watch -n2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
echo "  watch -n10 'ls ${JOB_DIR}/labels_standalone | wc -l'"
echo ""
echo "=== stop cleanly ==="
echo "  kill -TERM -${MASTER_PID}    # note leading dash = kill whole process group"
echo ""
echo "=== if any shard OOMs ==="
echo "  kill the job, then rerun with smaller batch:"
echo "    BATCH_SIZE=4 scripts_bench/launch_sam3_dart_fl_pj.sh"
echo "  (--resume will skip images already done)"
