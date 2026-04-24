#!/usr/bin/env bash
# Import fl_pj's standalone labels_standalone/*.txt into pipeline.db so
# filter/evaluate/finalize can run without re-doing the SAM3-DART GPU
# inference (~1-2h on 8 GPUs).
#
# Equivalent to running detect via the v4 pipeline but ~instant (reads
# ~38k .txt files + inserts rows, ~5-10s end-to-end).
#
# Pre-reqs:
#   - output/auto_annotation_v4/fl_pj_frames_dedup_v1_cls_0_85/ has
#     config.yaml + labels_standalone/*.txt (standalone SAM3-DART output)
#
# After this finishes, run:
#   ./scripts_bench/run_flpj_evaluate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOB_ID="fl_pj_frames_dedup_v1_cls_0_85"
IMAGE_DIR="/media/data_2/datasets/datasets_pavan/fl_pj/frames_dedup_v1_cls_0.85"
OVERRIDE="data_miner/auto_annotation_v4/configs/overrides/flpj_rules.yaml"
JOB_DIR="output/auto_annotation_v4/${JOB_ID}"
LABELS_DIR="${JOB_DIR}/labels_standalone"

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "ERROR: image dir not found: $IMAGE_DIR" >&2
    exit 1
fi
if [[ ! -d "$LABELS_DIR" ]]; then
    echo "ERROR: labels dir not found: $LABELS_DIR" >&2
    echo "Run launch_sam3_dart_fl_pj.sh first to produce labels_standalone/" >&2
    exit 1
fi
if [[ ! -f "$JOB_DIR/config.yaml" ]]; then
    echo "ERROR: job config.yaml not found: $JOB_DIR/config.yaml" >&2
    exit 1
fi

# Warn if pipeline.db already exists — importer is idempotent (INSERT OR
# REPLACE) but the operator should know we're overwriting.
if [[ -f "$JOB_DIR/pipeline.db" ]]; then
    mkdir -p "$JOB_DIR/backups"
    TS="$(date +%Y%m%d-%H%M%S)"
    cp "$JOB_DIR/pipeline.db" "$JOB_DIR/backups/pipeline-pre-import-${TS}.db"
    echo "existing pipeline.db backed up to: $JOB_DIR/backups/pipeline-pre-import-${TS}.db"
fi

N_LABELS=$(ls "$LABELS_DIR" | wc -l)
N_IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)
echo "labels: $N_LABELS"
echo "images: $N_IMAGES"

python -m data_miner.auto_annotation_v4.scripts.import_yolo_to_proposals \
    "$JOB_DIR" \
    --image-dir "$IMAGE_DIR" \
    --labels-dir "$LABELS_DIR" \
    --config "$OVERRIDE" \
    --model sam3_dart \
    --skip-image-size

echo ""
echo "Import complete. Next:"
echo "  1. Start the VLM tunnel:   ./scripts_bench/vlm_tunnel_flpj.sh --detach"
echo "  2. Run filter+evaluate:    ./scripts_bench/run_flpj_evaluate.sh"
