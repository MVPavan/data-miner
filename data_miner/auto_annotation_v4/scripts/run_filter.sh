#!/usr/bin/env bash
#
# run_filter.sh
#
# Run the filter stage (CPU-only, no GPU servers required) on a detect-complete
# job, with the sam3_dart-only source_model allowlist + 0.85 high-confidence
# auto-accept. After the pipeline exits, takes a WAL-safe backup of
# pipeline.db into {job_dir}/backups/pipeline-post-filter-<TS>.db.
#
# Usage:
#   scripts/run_filter.sh <loco|datatang> [OVERRIDES...]
#
# Example — both datasets in parallel:
#   scripts/run_filter.sh loco     > /tmp/flt-loco.log 2>&1 &
#   scripts/run_filter.sh datatang > /tmp/flt-dt.log 2>&1 &
#   wait

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <loco|datatang|ava_sampled|animal_person|nwpu_campus> [OVERRIDES...]" >&2
    exit 2
fi

DATASET="$1"; shift
case "$DATASET" in
    loco)
        IMG_DIR=/media/data_2/datasets/datasets_pavan/loco_unannotated_full_sam_filtered
        JOB_ID=loco_unannotated_full_sam_filtered_detect
        ;;
    datatang)
        IMG_DIR=/media/data_2/datasets/datasets_pavan/DataTang_val
        JOB_ID=datatang_val_detect
        ;;
    ava_sampled)
        IMG_DIR=/media/data_2/datasets/datasets_pavan/AVA_sampled
        JOB_ID=ava_sampled_detect
        ;;
    animal_person)
        IMG_DIR=/media/data_2/datasets/datasets_pavan/animal_person_manual_verification
        JOB_ID=animal_person_detect
        ;;
    nwpu_campus)
        IMG_DIR=/media/data_2/datasets/datasets_pavan/NWPUCampus_sampled
        JOB_ID=nwpu_campus_detect
        ;;
    *)
        echo "ERROR: unknown dataset '$DATASET'" >&2
        exit 2
        ;;
esac

JOB_OUT="${JOB_OUT:-output/auto_annotation_v4/${JOB_ID}}"
LOG_DIR="${JOB_OUT}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/filter-${JOB_ID}-${TS}.log"
mkdir -p "$LOG_DIR"

echo "[filter] dataset=${DATASET} job=${JOB_ID} log=${LOG_FILE}"

# runtime.stages=[filter] skips detect entirely; the submitter's per-stage
# hash check only fires for stages in runtime.stages, so the (already-done)
# detect stage is NOT invalidated by the config hash bump.
#
# filtering.allowed_source_models=[sam3_dart] is redundant with servers.yaml
# gdino enabled, but belt-and-braces: FilterPipeline step 1 drops gdino
# cands; merge-time filter is a no-op since detect.json already has gdino.
# Caller args go FIRST so a leading `--config X` (argparse optional) is parsed
# before the positional dotlist overrides — argparse with `nargs='*'`
# positionals rejects optionals appearing in the middle of the positional list.
python -u -m data_miner.auto_annotation_v4 \
    "$@" \
    runtime.image_dir="${IMG_DIR}" \
    runtime.job_id="${JOB_ID}" \
    'runtime.stages=[filter]' \
    'filtering.allowed_source_models=[sam3_dart]' \
    2>&1 | tee -a "$LOG_FILE"

# ---------- post-completion backup ----------
# WAL-safe online backup via Python sqlite3 .backup(); no sqlite3 CLI needed.
BAK_DIR="${JOB_OUT}/backups"
mkdir -p "$BAK_DIR"
BAK_PATH="${BAK_DIR}/pipeline-post-filter-${TS}.db"
python - <<PY
import sqlite3, os
src = "${JOB_OUT}/pipeline.db"
dst = "${BAK_PATH}"
src_con = sqlite3.connect(src)
dst_con = sqlite3.connect(dst)
with dst_con:
    src_con.backup(dst_con)
src_con.close(); dst_con.close()
print(f"[backup] {dst}  ({os.path.getsize(dst)/(1024*1024):.1f} MiB)")
PY

echo "[filter] done dataset=${DATASET}"
