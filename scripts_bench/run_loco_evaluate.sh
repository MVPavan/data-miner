#!/usr/bin/env bash
# Run evaluate + finalize on the LOCO job with the updated v3 prompts,
# split-threshold routing, and JPEG q=90 image prep.
#
# Pre-reqs:
#   - VLM LB running on http://localhost:8956 (Qwen/Qwen3.5-27B-FP8)
#   - filter stage already applied with loco_rules.yaml (existing DB)
#
# What this does:
#   1. Backs up pipeline.db before touching it.
#   2. Pings the VLM LB to confirm it's up.
#   3. Launches evaluate + finalize in the background with force_stages=[evaluate]
#      so any previous evaluate checkpoints are discarded (so the new v3 prompt
#      + new routing matrix run fresh).
#
# Usage:
#   ./scripts_bench/run_loco_evaluate.sh
#
# Monitor:
#   tail -f logs/evaluate-loco-*.log
#   curl http://localhost:8956/health
#   sqlite3 output/auto_annotation_v4/loco_unannotated_full_sam_filtered_detect/pipeline.db \
#     "SELECT stage, COUNT(*) FROM stages GROUP BY stage;"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOB_ID="loco_unannotated_full_sam_filtered_detect"
IMAGE_DIR="/media/data_2/datasets/datasets_pavan/loco_unannotated_full_sam_filtered"
OVERRIDE="data_miner/auto_annotation_v4/configs/overrides/loco_rules.yaml"
VLM_URL="http://localhost:8956/v1"
VLM_MODEL="Qwen/Qwen3.5-27B-FP8"

DB_DIR="output/auto_annotation_v4/${JOB_ID}"
DB_PATH="${DB_DIR}/pipeline.db"
BACKUP_DIR="${DB_DIR}/backups"
TS="$(date +%Y%m%d-%H%M%S)"

# ---- 1. sanity checks ----
if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: pipeline.db not found at $DB_PATH" >&2
    echo "Run detect + filter first." >&2
    exit 1
fi

if ! curl -fsS -o /dev/null --max-time 3 "${VLM_URL%/v1}/health"; then
    echo "ERROR: VLM LB not responding at ${VLM_URL%/v1}/health" >&2
    echo "Start it first (see scripts_bench/launch_sam3_dart_fl_pj.sh or the nginx conf)." >&2
    exit 1
fi

# Stale lock guard — pre-flight (the pipeline will also fail clearly if held).
LOCK="${DB_DIR}/pipeline.lock"
if [[ -f "$LOCK" ]]; then
    LOCK_PID="$(cat "$LOCK" 2>/dev/null || echo '')"
    if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "ERROR: pipeline.lock held by live PID $LOCK_PID — another run is active." >&2
        exit 1
    fi
    echo "WARN: stale pipeline.lock (PID ${LOCK_PID:-unknown} gone); removing."
    rm -f "$LOCK"
fi

# ---- 2. config pre-flight ----
# The H2 cross-section validator catches typos (e.g. per_class_min_area.Head,
# per_model_score.sam3_drt) and coexistence-without-score-gate. Validate
# BEFORE we spend VLM compute.
echo "Validating LOCO config..."
python - <<PY || { echo "Config failed validation — fix before launching."; exit 1; }
import sys, yaml
from pathlib import Path
from data_miner.auto_annotation_v4.configs import AutoAnnotationV4Config

def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

merged = {}
for p in (
    "data_miner/auto_annotation_v4/configs/default.yaml",
    "data_miner/auto_annotation_v4/configs/class_config.yaml",
    "${OVERRIDE}",
):
    with open(p) as f:
        merged = merge(merged, yaml.safe_load(f) or {})
AutoAnnotationV4Config.model_validate(merged)
print("  OK — thresholds, registry keys, detector keys, coexistence coupling all valid.")
PY

# Candidate breakdown so the operator knows what's about to hit the VLM.
# Uses python's sqlite3 (stdlib) so this works on boxes without the CLI.
# Failures here are informational only — never block the launch.
echo "Current routing split (from filter checkpoint):"
python - "$DB_PATH" <<'PY' || true
import json, sqlite3, sys
db = sys.argv[1]
try:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    auto = vlm = 0
    for (data,) in con.execute("SELECT data FROM stages WHERE stage='filter'"):
        d = json.loads(data)
        r = d.get("routing", {})
        auto += len(r.get("auto_accepted", []))
        vlm += len(r.get("needs_evaluation", []))
    print(f"  auto_accepted: {auto}    needs_evaluation (→VLM): {vlm}")
except Exception as e:
    print(f"  (breakdown unavailable: {e})")
PY

# ---- 3. backup DB ----
mkdir -p "$BACKUP_DIR" logs
cp "$DB_PATH" "${BACKUP_DIR}/pipeline-pre-evaluate-v3-${TS}.db"
echo "Backup: ${BACKUP_DIR}/pipeline-pre-evaluate-v3-${TS}.db"

# ---- 4. launch evaluate + finalize ----
LOG="logs/evaluate-loco-${TS}.log"
echo "Launching evaluate + finalize → ${LOG}"

setsid nohup python -u -m data_miner.auto_annotation_v4 \
    --config "$OVERRIDE" \
    runtime.image_dir="$IMAGE_DIR" \
    runtime.job_id="$JOB_ID" \
    'runtime.stages=[evaluate,finalize]' \
    'runtime.force_stages=[evaluate]' \
    servers.vlm.url="$VLM_URL" \
    servers.vlm.model="$VLM_MODEL" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "PID: $PID"
echo ""
echo "Monitor:"
echo "  tail -f $LOG"
echo "  curl ${VLM_URL%/v1}/health"
echo "  pgrep -af 'data_miner.auto_annotation_v4.*${JOB_ID}'"
