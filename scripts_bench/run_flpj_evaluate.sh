#!/usr/bin/env bash
# Run filter + evaluate + finalize for fl_pj through a local SSH tunnel
# to the remote VLM at pavan@10.160.97.10:8956. The tunnel forwards
# localhost:8958 -> remote 127.0.0.1:8956 and must be running before
# this script is invoked.
#
# Pre-reqs (run in order):
#   1. ./scripts_bench/run_flpj_import.sh           # seeds pipeline.db
#   2. ./scripts_bench/vlm_tunnel_flpj.sh --detach  # opens :8958 tunnel
#   3. this script
#
# If the remote VLM model/port differs, override via env:
#   VLM_URL=http://localhost:8958/v1 \
#   VLM_MODEL=Qwen/Qwen3.5-27B-FP8 \
#     ./scripts_bench/run_flpj_evaluate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOB_ID="fl_pj_frames_dedup_v1_cls_0_85"
IMAGE_DIR="/media/data_2/datasets/datasets_pavan/fl_pj/frames_dedup_v1_cls_0.85"
OVERRIDE="data_miner/auto_annotation_v4/configs/overrides/flpj_rules.yaml"

# Local tunnel endpoint (vlm_tunnel_flpj.sh forwards 8958 -> remote 8956).
VLM_URL="${VLM_URL:-http://localhost:8958/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.5-27B-FP8}"

DB_DIR="output/auto_annotation_v4/${JOB_ID}"
DB_PATH="${DB_DIR}/pipeline.db"
BACKUP_DIR="${DB_DIR}/backups"
TS="$(date +%Y%m%d-%H%M%S)"

# ---- 1. sanity checks ----
if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: pipeline.db not found at $DB_PATH" >&2
    echo "Import detect proposals first:  ./scripts_bench/run_flpj_import.sh" >&2
    exit 1
fi

if ! curl -fsS -o /dev/null --max-time 5 "${VLM_URL%/v1}/health"; then
    echo "ERROR: VLM tunnel not responding at ${VLM_URL%/v1}/health" >&2
    echo "Start it first:" >&2
    echo "  ./scripts_bench/vlm_tunnel_flpj.sh --detach" >&2
    echo "  ./scripts_bench/vlm_tunnel_flpj.sh --status" >&2
    echo "Or override with:  VLM_URL=http://<host>:<port>/v1 $0" >&2
    exit 1
fi

# Stale lock check
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
echo "Validating fl_pj config..."
python - <<PY || { echo "Config failed validation — fix before launching."; exit 1; }
import yaml
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

# Pre-filter stats so the operator knows what's about to hit the VLM.
# After filter runs, these numbers change; this is the detect-stage
# proposal count.
echo "Detect-stage proposals (pre-filter):"
python - "$DB_PATH" <<'PY' || true
import json, sqlite3, sys
db = sys.argv[1]
try:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n_detect = con.execute("SELECT COUNT(*) FROM stages WHERE stage='detect'").fetchone()[0]
    total_cands = 0
    for (data,) in con.execute("SELECT data FROM stages WHERE stage='detect'"):
        d = json.loads(data)
        total_cands += len(d.get("candidates", []))
    print(f"  images with detect: {n_detect}")
    print(f"  total proposals: {total_cands}")
except Exception as e:
    print(f"  (unavailable: {e})")
PY

# ---- 3. backup DB ----
mkdir -p "$BACKUP_DIR" logs
cp "$DB_PATH" "${BACKUP_DIR}/pipeline-pre-evaluate-v3-${TS}.db"
echo "Backup: ${BACKUP_DIR}/pipeline-pre-evaluate-v3-${TS}.db"

# ---- 4. launch filter + evaluate + finalize ----
LOG="logs/evaluate-flpj-${TS}.log"
echo "Launching filter + evaluate + finalize → ${LOG}"
echo "VLM: ${VLM_URL} model=${VLM_MODEL}"

setsid nohup python -u -m data_miner.auto_annotation_v4 \
    --config "$OVERRIDE" \
    runtime.image_dir="$IMAGE_DIR" \
    runtime.job_id="$JOB_ID" \
    'runtime.stages=[filter,evaluate,finalize]' \
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
