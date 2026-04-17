#!/usr/bin/env bash
#
# stop_detect_servers.sh
#
# Terminate ALL detector model-server processes, whether or not the pidfile
# is still current. Pidfile-only kills leave orphans when the start script
# is run more than once; we hunt by command-line pattern for belt-and-suspenders
# safety, then wait for GPU contexts to drain.
#
# Usage:
#   scripts/stop_detect_servers.sh [LOG_DIR]
#
# Defaults: LOG_DIR=logs/aav4_detect_servers

set -u

LOG_DIR="${1:-logs/aav4_detect_servers}"
MODULE_PATTERN='data_miner.auto_annotation_v4.model_servers'

# ---------- step 1: pidfile-based SIGTERM (fast path for clean stops) ----------
kill_pidfile() {
    local pidfile="$1" name="$2"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[stop] ${name} pid=${pid} (pidfile) — SIGTERM"
            kill -TERM "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}
kill_pidfile "${LOG_DIR}/sam3_dart.pid"      sam3_dart
kill_pidfile "${LOG_DIR}/grounding_dino.pid" grounding_dino

# ---------- step 2: SIGTERM every process matching the module pattern ----------
# Catches launchers spawned by earlier start_detect_servers.sh calls whose
# pidfiles were overwritten by later calls, plus LitServe worker children.
TERMED=$(pgrep -f "$MODULE_PATTERN" | tr '\n' ' ')
if [[ -n "${TERMED// }" ]]; then
    echo "[stop] SIGTERM pattern matches: $TERMED"
    # shellcheck disable=SC2086
    kill -TERM $TERMED 2>/dev/null || true
fi

# ---------- step 3: wait up to 15s for graceful exit, then SIGKILL ----------
for _ in $(seq 1 15); do
    REMAINING=$(pgrep -f "$MODULE_PATTERN" | tr '\n' ' ')
    if [[ -z "${REMAINING// }" ]]; then
        break
    fi
    sleep 1
done

REMAINING=$(pgrep -f "$MODULE_PATTERN" | tr '\n' ' ')
if [[ -n "${REMAINING// }" ]]; then
    echo "[stop] SIGKILL stragglers: $REMAINING"
    # shellcheck disable=SC2086
    kill -KILL $REMAINING 2>/dev/null || true
fi

# ---------- step 4: wait briefly for CUDA contexts to drain ----------
sleep 3

# ---------- step 5: verify ----------
FINAL=$(pgrep -f "$MODULE_PATTERN" | tr '\n' ' ')
if [[ -n "${FINAL// }" ]]; then
    echo "[stop] WARNING: still running after kill — pids: $FINAL" >&2
    exit 1
fi
echo "[stop] all model-server processes terminated."
