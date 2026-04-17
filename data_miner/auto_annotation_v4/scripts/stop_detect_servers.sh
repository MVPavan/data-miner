#!/usr/bin/env bash
#
# stop_detect_servers.sh
#
# Terminate the detector servers launched by start_detect_servers.sh.
# Reads PID files from LOG_DIR; falls back to pkill by module name.
#
# Usage:
#   scripts/stop_detect_servers.sh [LOG_DIR]
#
# Defaults: LOG_DIR=logs/aav4_detect_servers

set -u

LOG_DIR="${1:-logs/aav4_detect_servers}"

kill_pidfile() {
    local pidfile="$1" name="$2"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[stop] ${name} pid=${pid} — SIGTERM"
            kill -TERM "$pid"
        fi
        rm -f "$pidfile"
    fi
}

kill_pidfile "${LOG_DIR}/sam3_dart.pid"      sam3_dart
kill_pidfile "${LOG_DIR}/grounding_dino.pid" grounding_dino

# belt-and-suspenders: any stray workers
pkill -TERM -f "data_miner.auto_annotation_v4.model_servers.serve" 2>/dev/null || true

echo "[stop] done."
