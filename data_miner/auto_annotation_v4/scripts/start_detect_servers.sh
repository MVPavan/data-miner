#!/usr/bin/env bash
#
# start_detect_servers.sh
#
# Launch the two detector model servers used for the detect stage:
#
#   SAM3-DART   port 3013   GPUs 4,5
#   GDINO       port 3001   GPUs 6,7
#
# LitServe spawns one worker per GPU per server, sharing a batch queue
# on the given port. Logs + PIDs land under LOG_DIR.
#
# Usage:
#   scripts/start_detect_servers.sh [LOG_DIR]
#
# Defaults: LOG_DIR=logs/aav4_detect_servers
#
# Stop with scripts/stop_detect_servers.sh

set -euo pipefail

LOG_DIR="${1:-logs/aav4_detect_servers}"
mkdir -p "$LOG_DIR"

PY="${PY:-python}"
MODULE="data_miner.auto_annotation_v4.model_servers.serve"

# # ---------- SAM3-DART on GPUs 4,5 ----------
# # CUDA_VISIBLE_DEVICES isolates this process to GPUs 4,5 only — prevents
# # it from creating ~256 MiB CUDA "ghost contexts" on every other GPU
# # (including 6,7 where GDINO lives, which previously ate GDINO's OOM
# # headroom). Inside the process the visible pair is renumbered 0,1.
# SAM3_LOG="${LOG_DIR}/sam3_dart.log"
# SAM3_PID="${LOG_DIR}/sam3_dart.pid"
# echo "[start] sam3_dart  port=3013 physical-gpus=4,5 (logical 0,1)  log=${SAM3_LOG}"
# CUDA_VISIBLE_DEVICES=4,5 \
# nohup "$PY" -m "$MODULE" \
#     --model sam3_dart \
#     --port 3013 \
#     --gpu 0,1 \
#     --max-batch-size 8 \
#     > "$SAM3_LOG" 2>&1 &
# echo $! > "$SAM3_PID"

# ---------- GDINO on GPUs 6,7 ----------
# Multi-image batched: max-batch=8 + prompt-chunk-size=1 makes every Swin
# forward (8,3,H,W) ~11 GiB peak per worker on 24 GB 3090s. Harness numbers
# (scripts/harness_gdino.py, 43 prompts):
#     B=8 chunk=1  ~11 GiB  ~4.4 s per image  (this config)
#     B=4 chunk=2  ~11 GiB  ~4.7 s per image  (use if requests rarely fill 8)
#     B=1 chunk=4   ~6 GiB  ~6.7 s per image  (single-image baseline)
# CUDA_VISIBLE_DEVICES isolates to GPUs 6,7 only so SAM3-DART ghost contexts
# from GPUs 4,5 don't steal headroom (~512 MiB each before isolation).
GDINO_LOG="${LOG_DIR}/grounding_dino.log"
GDINO_PID="${LOG_DIR}/grounding_dino.pid"
echo "[start] grounding_dino  port=3001 physical-gpus=6,7 (logical 0,1)  log=${GDINO_LOG}"
# CUDA_VISIBLE_DEVICES=6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "$PY" -m "$MODULE" \
    --model grounding_dino \
    --port 3001 \
    --gpu 0,1,2,3,4,5,6,7 \
    --max-batch-size 8 \
    --prompt-chunk-size 1 \
    > "$GDINO_LOG" 2>&1 &
echo $! > "$GDINO_PID"

# ---------- health wait ----------
wait_ready() {
    local name="$1" port="$2" pidfile="$3" log="$4"
    local url="http://localhost:${port}/health"
    for i in $(seq 1 180); do
        if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "[start] ERROR: ${name} died during startup — see ${log}" >&2
            exit 1
        fi
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "[start] ${name} ready on :${port} (pid=$(cat "$pidfile"))"
            return 0
        fi
        sleep 2
    done
    echo "[start] ERROR: ${name} not ready after 360s — see ${log}" >&2
    exit 1
}

# wait_ready sam3_dart       3013 "$SAM3_PID"  "$SAM3_LOG"
wait_ready grounding_dino  3001 "$GDINO_PID" "$GDINO_LOG"

echo "[start] both detectors up."
