#!/usr/bin/env bash
# Robust SSH tunnel from local :8958 → remote 127.0.0.1:8956 on
# pavan@10.160.97.10 (where the fl_pj VLM LB listens).
#
# The tunnel survives transient network drops via a reconnect loop +
# SSH keepalives. Prefers `autossh` if installed; falls back to a
# POSIX retry loop with exponential backoff.
#
# Usage:
#   ./scripts_bench/vlm_tunnel_flpj.sh              # foreground
#   ./scripts_bench/vlm_tunnel_flpj.sh --detach     # background + logfile
#   ./scripts_bench/vlm_tunnel_flpj.sh --stop       # kill any running tunnel
#   ./scripts_bench/vlm_tunnel_flpj.sh --status     # check health
#
# Env overrides:
#   LOCAL_PORT   (default 8958)
#   REMOTE_HOST  (default pavan@10.160.97.10)
#   REMOTE_PORT  (default 8956)
#
# Health check (tunnel works when this returns "ok"):
#   curl http://localhost:8958/health

set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8958}"
REMOTE_HOST="${REMOTE_HOST:-pavan@10.160.97.10}"
REMOTE_PORT="${REMOTE_PORT:-8956}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
PIDFILE="logs/vlm_tunnel_flpj.pid"
LOGFILE="logs/vlm_tunnel_flpj.log"

# Common ssh options shared by both paths:
#   ServerAliveInterval 30, ServerAliveCountMax 3 — drops the tunnel after
#     ~90s of no response from the remote sshd, causing our reconnect loop
#     to kick in.
#   ExitOnForwardFailure yes — if the local port is already in use, exit
#     immediately instead of silently keeping a dead ssh session.
#   StrictHostKeyChecking accept-new — auto-accept new hosts but still
#     detect MITM changes.
SSH_OPTS=(
    -o "ServerAliveInterval=30"
    -o "ServerAliveCountMax=3"
    -o "ExitOnForwardFailure=yes"
    -o "StrictHostKeyChecking=accept-new"
    -o "TCPKeepAlive=yes"
    -N
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
)

# ---- Subcommands ----

_stop() {
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid="$(cat "$PIDFILE")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "stopping tunnel (PID $pid) ..."
            kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        else
            echo "no live process for PID ${pid:-unknown}"
        fi
        rm -f "$PIDFILE"
    else
        echo "no pidfile at $PIDFILE"
    fi
    # Also sweep any stray ssh -L ${LOCAL_PORT}: processes we might have left.
    pkill -f "ssh .*-L *${LOCAL_PORT}:" 2>/dev/null || true
}

_status() {
    local alive=false
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid="$(cat "$PIDFILE")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            alive=true
            echo "process: alive (PID $pid)"
        else
            echo "process: dead (stale pidfile)"
        fi
    else
        echo "process: not started"
    fi
    # Probe the local port
    local resp
    if resp="$(curl -fsS --max-time 3 "http://localhost:${LOCAL_PORT}/health" 2>/dev/null)"; then
        echo "tunnel health: ${resp} (via http://localhost:${LOCAL_PORT})"
    else
        echo "tunnel health: unreachable on http://localhost:${LOCAL_PORT}/health"
    fi
    $alive || return 1
}

_run_loop() {
    # Body of the reconnect loop. Runs until externally killed.
    # Prefers autossh (purpose-built for this) when present.
    local attempt=0
    local max_backoff=60
    while true; do
        attempt=$((attempt + 1))
        echo "[$(date '+%H:%M:%S')] attempt $attempt: connecting to ${REMOTE_HOST} (local :${LOCAL_PORT} -> remote :${REMOTE_PORT})"
        if command -v autossh >/dev/null 2>&1; then
            # autossh handles the reconnect itself, so the outer loop only
            # fires if autossh itself exits.
            AUTOSSH_POLL=30 AUTOSSH_GATETIME=10 autossh -M 0 "${SSH_OPTS[@]}" "$REMOTE_HOST" || true
        else
            ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" || true
        fi
        # Back off a bit, capped, so a dead remote doesn't hammer.
        local backoff=$(( attempt < 5 ? attempt * 2 : max_backoff ))
        (( backoff > max_backoff )) && backoff=$max_backoff
        echo "[$(date '+%H:%M:%S')] ssh exited; sleeping ${backoff}s before retry"
        sleep "$backoff"
    done
}

# ---- Argument handling ----

case "${1:-}" in
    --stop|stop)
        _stop
        exit 0
        ;;
    --status|status)
        _status
        exit $?
        ;;
    --detach|detach)
        # Start the loop in its own process group so _stop can kill
        # the whole subtree (ssh clients can fork multiplex masters).
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "tunnel already running (PID $(cat "$PIDFILE")) — use --stop first"
            exit 1
        fi
        setsid nohup bash -c '_run_loop() { :; }; '"$(declare -f _run_loop)"'; _run_loop' \
            </dev/null >"$LOGFILE" 2>&1 &
        PID=$!
        echo "$PID" > "$PIDFILE"
        echo "tunnel detached, PID $PID"
        echo "  log:     $LOGFILE"
        echo "  pidfile: $PIDFILE"
        echo ""
        echo "Health check:"
        echo "  curl http://localhost:${LOCAL_PORT}/health"
        exit 0
        ;;
    ""|--foreground|foreground)
        # Run in foreground; Ctrl-C exits.
        trap 'echo "caught SIGINT"; exit 0' INT TERM
        _run_loop
        ;;
    *)
        echo "usage: $0 [--detach|--stop|--status]" >&2
        exit 2
        ;;
esac
