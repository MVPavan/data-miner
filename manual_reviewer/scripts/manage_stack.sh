#!/usr/bin/env bash
# manage_stack.sh — bring up / tear down / inspect the manual_reviewer stack.
#
# Services managed:
#   sam3_1      LitServe model server on :3014 (GPU)
#   ls          Label Studio on :8080
#   ml_backend  manual_reviewer ML backend on :9090
#
# Usage:
#   ./manage_stack.sh start                  # bring up sam3_1 → ls → ml_backend
#   ./manage_stack.sh stop                   # tear down in reverse order
#   ./manage_stack.sh restart [<service>]    # restart all, or just one
#   ./manage_stack.sh status                 # PIDs + reachability check
#   ./manage_stack.sh logs <service>         # tail -f the log
#
# Configurable via env vars (sensible defaults shown):
#   LS_PORT=8080   LS_HOST=127.0.0.1   LS_USER=admin@example.com
#   LS_PASS=changeme123
#   LS_TOKEN=datatang-demo-token-1234567890abcdef
#   LS_DATA_DIR=/tmp/datatang_review/.ls_data
#   ML_PORT=9090   ML_HOST=127.0.0.1
#   SAM3_PORT=3014 SAM3_GPU=cuda:0
#   SAM3_PYTHONPATH=$PROJECT_ROOT/scratchpad/DART
#   AAV4_PIPELINE_DB=/tmp/datatang_review/pipeline.db
#   LS_BACKUP_DIR=$PROJECT_ROOT/manual_reviewer/.ls_backup
#   LOG_DIR=/tmp/datatang_review
#   PIDFILE_DIR=/tmp/datatang_review/pids
#   ENABLE_BATCH_PROPOSALS=false
#
# Pidfiles live in $PIDFILE_DIR. Logs append (never truncate) to $LOG_DIR.

set -u

# ── Resolve paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
LS_BIN="$PROJECT_ROOT/.venv/bin/label-studio"

# ── Defaults ──
: "${LS_PORT:=8080}"
: "${LS_HOST:=127.0.0.1}"
: "${LS_USER:=admin@example.com}"
: "${LS_PASS:=changeme123}"
: "${LS_TOKEN:=datatang-demo-token-1234567890abcdef}"
: "${LS_DATA_DIR:=/tmp/datatang_review/.ls_data}"

: "${ML_PORT:=9090}"
: "${ML_HOST:=127.0.0.1}"

: "${SAM3_PORT:=3014}"
: "${SAM3_GPU:=cuda:0}"
: "${SAM3_PYTHONPATH:=$PROJECT_ROOT/scratchpad/DART}"

: "${AAV4_PIPELINE_DB:=/tmp/datatang_review/pipeline.db}"
: "${LS_BACKUP_DIR:=$PROJECT_ROOT/manual_reviewer/.ls_backup}"
: "${ENABLE_BATCH_PROPOSALS:=false}"

: "${LOG_DIR:=/tmp/datatang_review}"
: "${PIDFILE_DIR:=/tmp/datatang_review/pids}"

# Colors (only when stdout is a TTY).
if [ -t 1 ]; then
  G="$(printf '\033[0;32m')" R="$(printf '\033[0;31m')" Y="$(printf '\033[0;33m')" N="$(printf '\033[0m')"
else
  G="" R="" Y="" N=""
fi

mkdir -p "$LOG_DIR" "$PIDFILE_DIR"

# ── Helpers ──

is_running() {
  local name="$1"
  local pidfile="$PIDFILE_DIR/$name.pid"
  [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

read_pid() { cat "$PIDFILE_DIR/$1.pid" 2>/dev/null; }

log_path() { echo "$LOG_DIR/$1.log"; }

# Wait until a TCP port accepts connections (rough liveness probe).
# Args: host port timeout_seconds
wait_for_port() {
  local host="$1" port="$2" deadline="${3:-15}"
  for _ in $(seq 1 "$deadline"); do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      exec 3<&- 3>&-
      return 0
    fi
    sleep 1
  done
  return 1
}

# ── Start methods ──

start_sam3_1() {
  if is_running sam3_1; then
    echo "${Y}sam3_1 already running (PID $(read_pid sam3_1))${N}"
    return 0
  fi
  echo "Starting sam3_1 on :$SAM3_PORT (gpu=$SAM3_GPU)…"
  cd "$PROJECT_ROOT"
  PYTHONPATH="$SAM3_PYTHONPATH" \
  nohup "$VENV_PY" -m data_miner.auto_annotation_v4.model_servers.sam3_1 \
    --port "$SAM3_PORT" --gpu "$SAM3_GPU" \
    >> "$(log_path sam3_1)" 2>&1 &
  echo $! > "$PIDFILE_DIR/sam3_1.pid"
  echo "${G}sam3_1 spawned (PID $!)${N}"
  # SAM 3.1 model load takes a while; just a port-up probe.
  if wait_for_port localhost "$SAM3_PORT" 60; then
    echo "${G}  → sam3_1 listening${N}"
  else
    echo "${R}  → sam3_1 didn't bind in 60s; check $(log_path sam3_1)${N}"
    return 1
  fi
}

start_ls() {
  if is_running ls; then
    echo "${Y}ls already running (PID $(read_pid ls))${N}"
    return 0
  fi
  echo "Starting Label Studio on $LS_HOST:$LS_PORT (data=$LS_DATA_DIR)…"
  mkdir -p "$LS_DATA_DIR"
  # LOCAL_FILES_SERVING_ENABLED=true is required for /api/storages/localfiles
  # to work — otherwise create_ls_project's attach-storage step fails with
  # "Serving local files from the host filesystem can be a security risk".
  # LOCAL_FILES_DOCUMENT_ROOT scopes the allowed serving prefix; defaulting
  # to the dataset root keeps storage POSTs from accidentally exposing /.
  LABEL_STUDIO_BASE_DATA_DIR="$LS_DATA_DIR" \
  LOCAL_FILES_SERVING_ENABLED="true" \
  LOCAL_FILES_DOCUMENT_ROOT="${LS_LOCAL_FILES_ROOT:-/}" \
  nohup "$LS_BIN" start \
    --host "$LS_HOST" --port "$LS_PORT" \
    --username "$LS_USER" --password "$LS_PASS" \
    --user-token "$LS_TOKEN" --enable-legacy-api-token --no-browser \
    >> "$(log_path ls)" 2>&1 &
  echo $! > "$PIDFILE_DIR/ls.pid"
  echo "${G}ls spawned (PID $!)${N}"
  if wait_for_port "$LS_HOST" "$LS_PORT" 30; then
    echo "${G}  → ls listening${N}"
  else
    echo "${R}  → ls didn't bind in 30s; check $(log_path ls)${N}"
    return 1
  fi
}

start_ml_backend() {
  if is_running ml_backend; then
    echo "${Y}ml_backend already running (PID $(read_pid ml_backend))${N}"
    return 0
  fi
  echo "Starting ml_backend on $ML_HOST:$ML_PORT…"
  cd "$PROJECT_ROOT"
  LABEL_STUDIO_ML_PORT="$ML_PORT" \
  LABEL_STUDIO_ML_HOST="$ML_HOST" \
  SAM3_1_URL="http://localhost:$SAM3_PORT/predict" \
  AAV4_PIPELINE_DB="$AAV4_PIPELINE_DB" \
  LS_BACKUP_DIR="$LS_BACKUP_DIR" \
  ENABLE_BATCH_PROPOSALS="$ENABLE_BATCH_PROPOSALS" \
  LS_URL="http://localhost:$LS_PORT" \
  LS_TOKEN="$LS_TOKEN" \
  nohup "$VENV_PY" -m manual_reviewer.ml_backend.server \
    >> "$(log_path ml_backend)" 2>&1 &
  echo $! > "$PIDFILE_DIR/ml_backend.pid"
  echo "${G}ml_backend spawned (PID $!)${N}"
  if wait_for_port "$ML_HOST" "$ML_PORT" 30; then
    echo "${G}  → ml_backend listening${N}"
  else
    echo "${R}  → ml_backend didn't bind in 30s; check $(log_path ml_backend)${N}"
    return 1
  fi
}

# ── Stop methods ──

stop_one() {
  local name="$1"
  local pidfile="$PIDFILE_DIR/$name.pid"
  local pattern
  case "$name" in
    sam3_1)     pattern="model_servers.sam3_1" ;;
    ls)         pattern="label-studio start" ;;
    ml_backend) pattern="manual_reviewer.ml_backend.server" ;;
    *) echo "${R}unknown service: $name${N}"; return 2 ;;
  esac

  # First try the pidfile.
  if [ -f "$pidfile" ]; then
    local pid; pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (PID $pid)…"
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        sleep 1
        kill -0 "$pid" 2>/dev/null || break
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi

  # Sweep stragglers (orphaned subprocesses, e.g. LitServe workers).
  pkill -f "$pattern" 2>/dev/null || true
  sleep 1
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    pkill -9 -f "$pattern" 2>/dev/null || true
  fi
  echo "${G}$name stopped${N}"
}

# ── Status ──

http_ok() {
  # Returns 0 if a HEAD/GET to URL replies anything (incl. 4xx).
  curl -s -o /dev/null -m 2 "$1" 2>/dev/null
}

status() {
  for s in sam3_1 ls ml_backend; do
    if is_running "$s"; then
      echo "${G}● $s${N}        running (PID $(read_pid "$s"))"
    else
      echo "${R}○ $s${N}        not running"
    fi
  done
  echo ""
  echo "Reachability:"
  if http_ok "http://localhost:$SAM3_PORT/predict"; then
    echo "  ${G}✓${N} sam3_1     http://localhost:$SAM3_PORT"
  else
    echo "  ${R}✗${N} sam3_1     http://localhost:$SAM3_PORT"
  fi
  if http_ok "http://$LS_HOST:$LS_PORT"; then
    echo "  ${G}✓${N} ls         http://$LS_HOST:$LS_PORT"
  else
    echo "  ${R}✗${N} ls         http://$LS_HOST:$LS_PORT"
  fi
  if http_ok "http://$ML_HOST:$ML_PORT/"; then
    echo "  ${G}✓${N} ml_backend http://$ML_HOST:$ML_PORT"
  else
    echo "  ${R}✗${N} ml_backend http://$ML_HOST:$ML_PORT"
  fi
}

# ── Top-level dispatch ──

cmd_start() {
  start_sam3_1
  start_ls
  start_ml_backend
}

cmd_stop() {
  stop_one ml_backend
  stop_one ls
  stop_one sam3_1
}

cmd_restart() {
  case "${1:-all}" in
    sam3_1)     stop_one sam3_1;     start_sam3_1 ;;
    ls)         stop_one ls;         start_ls ;;
    ml_backend) stop_one ml_backend; start_ml_backend ;;
    all)        cmd_stop; sleep 1; cmd_start ;;
    *) echo "Unknown service: $1"; usage; exit 2 ;;
  esac
}

cmd_logs() {
  local svc="${1:-}"
  case "$svc" in
    sam3_1|ls|ml_backend) tail -f "$(log_path "$svc")" ;;
    "") echo "Usage: $0 logs <sam3_1|ls|ml_backend>"; exit 2 ;;
    *) echo "Unknown service: $svc"; exit 2 ;;
  esac
}

usage() {
  sed -n '2,/^# Pidfiles/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

case "${1:-}" in
  start)   shift; cmd_start "$@" ;;
  stop)    shift; cmd_stop "$@" ;;
  restart) shift; cmd_restart "${1:-all}" ;;
  status)  status ;;
  logs)    shift; cmd_logs "$@" ;;
  ""|-h|--help) usage ;;
  *) echo "Unknown command: $1"; usage; exit 2 ;;
esac
