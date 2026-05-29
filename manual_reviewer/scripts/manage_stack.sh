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
#   ./manage_stack.sh backup                 # snapshot LS sqlite + pipeline.db,
#                                            # sync annotations for LS_BACKUP_PROJECTS
#   ./manage_stack.sh install-cron           # print a 5-min crontab line for `backup`
#   ./manage_stack.sh watch-backup [start|stop|status]
#                                            # cron-less alternative: a managed
#                                            # background loop that runs `backup`
#                                            # every BACKUP_INTERVAL seconds.
#                                            # Use this when the host has no
#                                            # cron daemon (containers, minimal
#                                            # base images).
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
#   LS_BACKUP_PROJECTS=""   # comma-separated LS project IDs to sync each backup
#   BACKUP_KEEP_SNAPSHOTS=288  # 24h × 12/hour at 5-min cadence
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
: "${LS_BACKUP_PROJECTS:=}"
: "${BACKUP_KEEP_SNAPSHOTS:=288}"
: "${BACKUP_INTERVAL:=300}"
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
    sam3_1)       pattern="model_servers.sam3_1" ;;
    ls)           pattern="label-studio start" ;;
    ml_backend)   pattern="manual_reviewer.ml_backend.server" ;;
    backup_loop)  pattern="manage_stack.sh backup" ;;
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

# ── Backup ──
#
# Three layers (each captures a different failure mode):
#   1. LS sqlite snapshot — recovers projects, users, ML backend wiring.
#   2. pipeline.db snapshot — recovers aa_v4 stage truth.
#   3. sync_ls_to_disk.py — appends per-annotation events for fine-grained
#      replay (the backup we'd *use* when LS is alive but we lost a day's
#      worth of edits to a corrupted draft).
#
# `sqlite3 .backup` is online-safe even with WAL writers — the engine
# coordinates page-level snapshots, so it's the right tool here. A naive
# `cp` of the .sqlite3 file mid-write can produce an unreadable copy.
cmd_backup() {
  local ts; ts="$(date +%Y%m%d-%H%M%S)"
  local snap_dir="$LS_BACKUP_DIR/db_snapshots"
  mkdir -p "$snap_dir"

  local ls_db="$LS_DATA_DIR/label_studio.sqlite3"
  if [ -f "$ls_db" ]; then
    if _sqlite_backup "$ls_db" "$snap_dir/label_studio.$ts.sqlite3"; then
      echo "${G}LS sqlite snapshot${N}      → $snap_dir/label_studio.$ts.sqlite3"
    else
      echo "${R}LS sqlite snapshot failed${N} (see $(log_path backup))"
    fi
  else
    echo "${Y}LS sqlite missing at $ls_db — start has likely never run${N}"
  fi

  if [ -f "$AAV4_PIPELINE_DB" ]; then
    if _sqlite_backup "$AAV4_PIPELINE_DB" "$snap_dir/pipeline.$ts.sqlite3"; then
      echo "${G}pipeline.db snapshot${N}    → $snap_dir/pipeline.$ts.sqlite3"
    else
      echo "${R}pipeline.db snapshot failed${N} (see $(log_path backup))"
    fi
  else
    echo "${Y}pipeline.db missing at $AAV4_PIPELINE_DB${N}"
  fi

  if [ -n "$LS_BACKUP_PROJECTS" ]; then
    IFS=',' read -ra _projs <<< "$LS_BACKUP_PROJECTS"
    for p in "${_projs[@]}"; do
      p="${p// /}"
      [ -z "$p" ] && continue
      LS_TOKEN="$LS_TOKEN" LS_BACKUP_DIR="$LS_BACKUP_DIR" \
      "$VENV_PY" -m manual_reviewer.scripts.sync_ls_to_disk \
        --ls-url "http://$LS_HOST:$LS_PORT" --project "$p" \
        >> "$(log_path backup)" 2>&1 \
        && echo "${G}sync_ls_to_disk${N}        → project $p" \
        || echo "${R}sync_ls_to_disk failed${N}  → project $p (see $(log_path backup))"
    done
  else
    echo "${Y}LS_BACKUP_PROJECTS unset — annotation diff sync skipped${N}"
  fi

  _rotate_snapshots "$snap_dir"
}

# Online-safe sqlite snapshot via Python's Connection.backup() — equivalent
# to the `sqlite3 .backup` CLI but doesn't need the sqlite3 binary on PATH
# (we ship Python with sqlite3 stdlib, not the CLI tool).
_sqlite_backup() {
  local src="$1" dst="$2"
  "$VENV_PY" -c "
import sys, sqlite3
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2])
try:
    src.backup(dst)
finally:
    dst.close(); src.close()
" "$src" "$dst" 2>>"$(log_path backup)"
}

_rotate_snapshots() {
  local snap_dir="$1"
  local keep="$BACKUP_KEEP_SNAPSHOTS"
  for prefix in label_studio pipeline; do
    # ls -1t sorts newest first; tail -n +N skips the first N-1.
    # Empty glob exits cleanly because of the 2>/dev/null + xargs -r.
    local stale
    stale="$(ls -1t "$snap_dir/$prefix."*.sqlite3 2>/dev/null | tail -n +"$((keep + 1))" || true)"
    if [ -n "$stale" ]; then
      echo "$stale" | xargs -r rm -f
    fi
  done
}

cmd_install_cron() {
  local script_path; script_path="$SCRIPT_DIR/manage_stack.sh"
  cat <<EOF
# Add this to your crontab via \`crontab -e\`.
# 5-minute cadence caps worst-case data loss between backups at 5 min.
# LS_BACKUP_PROJECTS must list every LS project ID you want sync'd.
*/5 * * * * LS_TOKEN="$LS_TOKEN" LS_BACKUP_PROJECTS="$LS_BACKUP_PROJECTS" LS_DATA_DIR="$LS_DATA_DIR" AAV4_PIPELINE_DB="$AAV4_PIPELINE_DB" LS_BACKUP_DIR="$LS_BACKUP_DIR" $script_path backup >> $LOG_DIR/backup.cron.log 2>&1
EOF
}

# Cron-less alternative: a self-respawning background loop with the same
# pidfile/log discipline as sam3_1/ls/ml_backend. Inherits the env vars
# from the invoking shell, so callers must export LS_TOKEN/LS_DATA_DIR/
# AAV4_PIPELINE_DB/LS_BACKUP_PROJECTS/LS_BACKUP_DIR before `start`.
cmd_watch_backup() {
  local sub="${1:-status}"
  case "$sub" in
    start)
      if is_running backup_loop; then
        echo "${Y}backup_loop already running (PID $(read_pid backup_loop))${N}"
        return 0
      fi
      echo "Starting backup_loop (interval=${BACKUP_INTERVAL}s)…"
      # Capture the env we need; child loop inherits via export.
      export LS_TOKEN LS_BACKUP_PROJECTS LS_DATA_DIR AAV4_PIPELINE_DB LS_BACKUP_DIR
      nohup bash -c "
        while true; do
          \"$SCRIPT_DIR/manage_stack.sh\" backup
          sleep \"$BACKUP_INTERVAL\"
        done
      " >> "$(log_path backup_loop)" 2>&1 &
      echo $! > "$PIDFILE_DIR/backup_loop.pid"
      echo "${G}backup_loop spawned (PID $!)${N}"
      echo "  → log: $(log_path backup_loop)"
      ;;
    stop)
      stop_one backup_loop
      ;;
    status)
      if is_running backup_loop; then
        echo "${G}● backup_loop${N}  running (PID $(read_pid backup_loop), interval=${BACKUP_INTERVAL}s)"
        echo "  log: $(log_path backup_loop)"
        echo "  recent runs:"
        tail -3 "$(log_path backup_loop)" 2>/dev/null | sed 's/^/    /'
      else
        echo "${R}○ backup_loop${N}  not running"
      fi
      ;;
    *) echo "Usage: $0 watch-backup <start|stop|status>"; exit 2 ;;
  esac
}

usage() {
  sed -n '2,/^# Pidfiles/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

case "${1:-}" in
  start)        shift; cmd_start "$@" ;;
  stop)         shift; cmd_stop "$@" ;;
  restart)      shift; cmd_restart "${1:-all}" ;;
  status)       status ;;
  logs)         shift; cmd_logs "$@" ;;
  backup)       cmd_backup ;;
  install-cron) cmd_install_cron ;;
  watch-backup) shift; cmd_watch_backup "${1:-status}" ;;
  ""|-h|--help) usage ;;
  *) echo "Unknown command: $1"; usage; exit 2 ;;
esac
