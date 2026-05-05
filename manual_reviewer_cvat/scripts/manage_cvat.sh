#!/usr/bin/env bash
# manage_cvat.sh — bring up / tear down / inspect the CVAT review stack.
#
# Mirrors manual_reviewer/scripts/manage_stack.sh in spirit so muscle memory
# transfers, but wraps `docker compose` instead of foreground processes.
#
# Usage:
#   ./manage_cvat.sh start
#   ./manage_cvat.sh stop
#   ./manage_cvat.sh restart [<service>]
#   ./manage_cvat.sh status
#   ./manage_cvat.sh logs <service>           # tail -f
#   ./manage_cvat.sh ps                       # docker compose ps
#   ./manage_cvat.sh exec <service> <cmd...>  # exec inside a container
#   ./manage_cvat.sh create-superuser [<user> [<email>]]
#   ./manage_cvat.sh backup                   # snapshot Postgres + cvat_data
#   ./manage_cvat.sh install-cron             # print 5-min crontab line
#   ./manage_cvat.sh watch-backup [start|stop|status]
#                                             # cron-less alternative

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$STACK_DIR/docker-compose.cvat.yml"
ENV_FILE="$STACK_DIR/configs/stack.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE — copy configs/stack.env.example first." >&2
  exit 2
fi

# shellcheck source=/dev/null
set -a; . "$ENV_FILE"; set +a

: "${COMPOSE_PROJECT_NAME:=cvat_review}"
: "${CVAT_BACKUP_DIR:=$STACK_DIR/.cvat_backup}"
: "${BACKUP_KEEP_SNAPSHOTS:=288}"
: "${BACKUP_INTERVAL:=300}"
: "${PIDFILE_DIR:=/tmp/datatang_review/pids}"
: "${LOG_DIR:=/tmp/datatang_review}"

mkdir -p "$CVAT_BACKUP_DIR" "$PIDFILE_DIR" "$LOG_DIR"

if [ -t 1 ]; then
  G="$(printf '\033[0;32m')" R="$(printf '\033[0;31m')" Y="$(printf '\033[0;33m')" N="$(printf '\033[0m')"
else
  G=""; R=""; Y=""; N=""
fi

DC="docker compose -p $COMPOSE_PROJECT_NAME -f $COMPOSE_FILE --env-file $ENV_FILE"

cmd_start()   { echo "${G}→${N} starting CVAT stack..."; $DC up -d; cmd_status; }
cmd_stop()    { echo "${G}→${N} stopping CVAT stack..."; $DC down; }
cmd_restart() { if [ $# -ge 1 ]; then $DC restart "$@"; else $DC down; $DC up -d; fi; }
cmd_ps()      { $DC ps; }
cmd_logs()    { [ $# -ge 1 ] || { echo "logs <service>" >&2; exit 2; }; $DC logs -f --tail=200 "$1"; }
cmd_exec()    { [ $# -ge 2 ] || { echo "exec <service> <cmd...>" >&2; exit 2; }; svc="$1"; shift; $DC exec "$svc" "$@"; }

cmd_status() {
  $DC ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}' || true
  printf '\n%s reachability:%s\n' "$Y" "$N"
  if curl -fsS "http://${CVAT_HOST:-127.0.0.1}:${CVAT_HOST_PORT:-8081}/api/server/about" >/dev/null 2>&1; then
    echo "  ${G}✔${N} http://${CVAT_HOST:-127.0.0.1}:${CVAT_HOST_PORT:-8081} — reachable"
  else
    echo "  ${R}✘${N} http://${CVAT_HOST:-127.0.0.1}:${CVAT_HOST_PORT:-8081} — not yet ready (give it ~30s on first start)"
  fi
}

cmd_create_superuser() {
  local user="${1:-${CVAT_ADMIN_USER:-admin}}"
  local email="${2:-${CVAT_ADMIN_EMAIL:-admin@example.com}}"
  echo "${G}→${N} creating Django superuser '$user' ($email)"
  $DC exec cvat_server python3 /home/django/manage.py createsuperuser --username "$user" --email "$email"
}

# ─── Backup ───
#
# We dump Postgres via pg_dump (clean SQL — restorable on a fresh stack) and
# tar the cvat_data named volume (uploaded media + cache). Both land in
# $CVAT_BACKUP_DIR/<utc-iso>/.

cmd_backup() {
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local dest="$CVAT_BACKUP_DIR/$stamp"
  mkdir -p "$dest"
  echo "${G}→${N} pg_dump → $dest/cvat.sql"
  $DC exec -T cvat_db pg_dump --clean --if-exists -U root -d cvat > "$dest/cvat.sql" || {
    echo "${R}✘${N} pg_dump failed"; rm -rf "$dest"; return 1; }
  echo "${G}→${N} cvat_data → $dest/cvat_data.tar.gz"
  docker run --rm \
    -v "${COMPOSE_PROJECT_NAME}_cvat_data:/src:ro" \
    -v "$dest:/dest" \
    alpine:3.19 tar -czf /dest/cvat_data.tar.gz -C /src . || {
    echo "${R}✘${N} cvat_data tar failed"; return 1; }
  echo "${G}✔${N} snapshot complete: $dest"
  cmd_prune_backups
}

cmd_prune_backups() {
  local keep="$BACKUP_KEEP_SNAPSHOTS"
  mapfile -t snaps < <(ls -1 "$CVAT_BACKUP_DIR" 2>/dev/null | sort)
  local n=${#snaps[@]}
  if (( n > keep )); then
    local to_remove=$((n - keep))
    for i in $(seq 0 $((to_remove - 1))); do
      rm -rf "$CVAT_BACKUP_DIR/${snaps[i]}"
    done
    echo "${Y}pruned${N} $to_remove old snapshot(s)"
  fi
}

cmd_install_cron() {
  local me; me="$(realpath "${BASH_SOURCE[0]}")"
  echo "# add to your crontab — runs every 5 minutes:"
  echo "*/5 * * * * cd $STACK_DIR && bash $me backup >> $LOG_DIR/cvat_backup.log 2>&1"
}

# ─── cron-less watch-backup loop (matches manage_stack.sh pattern) ───
cmd_watch_backup() {
  local action="${1:-status}"
  local pidfile="$PIDFILE_DIR/cvat_watch_backup.pid"
  case "$action" in
    start)
      if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "${Y}already running${N} (pid $(cat "$pidfile"))"; return 0
      fi
      nohup bash -c "while true; do bash '$0' backup; sleep $BACKUP_INTERVAL; done" \
        >>"$LOG_DIR/cvat_watch_backup.log" 2>&1 &
      echo $! > "$pidfile"
      echo "${G}✔${N} watch-backup started (pid $!) every ${BACKUP_INTERVAL}s"
      ;;
    stop)
      if [ -f "$pidfile" ]; then
        kill "$(cat "$pidfile")" 2>/dev/null && rm -f "$pidfile"
        echo "${G}✔${N} watch-backup stopped"
      else
        echo "${Y}not running${N}"
      fi
      ;;
    status)
      if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "${G}running${N} (pid $(cat "$pidfile"), every ${BACKUP_INTERVAL}s)"
      else
        echo "${R}not running${N}"
      fi
      ;;
    *)
      echo "watch-backup [start|stop|status]" >&2; exit 2 ;;
  esac
}

# ─── Dispatch ───
sub="${1:-status}"; shift || true
case "$sub" in
  start)             cmd_start "$@" ;;
  stop)              cmd_stop "$@" ;;
  restart)           cmd_restart "$@" ;;
  status)            cmd_status "$@" ;;
  ps)                cmd_ps "$@" ;;
  logs)              cmd_logs "$@" ;;
  exec)              cmd_exec "$@" ;;
  create-superuser)  cmd_create_superuser "$@" ;;
  backup)            cmd_backup "$@" ;;
  install-cron)      cmd_install_cron "$@" ;;
  watch-backup)      cmd_watch_backup "$@" ;;
  *)
    echo "usage: $0 {start|stop|restart|status|ps|logs|exec|create-superuser|backup|install-cron|watch-backup}" >&2
    exit 2
    ;;
esac