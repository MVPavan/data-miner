"""Launch all auto-annotation v3 model servers defined in the config.

Usage:
    # Launch everything from config
    python launch_all.py

    # Launch a subset
    python launch_all.py --servers grounding_dino sam3_dart

    # Override GPU assignment for a specific server
    python launch_all.py --override grounding_dino.gpu=1 sam3.gpu=3

    # Custom config file
    python launch_all.py --config /path/to/config.yaml

    # Skip health-check loop (useful in CI / test environments)
    python launch_all.py --no-health-check

Each server is launched as a subprocess. Servers already running on their
configured port are detected via /health and adopted (not re-launched).
All servers — both launched and adopted — are terminated on exit (Ctrl-C,
unexpected crash, health timeout, or any other shutdown path).

Press Ctrl-C to stop all servers.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

try:
    import yaml
except ImportError as exc:
    sys.exit(
        "PyYAML is required for launch_all.py. Install it with: pip install pyyaml\n"
        f"Original error: {exc}"
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
_DEFAULT_CONFIG = _HERE.parent / "configs" / "default.yaml"
_HEALTH_TIMEOUT_S = 60        # max seconds per health-check attempt
_HEALTH_MAX_RETRIES = 3       # retry health-check loop this many times
_HEALTH_POLL_INTERVAL_S = 2   # seconds between /health polls
_HEALTH_ENDPOINT = "/health"


# ---------------------------------------------------------------------------
# Config loading + override helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load detectors from ``configs/default.yaml → servers.detectors``.

    Returns ``{detector_name: cfg_dict}`` with disabled detectors filtered out.
    Normalises the ``gpu`` field ("cuda:1" → "1") for CUDA_VISIBLE_DEVICES.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "servers" not in raw:
        raise ValueError(f"Config must contain a top-level 'servers' mapping: {config_path}")
    servers = raw["servers"]
    detectors = servers.get("detectors")
    if not isinstance(detectors, dict):
        raise ValueError(
            f"Expected servers.detectors mapping in {config_path}; got {type(detectors)}"
        )
    out: dict[str, dict] = {}
    for name, cfg in detectors.items():
        if not cfg.get("enabled", True):
            logger.info("Detector '%s' is disabled — skipping", name)
            continue
        gpu = str(cfg.get("gpu", "0"))
        if gpu.startswith("cuda:"):
            gpu = gpu.split(":", 1)[1]
        cfg = dict(cfg)
        cfg["gpu"] = gpu
        cfg.setdefault("batch_timeout", cfg.get("batch_timeout_ms", 50) / 1000.0)
        out[name] = cfg
    if not out:
        raise ValueError(f"No enabled detectors found in {config_path}")
    return out


def apply_overrides(
    server_configs: dict[str, dict],
    overrides: list[str],
) -> None:
    """Apply dotted key=value overrides, e.g. 'grounding_dino.gpu=1'.

    Supported value types: int, float, str.
    """
    for override in overrides:
        try:
            key_path, value_str = override.split("=", 1)
        except ValueError:
            raise ValueError(
                f"Invalid override '{override}'. Expected format: server.field=value"
            )
        parts = key_path.strip().split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Override key must be 'server_name.field', got '{key_path}'"
            )
        server_name, field = parts
        if server_name not in server_configs:
            raise KeyError(
                f"Override references unknown server '{server_name}'. "
                f"Known: {sorted(server_configs)}"
            )
        # Coerce value to int / float / str
        value: Any
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        logger.info("Override: %s.%s = %r", server_name, field, value)
        server_configs[server_name][field] = value


# ---------------------------------------------------------------------------
# Port → PID resolution
# ---------------------------------------------------------------------------

def find_pid_on_port(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
    try:
        import psutil
    except ImportError:
        return None
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.status == "LISTEN":
            return conn.pid
    return None


def _kill_pid(pid: int, name: str, wait_s: float = 8.0) -> None:
    """SIGTERM a PID, wait, then SIGKILL if still alive."""
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to '%s' (pid=%d)", name, pid)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # probe — raises if gone
        except ProcessLookupError:
            return
        time.sleep(0.3)
    try:
        os.kill(pid, signal.SIGKILL)
        logger.warning("Sent SIGKILL to '%s' (pid=%d) — did not exit on SIGTERM", name, pid)
    except ProcessLookupError:
        pass


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def build_command(name: str, cfg: dict[str, Any]) -> list[str]:
    """Build the subprocess command for a single server."""
    script = cfg.get("script")
    if not script:
        raise ValueError(f"Server '{name}' has no 'script' field")

    script_path = _HERE / script
    if not script_path.exists():
        raise FileNotFoundError(f"Server script not found: {script_path}")

    # Launch as a module so relative imports (from ..contracts) resolve.
    module = f"data_miner.auto_annotation_v3.servers.{script_path.stem}"
    cmd = [
        sys.executable, "-m", module,
        "--port", str(cfg["port"]),
        "--device", "0",  # CUDA_VISIBLE_DEVICES masks to single GPU indexed as 0
        "--max-batch-size", str(cfg.get("max_batch_size", 8)),
        "--batch-timeout", str(cfg.get("batch_timeout", 0.05)),
    ]
    # Forward DetectorConfig.options as extra --key value CLI args. Keys map
    # to dashed flags (e.g. detection_only → --detection-only). Only flat
    # scalars (bool/int/float/str) are passed.
    for k, v in (cfg.get("options") or {}).items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    return cmd


def launch_server(
    name: str,
    cfg: dict[str, Any],
    log_dir: Path | None,
) -> subprocess.Popen:
    """Start a server subprocess and return the Popen handle."""
    cmd = build_command(name, cfg)
    env = os.environ.copy()

    gpu_id = cfg.get("gpu", 0)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    stdout = stderr = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        fh = log_file.open("a")
        stdout = fh
        stderr = fh
        logger.info("Server '%s' logs → %s", name, log_file)

    logger.info(
        "Launching '%s': port=%s gpu=%s cmd=%s",
        name,
        cfg.get("port"),
        gpu_id,
        " ".join(cmd),
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,  # isolate signal group so Ctrl-C propagates cleanly
    )
    return proc


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def _check_health(port: int) -> bool:
    """Return True if the server at *port* responds 200 on /health."""
    url = f"http://localhost:{port}{_HEALTH_ENDPOINT}"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_for_health(
    server_ports: dict[str, int],
    timeout_s: float = _HEALTH_TIMEOUT_S,
    poll_s: float = _HEALTH_POLL_INTERVAL_S,
) -> bool:
    """Poll /health on each server until all pass or *timeout_s* elapses.

    Returns True if all servers are healthy, False on timeout.
    """
    deadline = time.monotonic() + timeout_s
    pending = set(server_ports)

    logger.info(
        "Waiting up to %.0fs for servers to become healthy: %s",
        timeout_s,
        sorted(pending),
    )

    while pending and time.monotonic() < deadline:
        for name in list(pending):
            if _check_health(server_ports[name]):
                logger.info("Server '%s' is healthy (port %d)", name, server_ports[name])
                pending.discard(name)
        if pending:
            time.sleep(poll_s)

    if pending:
        logger.error(
            "Timed out after %.0fs. Unhealthy servers: %s", timeout_s, sorted(pending)
        )
        return False

    logger.info("All servers healthy.")
    return True


# ---------------------------------------------------------------------------
# Unified server tracker — owns both launched and adopted processes
# ---------------------------------------------------------------------------

class ServerTracker:
    """Tracks all server PIDs (launched + adopted) for unified shutdown."""

    def __init__(self) -> None:
        # Servers we launched as subprocesses (we own the Popen handle).
        self.launched: dict[str, subprocess.Popen] = {}
        # Servers already running that we adopted (we only know the PID).
        self.adopted: dict[str, int] = {}

    def all_pids(self) -> dict[str, int]:
        """Return {name: pid} for every tracked server."""
        pids: dict[str, int] = {}
        for name, proc in self.launched.items():
            pids[name] = proc.pid
        pids.update(self.adopted)
        return pids

    def terminate_all(self, wait_s: float = 15.0) -> None:
        """Kill every tracked server — both launched and adopted.

        Blocks until every process has exited (or been SIGKILL'd).
        """
        if not self.launched and not self.adopted:
            return

        all_pids = self.all_pids()
        if not all_pids:
            return

        logger.info("Shutting down %d server(s) …", len(all_pids))

        # 1. Send SIGTERM to everything.
        for name, proc in self.launched.items():
            if proc.poll() is None:
                logger.info("  SIGTERM → '%s' (pid=%d)", name, proc.pid)
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass

        for name, pid in self.adopted.items():
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info("  SIGTERM → '%s' (pid=%d)", name, pid)
            except ProcessLookupError:
                logger.info("  '%s' (pid=%d) already gone", name, pid)

        # 2. Wait for launched procs — poll with feedback.
        deadline = time.monotonic() + wait_s
        alive_launched = {
            n: p for n, p in self.launched.items() if p.poll() is None
        }
        while alive_launched and time.monotonic() < deadline:
            time.sleep(0.5)
            for name in list(alive_launched):
                if alive_launched[name].poll() is not None:
                    logger.info("  '%s' exited (code=%d)", name, alive_launched[name].returncode)
                    del alive_launched[name]

        # SIGKILL stragglers.
        for name, proc in alive_launched.items():
            logger.warning("  '%s' (pid=%d) did not exit — SIGKILL", name, proc.pid)
            try:
                proc.kill()
                proc.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        # 3. Wait for adopted PIDs.
        alive_adopted = {}
        for name, pid in self.adopted.items():
            try:
                os.kill(pid, 0)
                alive_adopted[name] = pid
            except ProcessLookupError:
                logger.info("  '%s' (pid=%d) exited", name, pid)

        while alive_adopted and time.monotonic() < deadline:
            time.sleep(0.5)
            for name in list(alive_adopted):
                try:
                    os.kill(alive_adopted[name], 0)
                except ProcessLookupError:
                    logger.info("  '%s' (pid=%d) exited", name, alive_adopted[name])
                    del alive_adopted[name]

        for name, pid in alive_adopted.items():
            logger.warning("  '%s' (pid=%d) did not exit — SIGKILL", name, pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        logger.info("All servers stopped.")
        self.launched.clear()
        self.adopted.clear()

    def check_launched(self) -> str | None:
        """Return name of first launched process that exited, or None."""
        for name, proc in self.launched.items():
            if proc.poll() is not None:
                return name
        return None


def _wait_pid(pid: int, name: str, timeout: float) -> None:
    """Wait for a PID to die; SIGKILL if it doesn't within timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.3)
    try:
        os.kill(pid, signal.SIGKILL)
        logger.warning("Sent SIGKILL to '%s' (pid=%d)", name, pid)
    except ProcessLookupError:
        pass


# Global tracker — atexit and signal handlers reference this.
_tracker = ServerTracker()


def _cleanup() -> None:
    """atexit handler — ensures all servers die when the launcher exits."""
    _tracker.terminate_all()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch all auto-annotation v3 model servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to serve_config.yaml (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--servers",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Launch only these servers (default: all defined in config)",
    )
    parser.add_argument(
        "--override",
        nargs="+",
        metavar="server.field=value",
        default=[],
        help=(
            "Override config fields per server, e.g. "
            "--override grounding_dino.gpu=1 sam3.port=3033"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for per-server log files. "
            "If omitted, server output goes to this process's stdout/stderr."
        ),
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=_HEALTH_TIMEOUT_S,
        metavar="SECONDS",
        help=f"Seconds to wait for all servers to pass /health (default: {_HEALTH_TIMEOUT_S})",
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip /health polling after launching servers",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Register atexit — covers sys.exit, unhandled exceptions, etc.
    atexit.register(_cleanup)

    # Signal handler sets a flag; the main loop checks it and calls
    # terminate_all synchronously so we block until all servers are dead.
    _shutdown_requested = False

    def _handle_signal(signum, _frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Second Ctrl-C — force exit immediately.
            logger.warning("Forced exit (second signal)")
            _tracker.terminate_all(wait_s=3)
            os._exit(1)
        _shutdown_requested = True
        logger.info("Signal %d received — will shut down after current tick …", signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- Load + filter config ----
    logger.info("Loading config: %s", args.config)
    try:
        all_configs = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Config error: %s", exc)
        return 1

    if args.servers:
        unknown = set(args.servers) - set(all_configs)
        if unknown:
            logger.error(
                "Unknown server(s) requested: %s. Known: %s",
                sorted(unknown),
                sorted(all_configs),
            )
            return 1
        selected = {k: all_configs[k] for k in args.servers}
    else:
        selected = all_configs

    if not selected:
        logger.error("No servers to launch.")
        return 1

    # ---- Apply overrides ----
    try:
        apply_overrides(selected, args.override)
    except (ValueError, KeyError) as exc:
        logger.error("Override error: %s", exc)
        return 1

    # ---- Pre-launch: detect already-running servers ----
    to_launch: dict[str, dict] = {}
    for name, cfg in selected.items():
        port = cfg["port"]
        if _check_health(port):
            pid = find_pid_on_port(port)
            if pid is not None:
                logger.info(
                    "Server '%s' already healthy on port %d (pid=%d) — adopting",
                    name, port, pid,
                )
                _tracker.adopted[name] = pid
            else:
                logger.info(
                    "Server '%s' already healthy on port %d (pid unknown) — adopting",
                    name, port,
                )
                # Can't track PID, but don't re-launch. Won't be killed on exit
                # since we have no PID — log a warning.
                logger.warning(
                    "Cannot resolve PID for '%s' on port %d (psutil unavailable or "
                    "permission denied). This server will NOT be killed on exit.",
                    name, port,
                )
        else:
            # Port not healthy — check if something is occupying it without /health
            pid = find_pid_on_port(port)
            if pid is not None:
                logger.warning(
                    "Port %d is occupied by pid=%d but /health failed — "
                    "killing stale process before launching '%s'",
                    port, pid, name,
                )
                _kill_pid(pid, f"stale:{name}")
                time.sleep(1)  # brief pause for port release
            to_launch[name] = cfg

    # ---- Launch missing servers ----
    failed_launch: list[str] = []
    for name, cfg in to_launch.items():
        try:
            proc = launch_server(name, cfg, log_dir=args.log_dir)
            _tracker.launched[name] = proc
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Failed to launch '%s': %s", name, exc)
            failed_launch.append(name)

    if failed_launch:
        logger.error("Could not launch: %s — terminating all servers", failed_launch)
        _tracker.terminate_all()
        return 1

    total = len(_tracker.launched) + len(_tracker.adopted)
    logger.info(
        "Servers: %d launched, %d adopted (%d total)",
        len(_tracker.launched),
        len(_tracker.adopted),
        total,
    )

    # ---- Health checks (only for newly launched servers) ----
    if _tracker.launched and not args.no_health_check:
        new_ports = {
            name: selected[name]["port"] for name in _tracker.launched
        }
        healthy = False
        for attempt in range(1, _HEALTH_MAX_RETRIES + 1):
            logger.info(
                "Health check attempt %d/%d (timeout=%ds)",
                attempt, _HEALTH_MAX_RETRIES, int(args.health_timeout),
            )
            if wait_for_health(new_ports, timeout_s=args.health_timeout):
                healthy = True
                break
            if attempt < _HEALTH_MAX_RETRIES:
                # Check that unhealthy servers haven't crashed entirely.
                for name in list(new_ports):
                    proc = _tracker.launched.get(name)
                    if proc and proc.poll() is not None:
                        logger.error(
                            "Server '%s' (pid=%d) crashed (code=%d) — "
                            "no point retrying",
                            name, proc.pid, proc.returncode,
                        )
                        _tracker.terminate_all()
                        return 1
                logger.info("Retrying health check …")
        if not healthy:
            logger.error(
                "Health check failed after %d attempts — terminating all servers",
                _HEALTH_MAX_RETRIES,
            )
            _tracker.terminate_all()
            return 1
    elif args.no_health_check:
        logger.info("Health check skipped (--no-health-check)")

    # ---- Keep running, monitor for unexpected exits ----
    logger.info(
        "All %d server(s) running. Press Ctrl-C to stop.", total,
    )
    while True:
        time.sleep(2)

        # Check for shutdown signal (Ctrl-C / SIGTERM).
        if _shutdown_requested:
            logger.info("Shutting down all servers …")
            _tracker.terminate_all()
            return 0

        # Check for unexpected process death.
        dead = _tracker.check_launched()
        if dead is not None:
            proc = _tracker.launched[dead]
            logger.error(
                "Server '%s' (pid=%d) exited unexpectedly with code %d",
                dead,
                proc.pid,
                proc.returncode,
            )
            _tracker.terminate_all()
            return 1


if __name__ == "__main__":
    sys.exit(main())
