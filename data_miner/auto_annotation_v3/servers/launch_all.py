"""Launch all auto-annotation v3 model servers defined in serve_config.yaml.

Usage:
    # Launch everything from config
    python launch_all.py

    # Launch a subset
    python launch_all.py --servers grounding_dino owlvit2

    # Override GPU assignment for a specific server
    python launch_all.py --override grounding_dino.gpu=1 sam3.gpu=3

    # Custom config file
    python launch_all.py --config /path/to/serve_config.yaml

    # Skip health-check loop (useful in CI / test environments)
    python launch_all.py --no-health-check

Each server is launched as a subprocess. The launcher waits for all servers to
pass a /health HTTP check before returning (or raises after a timeout).

Press Ctrl-C to stop all servers.
"""

from __future__ import annotations

import argparse
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
_HEALTH_TIMEOUT_S = 120       # max seconds to wait for all servers to come up
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
# Shutdown
# ---------------------------------------------------------------------------

def terminate_all(procs: dict[str, subprocess.Popen], wait_s: float = 10.0) -> None:
    """Send SIGTERM to all processes, then SIGKILL stragglers."""
    for name, proc in procs.items():
        if proc.poll() is None:
            logger.info("Sending SIGTERM to '%s' (pid=%d)", name, proc.pid)
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + wait_s
    for name, proc in procs.items():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.warning("'%s' did not exit cleanly; sending SIGKILL", name)
            try:
                proc.kill()
            except ProcessLookupError:
                pass


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

    # ---- Launch all servers ----
    procs: dict[str, subprocess.Popen] = {}

    def _handle_signal(signum, _frame):
        logger.info("Signal %d received — shutting down all servers …", signum)
        terminate_all(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    failed_launch: list[str] = []
    for name, cfg in selected.items():
        try:
            proc = launch_server(name, cfg, log_dir=args.log_dir)
            procs[name] = proc
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Failed to launch '%s': %s", name, exc)
            failed_launch.append(name)

    if failed_launch:
        logger.error("Could not launch: %s — terminating running servers", failed_launch)
        terminate_all(procs)
        return 1

    # ---- Health checks ----
    if not args.no_health_check:
        server_ports = {name: cfg["port"] for name, cfg in selected.items()}
        healthy = wait_for_health(server_ports, timeout_s=args.health_timeout)
        if not healthy:
            logger.error("Health check failed — terminating all servers")
            terminate_all(procs)
            return 1
    else:
        logger.info("Health check skipped (--no-health-check)")

    # ---- Keep running, monitor for unexpected exits ----
    logger.info(
        "All %d server(s) running. Press Ctrl-C to stop.",
        len(procs),
    )
    try:
        while True:
            time.sleep(5)
            for name, proc in list(procs.items()):
                rc = proc.poll()
                if rc is not None:
                    logger.error(
                        "Server '%s' (pid=%d) exited unexpectedly with code %d",
                        name,
                        proc.pid,
                        rc,
                    )
                    # Terminate remaining servers on unexpected exit
                    terminate_all(procs)
                    return 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down …")
        terminate_all(procs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
