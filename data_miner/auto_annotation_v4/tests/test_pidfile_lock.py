"""Tier 1 unit tests for the pidfile/flock lock semantics used by pipeline.py.

Exercises the same ``fcntl.flock(lock_fd, LOCK_EX | LOCK_NB)`` pattern that
pipeline.Pipeline.run() uses, in isolation — no workers, no DB connect.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


_ACQUIRE_SNIPPET = textwrap.dedent(
    """
    import fcntl
    import os
    import signal
    import sys
    import time

    lock_path = sys.argv[1]
    hold_seconds = float(sys.argv[2])

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("CONTENTION", flush=True)
        os.close(fd)
        sys.exit(2)

    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\\n".encode())
    print("ACQUIRED", flush=True)

    # Install SIGTERM handler to exit cleanly (releases the lock).
    def _bye(*_):
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _bye)

    deadline = time.time() + hold_seconds
    while time.time() < deadline:
        time.sleep(0.05)

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    print("RELEASED", flush=True)
    """
)


def _spawn(lock_path: Path, hold_seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _ACQUIRE_SNIPPET, str(lock_path), str(hold_seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_line(proc: subprocess.Popen, expected: str, timeout: float = 5.0) -> str:
    """Read stdout lines until *expected* is seen or process exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return ""
            time.sleep(0.02)
            continue
        line = line.strip()
        if line == expected:
            return line
        if line in ("CONTENTION", "RELEASED"):
            return line
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_second_process_sees_contention(tmp_path):
    lock_path = tmp_path / "pipeline.lock"

    p1 = _spawn(lock_path, hold_seconds=3.0)
    try:
        assert _wait_line(p1, "ACQUIRED") == "ACQUIRED", (
            f"p1 stderr: {p1.stderr.read() if p1.stderr else ''}"
        )

        p2 = _spawn(lock_path, hold_seconds=0.1)
        out2, err2 = p2.communicate(timeout=5)
        assert p2.returncode == 2, (
            f"p2 returncode={p2.returncode}, out={out2!r}, err={err2!r}"
        )
        assert "CONTENTION" in out2
    finally:
        p1.terminate()
        p1.wait(timeout=5)


def test_lock_released_on_clean_exit(tmp_path):
    lock_path = tmp_path / "pipeline.lock"

    p1 = _spawn(lock_path, hold_seconds=0.2)
    out1, err1 = p1.communicate(timeout=5)
    assert p1.returncode == 0, f"p1 returncode={p1.returncode}, err={err1!r}"
    assert "ACQUIRED" in out1 and "RELEASED" in out1

    # Now a second process can acquire the lock cleanly.
    p2 = _spawn(lock_path, hold_seconds=0.1)
    out2, err2 = p2.communicate(timeout=5)
    assert p2.returncode == 0, f"p2 returncode={p2.returncode}, err={err2!r}"
    assert "ACQUIRED" in out2


def test_lock_released_on_sigkill(tmp_path):
    lock_path = tmp_path / "pipeline.lock"

    p1 = _spawn(lock_path, hold_seconds=30.0)
    try:
        assert _wait_line(p1, "ACQUIRED") == "ACQUIRED"
        # Kernel-held flock is auto-released on process death, even SIGKILL.
        p1.send_signal(signal.SIGKILL)
        p1.wait(timeout=5)
    finally:
        if p1.poll() is None:
            p1.kill()
            p1.wait(timeout=5)

    p2 = _spawn(lock_path, hold_seconds=0.1)
    out2, err2 = p2.communicate(timeout=5)
    assert p2.returncode == 0, (
        f"After SIGKILL of holder, p2 could not acquire lock. "
        f"returncode={p2.returncode}, out={out2!r}, err={err2!r}"
    )
    assert "ACQUIRED" in out2
