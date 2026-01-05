"""SSH and rsync utilities for remote operations."""

import shutil
import subprocess


def test_ssh_connection(remote_dest: str, timeout: int = 10) -> bool:
    """
    Test SSH connection to remote destination (passwordless).
    
    Args:
        remote_dest: SSH destination in format "user@host:/path" or "user@host"
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    remote_host = remote_dest.split(":")[0]
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
             remote_host, "echo 'ok'"],
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def check_rsync_installed() -> bool:
    """Check if rsync is installed and available."""
    return shutil.which("rsync") is not None
