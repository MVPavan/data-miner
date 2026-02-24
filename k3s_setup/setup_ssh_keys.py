#!/usr/bin/env python3
"""Setup SSH keys for K3s cluster nodes.

Reads all node IPs, users, and key path from cluster_config.yaml.
Generates the key if it doesn't exist, then copies to all nodes.

Usage:
    python setup_ssh_keys.py              # Generate key (if needed) + copy to all nodes
    python setup_ssh_keys.py --test       # Test SSH connectivity to all nodes
    python setup_ssh_keys.py --clean      # Delete existing key + clean known_hosts entries
    python setup_ssh_keys.py --regenerate # Delete, regenerate, and copy to all nodes
"""

import argparse
import subprocess
import sys
from pathlib import Path

from cluster import cfg, nodes


def ssh_key_path():
    """Resolve SSH key path from cluster config."""
    return Path(cfg().cluster.ssh_key).expanduser()


def host_list():
    """Return [(ip, user, hostname), ...] from cluster config."""
    return [(str(n.ip), n.ssh_user, name) for name, n in nodes().items()]


def generate_key(key_path):
    """Generate an ed25519 SSH key if it doesn't already exist."""
    if key_path.exists():
        print(f"Key already exists: {key_path}")
        return True

    key_path.parent.mkdir(parents=True, exist_ok=True)
    comment = key_path.name
    r = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", comment],
    )
    if r.returncode != 0:
        print(f"FAILED: ssh-keygen returned {r.returncode}")
        return False

    print(f"Generated: {key_path}")
    return True


def delete_key(key_path):
    """Delete existing SSH key pair if it exists."""
    deleted = []
    for suffix in ["", ".pub"]:
        p = Path(f"{key_path}{suffix}")
        if p.exists():
            p.unlink()
            deleted.append(p.name)
            print(f"Deleted: {p}")

    if not deleted:
        print(f"No keys found at: {key_path}")
    return deleted


def clean_known_hosts(hosts):
    """Remove known_hosts entries for cluster nodes."""
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    if not known_hosts.exists():
        print("No known_hosts file found")
        return

    removed = []
    for ip, user, hostname in hosts:
        r = subprocess.run(
            ["ssh-keygen", "-R", ip],
            capture_output=True,
        )
        if r.returncode == 0:
            removed.append(f"{hostname} ({ip})")

    if removed:
        print(f"Removed known_hosts entries for: {', '.join(removed)}")
    else:
        print("No known_hosts entries found for cluster nodes")


def copy_keys(key_path, hosts):
    """Copy public key to all cluster nodes."""
    pub_key = f"{key_path}.pub"
    failed = []
    for ip, user, hostname in hosts:
        print(f"[{hostname} / {ip}] copying key...")
        r = subprocess.run(
            [
                "ssh-copy-id",
                "-i",
                pub_key,
                "-o",
                "StrictHostKeyChecking=no",
                f"{user}@{ip}",
            ],
        )
        if r.returncode != 0:
            print(f"[{hostname} / {ip}] FAILED")
            failed.append(hostname)
        else:
            print(f"[{hostname} / {ip}] OK")
    return failed


def test_connectivity(key_path, hosts):
    """Test SSH connectivity to all cluster nodes."""
    failed = []
    for ip, user, hostname in hosts:
        r = subprocess.run(
            [
                "ssh",
                "-i",
                str(key_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                f"{user}@{ip}",
                "echo OK",
            ],
            capture_output=True,
        )
        status = "OK" if r.returncode == 0 else "FAILED"
        print(f"[{hostname} / {ip}] {status}")
        if r.returncode != 0:
            failed.append(hostname)
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Setup SSH keys for K3s cluster nodes (reads from cluster_config.yaml)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test SSH connectivity only"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete existing key and clean known_hosts"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Delete existing key and regenerate + copy",
    )
    args = parser.parse_args()

    key_path = ssh_key_path()
    hosts = host_list()

    print(f"SSH key:  {key_path}")
    print(f"Nodes:    {', '.join(h[2] for h in hosts)}")
    print()

    if args.clean:
        delete_key(key_path)
        clean_known_hosts(hosts)
        print("\nCleanup complete.")
    elif args.test:
        failed = test_connectivity(key_path, hosts)
        if failed:
            print(f"\nFailed: {', '.join(failed)}")
            sys.exit(1)
        print("\nAll nodes reachable.")
    elif args.regenerate:
        delete_key(key_path)
        clean_known_hosts(hosts)
        print()
        if not generate_key(key_path):
            sys.exit(1)
        print()
        failed = copy_keys(key_path, hosts)
        if failed:
            print(f"\nFailed to copy to: {', '.join(failed)}")
            sys.exit(1)
        print("\nDone! Key regenerated and configured.")
    else:
        if not generate_key(key_path):
            sys.exit(1)
        print()
        failed = copy_keys(key_path, hosts)
        if failed:
            print(f"\nFailed to copy to: {', '.join(failed)}")
            sys.exit(1)
        print(f"\nDone! Key configured in cluster_config.yaml: {cfg().cluster.ssh_key}")


if __name__ == "__main__":
    main()
