# inventory.py — Pyinfra inventory (reads from cluster_config.yaml)
#
# Usage:
#   pyinfra inventory.py provision.py --data action=all
#   pyinfra inventory.py provision.py --data action=k3s
#
# For passwordless sudo on master:
#   sudo visudo → add: pavanmv ALL=(ALL) NOPASSWD: ALL

import subprocess
from pathlib import Path

from cluster import cfg, nodes, master_ip

ssh_key = str(Path(cfg().cluster.ssh_key).expanduser())

# Auto-read K3s token from master (if cluster already running)
try:
    result = subprocess.run(
        ["sudo", "cat", "/var/lib/rancher/k3s/server/node-token"],
        capture_output=True, text=True, timeout=5,
    )
    K3S_TOKEN = result.stdout.strip() if result.returncode == 0 else ""
    if K3S_TOKEN:
        print(f"Token found: {K3S_TOKEN}")
    else:
        print("No token found")
except Exception as e:
    K3S_TOKEN = ""
    print(f"No token found: {e}")

hosts = []
for name, node in nodes().items():
    hosts.append((str(node.ip), {
        "ssh_user": node.ssh_user,
        "ssh_key": ssh_key,
        "role": node.role,
        "hostname": name,
        "master_ip": master_ip(),
        "k3s_token": K3S_TOKEN,
        "labels": dict(node.get("labels", {})),
    }))
