#!/usr/bin/env python3
# Setup SSH keys: python setup_ssh_keys.py k3s_cluster [--test]

import subprocess, sys, os

HOSTS = [
    ("10.96.122.9", "pavanmv"),
    ("10.96.122.132", "pavanmv"),
    ("10.96.122.14", "pavan"),
]

key_name = sys.argv[1] if len(sys.argv) > 1 else "id_rsa_dm_k3s"
key_path = os.path.expanduser(f"~/.ssh/{key_name}")

if "--test" in sys.argv:
    for ip, user in HOSTS:
        r = subprocess.run(["ssh", "-i", key_path, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", f"{user}@{ip}", "echo OK"], capture_output=True)
        print(f"[{ip}] {'OK' if r.returncode == 0 else 'FAILED'}")
else:
    if not os.path.exists(key_path):
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "", "-C", key_name])
    for ip, user in HOSTS:
        print(f"[{ip}] copying...")
        subprocess.run(["ssh-copy-id", "-i", f"{key_path}.pub", "-o", "StrictHostKeyChecking=no", f"{user}@{ip}"])
    print(f'\nDone! Set ssh_key in inventory.py to: {key_path}')
