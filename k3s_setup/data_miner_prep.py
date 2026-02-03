# data_miner_prep.py
#
# Prepare nodes for Data Miner on K3s
#
# Usage:
#   pyinfra inventory.py data_miner_prep.py
#

from pyinfra import host
from pyinfra.operations import server, files

# Config
DATA_MINER_DB_DIR = "/data/data_miner_db"
SUBDIRS = ["postgres", "loki", "grafana"]

# User IDs for permissions (matching container UIDs)
# postgres: 999
# loki: 10001
# grafana: 472
UID_MAP = {
    "postgres": 999,
    "loki": 10001,
    "grafana": 472,
}

# Detect roles
is_master = host.data.get("role") == "master"

# ═══════════════════════════════════════════════════════════════════
# MASTER NODE SETUP
# ═══════════════════════════════════════════════════════════════════

if is_master:
    # 1. Label Master Node
    # --------------------
    server.shell(
        name="Label master node",
        commands=[
            "kubectl label node $(hostname | tr '[:upper:]' '[:lower:]') node-role.kubernetes.io/master=true --overwrite",
            "kubectl label node $(hostname | tr '[:upper:]' '[:lower:]') gpu=true --overwrite",
        ],
        _sudo=True,
    )

    # 2. Install NVIDIA Device Plugin
    # -------------------------------
    # Check if already installed to avoid error
    server.shell(
        name="Install NVIDIA Device Plugin",
        commands=[
            "kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml || true"
        ],
        _sudo=True,
    )

    # 3. Create Persistent Storage Directories
    # ----------------------------------------
    for subdir in SUBDIRS:
        full_path = f"{DATA_MINER_DB_DIR}/{subdir}"
        uid = UID_MAP.get(subdir, 0)
        gid = uid  # Assuming GID matches UID for simplicity in these standard images

        files.directory(
            name=f"Create {subdir} storage",
            path=full_path,
            user=str(uid),
            group=str(gid),
            mode="755",
            recursive=True,
            _sudo=True,
        )

# ═══════════════════════════════════════════════════════════════════
# ALL NODES (Verification)
# ═══════════════════════════════════════════════════════════════════

# Verify SeaweedFS mount exists (Diagnostic only)
server.shell(
    name="Verify SeaweedFS mount",
    commands=["ls -ld /swdfs_mnt/swshared || echo 'WARNING: SeaweedFS mount missing'"],
    _sudo=True,
)
