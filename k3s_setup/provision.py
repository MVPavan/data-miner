# provision.py — Unified pyinfra provisioning script
#
# All SSH/node-level operations in one file. Replaces:
#   k3s.py, data_miner_prep.py, docker_build.py, seaweedfs_prep.py
#
# Usage:
#   pyinfra inventory.py provision.py --data action=all       # Full provisioning
#   pyinfra inventory.py provision.py --data action=k3s       # K3s install only
#   pyinfra inventory.py provision.py --data action=nvidia    # NVIDIA runtime config
#   pyinfra inventory.py provision.py --data action=storage   # DB storage dirs
#   pyinfra inventory.py provision.py --data action=seaweedfs # SeaweedFS node prep
#   pyinfra inventory.py provision.py --data action=labels    # Node labels
#   pyinfra inventory.py provision.py --data action=docker    # Build & distribute image
#   pyinfra inventory.py provision.py --data action=clean     # Full cleanup
#   pyinfra inventory.py provision.py --data action=clean_k3s
#   pyinfra inventory.py provision.py --data action=clean_images
#   pyinfra inventory.py provision.py --data action=clean_seaweedfs  # Requires DELETE confirmation
#   pyinfra inventory.py provision.py --data action=clean_db         # Requires DELETE confirmation
#   pyinfra inventory.py provision.py --data action=clean_data       # Both SeaweedFS + DB

import os
import sys
import subprocess
import fcntl

from pyinfra import host, logger
from pyinfra.operations import apt, files, server, systemd
from pyinfra.facts.server import Which
from pyinfra.facts.files import File

sys.path.insert(0, os.path.dirname(__file__))
from cluster import cfg, master_hostname, master_ip, get_node_data_dir, get_node_image_type, get_image_config


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE Confirmation Helper
# ═══════════════════════════════════════════════════════════════════════════════

_delete_confirmed = False


def confirm_delete(what: str):
    """Require user to type DELETE to confirm destructive action.

    Aborts with sys.exit(1) if user doesn't type DELETE.
    """
    global _delete_confirmed
    if _delete_confirmed:
        return  # Already confirmed in this run

    print(f"\n⚠️  WARNING: This will permanently delete {what}!")
    print("Type DELETE to confirm: ", end="", flush=True)
    response = input().strip()
    if response != "DELETE":
        print("Aborted. You must type DELETE to proceed.")
        sys.exit(1)
    _delete_confirmed = True

# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════

is_master = host.data.get("role") == "master"
action = host.data.get("action", "all")
token = host.data.get("k3s_token", "")
host_master_ip = host.data.get("master_ip", master_ip())

K3S_VERSION = cfg().cluster.k3s_version
K3S_BINARY_URL = f"https://github.com/k3s-io/k3s/releases/download/{K3S_VERSION.replace('+', '%2B')}/k3s"
service = "k3s" if is_master else "k3s-agent"

NVIDIA_PLUGIN_VERSION = cfg().nvidia.plugin_version
NVIDIA_PLUGIN_URL = f"https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/{NVIDIA_PLUGIN_VERSION}/nvidia-device-plugin.yml"

# Image paths are now per-image-type, computed in build_and_distribute()
# Track which images have been built in this pyinfra run (avoids deadlock)
_images_built = set()

DB_PATH = cfg().storage.db_path
SEAWEED_DATA = cfg().seaweedfs.data_dir
SEAWEED_MOUNT = cfg().storage.seaweedfs_mount
MOUNT_PARENT = str(os.path.dirname(SEAWEED_MOUNT))

# Containerd config template — NVIDIA runtime + CNI paths (critical for node Ready)
CONTAINERD_CONFIG_TOML_TMPL = """\
[plugins."io.containerd.grpc.v1.cri".cni]
  bin_dir = "/opt/cni/bin"
  conf_dir = "/var/lib/rancher/k3s/agent/etc/cni/net.d"

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia"]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia".options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
"""


# ═══════════════════════════════════════════════════════════════════════════════
# K3s Install
# ═══════════════════════════════════════════════════════════════════════════════

def deploy_k3s():
    k3s_installed = host.get_fact(Which, command="k3s")

    if k3s_installed:
        systemd.service(
            name=f"Ensure {service} running",
            service=service,
            running=True,
            enabled=True,
            _sudo=True,
        )
        return

    # Prerequisites
    apt.update(name="Update apt", cache_time=3600, _sudo=True)
    apt.packages(name="Install curl", packages=["curl"], _sudo=True)
    server.shell(name="Disable swap", commands=["swapoff -a"], _sudo=True)

    if is_master:
        server.shell(name="Download K3s", commands=[
            f"curl -sfL https://get.k3s.io -o /tmp/k3s-install.sh && chmod +x /tmp/k3s-install.sh && "
            f"curl -Lo /tmp/k3s {K3S_BINARY_URL} && chmod +x /tmp/k3s"
        ], _sudo=True)
        server.shell(name="Install K3s master", commands=[
            "cp /tmp/k3s /usr/local/bin/k3s && "
            "INSTALL_K3S_SKIP_DOWNLOAD=true /tmp/k3s-install.sh server --write-kubeconfig-mode 644"
        ], _sudo=True)
        server.shell(name="Show token", commands=[
            "cat /var/lib/rancher/k3s/server/node-token"
        ], _sudo=True)
    elif token:
        # Copy k3s binary from master (local) instead of downloading
        files.put(
            name="Copy k3s binary from master",
            src="/usr/local/bin/k3s",
            dest="/tmp/k3s",
            mode="755",
            _sudo=True,
        )
        files.put(
            name="Copy k3s install script from master",
            src="/tmp/k3s-install.sh",
            dest="/tmp/k3s-install.sh",
            mode="755",
            _sudo=True, 
        )
        # Download install script and run with pre-copied binary
        server.shell(name="Install K3s agent", commands=[
            "chmod +x /tmp/k3s-install.sh && "
            "cp /tmp/k3s /usr/local/bin/k3s && "
            f"INSTALL_K3S_SKIP_DOWNLOAD=true K3S_URL=https://{host_master_ip}:6443 K3S_TOKEN={token} "
            "/tmp/k3s-install.sh agent"
        ], _sudo=True)
    else:
        server.shell(name="Skip — no token", commands=[
            "echo 'No token available — run master first'"
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# NVIDIA Runtime Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def setup_nvidia():
    if not is_master:
        return

    # Create containerd config directory
    files.directory(
        name="Create containerd config dir",
        path="/var/lib/rancher/k3s/agent/etc/containerd",
        _sudo=True,
    )

    # Write config.toml.tmpl
    server.shell(
        name="Write containerd config.toml.tmpl",
        commands=[
            f"cat > /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl << 'ENDOFCONFIG'\n"
            f"{CONTAINERD_CONFIG_TOML_TMPL}ENDOFCONFIG"
        ],
        _sudo=True,
    )

    # Generate CDI spec
    server.shell(
        name="Generate CDI spec",
        commands=[
            "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml 2>/dev/null || "
            "echo 'CDI generation skipped (nvidia-ctk not found)'"
        ],
        _sudo=True,
    )

    # Restart K3s to pick up config
    systemd.service(name="Restart K3s", service="k3s", restarted=True, _sudo=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Storage Directories (DB)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_storage():
    if not is_master:
        return

    for svc_name, uid in cfg().storage.db_services.items():
        full_path = f"{DB_PATH}/{svc_name}"
        files.directory(
            name=f"Create {svc_name} storage",
            path=full_path,
            user=str(uid),
            group=str(uid),
            mode="755",
            recursive=True,
            _sudo=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SeaweedFS Node Prep
# ═══════════════════════════════════════════════════════════════════════════════

def setup_seaweedfs():
    # Install FUSE
    apt.packages(name="Install fuse", packages=["fuse3"], _sudo=True)

    # Create directories
    files.directory(name=f"Create {SEAWEED_DATA}", path=SEAWEED_DATA, mode="755", _sudo=True)
    files.directory(name=f"Create {MOUNT_PARENT}", path=MOUNT_PARENT, mode="755", _sudo=True)
    files.directory(name=f"Create {SEAWEED_MOUNT}", path=SEAWEED_MOUNT, mode="777", _sudo=True)

    # FUSE config
    files.line(
        name="Enable FUSE user_allow_other",
        path="/etc/fuse.conf",
        line="user_allow_other",
        replace="^#?user_allow_other.*",
        present=True,
        _sudo=True,
    )

    # Sysctl tuning
    server.shell(name="Configure sysctl", commands=["""
cat > /etc/sysctl.d/99-seaweedfs.conf << 'EOF'
# SeaweedFS optimizations
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
net.core.somaxconn = 65535
vm.swappiness = 10
EOF
sysctl --system > /dev/null
"""], _sudo=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Node Labels
# ═══════════════════════════════════════════════════════════════════════════════

def apply_labels():
    if not is_master:
        return

    # Label master
    server.shell(
        name="Label master node",
        commands=[
            "kubectl label node $(hostname | tr '[:upper:]' '[:lower:]') "
            "node-role.kubernetes.io/master=true --overwrite",
        ],
        _sudo=True,
    )

    # Apply labels from cluster config
    for node_name, node_cfg in cfg().cluster.nodes.items():
        labels = node_cfg.get("labels", {})
        for label_key, label_val in labels.items():
            server.shell(
                name=f"Label {node_name}: {label_key}={label_val}",
                commands=[
                    f"kubectl label node {node_name} {label_key}={label_val} --overwrite"
                ],
                _sudo=True,
            )

    # Install NVIDIA device plugin (only on GPU nodes via nodeSelector)
    server.shell(
        name="Install NVIDIA Device Plugin",
        commands=[
            f"kubectl create -f {NVIDIA_PLUGIN_URL} 2>/dev/null || true",
            # Patch to only run on gpu=true nodes
            'kubectl patch daemonset nvidia-device-plugin-daemonset -n kube-system '
            '--type=json -p=\'[{"op": "add", "path": "/spec/template/spec/nodeSelector", '
            '"value": {"gpu": "true"}}]\' 2>/dev/null || true',
        ],
        _sudo=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Docker Image Build & Distribute
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_distribute():
    """Build and distribute the appropriate Docker image to this node.

    - Skips storage_only nodes (they don't run data-miner workers)
    - Builds base or gpu image based on node's 'image' config
    - Uses file lock to ensure image is built only once per type
    """
    hostname = host.data.hostname

    # Skip storage-only nodes — they don't run data-miner workers
    if host.data.get("storage_only", False):
        print(f"[{hostname}] Skipping — storage_only node, no image needed")
        return

    # Determine which image this node needs
    image_type = get_node_image_type(hostname)
    if not image_type:
        print(f"[{hostname}] Skipping — no image configured")
        return

    img_cfg = get_image_config(image_type)
    image_name = img_cfg["name"]
    full_image = img_cfg["full_name"]
    dockerfile = img_cfg["dockerfile"]
    tar_path = f"/tmp/{image_name}.tar"

    force = os.environ.get("FORCE", "false").lower() == "true"
    dockerfile_path = os.path.join(os.path.dirname(__file__), dockerfile)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build image if not already built in this run
    if image_type in _images_built:
        print(f"[{hostname}] Image already built in this run: {full_image}")
    elif os.path.exists(tar_path) and not force:
        print(f"[{hostname}] Image tar already exists: {tar_path}")
        _images_built.add(image_type)
    else:
        print(f"[{hostname}] Building Docker Image: {full_image}...")
        subprocess.check_call(
            f"docker build -f {dockerfile_path} -t {full_image} {project_root}",
            shell=True,
        )
        print(f"[{hostname}] Saving Image to {tar_path}...")
        subprocess.check_call(
            f"docker save {full_image} -o {tar_path}",
            shell=True,
        )
        print(f"[{hostname}] Image saved successfully")
        _images_built.add(image_type)

    # Copy to node
    remote_file = host.get_fact(File, path=tar_path)
    if force or not remote_file:
        files.put(
            name=f"Copy {image_type} image tar to node",
            src=tar_path,
            dest=tar_path,
        )

    # Import into K3s containerd
    server.shell(
        name=f"Import {image_type} image to K3s",
        commands=[f"k3s ctr images import {tar_path} --namespace k8s.io"],
        _sudo=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup Actions
# ═══════════════════════════════════════════════════════════════════════════════

def clean_k3s():
    """Uninstall K3s (force remove everything)."""
    server.shell(name="Stop services", commands=[
        f"systemctl stop {service} || true",
        "/usr/local/bin/k3s-killall.sh || true",
    ], _sudo=True)
    server.shell(name="Remove K3s files", commands=[
        "rm -rf /usr/local/bin/k3s* /usr/local/bin/kubectl /usr/local/bin/crictl",
        "rm -rf /var/lib/rancher/k3s /etc/rancher/k3s",
        "rm -rf /tmp/k3s* /tmp/k3s-install.sh",
        "rm -f /etc/systemd/system/k3s*.service",
        "systemctl daemon-reload",
    ], _sudo=True)


def clean_images():
    """Remove Docker images from K3s containerd for this node."""
    hostname = host.data.hostname
    image_type = get_node_image_type(hostname)

    if not image_type:
        print(f"[{hostname}] Skipping — storage_only node, no images to remove")
        return

    img_cfg = get_image_config(image_type)
    full_image = img_cfg["full_name"]

    server.shell(
        name=f"Remove {image_type} Docker image",
        commands=[f"k3s ctr -n k8s.io images rm docker.io/library/{full_image} 2>/dev/null; true"],
        _sudo=True,
    )


def clean_seaweedfs():
    """Unmount FUSE and remove SeaweedFS data for this node."""
    confirm_delete("SeaweedFS data on this node")

    hostname = host.data.hostname
    node_data_dir = get_node_data_dir(hostname)

    server.shell(name="Unmount FUSE", commands=[
        f"umount -l {SEAWEED_MOUNT} 2>/dev/null; true",
    ], _sudo=True)
    server.shell(name="Clean SeaweedFS data", commands=[
        f"rm -rf {node_data_dir}/master {node_data_dir}/filer {node_data_dir}/volume",
    ], _sudo=True)


def clean_db():
    """Remove database directories (postgres, loki, grafana). Master only."""
    if host.data.role != "master":
        return

    confirm_delete("database directories (postgres, loki, grafana)")

    db_path = cfg().storage.db_path
    server.shell(name="Clean DB data", commands=[
        f"rm -rf {db_path}/postgres {db_path}/loki {db_path}/grafana",
    ], _sudo=True)


def clean_data():
    """Clean all data: SeaweedFS + DB."""
    confirm_delete("ALL data (SeaweedFS + database)")
    # Skip individual confirmations since we already confirmed
    global _delete_confirmed
    _delete_confirmed = True
    clean_seaweedfs()
    clean_db()


def clean_all():
    """Full cleanup: SeaweedFS + images + K3s."""
    clean_seaweedfs()
    clean_images()
    clean_k3s()


# ═══════════════════════════════════════════════════════════════════════════════
# K3s Status/Restart/Stop (utility actions from old k3s.py)
# ═══════════════════════════════════════════════════════════════════════════════

def k3s_status():
    server.shell(name="Status", commands=[
        f"systemctl status {service} --no-pager || true"
    ], _sudo=True)
    if is_master:
        server.shell(name="Nodes", commands=["kubectl get nodes -o wide || true"], _sudo=True)


def k3s_restart():
    systemd.service(name="Restart", service=service, restarted=True, _sudo=True)


def k3s_stop():
    systemd.service(name="Stop", service=service, running=False, _sudo=True)


def k3s_uninstall():
    script = "/usr/local/bin/k3s-uninstall.sh" if is_master else "/usr/local/bin/k3s-agent-uninstall.sh"
    if host.get_fact(File, path=script):
        server.shell(name="Uninstall", commands=[script], _sudo=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Action Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

ACTIONS = {
    # Provisioning
    "k3s": deploy_k3s,
    "nvidia": setup_nvidia,
    "storage": setup_storage,
    "seaweedfs": setup_seaweedfs,
    "labels": apply_labels,
    "docker": build_and_distribute,
    "all": lambda: [f() for f in [deploy_k3s, setup_nvidia, setup_storage,
                                    setup_seaweedfs, apply_labels, build_and_distribute]],
    # Cleanup
    "clean": clean_all,
    "clean_k3s": clean_k3s,
    "clean_images": clean_images,
    "clean_seaweedfs": clean_seaweedfs,
    "clean_db": clean_db,
    "clean_data": clean_data,
    # Utilities (compat with old k3s.py actions)
    "status": k3s_status,
    "restart": k3s_restart,
    "stop": k3s_stop,
    "uninstall": k3s_uninstall,
}

if action not in ACTIONS:
    logger.error(f"Invalid action: {action}. Valid: {list(ACTIONS.keys())}")
    raise SystemExit(1)

ACTIONS[action]()
