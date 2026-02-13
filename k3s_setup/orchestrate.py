"""
K3s Data Miner — Full Teardown & Fresh Install Orchestrator

Usage:
    uv run python k3s_setup/orchestrate.py teardown               # Clean everything (keeps project data)
    uv run python k3s_setup/orchestrate.py teardown --wipe-data    # Clean everything INCLUDING project data
    uv run python k3s_setup/orchestrate.py setup                   # Fresh install
    uv run python k3s_setup/orchestrate.py all                     # Teardown (keep data) → setup
    uv run python k3s_setup/orchestrate.py all --wipe-data         # Full wipe → setup

Note: This script runs ON the master node (pavanjci) — kubectl runs locally.
"""

import subprocess
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
K3S_SETUP_DIR = PROJECT_ROOT / "k3s_setup"

NODES = [
    {"ip": "10.96.122.9", "user": "pavanmv", "role": "master", "name": "pavanjci"},
    {"ip": "10.96.122.132", "user": "pavanmv", "role": "worker", "name": "manthana"},
    {"ip": "10.96.122.14", "user": "pavan", "role": "worker", "name": "arsenal"},
]

MASTER = NODES[0]
WORKERS = [n for n in NODES if n["role"] == "worker"]
SSH_KEY = str(Path.home() / ".ssh" / "id_rsa_dm_k3s")
SSH_OPTS = f"-i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

NVIDIA_PLUGIN_URL = "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml"

# K3s containerd config template with NVIDIA runtime + correct CNI paths
# Missing CNI paths causes the node to go NotReady
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
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def banner(msg: str, char: str = "="):
    width = 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}{char * width}")
    print(f"  {msg}")
    print(f"{char * width}{Colors.END}\n")


def step(num: int, msg: str):
    print(f"{Colors.BOLD}{Colors.BLUE}[Step {num}]{Colors.END} {msg}")


def ok(msg: str = "OK"):
    print(f"  {Colors.GREEN}✓ {msg}{Colors.END}")


def warn(msg: str):
    print(f"  {Colors.YELLOW}⚠ {msg}{Colors.END}")


def fail(msg: str):
    print(f"  {Colors.RED}✗ {msg}{Colors.END}")


def run(
    cmd: str, check: bool = True, capture: bool = False, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a local command."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        if capture:
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
        fail(f"Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def run_sudo(
    cmd: str, check: bool = True, capture: bool = False, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a local sudo command (we're on master)."""
    return run(f"sudo {cmd}", check=check, capture=capture, timeout=timeout)


def ssh(
    ip: str,
    user: str,
    cmd: str,
    check: bool = True,
    capture: bool = False,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run a command on a remote node via SSH."""
    full_cmd = f"ssh {SSH_OPTS} {user}@{ip} {repr(cmd)}"
    print(f"  [{ip}] $ {cmd}")
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        if capture:
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
        fail(f"SSH command failed on {ip}")
        sys.exit(1)
    return result


def ssh_sudo(
    ip: str,
    user: str,
    cmd: str,
    check: bool = True,
    capture: bool = False,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run a sudo command on a remote node."""
    return ssh(ip, user, f"sudo {cmd}", check=check, capture=capture, timeout=timeout)


def on_workers(cmd: str, sudo: bool = True, check: bool = False):
    """Run a command on worker nodes only."""
    for node in WORKERS:
        if sudo:
            ssh_sudo(node["ip"], node["user"], cmd, check=check)
        else:
            ssh(node["ip"], node["user"], cmd, check=check)


def on_all_nodes(cmd: str, sudo: bool = True, check: bool = False):
    """Run a command on all nodes (local sudo for master, SSH for workers)."""
    # Master — local
    print(f"  [master] $ sudo {cmd}")
    subprocess.run(f"sudo {cmd}", shell=True, timeout=120)
    # Workers — SSH
    on_workers(cmd, sudo=sudo, check=check)


def kubectl(
    args: str, check: bool = True, capture: bool = False, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Run kubectl locally (we're on master)."""
    return run(f"kubectl {args}", check=check, capture=capture, timeout=timeout)


def pyinfra_run(script: str, extra_args: str = ""):
    """Run a pyinfra script from k3s_setup dir."""
    cmd = f"cd {K3S_SETUP_DIR} && uv run pyinfra inventory.py {script} -y {extra_args}"
    run(cmd, timeout=600)


def wait_for_nodes(count: int, timeout: int = 300):
    """Wait until the expected number of nodes are Ready."""
    print(f"  Waiting for {count} nodes to be Ready (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run(
            "kubectl get nodes --no-headers 2>/dev/null | grep ' Ready ' | wc -l",
            check=False,
            capture=True,
            timeout=15,
        )
        try:
            ready = int(result.stdout.strip())
        except (ValueError, AttributeError):
            ready = 0
        if ready >= count:
            ok(f"{ready}/{count} nodes Ready")
            return
        time.sleep(10)
    fail(f"Timed out waiting for {count} nodes")
    sys.exit(1)


def wait_for_pods(namespace: str, timeout: int = 300, min_pods: int = 1):
    """Wait until all pods in a namespace are Running."""
    print(f"  Waiting for pods in {namespace} to be Running (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run(
            f"kubectl get pods -n {namespace} --no-headers 2>/dev/null",
            check=False,
            capture=True,
            timeout=15,
        )
        lines = [l for l in (result.stdout or "").strip().split("\n") if l.strip()]
        if len(lines) >= min_pods:
            all_running = all("Running" in line for line in lines)
            not_ready = [l for l in lines if "Running" not in l]
            if all_running:
                ok(f"{len(lines)} pods Running in {namespace}")
                return
            pending = len(not_ready)
            print(f"    {len(lines) - pending}/{len(lines)} Running, waiting...")
        time.sleep(15)
    # Print final state on timeout
    kubectl(f"get pods -n {namespace} -o wide", check=False)
    fail(f"Timed out waiting for pods in {namespace}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TEARDOWN
# ═══════════════════════════════════════════════════════════════════════════════


def teardown(wipe_data: bool = False):
    if wipe_data:
        banner("TEARDOWN — Full Clean Slate (INCLUDING project data)")
    else:
        banner("TEARDOWN — Clean Slate (preserving project data)")

    # Check if K3s/kubectl is available for namespace cleanup
    k3s_check = run(
        "kubectl cluster-info 2>/dev/null && echo K3S_UP || echo K3S_DOWN",
        check=False,
        capture=True,
        timeout=15,
    )
    k3s_up = "K3S_UP" in (k3s_check.stdout or "")

    if k3s_up:
        step(1, "Delete data-miner namespace")
        kubectl("delete ns data-miner --ignore-not-found --timeout=120s", check=False)
        ok()

        step(2, "Delete seaweedfs namespace")
        kubectl("delete ns seaweedfs --ignore-not-found --timeout=120s", check=False)
        ok()

        step(3, "Delete NVIDIA device plugin")
        kubectl(f"delete -f {NVIDIA_PLUGIN_URL} --ignore-not-found", check=False)
        ok()

        step(4, "Remove node labels")
        kubectl(
            "label node pavanjci node-role.kubernetes.io/master- gpu- --ignore-not-found 2>/dev/null || true",
            check=False,
        )
        ok()
    else:
        warn("K3s not running — skipping namespace/label cleanup (steps 1-4)")

    step(5, "Wait for FUSE unmounts to settle")
    time.sleep(15)
    ok()

    step(6, "Clean SeaweedFS metadata on all nodes")
    on_all_nodes("rm -rf /data/seaweed/master /data/seaweed/filer /data/seaweed/volume")
    ok()

    step(7, "Unmount FUSE on all nodes")
    on_all_nodes("umount -f /swdfs_mnt/swshared 2>/dev/null; true")
    ok()

    if wipe_data:
        step(8, "WIPE: SeaweedFS data (data_miner_output) on all nodes")
        on_all_nodes("rm -rf /swdfs_mnt/swshared/data_miner_output")
        on_all_nodes("rm -rf /swdfs_mnt/swshared/*")
        ok("All SeaweedFS data wiped")

        step(9, "WIPE: Database dirs on master")
        run_sudo(
            "rm -rf /data/data_miner_db/postgres /data/data_miner_db/loki /data/data_miner_db/grafana",
            check=False,
        )
        ok("DB dirs wiped")
    else:
        step(8, "SKIP: Preserving SeaweedFS data (data_miner_output)")
        ok("Data preserved at /swdfs_mnt/swshared/data_miner_output")

        step(9, "SKIP: Preserving database dirs")
        ok("Data preserved at /data/data_miner_db/")

    step(10, "Remove Docker image from all nodes")
    on_all_nodes(
        "k3s ctr -n k8s.io images rm docker.io/library/data-miner:latest 2>/dev/null; true"
    )
    ok()

    step(11, "Remove local tar cache")
    run("rm -f /tmp/data-miner.tar /tmp/data_miner_build.lock", check=False)
    ok()

    step(12, "Uninstall K3s on all nodes (force clean)")
    try:
        pyinfra_run("k3s.py", "--data action=clean")
    except (SystemExit, subprocess.CalledProcessError):
        warn("Pyinfra clean had issues — trying manual cleanup")
        # Manual fallback for each node
        cleanup_cmd = (
            "systemctl stop k3s k3s-agent 2>/dev/null; "
            "/usr/local/bin/k3s-killall.sh 2>/dev/null; "
            "rm -rf /usr/local/bin/k3s* /usr/local/bin/kubectl /usr/local/bin/crictl "
            "/var/lib/rancher/k3s /etc/rancher/k3s /tmp/k3s* "
            "/etc/systemd/system/k3s*.service; "
            "systemctl daemon-reload"
        )
        # Master — local
        run_sudo(cleanup_cmd, check=False)
        # Workers — SSH
        on_workers(cleanup_cmd, sudo=True, check=False)
    ok()

    step(13, "Verify K3s removed from all nodes")
    # Master — local
    result = run("which k3s 2>/dev/null || echo GONE", check=False, capture=True)
    if "GONE" in (result.stdout or ""):
        ok("master: K3s removed")
    else:
        warn("master: K3s binary still found")

    for node in WORKERS:
        result = ssh(
            node["ip"],
            node["user"],
            "which k3s 2>/dev/null || echo GONE",
            check=False,
            capture=True,
            timeout=10,
        )
        if "GONE" in (result.stdout or ""):
            ok(f"{node['name']}: K3s removed")
        else:
            warn(f"{node['name']}: K3s binary still found")

    banner("TEARDOWN COMPLETE")


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════


def setup():
    banner("SETUP — Fresh Install from Scratch")

    # ── Step 1: Install K3s on master only ─────────────────────────────────
    step(1, "Install K3s on master node")
    # Target master directly — inventory.py would fail reading token since
    # K3s isn't installed yet. We run from k3s_setup dir so k3s.py is found.
    run(
        f"cd {K3S_SETUP_DIR} && uv run pyinfra "
        f"--user {MASTER['user']} --key {SSH_KEY} --sudo "
        f"{MASTER['ip']} k3s.py -y --data role=master",
        timeout=300,
    )
    ok("K3s master installed")

    # ── Step 2: Wait for master Ready + read token ─────────────────────────
    step(2, "Wait for master node to be Ready and read token")
    wait_for_nodes(1, timeout=120)

    token_result = run_sudo(
        "cat /var/lib/rancher/k3s/server/node-token",
        capture=True,
        timeout=15,
    )
    token = token_result.stdout.strip()
    if not token:
        fail("Could not read K3s token from master")
        sys.exit(1)
    ok(f"Token: {token[:20]}...")

    # ── Step 3: Install K3s on workers ─────────────────────────────────────
    step(3, "Install K3s on worker nodes")
    # Now inventory.py can read the token from master
    pyinfra_run("k3s.py")
    wait_for_nodes(3, timeout=180)

    # ── Step 4: Configure NVIDIA containerd runtime on master ──────────────
    step(4, "Configure NVIDIA containerd runtime on master")

    # Create the containerd config directory
    run_sudo("mkdir -p /var/lib/rancher/k3s/agent/etc/containerd")

    # Write the config template
    config_path = "/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl"
    run(
        f"sudo tee {config_path} > /dev/null << 'ENDOFCONFIG'\n{CONTAINERD_CONFIG_TOML_TMPL}ENDOFCONFIG",
    )
    ok("config.toml.tmpl written")

    # Generate CDI spec
    run_sudo(
        "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml 2>/dev/null || "
        "echo 'CDI generation skipped (nvidia-ctk not found)'",
        check=False,
    )

    # Restart K3s to pick up the new config
    run_sudo("systemctl restart k3s")
    time.sleep(10)
    wait_for_nodes(3, timeout=120)
    ok("NVIDIA runtime configured, all nodes Ready")

    # ── Step 5: Prepare SeaweedFS on all nodes ─────────────────────────────
    step(5, "Prepare SeaweedFS on all nodes (fuse, dirs, sysctl)")
    pyinfra_run("seaweedfs_prep.py")
    ok()

    # ── Step 6: Deploy SeaweedFS ───────────────────────────────────────────
    step(6, "Deploy SeaweedFS to K3s")
    kubectl(f"apply -f {K3S_SETUP_DIR / 'seaweedfs.yaml'}")
    # SeaweedFS has: master, filer, volume (DaemonSet x3 nodes), mount (DaemonSet x3 nodes) = ~8 pods
    wait_for_pods("seaweedfs", timeout=300, min_pods=4)

    # ── Step 7: Verify FUSE mount on all nodes ─────────────────────────────
    step(7, "Verify FUSE mount on all nodes")
    retries = 12  # ~2 minutes
    for attempt in range(retries):
        all_mounted = True
        # Master — local check
        master_check = run(
            "ls /swdfs_mnt/swshared 2>/dev/null && echo MOUNTED || echo NOT_MOUNTED",
            check=False,
            capture=True,
            timeout=5,
        )
        if "NOT_MOUNTED" in (master_check.stdout or ""):
            all_mounted = False

        # Workers — SSH check
        for node in WORKERS:
            result = ssh(
                node["ip"],
                node["user"],
                "ls /swdfs_mnt/swshared 2>/dev/null && echo MOUNTED || echo NOT_MOUNTED",
                check=False,
                capture=True,
                timeout=10,
            )
            if "NOT_MOUNTED" in (result.stdout or ""):
                all_mounted = False

        if all_mounted:
            ok("FUSE mount accessible on all nodes")
            break
        if attempt < retries - 1:
            print(f"    Mount not ready, retrying ({attempt + 1}/{retries})...")
            time.sleep(10)
    else:
        fail("FUSE mount not accessible on all nodes after retries")
        sys.exit(1)

    # ── Step 8: Prepare Data Miner nodes ───────────────────────────────────
    step(8, "Prepare Data Miner nodes (labels, NVIDIA plugin, storage dirs)")
    pyinfra_run("data_miner_prep.py")
    ok()

    # ── Step 9: Build & distribute Docker image ────────────────────────────
    step(9, "Build and distribute Docker image to all nodes")
    run(
        f"cd {K3S_SETUP_DIR} && FORCE=true uv run pyinfra inventory.py docker_build.py -y",
        timeout=600,
    )
    ok("Image built and distributed")

    # ── Step 10: Create namespace ──────────────────────────────────────────
    step(10, "Create data-miner namespace")
    kubectl(f"apply -f {K3S_SETUP_DIR / 'manifests' / 'namespace.yaml'}")
    ok()

    # ── Step 11: Create HF secret ─────────────────────────────────────────
    step(11, "Create HF token secret")
    env_file = PROJECT_ROOT / ".env"
    hf_token = ""
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip()
                break
    if hf_token:
        kubectl(
            f"create secret generic hf-secret "
            f"--from-literal=token={hf_token} "
            f"-n data-miner --dry-run=client -o yaml | kubectl apply -f -"
        )
        ok("HF secret created")
    else:
        warn("HF_TOKEN not found in .env — skipping secret creation")

    # ── Step 12: Deploy config ─────────────────────────────────────────────
    step(12, "Deploy ConfigMap")
    kubectl(f"apply -f {K3S_SETUP_DIR / 'manifests' / 'config/'}")
    ok()

    # ── Step 13: Deploy infrastructure ─────────────────────────────────────
    step(13, "Deploy infrastructure (Postgres, Loki, Grafana)")
    kubectl(f"apply -f {K3S_SETUP_DIR / 'manifests' / 'infrastructure/'}")
    wait_for_pods("data-miner", timeout=180, min_pods=3)

    # ── Step 14: Deploy workers ────────────────────────────────────────────
    step(14, "Deploy workers")
    kubectl(f"apply -f {K3S_SETUP_DIR / 'manifests' / 'workers/'}")
    # Expected: 3 infra + 3 download + 2 extract + 1 filter + 1 dedup + 1 detect + 1 monitor = 12
    wait_for_pods("data-miner", timeout=600, min_pods=12)

    # ── Step 15: Initialize database ───────────────────────────────────────
    step(15, "Initialize database schema")
    kubectl(
        "exec -n data-miner deploy/monitor-worker -- python -m data_miner.cli init-db",
        timeout=60,
    )
    ok("Database initialized")

    # ── Step 16: Populate project data ─────────────────────────────────────
    step(16, "Populate project data")
    kubectl(
        "exec -n data-miner deploy/monitor-worker -- python -m data_miner.cli populate --config /config/config.yaml",
        timeout=120,
    )
    ok("Project data populated")

    # ── Step 17: Final verification ────────────────────────────────────────
    step(17, "Final verification")
    print()
    print(f"{Colors.BOLD}Cluster Nodes:{Colors.END}")
    kubectl("get nodes -o wide")
    print()
    print(f"{Colors.BOLD}SeaweedFS Pods:{Colors.END}")
    kubectl("get pods -n seaweedfs -o wide")
    print()
    print(f"{Colors.BOLD}Data Miner Pods:{Colors.END}")
    kubectl("get pods -n data-miner -o wide")
    print()

    banner("SETUP COMPLETE")
    print("  Grafana:  http://pavanjci:30300")
    print(
        "  Status:   kubectl exec -n data-miner deploy/monitor-worker -- python -m data_miner.cli status"
    )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

USAGE = """
K3s Data Miner — Full Teardown & Fresh Install Orchestrator

Usage:
    uv run python k3s_setup/orchestrate.py <command> [options]

Commands:
    teardown              Destroy K3s, SeaweedFS, images (preserves project data by default)
    setup                 Fresh install from scratch
    all                   Teardown then setup

Options:
    --wipe-data           Also delete project data (SeaweedFS files + database dirs)
""".strip()


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    command = args[0]
    wipe_data = "--wipe-data" in args

    if command not in ("teardown", "setup", "all"):
        print(USAGE)
        sys.exit(1)

    if command == "teardown":
        teardown(wipe_data=wipe_data)
    elif command == "setup":
        setup()
    elif command == "all":
        teardown(wipe_data=wipe_data)
        setup()


if __name__ == "__main__":
    main()
