"""
K3s Data Miner — K8s Orchestrator (kubectl only, no SSH)

Handles manifest generation, deployment, and teardown via kubectl.
Node-level provisioning (K3s install, NVIDIA, Docker image) is done
separately by provision.py via pyinfra.

Usage:
    python k3s_setup/orchestrate.py setup --run-config run_configs/glass_door.yaml
    python k3s_setup/orchestrate.py teardown [--wipe-data]
    python k3s_setup/orchestrate.py all --run-config run_configs/glass_door.yaml [--wipe-data]

Full workflow:
    # Phase 1: Provision nodes (SSH/pyinfra)
    pyinfra k3s_setup/inventory.py k3s_setup/provision.py --data action=all

    # Phase 2: Deploy to K8s (kubectl)
    python k3s_setup/orchestrate.py setup --run-config run_configs/glass_door.yaml
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cluster import cfg, master_hostname

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (from cluster config)
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
K3S_SETUP_DIR = Path(__file__).resolve().parent
MANIFESTS_DIR = K3S_SETUP_DIR / "manifests"

NAMESPACE = cfg().cluster.namespace
SEAWEEDFS_NS = cfg().seaweedfs.namespace
NVIDIA_PLUGIN_URL = (
    f"https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/"
    f"{cfg().nvidia.plugin_version}/nvidia-device-plugin.yml"
)


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


def kubectl(
    args: str, check: bool = True, capture: bool = False, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Run kubectl locally."""
    return run(f"kubectl {args}", check=check, capture=capture, timeout=timeout)


def wait_for_nodes(count: int, timeout: int = 300):
    """Wait until the expected number of nodes are Ready."""
    print(f"  Waiting for {count} nodes to be Ready (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run(
            "kubectl get nodes --no-headers 2>/dev/null | grep ' Ready ' | wc -l",
            check=False, capture=True, timeout=15,
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
            check=False, capture=True, timeout=15,
        )
        lines = [l for l in (result.stdout or "").strip().split("\n") if l.strip()]
        if len(lines) >= min_pods:
            all_running = all("Running" in line for line in lines)
            if all_running:
                ok(f"{len(lines)} pods Running in {namespace}")
                return
            not_ready = [l for l in lines if "Running" not in l]
            pending = len(not_ready)
            print(f"    {len(lines) - pending}/{len(lines)} Running, waiting...")
        time.sleep(15)
    kubectl(f"get pods -n {namespace} -o wide", check=False)
    fail(f"Timed out waiting for pods in {namespace}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TEARDOWN (kubectl only)
# ═══════════════════════════════════════════════════════════════════════════════


def teardown(wipe_data: bool = False):
    if wipe_data:
        banner("TEARDOWN — Full Clean Slate (INCLUDING project data)")
    else:
        banner("TEARDOWN — Clean Slate (preserving project data)")

    # Check if K3s/kubectl is available
    k3s_check = run(
        "kubectl cluster-info 2>/dev/null && echo K3S_UP || echo K3S_DOWN",
        check=False, capture=True, timeout=15,
    )
    k3s_up = "K3S_UP" in (k3s_check.stdout or "")

    if k3s_up:
        step(1, f"Delete {NAMESPACE} namespace")
        kubectl(f"delete ns {NAMESPACE} --ignore-not-found --timeout=120s", check=False)
        ok()

        step(2, f"Delete {SEAWEEDFS_NS} namespace")
        kubectl(f"delete ns {SEAWEEDFS_NS} --ignore-not-found --timeout=120s", check=False)
        ok()

        step(3, "Delete NVIDIA device plugin")
        kubectl(f"delete -f {NVIDIA_PLUGIN_URL} --ignore-not-found", check=False)
        ok()

        step(4, "Remove node labels")
        kubectl(
            f"label node {master_hostname()} node-role.kubernetes.io/master- gpu- "
            f"--ignore-not-found 2>/dev/null || true",
            check=False,
        )
        ok()
    else:
        warn("K3s not running — skipping namespace/label cleanup (steps 1-4)")

    if wipe_data:
        step(5, "WIPE: Call provision.py for node cleanup")
        run(
            f"cd {K3S_SETUP_DIR} && uv run pyinfra inventory.py provision.py -y "
            f"--data action=clean_seaweedfs",
            check=False, timeout=300,
        )
        ok()

        step(6, "WIPE: Database dirs on master")
        db_path = cfg().storage.db_path
        run(f"sudo rm -rf {db_path}/postgres {db_path}/loki {db_path}/grafana", check=False)
        ok("DB dirs wiped")
    else:
        step(5, "SKIP: Preserving project data")
        ok()

    banner("TEARDOWN COMPLETE")


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP (generate manifests + kubectl apply)
# ═══════════════════════════════════════════════════════════════════════════════


def setup(run_config: str):
    banner("SETUP — Deploy to K8s")

    # ── Step 1: Generate manifests ───────────────────────────────────────
    step(1, "Generate manifests")
    run(
        f"python {K3S_SETUP_DIR / 'generate_manifests.py'} --run-config {run_config}",
        timeout=30,
    )
    ok("Manifests generated")

    # ── Step 2: Verify nodes ready ───────────────────────────────────────
    step(2, "Verify cluster nodes are Ready")
    wait_for_nodes(3, timeout=60)

    # ── Step 3: Deploy SeaweedFS ─────────────────────────────────────────
    step(3, "Deploy SeaweedFS")
    kubectl(f"apply -f {MANIFESTS_DIR / 'seaweedfs/'}")
    wait_for_pods(SEAWEEDFS_NS, timeout=300, min_pods=4)

    # ── Step 4: Verify FUSE mount ────────────────────────────────────────
    step(4, "Verify FUSE mount via pod exec")
    mount_path = cfg().storage.seaweedfs_mount
    retries = 12
    for attempt in range(retries):
        result = kubectl(
            f"exec -n {SEAWEEDFS_NS} ds/mount -- ls {mount_path}",
            check=False, capture=True, timeout=10,
        )
        if result.returncode == 0:
            ok("FUSE mount accessible")
            break
        if attempt < retries - 1:
            print(f"    Mount not ready, retrying ({attempt + 1}/{retries})...")
            time.sleep(10)
    else:
        warn("Could not verify FUSE mount via pod exec — continuing anyway")

    # ── Step 5: Create namespace ─────────────────────────────────────────
    step(5, f"Create {NAMESPACE} namespace")
    kubectl(f"apply -f {MANIFESTS_DIR / 'namespace.yaml'}")
    ok()

    # ── Step 6: Create HF secret ─────────────────────────────────────────
    step(6, "Create HF token secret")
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
            f"-n {NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
        )
        ok("HF secret created")
    else:
        warn("HF_TOKEN not found in .env — skipping secret creation")

    # ── Step 7: Deploy ConfigMap ─────────────────────────────────────────
    step(7, "Deploy ConfigMap")
    kubectl(f"apply -f {MANIFESTS_DIR / 'config/'}")
    ok()

    # ── Step 8: Deploy infrastructure ────────────────────────────────────
    step(8, "Deploy infrastructure (Postgres, Loki, Grafana)")
    kubectl(f"apply -f {MANIFESTS_DIR / 'infrastructure/'}")
    wait_for_pods(NAMESPACE, timeout=180, min_pods=3)

    # ── Step 9: Deploy workers ───────────────────────────────────────────
    step(9, "Deploy workers")
    kubectl(f"apply -f {MANIFESTS_DIR / 'workers/'}")
    # Expected: 3 infra + 3 download + 2 extract + 1 filter + 1 dedup + 1 detect + 1 monitor = 12
    wait_for_pods(NAMESPACE, timeout=600, min_pods=12)

    # ── Step 10: Initialize database ─────────────────────────────────────
    step(10, "Initialize database schema")
    kubectl(
        f"exec -n {NAMESPACE} deploy/monitor-worker -- python -m data_miner.cli init-db",
        timeout=60,
    )
    ok("Database initialized")

    # ── Step 11: Populate project data ───────────────────────────────────
    step(11, "Populate project data")
    kubectl(
        f"exec -n {NAMESPACE} deploy/monitor-worker -- "
        f"python -m data_miner.cli populate --config /config/config.yaml",
        timeout=120,
    )
    ok("Project data populated")

    # ── Step 12: Final verification ──────────────────────────────────────
    step(12, "Final verification")
    print()
    print(f"{Colors.BOLD}Cluster Nodes:{Colors.END}")
    kubectl("get nodes -o wide")
    print()
    print(f"{Colors.BOLD}SeaweedFS Pods:{Colors.END}")
    kubectl(f"get pods -n {SEAWEEDFS_NS} -o wide")
    print()
    print(f"{Colors.BOLD}Data Miner Pods:{Colors.END}")
    kubectl(f"get pods -n {NAMESPACE} -o wide")
    print()

    banner("SETUP COMPLETE")
    print(f"  Grafana:  http://{master_hostname()}:30300")
    print(
        f"  Status:   kubectl exec -n {NAMESPACE} deploy/monitor-worker "
        f"-- python -m data_miner.cli status"
    )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="K3s Data Miner — K8s Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Full workflow:
  # Phase 1: Provision nodes
  pyinfra k3s_setup/inventory.py k3s_setup/provision.py --data action=all

  # Phase 2: Deploy to K8s
  python k3s_setup/orchestrate.py setup --run-config run_configs/glass_door.yaml
""",
    )
    parser.add_argument(
        "command",
        choices=["teardown", "setup", "all"],
        help="teardown: destroy K8s resources | setup: deploy | all: teardown + setup",
    )
    parser.add_argument(
        "--wipe-data",
        action="store_true",
        help="Also delete project data (SeaweedFS files + database dirs)",
    )
    parser.add_argument(
        "--run-config",
        default=str(PROJECT_ROOT / "run_configs" / "glass_door.yaml"),
        help="Path to run config YAML for ConfigMap generation (default: glass_door.yaml)",
    )
    args = parser.parse_args()

    if args.command == "teardown":
        teardown(wipe_data=args.wipe_data)
    elif args.command == "setup":
        setup(run_config=args.run_config)
    elif args.command == "all":
        teardown(wipe_data=args.wipe_data)
        setup(run_config=args.run_config)


if __name__ == "__main__":
    main()
