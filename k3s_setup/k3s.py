# k3s.py
#
# Usage:
#   pyinfra inventory.py k3s.py                          # Deploy all
#   pyinfra inventory.py k3s.py --data offline=true      # Offline workers
#   pyinfra inventory.py k3s.py --data action=status     # Status
#   pyinfra inventory.py k3s.py --data action=uninstall  # Uninstall (graceful)
#   pyinfra inventory.py k3s.py --data action=clean      # Clean (force remove all)
#   pyinfra inventory.py k3s.py --data action=restart    # Restart
#   pyinfra inventory.py k3s.py --data action=stop       # Stop

from pyinfra import host, logger
from pyinfra.operations import apt, files, server, systemd
from pyinfra.facts.server import Which
from pyinfra.facts.files import File

valid_actions = ["deploy", "status", "restart", "stop", "uninstall", "clean"]

# Config
K3S_VERSION = "v1.31.0+k3s1"
K3S_BINARY_URL = f"https://github.com/k3s-io/k3s/releases/download/{K3S_VERSION.replace('+', '%2B')}/k3s"

# Detect
is_master = host.data.get("role") == "master"
action = host.data.get("action", "deploy")
offline = host.data.get("offline", False)
token = host.data.get("k3s_token", "")
master_ip = host.data.get("master_ip")
service = "k3s" if is_master else "k3s-agent"
k3s_installed = host.get_fact(Which, command="k3s")

# ═══════════════════════════════════════════════════════════════════
# ACTIONS
# ═══════════════════════════════════════════════════════════════════
if action not in valid_actions:
    logger.error(f"Invalid action: {action}. Valid: {valid_actions}")
    raise SystemExit(1)


if action == "status":
    server.shell(name="Status", commands=[f"systemctl status {service} --no-pager || true"], _sudo=True)
    if is_master:
        server.shell(name="Nodes", commands=["kubectl get nodes -o wide || true"], _sudo=True)

elif action == "restart":
    systemd.service(name="Restart", service=service, restarted=True, _sudo=True)

elif action == "stop":
    systemd.service(name="Stop", service=service, running=False, _sudo=True)

elif action == "uninstall":
    script = "/usr/local/bin/k3s-uninstall.sh" if is_master else "/usr/local/bin/k3s-agent-uninstall.sh"
    if host.get_fact(File, path=script):
        server.shell(name="Uninstall", commands=[script], _sudo=True)

# ═══════════════════════════════════════════════════════════════════
# CLEAN (force - removes everything including partial installs)
# ═══════════════════════════════════════════════════════════════════

elif action == "clean":
    server.shell(name="Stop services", commands=[
        f"systemctl stop {service} || true",
        "/usr/local/bin/k3s-killall.sh || true"
    ], _sudo=True)
    server.shell(name="Remove K3s files", commands=[
        "rm -rf /usr/local/bin/k3s* /usr/local/bin/kubectl /usr/local/bin/crictl",
        "rm -rf /var/lib/rancher/k3s /etc/rancher/k3s",
        "rm -rf /tmp/k3s* /tmp/k3s-install.sh",
        "rm -f /etc/systemd/system/k3s*.service",
        "systemctl daemon-reload"
    ], _sudo=True)

# ═══════════════════════════════════════════════════════════════════
# DEPLOY
# ═══════════════════════════════════════════════════════════════════

elif not k3s_installed:
    # Prep
    apt.update(name="Update apt", cache_time=3600, _sudo=True)
    apt.packages(name="Install packages", packages=["curl"], _sudo=True)
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
        server.shell(name="Token", commands=["cat /var/lib/rancher/k3s/server/node-token"], _sudo=True)

    elif token:
        if offline:
            files.put(name="Copy K3s files", src="/tmp/k3s-install.sh", dest="/tmp/k3s-install.sh", mode="755", _sudo=True)
            files.put(name="Copy K3s binary", src="/tmp/k3s", dest="/tmp/k3s", mode="755", _sudo=True)
            server.shell(name="Install K3s agent (offline)", commands=[
                f"cp /tmp/k3s /usr/local/bin/k3s && "
                f"INSTALL_K3S_SKIP_DOWNLOAD=true K3S_URL=https://{master_ip}:6443 K3S_TOKEN={token} /tmp/k3s-install.sh agent"
            ], _sudo=True)
        else:
            server.shell(name="Install K3s agent", commands=[
                f"curl -sfL https://get.k3s.io | K3S_URL=https://{master_ip}:6443 K3S_TOKEN={token} sh -s - agent"
            ], _sudo=True)
    else:
        server.shell(name="Skip", commands=["echo 'No token - run master first'"])

elif k3s_installed:
    systemd.service(name=f"Ensure {service} running", service=service, running=True, enabled=True, _sudo=True)