"""
Shared cluster configuration loader.

All k3s_setup scripts import from this module to read cluster_config.yaml.
"""

from pathlib import Path
from omegaconf import OmegaConf

_CFG = None
CONFIG_FILE = Path(__file__).parent / "cluster_config.yaml"


def cfg():
    """Load and cache cluster config."""
    global _CFG
    if _CFG is None:
        _CFG = OmegaConf.load(CONFIG_FILE)
    return _CFG


def nodes(role=None):
    """Return {name: node_cfg} dict, optionally filtered by role."""
    all_nodes = cfg().cluster.nodes
    if role:
        return {k: v for k, v in all_nodes.items() if v.get("role") == role}
    return dict(all_nodes)


def master():
    """Return (name, node_cfg) tuple for the master node."""
    return next(iter(nodes("master").items()))


def master_hostname():
    """Return the master node's hostname."""
    return master()[0]


def master_ip():
    """Return the master node's IP address."""
    return str(master()[1].ip)


def compute_nodes():
    """Return nodes that should run data-miner workers (excludes storage_only)."""
    return {k: v for k, v in nodes().items() if not v.get("storage_only", False)}


def storage_nodes():
    """Return all nodes that run SeaweedFS volume servers (all nodes)."""
    return nodes()


def get_node_disk_limit(hostname: str) -> int:
    """Get SeaweedFS disk limit in MB for a specific node.

    Returns 0 for unlimited if no limit is configured.
    """
    node = nodes().get(hostname, {})
    default_limit = cfg().seaweedfs.get("default_disk_limit_mb", 0)
    return node.get("disk_limit_mb", default_limit)


def get_node_data_dir(hostname: str) -> str:
    """Get SeaweedFS data directory for a specific node.

    Falls back to seaweedfs.data_dir if not specified per-node.
    """
    node = nodes().get(hostname, {})
    default_dir = cfg().seaweedfs.data_dir
    return node.get("data_dir", default_dir)


def resolve_schedule(schedule_on):
    """Convert schedule_on string to K8s scheduling spec.

    Returns dict with either 'topology_spread' or 'node_selector' key.
    Storage-only nodes are excluded from worker scheduling.
    """
    if schedule_on == "all":
        # Only schedule on compute nodes (exclude storage_only)
        return {"topology_spread": True, "allowed_nodes": list(compute_nodes().keys())}
    if schedule_on == "master":
        return {"node_selector": {"kubernetes.io/hostname": master_hostname()}}
    if schedule_on == "gpu":
        gpu_nodes = {
            k: v for k, v in compute_nodes().items()
            if v.get("labels", {}).get("gpu") == "true"
        }
        hostname = next(iter(gpu_nodes))
        return {"node_selector": {"kubernetes.io/hostname": hostname, "gpu": "true"}}
    # Treat as explicit hostname
    return {"node_selector": {"kubernetes.io/hostname": schedule_on}}
