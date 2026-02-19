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
    """Return nodes that run SeaweedFS volume servers (excludes no_storage nodes)."""
    return {k: v for k, v in nodes().items() if not v.get("no_storage", False)}


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


# =============================================================================
# Image Management
# =============================================================================


def get_node_image_type(hostname: str) -> str | None:
    """Get the image type (base/gpu) for a node.

    Returns None for storage_only nodes that don't need an image.
    """
    node = nodes().get(hostname, {})
    if node.get("storage_only", False):
        return None
    return node.get("image", "base")  # Default to base if not specified


def get_image_config(image_type: str) -> dict:
    """Get image configuration (name, tag, dockerfile) for an image type.

    Args:
        image_type: "base" or "gpu"

    Returns:
        Dict with name, tag, dockerfile, full_name keys
    """
    img_cfg = cfg().images.get(image_type)
    if not img_cfg:
        raise ValueError(f"Unknown image type: {image_type}")
    return {
        "name": img_cfg.name,
        "tag": img_cfg.tag,
        "dockerfile": img_cfg.dockerfile,
        "full_name": f"{img_cfg.name}:{img_cfg.tag}",
    }


def get_required_images() -> dict[str, list[str]]:
    """Get which images need to be built and which nodes need them.

    Returns:
        Dict mapping image_type -> list of hostnames that need it
    """
    result = {}
    for hostname, node_cfg in compute_nodes().items():
        image_type = node_cfg.get("image", "base")
        if image_type not in result:
            result[image_type] = []
        result[image_type].append(hostname)
    return result


def get_worker_image(schedule_on: str) -> str:
    """Get the full image name for a worker based on its schedule.

    Workers inherit the image from the node they're scheduled on.
    For 'all' scheduling, uses the most capable image (gpu if any node has it).
    """
    if schedule_on == "all":
        # Use base image if any compute node has base — lowest common denominator
        # so the image is available on every node the pod might be scheduled on.
        # Only use gpu if ALL compute nodes have gpu.
        has_base = any(
            node_cfg.get("image", "base") == "base"
            for node_cfg in compute_nodes().values()
        )
        if has_base:
            return get_image_config("base")["full_name"]
        return get_image_config("gpu")["full_name"]
    elif schedule_on == "master":
        master_cfg = master()[1]
        image_type = master_cfg.get("image", "base")
        return get_image_config(image_type)["full_name"]
    elif schedule_on == "gpu":
        # GPU workers always use gpu image
        return get_image_config("gpu")["full_name"]
    else:
        # Explicit hostname
        node_cfg = nodes().get(schedule_on, {})
        image_type = node_cfg.get("image", "base")
        return get_image_config(image_type)["full_name"]
