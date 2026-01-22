"""
Fabric Deployment Script for Distributed Data Miner

Supports Docker Swarm mode for distributed deployments.

SWARM MODE:
    fab swarm-init                      # Initialize swarm on master
    fab swarm-join                      # Join all workers to swarm
    fab swarm-deploy                    # Deploy stack to swarm
    fab swarm-scale --service=download --replicas=30
    fab swarm-status                    # Check swarm and service status

Common:
    fab --list                          # Show available tasks
    fab logs --service=download         # View logs from a service
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from fabric import task, Connection
from invoke import Context


# =============================================================================
# Configuration
# =============================================================================

CLUSTER_CONFIG_PATH = Path(__file__).parent / "docker_configs" / "cluster.yaml"
PROJECT_FILES = [
    "pyproject.toml",
    "uv.lock",
    "data_miner/",
    "run_configs/",
    "docker_configs/",
]


def load_cluster_config() -> dict:
    """Load cluster configuration from YAML."""
    if not CLUSTER_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Cluster config not found: {CLUSTER_CONFIG_PATH}")
    
    with open(CLUSTER_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        elif isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env(i) for i in obj]
        return obj
    
    return expand_env(config)


def get_connection(host_config: dict) -> Connection:
    """Create Fabric connection from host config."""
    connect_kwargs = {}
    
    if "key_file" in host_config:
        key_path = os.path.expanduser(host_config["key_file"])
        connect_kwargs["key_filename"] = key_path
    elif "password" in host_config:
        connect_kwargs["password"] = host_config["password"]
    
    return Connection(
        host=host_config["host"],
        user=host_config["user"],
        connect_kwargs=connect_kwargs,
    )


# =============================================================================
# Installation Tasks
# =============================================================================

@task
def install_docker(c):
    """Install Docker on a remote machine."""
    print(f"Installing Docker on {c.host}...")
    
    # Check if Docker is already installed
    result = c.run("which docker", warn=True, hide=True)
    if result.ok:
        print("Docker already installed")
        return
    
    # Install Docker
    c.sudo("apt-get update")
    c.sudo("apt-get install -y ca-certificates curl gnupg")
    c.sudo("install -m 0755 -d /etc/apt/keyrings")
    c.run("curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg")
    c.sudo("chmod a+r /etc/apt/keyrings/docker.gpg")
    
    c.run('''echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null''')
    
    c.sudo("apt-get update")
    c.sudo("apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin")
    
    # Add user to docker group
    c.sudo(f"usermod -aG docker {c.user}")
    print("Docker installed successfully!")


@task
def install_nvidia_docker(c):
    """Install NVIDIA Container Toolkit for GPU support."""
    print(f"Installing NVIDIA Docker on {c.host}...")
    
    c.run("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
    c.run('''curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list''')
    
    c.sudo("apt-get update")
    c.sudo("apt-get install -y nvidia-container-toolkit")
    c.sudo("nvidia-ctk runtime configure --runtime=docker")
    c.sudo("systemctl restart docker")
    print("NVIDIA Docker installed successfully!")


# =============================================================================
# Deployment Tasks
# =============================================================================

@task
def sync_project(c, dest_path: str = "/opt/data_miner"):
    """Sync project files to remote machine."""
    print(f"Syncing project to {c.host}:{dest_path}...")
    
    # Create destination directory
    c.sudo(f"mkdir -p {dest_path}")
    c.sudo(f"chown {c.user}:{c.user} {dest_path}")
    
    # Use rsync for efficient sync
    project_root = Path(__file__).parent
    
    for item in PROJECT_FILES:
        local_path = project_root / item
        if local_path.exists():
            if local_path.is_dir():
                c.run(f"rsync -avz --delete {local_path}/ {c.host}:{dest_path}/{item}/")
            else:
                c.run(f"rsync -avz {local_path} {c.host}:{dest_path}/{item}")
    
    print("Project synced!")


@task
def deploy_standalone(c):
    """Deploy standalone stack (single machine, no swarm)."""
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    
    if c.host != master_config["host"]:
        print(f"Connecting to master at {master_config['host']}...")
        c = get_connection(master_config)
    
    dest_path = config["deployment"]["project_path"]
    persistent_dir = config.get("storage", {}).get("persistent_data_dir", "./data")
    
    print(f"Deploying standalone stack to {c.host}...")
    
    # Sync project files
    sync_project(c, dest_path)
    
    # Create data directories
    c.sudo(f"mkdir -p {dest_path}/{persistent_dir}/postgres")
    c.sudo(f"mkdir -p {dest_path}/{persistent_dir}/grafana")
    c.sudo(f"mkdir -p {dest_path}/{persistent_dir}/loki")
    c.sudo(f"mkdir -p {dest_path}/{persistent_dir}/output")
    
    # Build and start
    with c.cd(f"{dest_path}/docker_configs"):
        if config["deployment"].get("rebuild_image", True):
            c.run("docker compose -f docker-compose.standalone.yml build")
        
        c.run("docker compose -f docker-compose.standalone.yml up -d")
    
    print("Standalone deployment complete!")


# REMOVED: deploy_worker and deploy_all - use swarm mode instead
# For distributed deployments, use:
#   fab swarm-init
#   fab swarm-join  
#   fab swarm-deploy


# =============================================================================
# Management Tasks
# =============================================================================

@task
def status(c):
    """Check swarm status (alias for swarm-status)."""
    swarm_status(c)


@task
def logs(c, service: str = "download", lines: int = 50):
    """View logs from a swarm service."""
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    master_conn.run(f"docker service logs --tail {lines} dm_{service}")


@task
def stop(c):
    """Stop the swarm stack (alias for swarm-down)."""
    swarm_down(c)


# =============================================================================
# Database Tasks
# =============================================================================

@task
def init_db(c):
    """Initialize database on master."""
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print("Initializing database...")
    master_conn.run("docker exec dm_workers data-miner init-db")
    print("Database initialized!")


@task
def populate(c, config_file: str = "/app/config.yaml"):
    """Run populate command on master."""
    cfg = load_cluster_config()
    master_conn = get_connection(cfg["cluster"]["master"])
    
    print("Running populate...")
    master_conn.run(f"docker exec dm_workers data-miner populate --config {config_file}")
    print("Populate complete!")


@task  
def pipeline_status(c):
    """Check pipeline status."""
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    master_conn.run("docker exec dm_workers data-miner status")


# =============================================================================
# Docker Swarm Tasks
# =============================================================================

@task
def swarm_init(c):
    """Initialize Docker Swarm on master node."""
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    master_conn = get_connection(master_config)
    master_ip = master_config["host"]
    
    print(f"Initializing Docker Swarm on {master_ip}...")
    
    # Check if already in swarm
    result = master_conn.run("docker info --format '{{.Swarm.LocalNodeState}}'", hide=True)
    if result.stdout.strip() == "active":
        print("Swarm already initialized!")
        # Get join token
        result = master_conn.run("docker swarm join-token worker -q", hide=True)
        print(f"\nWorker join token: {result.stdout.strip()}")
        return
    
    # Initialize swarm
    master_conn.run(f"docker swarm init --advertise-addr {master_ip}")
    
    # Get join token
    result = master_conn.run("docker swarm join-token worker -q", hide=True)
    token = result.stdout.strip()
    
    print(f"\nSwarm initialized!")
    print(f"Worker join token: {token}")
    print(f"\nRun on workers: docker swarm join --token {token} {master_ip}:2377")


@task
def swarm_join(c):
    """Join all worker nodes to the swarm."""
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    master_conn = get_connection(master_config)
    master_ip = master_config["host"]
    
    # Get join token
    result = master_conn.run("docker swarm join-token worker -q", hide=True)
    token = result.stdout.strip()
    
    print(f"Joining workers to swarm at {master_ip}...")
    
    for worker_config in config["cluster"]["workers"]:
        worker_conn = get_connection(worker_config)
        print(f"\nJoining {worker_config['host']}...")
        
        # Check if already in swarm
        result = worker_conn.run("docker info --format '{{.Swarm.LocalNodeState}}'", hide=True, warn=True)
        if result.stdout.strip() == "active":
            print(f"  Already in swarm")
            continue
        
        # Leave any existing swarm
        worker_conn.run("docker swarm leave --force", warn=True, hide=True)
        
        # Join swarm
        worker_conn.run(f"docker swarm join --token {token} {master_ip}:2377")
        print(f"  Joined successfully!")
    
    print("\nAll workers joined! Checking nodes...")
    master_conn.run("docker node ls")


@task
def swarm_deploy(c, registry: str = ""):
    """Deploy the stack to Docker Swarm."""
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    master_conn = get_connection(master_config)
    dest_path = config["deployment"]["project_path"]
    
    print("Deploying stack to Docker Swarm...")
    
    # Sync project files to master
    sync_project(master_conn, dest_path)
    
    # Build image on master
    print("\nBuilding Docker image...")
    with master_conn.cd(dest_path):
        master_conn.run("docker build -t data-miner:latest .")
        
        # If registry provided, push image
        if registry:
            master_conn.run(f"docker tag data-miner:latest {registry}/data-miner:latest")
            master_conn.run(f"docker push {registry}/data-miner:latest")
            print(f"Image pushed to {registry}")
    
    # Deploy stack
    print("\nDeploying stack...")
    with master_conn.cd(dest_path):
        if registry:
            master_conn.run(f"REGISTRY={registry} docker stack deploy -c docker-compose.swarm.yml dm")
        else:
            # For local registry-less deployment, build on each node
            master_conn.run("docker stack deploy -c docker-compose.swarm.yml dm --resolve-image=never")
    
    print("\nStack deployed! Checking services...")
    master_conn.run("docker service ls")


@task
def swarm_scale(c, service: str, replicas: int):
    """Scale a swarm service.
    
    Example: fab swarm-scale --service=download --replicas=30
    """
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print(f"Scaling dm_{service} to {replicas} replicas...")
    master_conn.run(f"docker service scale dm_{service}={replicas}")
    print("Scaling complete!")


@task
def swarm_status(c):
    """Check Docker Swarm status."""
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print("\n" + "="*50)
    print("SWARM NODES")
    print("="*50)
    master_conn.run("docker node ls")
    
    print("\n" + "="*50)
    print("SERVICES")
    print("="*50)
    master_conn.run("docker service ls")
    
    print("\n" + "="*50)
    print("DOWNLOAD WORKERS")
    print("="*50)
    master_conn.run("docker service ps dm_download --no-trunc", warn=True)


@task
def swarm_logs(c, service: str = "download", lines: int = 50):
    """View logs from a swarm service."""
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    master_conn.run(f"docker service logs --tail {lines} dm_{service}")


@task
def swarm_down(c):
    """Remove the swarm stack."""
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print("Removing swarm stack...")
    master_conn.run("docker stack rm dm")
    print("Stack removed!")


@task
def swarm_leave_all(c):
    """Leave swarm on all nodes (destroys swarm)."""
    config = load_cluster_config()
    
    # Workers leave first
    for worker_config in config["cluster"]["workers"]:
        worker_conn = get_connection(worker_config)
        print(f"Leaving swarm on {worker_config['host']}...")
        worker_conn.run("docker swarm leave --force", warn=True)
    
    # Master leaves last
    master_conn = get_connection(config["cluster"]["master"])
    print("Leaving swarm on master...")
    master_conn.run("docker swarm leave --force", warn=True)
    
    print("All nodes left swarm!")


@task
def swarm_build_all(c):
    """Build the Docker image on all nodes (for registry-less deployment)."""
    config = load_cluster_config()
    dest_path = config["deployment"]["project_path"]
    
    # Build on master
    master_conn = get_connection(config["cluster"]["master"])
    print(f"Building on master...")
    sync_project(master_conn, dest_path)
    with master_conn.cd(dest_path):
        master_conn.run("docker build -t data-miner:latest .")
    
    # Build on workers
    for worker_config in config["cluster"]["workers"]:
        worker_conn = get_connection(worker_config)
        print(f"\nBuilding on {worker_config['host']}...")
        sync_project(worker_conn, dest_path)
        with worker_conn.cd(dest_path):
            worker_conn.run("docker build -t data-miner:latest .")
    
    print("\nImage built on all nodes!")

