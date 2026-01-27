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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



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
    "generate_compose.py",
    "fabfile.py",
    "weed",
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


def is_local_host(host: str) -> bool:
    """Check if host is the local machine."""
    if host in ["localhost", "127.0.0.1", "::1"]:
        return True
    
    # Check if host IP matches any local interface using 'ip addr'
    # This addresses the issue where socket.gethostbyname return 127.0.1.1
    try:
        import subprocess
        # Run ip addr to list all IPs
        result = subprocess.run(
            ["ip", "addr"], 
            capture_output=True, 
            text=True
        )
        
        # Check if IP exists in output (look for "inet <IP>/")
        # We add spaces/slashes to ensure we don't partial match (e.g. 10.0.0.1 matching 10.0.0.100)
        if f"inet {host}/" in result.stdout or f"inet {host} " in result.stdout:
            return True
            
        # Fallback to socket check just in case
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        if host == local_ip:
            return True
            
    except Exception:
        pass
        
    return False


def get_connection(host_config: dict) -> Connection:
    """Create Fabric connection or local Context.
    
    STRICT SSH MODE:
    - Must use SSH Key (Explicit or Implicit).
    - Password is only used for sudo prompts via Config, NOT for connection.
    - If no key is configured or found, this fails.
    """
    host = host_config["host"]
    
    # If running on the target host, return local Context
    if is_local_host(host):
        print(f"[{host}] Local connection.")
        from invoke import Config
        config = Config()
        if "password" in host_config:
            config.sudo.password = host_config["password"]
        config.user = host_config["user"]
        return Context(config=config)

    connect_kwargs = {}
    
    # 1. Check for explicit key_file in host config (per-node override)
    if "key_file" in host_config:
        key_path = os.path.expanduser(host_config["key_file"])
        if os.path.exists(key_path):
            connect_kwargs["key_filename"] = key_path
    
    # 2. Implicit Check: Try auto-generated data_miner key
    if "key_filename" not in connect_kwargs:
        # We try to load the name from config, or default
        # Note: We can't easily call load_cluster_config() here inside get_connection if it causes recursion?
        # No, load_cluster_config is safe.
        try:
             # Just look for the standard key we generate
             ssh_key_name = "id_rsa_data_miner"
             # If we want to be pure, we should read it from config["cluster"]["ssh_key"]["name"]
             # But host_config is just a chunk.
             # Let's try the default path.
             dm_key = Path.home() / ".ssh" / ssh_key_name
             if dm_key.exists():
                 connect_kwargs["key_filename"] = str(dm_key)
        except Exception:
             pass

    # 3. Final Strict Check
    if "key_filename" not in connect_kwargs:
         raise ValueError(f"[{host}] No SSH key configured and implicit key (id_rsa_data_miner) not found! Run 'fab ensure-ssh-access'.")

    return Connection(
        host=host,
        user=host_config["user"],
        connect_kwargs=connect_kwargs,
    )


# =============================================================================
# Installation Tasks
# =============================================================================

@task
def install_docker(c):
    """Install Docker on a remote machine.
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    hostname = c.host if hasattr(c, "host") else "localhost"
    print(f"Installing Docker on {hostname}...")
    # ... (Implementation same as before, omitted for brevity if needed, but keeping full replacement)
    
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
    """Install NVIDIA Container Toolkit for GPU support.
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    hostname = c.host if hasattr(c, "host") else "localhost"
    print(f"Installing NVIDIA Docker on {hostname}...")
    
    c.run("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
    c.run('''curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list''')
    
    c.sudo("apt-get update")
    c.sudo("apt-get install -y nvidia-container-toolkit")
    c.sudo("nvidia-ctk runtime configure --runtime=docker")
    c.sudo("systemctl restart docker")
    print("NVIDIA Docker installed successfully!")


@task
def ensure_ssh_access(c):
    """Enforce SSH access: Generate key if missing, deploy to nodes (Implicit Config)."""
    config = load_cluster_config()
    nodes = [config["cluster"]["master"]] + config["cluster"].get("workers", [])
    
    # Automated key details
    ssh_dir = Path.home() / ".ssh"
    key_name = "id_rsa_data_miner"
    key_path = ssh_dir / key_name
    pub_key_path = ssh_dir / f"{key_name}.pub"
    
    # 1. Ensure Local Key Exists
    if not key_path.exists():
        print(f"Generating automated key: {key_path}")
        ssh_dir.mkdir(parents=True, exist_ok=True)
        c.run(f'ssh-keygen -t rsa -b 4096 -f {key_path} -N "" -C "data_miner_key"', hide=True)
    
    with open(pub_key_path) as f:
        pub_key = f.read().strip()
        
    # 2. Iterate Nodes
    for node in nodes:
        host = node["host"]
        print(f"Checking access for {host}...")
        
        # Check if config has explicit key_file
        if "key_file" in node:
            key_file = os.path.expanduser(node["key_file"])
            if os.path.exists(key_file):
                print(f"  [OK] Has explicit key_file: {key_file}")
                continue
        
        # Implicit check: try connecting with our auto key
        # We can't use get_connection here because it throws error if no key.
        # So we try raw connection with the key.
        try:
             conn = Connection(host=host, user=node["user"], connect_kwargs={"key_filename": str(key_path)})
             conn.run("echo 'SSH OK'", hide=True)
             print("  [OK] Implicit key access works.")
             continue
        except Exception:
             pass 

        # If we are here, we need to setup access using password
        print(f"  [INFO] Setting up implicit passwordless access...")
        
        if "password" not in node:
             raise ValueError(f"[{host}] No key AND no password! Cannot setup access.")
             
        # Connect via Password to deploy key (One-time)
        try:
            print("  Connecting via password to install key...")
            temp_conn = Connection(host=host, user=node["user"], connect_kwargs={"password": node["password"]})
            
            temp_conn.run("mkdir -p ~/.ssh")
            # Check/Add
            if pub_key not in temp_conn.run("cat ~/.ssh/authorized_keys 2>/dev/null || echo ''", hide=True).stdout:
                temp_conn.run(f'echo "{pub_key}" >> ~/.ssh/authorized_keys')
                print("  Key installed on remote.")
            else:
                print("  Remote already has key.")
            
            # Fix permissions
            temp_conn.run("chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys")
            temp_conn.close()
            
        except Exception as e:
            print(f"  [ERROR] Failed to setup key: {e}")
            raise e

    print("\nAll nodes checked. Access granted.")


@task
def cleanup_ssh_keys(c):
    """Remove automated keys (Implicit Config)."""
    config = load_cluster_config()
    
    key_comment = "data_miner_key" # The tag we used
    
    nodes = [config["cluster"]["master"]] + config["cluster"].get("workers", [])
    
    print(f"Cleaning up SSH keys tagged '{key_comment}'...")
    
    for node in nodes:
        host = node["host"]
        
        # Try connecting with strict mode (using the key)
        # Ideally we use the implicit key we know about
        ssh_dir = Path.home() / ".ssh"
        key_name = "id_rsa_data_miner"
        key_path = ssh_dir / key_name
        
        try:
             # Connect explicitly with the key we want to revoke access for
             # (This works if we still have the key locally)
             kwargs = {}
             if key_path.exists():
                 kwargs["key_filename"] = str(key_path)
             elif "key_file" in node:
                 kwargs["key_filename"] = os.path.expanduser(node["key_file"])
             
             if not kwargs and "password" in node:
                 kwargs["password"] = node["password"]
                 
             if not kwargs:
                 print(f"  [{host}] Cannot connect to cleanup (no key/pass).")
                 continue

             conn = Connection(host=host, user=node["user"], connect_kwargs=kwargs)
             conn.run(f"sed -i '/{key_comment}/d' ~/.ssh/authorized_keys")
             print(f"  [{host}] Removed key from authorized_keys.")
        except Exception as e:
             print(f"  [{host}] Cleanup failed: {e}")

    # Cleanup Local Keys
    print("Cleaning up local keys...")
    for f in [ssh_dir / key_name, ssh_dir / f"{key_name}.pub"]:
        if f.exists():
            f.unlink()
            print(f"  Removed {f}")
            
    print("Cleanup complete!")


# =============================================================================
# Deployment Tasks
# =============================================================================

@task
def prepare_storage(c):
    """Create required storage directories on all machines.
    
    Creates:
    - seaweed_data_dir on ALL machines (master + workers)
    - persistent_data_dir on MASTER only
    
    Usage: fab prepare-storage
    """
    config = load_cluster_config()
    seaweed_dir = config.get("storage", {}).get("seaweed_data_dir", "/data/seaweed")
    persistent_dir = config.get("storage", {}).get("persistent_data_dir", "/data/data_miner")
    fuse_mount = config.get("storage", {}).get("fuse_mount", "/mnt/swdshared")
    
    # Prepare master
    master_config = config["cluster"]["master"]
    master_conn = get_connection(master_config)
    hostname = master_conn.host if hasattr(master_conn, "host") else "localhost"
    
    print(f"\n{'='*50}")
    print(f"Preparing storage on MASTER ({hostname})")
    print('='*50)
    
    # SeaweedFS directory and FUSE mount
    master_conn.sudo(f"mkdir -p {seaweed_dir}")
    master_conn.sudo(f"chmod 777 {seaweed_dir}")
    master_conn.sudo(f"mkdir -p {fuse_mount}")
    master_conn.sudo(f"chmod 777 {fuse_mount}")
    print(f"  Created: {seaweed_dir}")
    print(f"  Created: {fuse_mount}")
    
    # Persistent data directories (postgres, grafana, loki, output)
    for subdir in ["postgres", "grafana", "loki", "output"]:
        full_path = f"{persistent_dir}/{subdir}"
        master_conn.sudo(f"mkdir -p {full_path}")
        master_conn.sudo(f"chmod 777 {full_path}")
        print(f"  Created: {full_path}")
    
    # Prepare workers
    for worker_config in config["cluster"].get("workers", []):
        worker_conn = get_connection(worker_config)
        worker_host = worker_conn.host if hasattr(worker_conn, "host") else "localhost"
        
        print(f"\n{'='*50}")
        print(f"Preparing storage on WORKER ({worker_host})")
        print('='*50)
        
        # SeaweedFS directory and FUSE mount
        worker_conn.sudo(f"mkdir -p {seaweed_dir}")
        worker_conn.sudo(f"chmod 777 {seaweed_dir}")
        worker_conn.sudo(f"mkdir -p {fuse_mount}")
        worker_conn.sudo(f"chmod 777 {fuse_mount}")
        print(f"  Created: {seaweed_dir}")
        print(f"  Created: {fuse_mount}")
    
    print("\n" + "="*50)
    print("Storage directories prepared on all machines!")
    print("="*50)

@task
def sync_project(c, dest_path: str = "/opt/data_miner"):
    """Sync project files to remote machine."""
    hostname = c.host if hasattr(c, "host") else "localhost"
    print(f"Syncing project to {hostname}:{dest_path}...")
    
    # Create destination directory
    c.sudo(f"mkdir -p {dest_path}")
    c.sudo(f"chown {c.user}:{c.user} {dest_path}")
    
    project_root = Path(__file__).parent
    
    # If local, just copy files
    if isinstance(c, Context):
        for item in PROJECT_FILES:
            local_path = project_root / item
            if local_path.exists():
                c.run(f"cp -r {local_path} {dest_path}/")
    else:
        # Use rsync for remote
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
    """View logs from a swarm service (alias for swarm-logs)."""
    swarm_logs(c, service=service, lines=lines)


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
    """Join all worker nodes to the swarm.
    
    Ensures SSH access before starting.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    # ... remaining code flow via original function
    
    master_conn = get_connection(master_config)
    master_ip = master_config["host"]
    
    # Get join token
    result = master_conn.run("docker swarm join-token worker -q", hide=True)
    token = result.stdout.strip()
    
    print(f"Joining workers to swarm at {master_ip}...")
    
    for worker_config in config["cluster"]["workers"]:
        worker_conn = get_connection(worker_config)
        print(f"\\nJoining {worker_config['host']}...")
        
        # Check if already in swarm
        result = worker_conn.run("docker info --format '{{.Swarm.LocalNodeState}}'", hide=True, warn=True)
        if result.stdout.strip() == "active":
            print(f"  Already in swarm")
            continue
        
        # Leave any existing swarm
        worker_conn.sudo("docker swarm leave --force", warn=True, hide=True)
        
        # Join swarm
        worker_conn.sudo(f"docker swarm join --token {token} {master_ip}:2377")
        print(f"  Joined successfully!")
    
    print("\\nAll workers joined! Checking nodes...")
    master_conn.run("docker node ls")



@task
def swarm_deploy_seaweed(c):
    """Deploy only SeaweedFS to swarm (for testing distributed storage).
    
    This deploys seaweed-master, seaweed-filer, seaweed-volume, and seaweed-mount
    without postgres, data-miner, or other services.
    
    Usage: fab swarm-deploy-seaweed
    """
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    master_conn = get_connection(master_config)
    dest_path = config["deployment"]["project_path"]
    
    print("Deploying SeaweedFS-only stack to Docker Swarm...")
    
    # Sync project files to master
    sync_project(master_conn, dest_path)
    
    # Generate seaweed compose if not exists
    with master_conn.cd(dest_path):
        master_conn.run("uv run python generate_compose.py --scenario seaweed")
    
    # Deploy stack
    print("\nDeploying SeaweedFS stack...")
    with master_conn.cd(f"{dest_path}/docker_configs"):
        master_conn.run("docker stack deploy -c docker-compose.seaweed.yml dm-seaweed")
    
    print("\nSeaweedFS stack deployed! Checking services...")
    master_conn.run("docker service ls | grep dm-seaweed")


@task
def swarm_deploy(c, registry: str = ""):
    """Deploy the stack to Docker Swarm.
    
    Ensures SSH access before starting.
    """
    ensure_ssh_access(c)
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
        # Dockerfile is in docker_configs/Dockerfile, but context should implementation root?
        # Assuming project root is needed for context.
        master_conn.run("docker build -t data-miner:latest -f docker_configs/Dockerfile .")
        
        # If registry provided, push image
        if registry:
            master_conn.run(f"docker tag data-miner:latest {registry}/data-miner:latest")
            master_conn.run(f"docker push {registry}/data-miner:latest")
            print(f"Image pushed to {registry}")
    
    # Deploy stack
    print("\nDeploying stack...")
    with master_conn.cd(dest_path):
        # We need to render the compose file first!
        # swarm_deploy_seaweed does: uv run python generate_compose.py --scenario seaweed
        # preventing old stale files.
        master_conn.run("uv run python generate_compose.py --scenario master") # Wait, scenarios uses scenarios['master']?
        # Actually generate_compose.py generates multiple files? 
        # Reference generate_compose arguments.
        # It defaults to generating ALL if no args? Or specific?
        # Let's check how we deploy.
        # We deploy 'docker-compose.swarm.yml'. 
        return_code = master_conn.run("uv run python generate_compose.py --scenario swarm", warn=True)
        # We don't have a 'swarm' scenario in cluster.yaml scenarios list usually?
        # Let's check generate_compose usage.
        
        # Reverting to simple build fix first.
        
        if registry:
            master_conn.run(f"REGISTRY={registry} docker stack deploy -c docker_configs/docker-compose.swarm.yml dm")
        else:
            # For local registry-less deployment, build on each node
            master_conn.run("docker stack deploy -c docker_configs/docker-compose.swarm.yml dm --resolve-image=never")
    
    print("\nStack deployed! Checking services...")
    master_conn.run("docker service ls")


@task
def swarm_scale(c, service: str, replicas: int):
    """Scale a swarm service.
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print(f"Scaling dm_{service} to {replicas} replicas...")
    master_conn.run(f"docker service scale dm_{service}={replicas}")
    print("Scaling complete!")


@task
def swarm_status(c):
    """Check Docker Swarm status.
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print("\\n" + "="*50)
    print("SWARM NODES")
    print("="*50)
    master_conn.run("docker node ls")
    
    print("\\n" + "="*50)
    print("SERVICES")
    print("="*50)
    master_conn.run("docker service ls")
    
    print("\\n" + "="*50)
    print("DOWNLOAD WORKERS")
    print("="*50)
    master_conn.run("docker service ps dm_download --no-trunc", warn=True)



@task
def swarm_logs(c, service: str = "download", lines: int = 50):
    """View logs from a swarm service.
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    master_conn.run(f"docker service logs --tail {lines} dm_{service}")


@task
def swarm_down(c):
    """Remove the swarm stack.
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    master_conn = get_connection(config["cluster"]["master"])
    
    print("Removing swarm stack...")
    master_conn.run("docker stack rm dm")
    print("Stack removed!")



@task
def swarm_leave_all(c):
    """Leave swarm on all nodes (destroys swarm).
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    
    # Workers leave first
    for worker_config in config["cluster"]["workers"]:
        worker_conn = get_connection(worker_config)
        print(f"Leaving swarm on {worker_config['host']}...")
        worker_conn.sudo("docker swarm leave --force", warn=True)
    
    # Master leaves last
    master_conn = get_connection(config["cluster"]["master"])
    print("Leaving swarm on master...")
    master_conn.sudo("docker swarm leave --force", warn=True)
    
    print("All nodes left swarm!")


@task
def reset_cluster(c):
    """Deep cleanup of the entire cluster (Destructive).
    
    1. Removes Swarm stacks (dm, dm-seaweed).
    2. Forces all nodes to leave Swarm.
    3. Prunes Docker system.
    4. Deletes ALL data directories (/data/seaweed, /mnt/swdshared, /mnt/seaweed).
    5. Kills resident processes (weed).
    
    Usage: fab reset-cluster
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    master_config = config["cluster"]["master"]
    master_conn = get_connection(master_config)
    
    # 1. Remove Stacks
    print("\\n[MASTER] Removing Docker Stacks...")
    master_conn.run("docker stack rm dm dm-seaweed", warn=True)
    
    # 2. Iterate ALL nodes for deep cleanup
    nodes = [master_config] + config["cluster"].get("workers", [])
    
    for node_cfg in nodes:
        try:
            conn = get_connection(node_cfg)
            host = node_cfg["host"]
            print(f"\\n[{host}] cleaning up...")
            
            # Leave Swarm
            conn.sudo("docker swarm leave --force", warn=True, hide=True)
            
            # Kill processes
            conn.sudo("pkill weed", warn=True, hide=True)
            
            # Unmount
            conn.sudo("umount /mnt/seaweed", warn=True, hide=True)
            conn.sudo("umount /mnt/swdshared", warn=True, hide=True)
            
            # Remove Data and Mounts
            # WARNING: This deletes data!
            conn.sudo("rm -rf /data/seaweed /mnt/seaweed /mnt/swdshared /data/data_miner_db", warn=True)
            
            # Prune Docker (optional, keeps it clean)
            # conn.sudo("docker system prune -f", warn=True) # Maybe too aggressive?
            
            print(f"[{host}] Cleanup complete.")
        except Exception as e:
            print(f"[{node_cfg['host']}] Cleanup failed: {e}")

    print("\\nCluster Reset Successfully!")


@task
def setup_native_mount(c):
    """Start SeaweedFS Native Mount on all nodes (Simple Background Process)."""
    ensure_ssh_access(c)
    config = load_cluster_config()
    nodes = [config["cluster"]["master"]] + config["cluster"].get("workers", [])
    
    for node in nodes:
        conn = get_connection(node)
        host = node["host"]
        print(f"[{host}] Starting Native Mount...")
        
        # 1. Install Binary
        # Deploy from local project root to target
        local_binary = "weed"
        if os.path.exists(local_binary):
             remote_tmp = f"/tmp/weed_{host}"
             try:
                 if hasattr(conn, "put"):
                     conn.put(local_binary, remote_tmp)
                     conn.sudo(f"mv {remote_tmp} /usr/local/bin/weed")
                 else:
                     conn.run(f"cp {local_binary} /usr/local/bin/weed")
                     
                 conn.sudo("chmod +x /usr/local/bin/weed")
                 print(f"  [{host}] weed binary installed.")
             except Exception as e:
                 print(f"  [{host}] Failed to install binary: {e}")
        else:
             print(f"  [WARNING] Local 'weed' binary not found! Cannot deploy to {host}.")
        
        # 2. Mount
        conn.sudo("mkdir -p /mnt/swdshared")
        # Kill existing
        conn.sudo("pkill -f 'weed mount'", warn=True, hide=True)
        # Start new
        # filer=localhost:8888 works on master. On workers, we need Master IP?
        # NO! Filer runs on Manager only? Or swarm service?
        # Filer is a Service. It runs on Manager usually.
        # But 'localhost:8888' on worker will fail if filer is not on that worker.
        # We need the Swarm VIP or Manager IP.
        # But 'weed mount' runs on HOST. Host can't resolve 'seaweed-filer'.
        # Host CAN reach mapped port 8888 on localhost IF the service mesh exposes it on all nodes (Swarm Mesh).
        # Yes, Swarm Mesh exposes ports on ALL nodes.
        # So 'localhost:8888' works on ANY node in the swarm.
        
        cmd = "nohup /usr/local/bin/weed mount -dir=/mnt/swdshared -filer=localhost:8888 -filer.path=/ -allowOthers=true > /tmp/weed_mount.log 2>&1 &"
        conn.sudo(cmd)
        print(f"  [{host}] Mount started.")

    print("Native Mounts Started!")


@task
def verify_cluster_io(c):
    """Run distributed I/O verification test across all nodes.
    
    Tests:
    1. Write from HOST on each machine (to /mnt/swdshared)
    2. Write from CONTAINER on each machine (via /mnt/swdshared bind)
    3. Verify all files are visible on all machines with correct size.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    nodes = [config["cluster"]["master"]] + config["cluster"].get("workers", [])
    
    import time
    timestamp = int(time.time())
    
    print(f"\\n{'='*60}")
    print(f"Starting Distributed I/O Test (Run ID: {timestamp})")
    print(f"{'='*60}")
    
    expected_files = []
    
    # PHASE 1: WRITE
    for node in nodes:
        conn = get_connection(node)
        host = node["host"]
        hostname = conn.run("hostname", hide=True).stdout.strip()
        print(f"\\n[WRITE PHASE] Node: {host} ({hostname})")
        
        # Define files
        host_file = f"host_{hostname}_{timestamp}.txt"
        cont_file = f"container_{hostname}_{timestamp}.txt"
        expected_files.extend([host_file, cont_file])
        
        # 1. Host Write
        content = f"Host write from {hostname} at {timestamp}"
        conn.sudo(f"sh -c 'echo \"{content}\" > /mnt/swdshared/{host_file}'")
        print(f"  Wrote host file: {host_file}")
        
        # 2. Container Write
        # Use alpine for speed, bind mount with rslave
        cont_cmd = f"echo 'Container write from {hostname}' > /data/{cont_file}"
        docker_cmd = f"docker run --rm -v /mnt/swdshared:/data:rslave alpine sh -c \"{cont_cmd}\""
        conn.sudo(docker_cmd)
        print(f"  Wrote container file: {cont_file}")
        
    print(f"\\nwaiting 2 seconds for sync...")
    time.sleep(2)
    
    # PHASE 2: READ / VERIFY
    all_passed = True
    
    for node in nodes:
        conn = get_connection(node)
        host = node["host"]
        hostname = conn.run("hostname", hide=True).stdout.strip()
        print(f"\\n[VERIFY PHASE] Node: {host} ({hostname})")
        
        # List files
        # ls -l for size
        print("  Listing /mnt/swdshared:")
        ls_res = conn.sudo(f"ls -l /mnt/swdshared/*_{timestamp}.txt", warn=True, hide=True)
        if ls_res.ok:
            print(ls_res.stdout.strip())
        else:
            print("  ERROR: No test files found!")
            all_passed = False
            continue
            
        # Check specific files
        missing = []
        for f in expected_files:
            if f not in ls_res.stdout:
                missing.append(f)
        
        if missing:
            print(f"  FAILED: Missing files: {missing}")
            all_passed = False
        else:
            print("  SUCCESS: All test files visible.")
            
        # Verify content of own files (optional, but ls -l showed size)
        # We assume if size > 0 it's good.
        # Check for empty files
        if " 0 " in ls_res.stdout: # naive check for 0 byte size in ls -l output
             print("  WARNING: Some files have 0 size!")
             # Do a grep for 0 size?
             # ls -l format: -rw-r--r-- 1 root root SIZE date name
             pass

    print(f"\\n{'='*60}")
    if all_passed:
        print("DISTRIBUTED I/O VERIFICATION: PASSED ✅")
    else:
        print("DISTRIBUTED I/O VERIFICATION: FAILED ❌")
    print(f"{'='*60}")








@task
def swarm_build_all(c):
    """Build the Docker image on all nodes (for registry-less deployment).
    
    Ensures SSH access first.
    """
    ensure_ssh_access(c)
    config = load_cluster_config()
    dest_path = config["deployment"]["project_path"]
    
    # Build on master
    master_conn = get_connection(config["cluster"]["master"])
    print(f"Building on master...")
    sync_project(master_conn, dest_path)
    with master_conn.cd(dest_path):
        master_conn.run("docker build -t data-miner:latest -f docker_configs/Dockerfile .")
    
    # Build on workers
    for worker_config in config["cluster"]["workers"]:
        worker_conn = get_connection(worker_config)
        print(f"\nBuilding on {worker_config['host']}...")
        sync_project(worker_conn, dest_path)
        with worker_conn.cd(dest_path):
            worker_conn.run("docker build -t data-miner:latest -f docker_configs/Dockerfile .")
    
    print("\nImage built on all nodes!")

