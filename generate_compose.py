#!/usr/bin/env python3
"""
Docker Compose Generator

Automatically generates Docker Compose files from cluster.yaml configuration.

Usage:
    python scripts/generate_compose.py                    # Generate all compose files
    python scripts/generate_compose.py --scenario master  # Generate specific scenario
    python scripts/generate_compose.py --output ./gen/    # Output to custom directory
    python scripts/generate_compose.py --dry-run          # Print to stdout
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Service Templates
# =============================================================================

SERVICE_POSTGRES = """
  postgres:
    image: postgres:16
    container_name: dm_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: {db_user}
      POSTGRES_PASSWORD: {db_password}
      POSTGRES_DB: {db_name}
    ports:
      - "{db_port}:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U {db_user} -d {db_name}"]
      interval: 10s
      timeout: 5s
      retries: 5
"""

SERVICE_SEAWEED_MASTER = """
  seaweed-master:
    image: chrislusf/seaweedfs
    container_name: dm_seaweed_master
    restart: unless-stopped
    command: master -ip=seaweed-master -defaultReplication={replication}
    ports:
      - "9333:9333"
      - "19333:19333"
    volumes:
      - ./data/seaweed/master:/data
"""

SERVICE_SEAWEED_VOLUME_MASTER = """
  seaweed-volume:
    image: chrislusf/seaweedfs
    container_name: dm_seaweed_volume
    restart: unless-stopped
    command: volume -mserver=seaweed-master:9333 -ip=seaweed-volume -dir=/data -max=100
    depends_on:
      - seaweed-master
    volumes:
      - ./data/seaweed/volume:/data
"""

SERVICE_SEAWEED_VOLUME_WORKER = """
  seaweed-volume:
    image: chrislusf/seaweedfs
    container_name: dm_seaweed_volume
    restart: unless-stopped
    command: volume -mserver=${MASTER_IP}:9333 -dir=/data -max=100
    environment:
      - MASTER_IP=${MASTER_IP}
    volumes:
      - ${LOCAL_DISK:-./data/seaweed}:/data
"""

SERVICE_SEAWEED_FILER = """
  seaweed-filer:
    image: chrislusf/seaweedfs
    container_name: dm_seaweed_filer
    restart: unless-stopped
    command: filer -master=seaweed-master:9333 -ip=seaweed-filer
    depends_on:
      - seaweed-master
      - seaweed-volume
    ports:
      - "8888:8888"
      - "18888:18888"
"""

SERVICE_SEAWEED_MOUNT_MASTER = """
  seaweed-mount:
    image: chrislusf/seaweedfs
    container_name: dm_seaweed_mount
    restart: unless-stopped
    command: mount -filer=seaweed-filer:8888 -dir=/mnt/seaweed -filer.path=/
    privileged: true
    devices:
      - /dev/fuse
    cap_add:
      - SYS_ADMIN
    depends_on:
      - seaweed-filer
    volumes:
      - seaweed-data:/mnt/seaweed:shared
"""

SERVICE_SEAWEED_MOUNT_WORKER = """
  seaweed-mount:
    image: chrislusf/seaweedfs
    container_name: dm_seaweed_mount
    restart: unless-stopped
    command: mount -filer=${MASTER_IP}:8888 -dir=/mnt/seaweed -filer.path=/
    privileged: true
    devices:
      - /dev/fuse
    cap_add:
      - SYS_ADMIN
    environment:
      - MASTER_IP=${MASTER_IP}
    volumes:
      - seaweed-data:/mnt/seaweed:shared
"""

SERVICE_DATAMINER_MASTER = """
  data-miner:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: dm_workers
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      seaweed-mount:
        condition: service_started
    environment:
      - DATA_MINER_CONFIG=/app/config.yaml
      - DATABASE_URL=postgresql://{db_user}:{db_password}@postgres:{db_port}/{db_name}
    volumes:
      - ./swarm_configs/{config_file}:/app/config.yaml:ro
      - seaweed-data:/mnt/shared:shared
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

SERVICE_DATAMINER_STANDALONE = """
  data-miner:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: dm_workers
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      seaweed-mount:
        condition: service_started
    environment:
      - DATA_MINER_CONFIG=/app/config.yaml
      - DATABASE_URL=postgresql://{db_user}:{db_password}@postgres:{db_port}/{db_name}
    volumes:
      - ./swarm_configs/{config_file}:/app/config.yaml:ro
      - seaweed-data:/mnt/shared:shared
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

SERVICE_DATAMINER_WORKER = """
  data-miner:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: dm_download_workers
    restart: unless-stopped
    depends_on:
      - seaweed-mount
    environment:
      - DATA_MINER_CONFIG=/app/config.yaml
      - DATABASE_URL=postgresql://{db_user}:{db_password}@${{MASTER_IP}}:{db_port}/{db_name}
      - MASTER_IP=${{MASTER_IP}}
    volumes:
      - ./swarm_configs/{config_file}:/app/config.yaml:ro
      - seaweed-data:/mnt/shared:shared
"""

SERVICE_GRAFANA = """
  grafana:
    image: grafana/grafana:10.0.0
    container_name: dm_grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./data/grafana:/var/lib/grafana
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
"""

SERVICE_ADMINER = """
  adminer:
    image: adminer
    container_name: dm_adminer
    restart: unless-stopped
    ports:
      - "8880:8080"
    depends_on:
      - postgres
"""

VOLUMES_SECTION = """
volumes:
  seaweed-data:
"""


# =============================================================================
# Generator Functions
# =============================================================================

def load_cluster_config(config_path: Path) -> dict:
    """Load cluster configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_standalone(config: dict) -> str:
    """Generate docker-compose for standalone (single machine) deployment."""
    db = config.get("database", {})
    storage = config.get("storage", {})
    scenarios = config.get("scenarios", {})
    standalone = scenarios.get("standalone", {})
    
    db_user = db.get("user", "postgres")
    db_password = db.get("password", "postgres")
    db_name = db.get("name", "data_miner")
    db_port = db.get("port", 5432)
    replication = storage.get("replication", "000")
    config_file = standalone.get("data_miner_config", "standalone.yaml")
    
    lines = [
        "# Standalone Single-Machine Stack",
        "# Auto-generated from cluster.yaml",
        "# Runs all services on one machine",
        "",
        "services:",
    ]
    
    # Add all services
    lines.append(SERVICE_POSTGRES.format(
        db_user=db_user, db_password=db_password, 
        db_name=db_name, db_port=db_port
    ))
    lines.append(SERVICE_SEAWEED_MASTER.format(replication=replication))
    lines.append(SERVICE_SEAWEED_VOLUME_MASTER)
    lines.append(SERVICE_SEAWEED_FILER)
    lines.append(SERVICE_SEAWEED_MOUNT_MASTER)
    lines.append(SERVICE_DATAMINER_STANDALONE.format(
        db_user=db_user, db_password=db_password,
        db_name=db_name, db_port=db_port, config_file=config_file
    ))
    lines.append(SERVICE_GRAFANA)
    lines.append(SERVICE_ADMINER)
    lines.append(VOLUMES_SECTION)
    
    return "\n".join(lines)


def generate_master(config: dict) -> str:
    """Generate docker-compose for master node in distributed deployment."""
    db = config.get("database", {})
    storage = config.get("storage", {})
    scenarios = config.get("scenarios", {})
    master = scenarios.get("master", {})
    
    db_user = db.get("user", "postgres")
    db_password = db.get("password", "postgres")
    db_name = db.get("name", "data_miner")
    db_port = db.get("port", 5432)
    replication = storage.get("replication", "000")
    config_file = master.get("data_miner_config", "master.yaml")
    
    lines = [
        "# Master Node Stack",
        "# Auto-generated from cluster.yaml",
        "# Runs: PostgreSQL, SeaweedFS, data-miner (processing workers)",
        "",
        "services:",
    ]
    
    lines.append(SERVICE_POSTGRES.format(
        db_user=db_user, db_password=db_password,
        db_name=db_name, db_port=db_port
    ))
    lines.append(SERVICE_SEAWEED_MASTER.format(replication=replication))
    lines.append(SERVICE_SEAWEED_VOLUME_MASTER)
    lines.append(SERVICE_SEAWEED_FILER)
    lines.append(SERVICE_SEAWEED_MOUNT_MASTER)
    lines.append(SERVICE_DATAMINER_MASTER.format(
        db_user=db_user, db_password=db_password,
        db_name=db_name, db_port=db_port, config_file=config_file
    ))
    lines.append(SERVICE_GRAFANA)
    lines.append(SERVICE_ADMINER)
    lines.append(VOLUMES_SECTION)
    
    return "\n".join(lines)


def generate_worker(config: dict) -> str:
    """Generate docker-compose for worker nodes in distributed deployment."""
    db = config.get("database", {})
    scenarios = config.get("scenarios", {})
    worker = scenarios.get("worker", {})
    
    db_user = db.get("user", "postgres")
    db_password = db.get("password", "postgres")
    db_name = db.get("name", "data_miner")
    db_port = db.get("port", 5432)
    config_file = worker.get("data_miner_config", "worker.yaml")
    
    lines = [
        "# Worker Node Stack",
        "# Auto-generated from cluster.yaml",
        "# Runs: SeaweedFS volume, data-miner (download workers only)",
        "#",
        "# Usage:",
        "#   MASTER_IP=192.168.1.100 LOCAL_DISK=/data/seaweed docker compose -f docker-compose.worker.yml up -d",
        "",
        "services:",
    ]
    
    lines.append(SERVICE_SEAWEED_VOLUME_WORKER)
    lines.append(SERVICE_SEAWEED_MOUNT_WORKER)
    lines.append(SERVICE_DATAMINER_WORKER.format(
        db_user=db_user, db_password=db_password,
        db_name=db_name, db_port=db_port, config_file=config_file
    ))
    lines.append(VOLUMES_SECTION)
    
    return "\n".join(lines)


def detect_deployment_mode(config: dict) -> str:
    """Detect deployment mode based on cluster configuration."""
    cluster = config.get("cluster", {})
    workers = cluster.get("workers", [])
    
    if not workers:
        return "standalone"
    return "distributed"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Docker Compose files from cluster.yaml"
    )
    parser.add_argument(
        "--config", "-c",
        default="swarm_configs/cluster.yaml",
        help="Path to cluster.yaml (default: swarm_configs/cluster.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        default="generated",
        help="Output directory (default: generated/)"
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=["standalone", "master", "worker", "all"],
        default="all",
        help="Scenario to generate (default: auto-detect based on config)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print to stdout instead of writing files"
    )
    parser.add_argument(
        "--in-place", "-i",
        action="store_true",
        help="Overwrite existing docker-compose files in project root"
    )
    
    args = parser.parse_args()
    
    # Find project root (script is now at repo root)
    project_root = Path(__file__).parent
    config_path = project_root / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    config = load_cluster_config(config_path)
    
    # Detect mode if auto
    mode = detect_deployment_mode(config)
    
    print(f"Detected deployment mode: {mode}")
    print(f"Config file: {config_path}")
    
    # Determine output directory
    if args.in_place:
        output_dir = project_root
    else:
        output_dir = project_root / args.output
        output_dir.mkdir(exist_ok=True)
    
    # Generate files based on mode and scenario
    files_to_generate = []
    
    if args.scenario == "all":
        if mode == "standalone":
            files_to_generate = [("docker-compose.standalone.yml", generate_standalone)]
        else:
            files_to_generate = [
                ("docker-compose.master.yml", generate_master),
                ("docker-compose.worker.yml", generate_worker),
            ]
    elif args.scenario == "standalone":
        files_to_generate = [("docker-compose.standalone.yml", generate_standalone)]
    elif args.scenario == "master":
        files_to_generate = [("docker-compose.master.yml", generate_master)]
    elif args.scenario == "worker":
        files_to_generate = [("docker-compose.worker.yml", generate_worker)]
    
    # Generate and output
    for filename, generator in files_to_generate:
        content = generator(config)
        
        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"# {filename}")
            print('='*60)
            print(content)
        else:
            output_path = output_dir / filename
            with open(output_path, "w") as f:
                f.write(content)
            print(f"Generated: {output_path}")
    
    if not args.dry_run:
        print(f"\nFiles written to: {output_dir}")


if __name__ == "__main__":
    main()
