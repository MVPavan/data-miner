#!/usr/bin/env python3
"""
Docker Compose Generator

Generates Docker Compose files from cluster.yaml configuration using a
data-driven service builder pattern.

Modes:
  - standalone: Single machine (all services on one host)
  - swarm: Distributed cluster using Docker Swarm

Usage:
    python generate_compose.py                    # Generate based on cluster.yaml
    python generate_compose.py --scenario swarm   # Force swarm generation
    python generate_compose.py --dry-run          # Print to stdout
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Service Definitions - Single Source of Truth
# =============================================================================
# Each service is defined ONCE. Variations for swarm/standalone are handled
# via the 'swarm_overrides' and 'variants' keys.

db_services = {
    "postgres": {
        "image": "postgres:16",
        "container_name": "dm_postgres",
        "restart": "unless-stopped",
        "environment": {
            "POSTGRES_USER": "{db_user}",
            "POSTGRES_PASSWORD": "{db_password}",
            "POSTGRES_DB": "{db_name}",
        },
        "ports": ["{db_port}:5432"],
        "volumes": ["{persistent_data_dir}/postgres:/var/lib/postgresql/data"],
        "healthcheck": {
            "test": ["CMD-SHELL", "pg_isready -U {db_user} -d {db_name}"],
            "interval": "10s",
            "timeout": "5s",
            "retries": 5,
        },
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "ports"],
            "networks": ["dm-network"],
            "deploy": {
                "mode": "replicated",
                "replicas": 1,
                "placement": {"constraints": ["node.role == manager"]},
                "restart_policy": {"condition": "on-failure"},
            },
        },
    },

    "grafana": {
        "image": "grafana/grafana:10.0.0",
        "container_name": "dm_grafana",
        "restart": "unless-stopped",
        "ports": ["3000:3000"],
        "volumes": ["{persistent_data_dir}/grafana:/var/lib/grafana"],
        "environment": {
            "GF_AUTH_ANONYMOUS_ENABLED": "true",
            "GF_AUTH_ANONYMOUS_ORG_ROLE": "Admin",
        },
        "swarm_overrides": {
            "_remove": ["container_name", "restart"],
            "networks": ["dm-network"],
            "deploy": {
                "mode": "replicated",
                "replicas": 1,
                "placement": {"constraints": ["node.role == manager"]},
            },
        },
    },

    "adminer": {
        "image": "adminer",
        "container_name": "dm_adminer",
        "restart": "unless-stopped",
        "ports": ["8880:8080"],
        "depends_on": ["postgres"],
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "depends_on"],
            "networks": ["dm-network"],
            "deploy": {
                "mode": "replicated",
                "replicas": 1,
                "placement": {"constraints": ["node.role == manager"]},
            },
        },
    },

    "loki": {
        "image": "grafana/loki:2.9.0",
        "container_name": "dm_loki",
        "command": "-config.file=/etc/loki/local-config.yaml",
        "ports": ["3100:3100"],
        "volumes": ["{persistent_data_dir}/loki:/loki"],
        "restart": "unless-stopped",
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "ports"],
            "networks": ["dm-network"],
            "deploy": {
                "mode": "replicated",
                "replicas": 1,
                "placement": {"constraints": ["node.role == manager"]},
            },
        },
    },
}

seaweed_services = {
    "seaweed-master": {
        "image": "chrislusf/seaweedfs",
        "container_name": "dm_seaweed_master",
        "restart": "unless-stopped",
        "command": "master -ip.bind=0.0.0.0 -ip=seaweed-master -defaultReplication={replication}",
        "ports": ["9333:9333", "19333:19333"],
        "volumes": ["{data_dir}/seaweed/master:/data"],
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "volumes"],
            "command": "master -ip.bind=0.0.0.0 -ip=seaweed-master -defaultReplication={replication} -volumeSizeLimitMB=1000",
            "networks": ["dm-network"],
            "deploy": {
                "mode": "replicated",
                "replicas": 1,
                "placement": {"constraints": ["node.role == manager"]},
                "restart_policy": {"condition": "on-failure"},
            },
        },
    },
    
    "seaweed-volume": {
        "image": "chrislusf/seaweedfs",
        "container_name": "dm_seaweed_volume",
        "restart": "unless-stopped",
        "command": "volume -mserver=seaweed-master:9333 -ip=seaweed-volume -dir=/data/seaweed -max=100",
        "depends_on": ["seaweed-master"],
        "volumes": ["{seaweed_data_dir}/volume:/data/seaweed"],
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "depends_on"],
            "command": "volume -mserver=seaweed-master:9333 -dir=/data/seaweed -max=100",
            "volumes": ["{seaweed_data_dir}:/data/seaweed"],
            "networks": ["dm-network"],
            "deploy": {
                "mode": "global",
                "restart_policy": {"condition": "on-failure"},
            },
        },
    },
    
    "seaweed-filer": {
        "image": "chrislusf/seaweedfs",
        "container_name": "dm_seaweed_filer",
        "restart": "unless-stopped",
        "command": "filer -master=seaweed-master:9333 -ip.bind=0.0.0.0 -ip=seaweed-filer",
        "depends_on": ["seaweed-master", "seaweed-volume"],
        "ports": ["8888:8888", "18888:18888"],
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "depends_on"],
            "command": "filer -master=seaweed-master:9333 -ip.bind=0.0.0.0 -ip=seaweed-filer",
            "networks": ["dm-network"],
            "deploy": {
                "mode": "replicated",
                "replicas": 1,
                "placement": {"constraints": ["node.role == manager"]},
                "restart_policy": {"condition": "on-failure"},
            },
        },
    },
    
    "seaweed-mount": {
        "image": "chrislusf/seaweedfs",
        "container_name": "dm_seaweed_mount",
        "restart": "unless-stopped",
        "command": "mount -filer=seaweed-filer:8888 -dir=/mnt/swdshared -filer.path=/",
        "privileged": True,
        "devices": ["/dev/fuse"],
        "cap_add": ["SYS_ADMIN"],
        "depends_on": ["seaweed-filer"],
        "volumes": ["seaweed-data:/mnt/swdshared:shared"],
        "swarm_overrides": {
            "_remove": ["container_name", "restart", "depends_on"],
            "volumes": ["/mnt/swdshared:/mnt/swdshared"],
            "networks": ["dm-network"],
            "deploy": {
                "mode": "global",
                "restart_policy": {"condition": "on-failure"},
            },
        },
    },

}


data_miner_services = {
# Data miner base - variants handle different deployment roles
    "data-miner": {
        "build": {
            "context": "..",
            "dockerfile": "docker_configs/Dockerfile",
        },
        "restart": "unless-stopped",
        "environment": {
            "DATA_MINER_CONFIG": "/app/config.yaml",
            "DATABASE_URL": "postgresql://{db_user}:{db_password}@postgres:{db_port}/{db_name}",
        },
        "volumes": [
            "./{config_file}:/app/config.yaml:ro",
            "seaweed-data:/mnt/swdshared:shared",
        ],
        "depends_on": {
            "postgres": {"condition": "service_healthy"},
            "seaweed-mount": {"condition": "service_started"},
        },
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                }
            }
        },
        # Variants for different roles
        "variants": {
            "standalone": {
                "container_name": "dm_workers",
                "_remove": ["depends_on"],
                "depends_on": {
                    "postgres": {"condition": "service_healthy"},
                },
                "volumes": [
                    "./{config_file}:/app/config.yaml:ro",
                    "{persistent_data_dir}/output:/mnt/swdshared",
                ],
            },
            "download": {
                "_swarm": True,
                "_service_name": "download",
                "_remove": ["build", "restart", "depends_on", "deploy"],
                "image": "${REGISTRY:-localhost:5000}/data-miner:${TAG:-latest}",
                "configs": [{"source": "worker-config", "target": "/app/config.yaml"}],
                "volumes": ["/mnt/swdshared:/mnt/swdshared:rslave"],
                "networks": ["dm-network"],
                "deploy": {
                    "mode": "replicated",
                    "replicas": "{download_replicas}",
                    "placement": {"constraints": ["node.role == worker"]},
                    "restart_policy": {"condition": "on-failure"},
                    "resources": {"limits": {"memory": "2G"}},
                },
            },
            "master-workers": {
                "_swarm": True,
                "_service_name": "master-workers",
                "_remove": ["build", "restart", "depends_on"],
                "image": "${REGISTRY:-localhost:5000}/data-miner:${TAG:-latest}",
                "configs": [{"source": "master-config", "target": "/app/config.yaml"}],
                "volumes": ["/mnt/swdshared:/mnt/swdshared:rslave"],
                "networks": ["dm-network"],
                "deploy": {
                    "mode": "replicated",
                    "replicas": 1,
                    "placement": {"constraints": ["node.role == manager"]},
                    "restart_policy": {"condition": "on-failure"},
                    "resources": {
                        "reservations": {
                            "generic_resources": [{"discrete_resource_spec": {"kind": "gpu", "value": 1}}]
                        }
                    },
                },
            },
        },
    },

}

SERVICES = {
    **db_services,
    **seaweed_services,
    **data_miner_services,
}


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIOS = {
    "standalone": {
        "services": [
            "postgres",
            ("data-miner", "standalone"),
            "grafana",
            "loki",
            "adminer",
        ],
        "volumes": [],
    },
    "swarm": {
        "services": [
            "postgres",
            "seaweed-master",
            "seaweed-filer",
            "seaweed-volume",
            "seaweed-mount",
            ("data-miner", "download"),
            ("data-miner", "master-workers"),
            "grafana",
            "loki",
            "adminer",
        ],
        "volumes": ["shared-data"],
        "networks": {
            "dm-network": {"driver": "overlay", "attachable": True}
        },
        "configs": {
            "master-config": {"file": "./master.yaml"},
            "worker-config": {"file": "./worker.yaml"},
        },
    },
    "seaweed": {
        "services": [
            "seaweed-master",
            "seaweed-filer",
            "seaweed-volume",
            "seaweed-mount",
        ],
        "volumes": ["seaweed-data", "shared-data"],
        "networks": {
            "dm-network": {"driver": "overlay", "attachable": True}
        },
    },
}


# =============================================================================
# Service Renderer
# =============================================================================

def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def format_value(value: Any, params: dict) -> Any:
    """Recursively format string placeholders in a value."""
    if isinstance(value, str):
        try:
            return value.format(**params)
        except KeyError:
            return value  # Leave unformatted if key not found
    elif isinstance(value, dict):
        return {k: format_value(v, params) for k, v in value.items()}
    elif isinstance(value, list):
        return [format_value(v, params) for v in value]
    return value


def render_service(name: str, config: dict, mode: str = "compose", variant: str = None) -> tuple[str, dict]:
    """
    Render a service definition.
    
    Args:
        name: Service name from SERVICES dict
        config: Config dict with parameters
        mode: "compose" or "swarm"
        variant: Optional variant name for services with variants
        
    Returns:
        Tuple of (service_name, service_definition)
    """
    if name not in SERVICES:
        raise ValueError(f"Unknown service: {name}")
    
    service = copy.deepcopy(SERVICES[name])
    service_name = name
    
    # Apply variant if specified
    if variant and "variants" in service:
        variants = service.pop("variants")
        if variant in variants:
            variant_def = variants[variant]
            
            # Get custom service name if specified
            if "_service_name" in variant_def:
                service_name = variant_def.pop("_service_name")
            
            # Remove fields marked for removal
            if "_remove" in variant_def:
                for field in variant_def.pop("_remove"):
                    service.pop(field, None)
            
            # Check if this is a swarm-only variant
            variant_def.pop("_swarm", None)
            
            # Merge variant into service
            service = deep_merge(service, variant_def)
    else:
        # Remove variants key if not using a variant
        service.pop("variants", None)
    
    # Apply swarm overrides if in swarm mode
    if mode == "swarm" and "swarm_overrides" in service:
        overrides = service.pop("swarm_overrides")
        
        # Remove fields marked for removal
        if "_remove" in overrides:
            for field in overrides.pop("_remove"):
                service.pop(field, None)
        
        # Merge overrides
        service = deep_merge(service, overrides)
    else:
        service.pop("swarm_overrides", None)
    
    # Format placeholders with config values
    params = {
        "db_user": config.get("database", {}).get("user", "postgres"),
        "db_password": config.get("database", {}).get("password", "postgres"),
        "db_name": config.get("database", {}).get("name", "data_miner"),
        "db_port": config.get("database", {}).get("port", 5432),
        "replication": config.get("storage", {}).get("replication", "000"),
        "data_dir": ".." if mode == "compose" else "",
        "seaweed_data_dir": config.get("storage", {}).get("seaweed_data_dir", "/data/seaweed"),
        "persistent_data_dir": config.get("storage", {}).get("persistent_data_dir", "../data"),
        "config_file": config.get("scenarios", {}).get("standalone", {}).get("data_miner_config", "standalone.yaml"),
        "download_replicas": config.get("scenarios", {}).get("swarm", {}).get("download_replicas", 10),
    }
    
    service = format_value(service, params)
    
    return service_name, service


def generate_compose_file(scenario: str, config: dict) -> str:
    """Generate a complete compose file for a scenario."""
    scenario_def = SCENARIOS[scenario]
    mode = "swarm" if scenario in ("swarm", "seaweed") else "compose"
    
    # Build services dict
    services = {}
    for service_spec in scenario_def["services"]:
        if isinstance(service_spec, tuple):
            name, variant = service_spec
        else:
            name, variant = service_spec, None
        
        service_name, service_def = render_service(name, config, mode, variant)
        services[service_name] = service_def
    
    # Build compose file structure
    compose = {"services": services}
    
    # Add volumes
    if "volumes" in scenario_def:
        compose["volumes"] = {v: None for v in scenario_def["volumes"]}
    
    # Add networks (swarm only)
    if "networks" in scenario_def:
        compose["networks"] = scenario_def["networks"]
    
    # Add configs (swarm only)
    if "configs" in scenario_def:
        compose["configs"] = scenario_def["configs"]
    
    # Add version for swarm
    if mode == "swarm":
        header = '''# Docker Swarm Stack for Distributed Data Miner
# Auto-generated from cluster.yaml
#
# Deployment:
#   1. Initialize swarm on master: docker swarm init --advertise-addr MASTER_IP
#   2. Join workers: docker swarm join --token TOKEN MASTER_IP:2377
#   3. Deploy: docker stack deploy -c docker-compose.swarm.yml dm
#   4. Scale: docker service scale dm_download=N

'''
    else:
        header = '''# Standalone Single-Machine Stack
# Auto-generated from cluster.yaml
# Runs all services on one machine
#
# Usage:
#   cd docker_configs && docker compose -f docker-compose.standalone.yml up -d

'''
    
    # Generate YAML
    yaml_output = yaml.dump(compose, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return header + yaml_output


# =============================================================================
# CLI
# =============================================================================

def load_cluster_config(config_path: Path) -> dict:
    """Load cluster configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


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
        default="docker_configs/cluster.yaml",
        help="Path to cluster.yaml (default: docker_configs/cluster.yaml)"
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=["standalone", "swarm", "seaweed", "all"],
        default="all",
        help="Scenario to generate (standalone, swarm, seaweed for testing)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print to stdout instead of writing files"
    )
    
    args = parser.parse_args()
    
    # Find project root (script is at repo root)
    project_root = Path(__file__).parent
    config_path = project_root / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    config = load_cluster_config(config_path)
    
    # Detect mode if auto
    mode = detect_deployment_mode(config)
    
    # Output to docker_configs/
    output_dir = project_root / "docker_configs"
    
    print(f"Detected deployment mode: {mode}")
    print(f"Config file: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Determine which files to generate
    files_to_generate = []
    
    if args.scenario == "all":
        if mode == "standalone":
            files_to_generate = [("docker-compose.standalone.yml", "standalone")]
        else:
            files_to_generate = [("docker-compose.swarm.yml", "swarm")]
    elif args.scenario == "standalone":
        files_to_generate = [("docker-compose.standalone.yml", "standalone")]
    elif args.scenario == "swarm":
        files_to_generate = [("docker-compose.swarm.yml", "swarm")]
    elif args.scenario == "seaweed":
        files_to_generate = [("docker-compose.seaweed.yml", "seaweed")]
    
    # Generate and output
    for filename, scenario in files_to_generate:
        content = generate_compose_file(scenario, config)
        
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
