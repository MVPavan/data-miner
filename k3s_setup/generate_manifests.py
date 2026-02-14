"""
Generate K8s manifests from cluster_config.yaml + run config.

Usage:
    python k3s_setup/generate_manifests.py --run-config run_configs/glass_door.yaml
    python k3s_setup/generate_manifests.py  # uses default run_configs/glass_door.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# Ensure k3s_setup is importable
sys.path.insert(0, str(Path(__file__).parent))
from cluster import cfg, resolve_schedule, master_hostname

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


# =============================================================================
# Worker Manifest Generation
# =============================================================================


def build_worker_manifest(name, worker_cfg):
    """Build a single worker Deployment/StatefulSet manifest."""
    kind = OmegaConf.to_container(worker_cfg).get("kind", "Deployment")
    app_name = f"{name}-worker"
    namespace = cfg().cluster.namespace
    image = f"{cfg().image.name}:{cfg().image.tag}"
    replicas = worker_cfg.replicas
    schedule = resolve_schedule(worker_cfg.schedule_on)

    # Resources
    resources = OmegaConf.to_container(worker_cfg.resources, resolve=True)

    # Base env vars
    env = [
        {"name": "DATA_MINER_CONFIG", "value": "/config/config.yaml"},
        {
            "name": "DATABASE_URL",
            "valueFrom": {
                "configMapKeyRef": {"name": "data-miner-config", "key": "DATABASE_URL"}
            },
        },
        {
            "name": "LOKI_URL",
            "valueFrom": {
                "configMapKeyRef": {"name": "data-miner-config", "key": "LOKI_URL"}
            },
        },
    ]

    # HF token (optional)
    if worker_cfg.get("hf_token", False):
        env.append({
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {"name": "hf-secret", "key": "token", "optional": True}
            },
        })

    # Extra env vars
    if worker_cfg.get("extra_env"):
        env.extend(OmegaConf.to_container(worker_cfg.extra_env, resolve=True))

    # Container spec
    container = {
        "name": app_name,
        "image": image,
        "imagePullPolicy": cfg().image.pull_policy,
        "command": ["python", "-m", f"data_miner.workers.{name}"],
        "env": env,
        "volumeMounts": [
            {"name": "config", "mountPath": "/config"},
            {"name": "seaweedfs", "mountPath": cfg().storage.seaweedfs_mount},
        ],
        "resources": resources,
    }

    # Liveness probe
    if worker_cfg.get("liveness_probe"):
        probe = worker_cfg.liveness_probe
        container["livenessProbe"] = {
            "exec": {"command": OmegaConf.to_container(probe.command)},
            "initialDelaySeconds": probe.initial_delay,
            "periodSeconds": probe.period,
        }
        if probe.get("failure_threshold"):
            container["livenessProbe"]["failureThreshold"] = probe.failure_threshold

    # Pod spec
    pod_spec = {
        "containers": [container],
        "volumes": [
            {"name": "config", "configMap": {"name": "data-miner-config"}},
            {
                "name": "seaweedfs",
                "hostPath": {
                    "path": cfg().storage.seaweedfs_mount,
                    "type": "Directory",
                },
            },
        ],
    }

    # Scheduling
    if "node_selector" in schedule:
        pod_spec["nodeSelector"] = schedule["node_selector"]
    elif schedule.get("topology_spread"):
        pod_spec["topologySpreadConstraints"] = [
            {
                "maxSkew": 1,
                "topologyKey": "kubernetes.io/hostname",
                "whenUnsatisfiable": "DoNotSchedule",
                "labelSelector": {"matchLabels": {"app": app_name}},
            }
        ]

    # Build manifest
    manifest = {
        "apiVersion": "apps/v1",
        "kind": kind,
        "metadata": {"name": app_name, "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": app_name}},
            "template": {
                "metadata": {"labels": {"app": app_name}},
                "spec": pod_spec,
            },
        },
    }

    # StatefulSet needs serviceName
    if kind == "StatefulSet":
        manifest["spec"]["serviceName"] = app_name

    return manifest


def generate_workers():
    """Generate all worker manifests."""
    workers_dir = MANIFESTS_DIR / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    for name, worker_cfg in cfg().workers.items():
        manifest = build_worker_manifest(name, worker_cfg)
        kind = manifest["kind"].lower()
        # Match existing file naming: download-statefulset.yaml, filter-deployment.yaml
        filename = f"{name}-{kind}.yaml"
        out_path = workers_dir / filename
        with open(out_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated {out_path.relative_to(MANIFESTS_DIR.parent)}")


# =============================================================================
# ConfigMap Generation
# =============================================================================


def generate_configmap(run_config_path):
    """Generate ConfigMap with merged app config + DATABASE_URL + LOKI_URL."""
    config_dir = MANIFESTS_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Three-layer merge: default.yaml + run_config + k3s_overrides
    default_config = PROJECT_ROOT / "data_miner" / "config" / "default.yaml"
    base = OmegaConf.load(default_config)
    run = OmegaConf.load(run_config_path)

    # k3s_app_overrides — resolve storage.seaweedfs_mount but keep app-level
    # ${var} interpolations unresolved (they resolve at runtime in the app)
    k3s_overrides = OmegaConf.to_container(cfg().k3s_app_overrides, resolve=True)
    k8s = OmegaConf.create(k3s_overrides)

    merged = OmegaConf.merge(base, run, k8s)

    # Also set database.url and logging.loki_url in the merged config
    database_url = OmegaConf.to_container(cfg(), resolve=True)["database_url"]
    loki_url = OmegaConf.to_container(cfg(), resolve=True)["loki_url"]
    merged.database.url = database_url
    merged.logging.loki_url = loki_url

    # Serialize to YAML string — use OmegaConf to preserve ${var} interpolations
    config_yaml = OmegaConf.to_yaml(merged, resolve=False)

    # Build ConfigMap
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "data-miner-config",
            "namespace": cfg().cluster.namespace,
        },
        "data": {
            "DATABASE_URL": database_url,
            "LOKI_URL": loki_url,
            "config.yaml": config_yaml,
        },
    }

    out_path = config_dir / "configmap.yaml"
    with open(out_path, "w") as f:
        yaml.dump(configmap, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated {out_path.relative_to(MANIFESTS_DIR.parent)}")


# =============================================================================
# Namespace Generation
# =============================================================================


def generate_namespace():
    """Generate namespace manifest."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": cfg().cluster.namespace},
    }
    out_path = MANIFESTS_DIR / "namespace.yaml"
    with open(out_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated {out_path.relative_to(MANIFESTS_DIR.parent)}")


# =============================================================================
# Infrastructure Generation (postgres, loki, grafana, adminer)
# =============================================================================


def build_infrastructure_manifest(name, infra_cfg):
    """Build infrastructure Deployment/StatefulSet + Service manifests."""
    kind = infra_cfg.get("kind", "Deployment")
    namespace = cfg().cluster.namespace
    image = infra_cfg.image
    port = infra_cfg.port
    schedule = resolve_schedule(infra_cfg.schedule_on)
    resources = OmegaConf.to_container(infra_cfg.resources, resolve=True)

    # Container spec
    container = {
        "name": name,
        "image": image,
        "ports": [{"containerPort": port}],
        "resources": resources,
    }

    # Args (optional)
    if infra_cfg.get("args"):
        container["args"] = OmegaConf.to_container(infra_cfg.args)

    # Env vars (optional)
    if infra_cfg.get("env"):
        container["env"] = OmegaConf.to_container(infra_cfg.env)

    # Volume mount (optional - for persistent data)
    volumes = []
    if infra_cfg.get("volume_mount") and infra_cfg.get("data_dir"):
        data_dir = OmegaConf.to_container(infra_cfg, resolve=True)["data_dir"]
        container["volumeMounts"] = [{"name": f"{name}-data", "mountPath": infra_cfg.volume_mount}]
        volumes.append({
            "name": f"{name}-data",
            "hostPath": {"path": data_dir, "type": "DirectoryOrCreate"},
        })

    # Liveness probe (optional)
    if infra_cfg.get("liveness_probe"):
        probe = infra_cfg.liveness_probe
        container["livenessProbe"] = {
            "exec": {"command": OmegaConf.to_container(probe.command)},
            "initialDelaySeconds": probe.initial_delay,
            "periodSeconds": probe.period,
        }

    # Readiness probe (optional)
    if infra_cfg.get("readiness_probe"):
        probe = infra_cfg.readiness_probe
        container["readinessProbe"] = {
            "exec": {"command": OmegaConf.to_container(probe.command)},
            "initialDelaySeconds": probe.initial_delay,
            "periodSeconds": probe.period,
        }

    # Pod spec
    pod_spec = {"containers": [container]}
    if volumes:
        pod_spec["volumes"] = volumes
    if "node_selector" in schedule:
        pod_spec["nodeSelector"] = schedule["node_selector"]

    # Workload manifest
    workload = {
        "apiVersion": "apps/v1",
        "kind": kind,
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": pod_spec,
            },
        },
    }
    if kind == "StatefulSet":
        workload["spec"]["serviceName"] = name

    # Service manifest
    service_type = infra_cfg.get("service_type", "ClusterIP")
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "selector": {"app": name},
            "ports": [{"port": port, "targetPort": port}],
        },
    }

    if service_type == "headless":
        service["spec"]["clusterIP"] = "None"
    elif service_type == "NodePort":
        service["spec"]["type"] = "NodePort"
        if infra_cfg.get("node_port"):
            service["spec"]["ports"][0]["nodePort"] = infra_cfg.node_port

    return workload, service


def generate_infrastructure():
    """Generate infrastructure manifests (postgres, loki, grafana, adminer)."""
    infra_dir = MANIFESTS_DIR / "infrastructure"
    infra_dir.mkdir(parents=True, exist_ok=True)

    for name, infra_cfg in cfg().infrastructure.items():
        workload, service = build_infrastructure_manifest(name, infra_cfg)

        # Write workload (deployment/statefulset)
        kind = workload["kind"].lower()
        workload_path = infra_dir / f"{name}-{kind}.yaml"
        with open(workload_path, "w") as f:
            yaml.dump(workload, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated {workload_path.relative_to(MANIFESTS_DIR.parent)}")

        # Write service
        service_path = infra_dir / f"{name}-service.yaml"
        with open(service_path, "w") as f:
            yaml.dump(service, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated {service_path.relative_to(MANIFESTS_DIR.parent)}")


# =============================================================================
# SeaweedFS Generation
# =============================================================================


def generate_seaweedfs():
    """Generate SeaweedFS manifests (namespace + master + filer + volume + mount)."""
    sw = cfg().seaweedfs
    schedule = resolve_schedule(sw.schedule_on)
    node_selector = schedule.get("node_selector", {})
    image = sw.image
    data_dir = sw.data_dir
    mount_path = cfg().storage.seaweedfs_mount
    mount_parent = str(Path(mount_path).parent)
    namespace = sw.namespace

    documents = [
        # Namespace
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": namespace},
        },
        # Master StatefulSet
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": "master", "namespace": namespace},
            "spec": {
                "serviceName": "master",
                "replicas": 1,
                "selector": {"matchLabels": {"app": "seaweedfs", "component": "master"}},
                "template": {
                    "metadata": {"labels": {"app": "seaweedfs", "component": "master"}},
                    "spec": {
                        "nodeSelector": node_selector,
                        "containers": [{
                            "name": "master",
                            "image": image,
                            "args": [
                                "master", "-ip=$(POD_IP)", "-ip.bind=0.0.0.0",
                                "-port=9333", "-mdir=/data", "-defaultReplication=000",
                            ],
                            "env": [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}],
                            "ports": [{"containerPort": 9333}, {"containerPort": 19333}],
                            "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        }],
                        "volumes": [{"name": "data", "hostPath": {"path": f"{data_dir}/master"}}],
                    },
                },
            },
        },
        # Master Service
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "master", "namespace": namespace},
            "spec": {
                "clusterIP": "None",
                "selector": {"app": "seaweedfs", "component": "master"},
                "ports": [
                    {"name": "http", "port": 9333},
                    {"name": "grpc", "port": 19333},
                ],
            },
        },
        # Filer StatefulSet
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": "filer", "namespace": namespace},
            "spec": {
                "serviceName": "filer",
                "replicas": 1,
                "selector": {"matchLabels": {"app": "seaweedfs", "component": "filer"}},
                "template": {
                    "metadata": {"labels": {"app": "seaweedfs", "component": "filer"}},
                    "spec": {
                        "nodeSelector": node_selector,
                        "containers": [{
                            "name": "filer",
                            "image": image,
                            "args": ["filer", "-master=master:9333", "-port=8888"],
                            "ports": [{"containerPort": 8888}, {"containerPort": 18888}],
                            "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        }],
                        "volumes": [{"name": "data", "hostPath": {"path": f"{data_dir}/filer"}}],
                    },
                },
            },
        },
        # Filer Service
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "filer", "namespace": namespace},
            "spec": {
                "clusterIP": "None",
                "selector": {"app": "seaweedfs", "component": "filer"},
                "ports": [
                    {"name": "http", "port": 8888},
                    {"name": "grpc", "port": 18888},
                ],
            },
        },
        # Volume DaemonSet
        {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {"name": "volume", "namespace": namespace},
            "spec": {
                "selector": {"matchLabels": {"app": "seaweedfs", "component": "volume"}},
                "template": {
                    "metadata": {"labels": {"app": "seaweedfs", "component": "volume"}},
                    "spec": {
                        "containers": [{
                            "name": "volume",
                            "image": image,
                            "args": [
                                "volume", "-mserver=master:9333", "-ip=$(POD_IP)",
                                "-port=8080", "-dir=/data", "-max=0",
                            ],
                            "env": [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}],
                            "ports": [{"containerPort": 8080}, {"containerPort": 18080}],
                            "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        }],
                        "volumes": [{"name": "data", "hostPath": {"path": f"{data_dir}/volume"}}],
                    },
                },
            },
        },
        # Mount DaemonSet
        {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {"name": "mount", "namespace": namespace},
            "spec": {
                "selector": {"matchLabels": {"app": "seaweedfs", "component": "mount"}},
                "template": {
                    "metadata": {"labels": {"app": "seaweedfs", "component": "mount"}},
                    "spec": {
                        "containers": [{
                            "name": "mount",
                            "image": image,
                            "command": ["weed"],
                            "args": [
                                "mount", "-filer=filer:8888",
                                f"-dir={mount_path}", "-allowOthers", "-umask=000",
                            ],
                            "securityContext": {"privileged": True},
                            "volumeMounts": [
                                {"name": "mnt", "mountPath": mount_parent, "mountPropagation": "Bidirectional"},
                                {"name": "fuse", "mountPath": "/dev/fuse"},
                            ],
                        }],
                        "volumes": [
                            {"name": "mnt", "hostPath": {"path": mount_parent, "type": "DirectoryOrCreate"}},
                            {"name": "fuse", "hostPath": {"path": "/dev/fuse"}},
                        ],
                    },
                },
            },
        },
    ]

    # Write one file per K8s object (matching infrastructure/ pattern)
    seaweedfs_dir = MANIFESTS_DIR / "seaweedfs"
    seaweedfs_dir.mkdir(parents=True, exist_ok=True)

    for doc in documents:
        kind = doc["kind"].lower()
        name = doc["metadata"]["name"]
        filename = f"{name}-{kind}.yaml" if kind != "namespace" else "namespace.yaml"
        out_path = seaweedfs_dir / filename
        with open(out_path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated {out_path.relative_to(MANIFESTS_DIR.parent)}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate K8s manifests from cluster config")
    parser.add_argument(
        "--run-config",
        default=str(PROJECT_ROOT / "run_configs" / "glass_door.yaml"),
        help="Path to run config YAML for ConfigMap generation",
    )
    args = parser.parse_args()

    run_config = Path(args.run_config)
    if not run_config.exists():
        print(f"Error: run config not found: {run_config}")
        sys.exit(1)

    print("Generating K8s manifests...")
    generate_namespace()
    generate_configmap(run_config)
    generate_workers()
    generate_infrastructure()
    generate_seaweedfs()
    print("Done.")


if __name__ == "__main__":
    main()
