# K3s Data Miner Deployment Guide

> **Version:** 2.0 (Multi-Image Architecture)

---

## Cluster Overview

The cluster consists of nodes with different roles:

| Role | Description | Image | SeaweedFS |
|------|-------------|-------|-----------|
| Master (GPU) | Control plane + GPU workers | `data-miner-gpu` | Mount only (`no_storage: true`) |
| Worker (Compute) | Download workers | `data-miner-base` | Mount only or contribute storage |
| Worker (Storage) | SeaweedFS volume servers | None | Contribute storage (`storage_only: true`) |

- **K3s version:** Configured in `cluster_config.yaml`
- **SSH key:** `~/.ssh/id_rsa_dm_k3s`
- **Inventory:** `k3s_setup/inventory.py`

---

## Architecture (Two-Phase Design)

| Phase | Tool | Script | What it does |
|---|---|---|---|
| 1. Provisioning | pyinfra (SSH) | `provision.py` | K3s install, NVIDIA config, storage dirs, SeaweedFS prep, Docker image build/distribute |
| 2. Orchestration | kubectl (K8s) | `orchestrate.py` | Generate manifests, deploy SeaweedFS, deploy infra/workers, init DB, verify |

```bash
# Phase 1: Provision all nodes
pyinfra k3s_setup/inventory.py k3s_setup/provision.py --data action=all

# Phase 2: Deploy to K8s
python k3s_setup/orchestrate.py setup --run-config run_configs/<your_config>.yaml
```

---

## Multi-Image Architecture

Two Docker images are built and distributed based on node configuration:

| Image | Dockerfile | Size | Contents | Nodes |
|-------|------------|------|----------|-------|
| `data-miner-base` | `Dockerfile.base` | ~500MB | Core deps (yt-dlp, av, opencv) | Compute workers |
| `data-miner-gpu` | `Dockerfile.gpu` | ~5GB | Core + ML (torch, transformers) | GPU nodes |

Nodes with `storage_only: true` receive no image (they only run SeaweedFS).

---

## Manifest Generation

All manifests are now **dynamically generated** from `cluster_config.yaml` via `generate_manifests.py`:

```
k3s_setup/manifests/
├── namespace.yaml                          # data-miner namespace
├── config/
│   └── configmap.yaml                      # 3-layer merge: default.yaml + run_config + k3s_overrides
├── infrastructure/                         # GENERATED from cluster_config.yaml
│   ├── postgres-statefulset.yaml           # PostgreSQL (master node, hostPath)
│   ├── postgres-service.yaml               # Headless service
│   ├── loki-statefulset.yaml               # Loki logging (master node)
│   ├── loki-service.yaml                   # ClusterIP
│   ├── grafana-deployment.yaml             # Grafana (master node)
│   ├── grafana-service.yaml                # NodePort :30300
│   ├── adminer-deployment.yaml             # Adminer DB admin (master node)
│   └── adminer-service.yaml                # NodePort :30080
├── workers/                                # GENERATED from cluster_config.yaml
│   ├── download-statefulset.yaml           # 3 replicas, spread across all nodes
│   ├── extract-deployment.yaml             # 2 replicas on master
│   ├── filter-deployment.yaml              # 1 replica, GPU node
│   ├── dedup-deployment.yaml               # 1 replica, GPU node
│   ├── detect-deployment.yaml              # 1 replica, GPU node
│   └── monitor-deployment.yaml             # 1 replica, master
└── seaweedfs/                              # GENERATED from cluster_config.yaml
    ├── namespace.yaml
    ├── master-statefulset.yaml
    ├── master-service.yaml
    ├── filer-statefulset.yaml
    ├── filer-service.yaml
    ├── volume-{hostname}-statefulset.yaml  # Per-node volume servers with disk limits
    └── mount-daemonset.yaml
```

### ConfigMap 3-Layer Merge
```
base = default.yaml                    # Application defaults
run = run_config.yaml                  # Project-specific (glass_door.yaml)
k8s = cluster_config.k3s_app_overrides # K8s environment (seaweedfs mount, device=auto)
merged = merge(base, run, k8s)
```

---

## Worker Distribution

Workers are scheduled based on `schedule_on` in `cluster_config.yaml`:

| Worker | Schedule | Image | Description |
|--------|----------|-------|-----------|
| download | `all` | Per-node | Spread across all compute nodes |
| extract | `master` | Per-node | On master node |
| filter | `gpu` | GPU | GPU node, requires HF token |
| dedup | `gpu` | GPU | GPU node, requires HF token |
| detect | `gpu` | GPU | GPU node, requires HF token |
| monitor | `master` | Per-node | On master node |

### SeaweedFS Pods (seaweedfs namespace)
| Pod | Scheduling | Description |
|-----|------------|-----------|
| master-0 | Master node | Metadata server |
| filer-0 | Master node | File interface |
| volume-{hostname} | Per storage node | One per node with storage |
| mount (DaemonSet) | All nodes | FUSE mount client |

### Infrastructure Pods (data-miner namespace)
| Pod | Kind | Scheduling | Storage | Port |
|-----|------|------------|---------|------|
| postgres-0 | StatefulSet | Master node | hostPath: `${storage.db_path}/postgres` | 5432 |
| loki-0 | StatefulSet | Master node | hostPath: `${storage.db_path}/loki` | 3100 |
| grafana | Deployment | Master node | None (ephemeral) | 3000 → NodePort 30300 |
| adminer | Deployment | Master node | None | 8080 → NodePort 30080 |

#### PostgreSQL Details
- **Image:** `postgres:16`
- **Database:** `data_miner`
- **Credentials:** `postgres` / `postgres`
- **Access from pods:** `postgres.data-miner.svc.cluster.local:5432`
- **DATABASE_URL:** `postgresql://postgres:postgres@postgres.data-miner.svc.cluster.local:5432/data_miner`
- **Storage:** Persistent via hostPath on master node

#### Loki Details
- **Image:** `grafana/loki:2.9.0`
- **Access from pods:** `http://loki.data-miner.svc.cluster.local:3100`
- **Storage:** Persistent via hostPath on master node
- **Python logging:** Uses `python-logging-loki` handler

---

## Service Access

| Service | URL | Purpose |
|---|---|---|
| Grafana | `http://<master>:30300` | Monitoring dashboard |
| Adminer | `http://<master>:30080` | PostgreSQL web admin |
| Loki | Internal only | Log aggregation (query via Grafana) |

### Adminer Connection Details
- System: PostgreSQL
- Server: `postgres.data-miner.svc.cluster.local`
- Username: `postgres`
- Password: `postgres`
- Database: `data_miner`

### Grafana Loki Queries
```
{application="data_miner"}                    # All logs
{application="data_miner", severity="error"}  # Errors only
{application="data_miner"} |~ "download"      # Download logs
{logger="data_miner.workers.base"}            # Worker lifecycle logs
```

---

## cluster_config.yaml Structure

```yaml
cluster:
  namespace: data-miner
  k3s_version: v1.31.0+k3s1
  ssh_key: "~/.ssh/id_rsa_dm_k3s"
  nodes:
    # GPU/Master node
    master_node:
      ip: "10.x.x.x"
      role: master
      ssh_user: user
      no_storage: true        # Mount only, no volume server
      image: gpu              # GPU image for ML workers
      labels:
        gpu: "true"
    
    # Compute worker (runs download workers)
    compute_node:
      ip: "10.x.x.x"
      role: worker
      ssh_user: user
      image: base             # Lightweight image
    
    # Storage-only worker
    storage_node:
      ip: "10.x.x.x"
      role: worker
      ssh_user: user
      storage_only: true      # No image, only SeaweedFS
      disk_limit_mb: 500000   # 500GB

storage:
  seaweedfs_mount: /swdfs_mnt/swshared
  db_path: /data/data_miner_db

images:
  base:
    name: data-miner-base
    tag: latest
    dockerfile: Dockerfile.base
  gpu:
    name: data-miner-gpu
    tag: latest
    dockerfile: Dockerfile.gpu
  pull_policy: Never

seaweedfs:
  namespace: seaweedfs
  image: chrislusf/seaweedfs:latest
  data_dir: /data/seaweed
  schedule_on: master
  default_disk_limit_mb: 0    # 0 = unlimited

services:
  postgres:
    kind: StatefulSet
    image: postgres:16
    port: 5432
    storage: hostPath         # Uses ${storage.db_path}/postgres
    env:
      POSTGRES_DB: data_miner
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
  loki:
    kind: StatefulSet
    image: grafana/loki:2.9.0
    port: 3100
    storage: hostPath         # Uses ${storage.db_path}/loki
  grafana:
    kind: Deployment
    image: grafana/grafana:10.0.0
    node_port: 30300
  adminer:
    kind: Deployment
    image: adminer:latest
    node_port: 30080

resource_presets:
  small:  { requests: {cpu: 250m, memory: 512Mi}, limits: {cpu: 1000m, memory: 2Gi} }
  medium: { requests: {cpu: 500m, memory: 1Gi}, limits: {cpu: 2000m, memory: 4Gi} }
  gpu:    { requests: {cpu: 500m, memory: 4Gi}, limits: {cpu: 2000m, memory: 16Gi} }

workers:
  download:  { kind: StatefulSet, replicas: 4, schedule_on: all, resources: ${resource_presets.medium} }
  extract:   { replicas: 3, schedule_on: master, resources: ${resource_presets.medium} }
  filter:    { replicas: 1, schedule_on: gpu, hf_token: true, resources: ${resource_presets.gpu} }
  dedup:     { replicas: 1, schedule_on: gpu, hf_token: true, resources: ${resource_presets.gpu} }
  detect:    { replicas: 1, schedule_on: gpu, hf_token: true, resources: ${resource_presets.gpu} }
  monitor:   { replicas: 1, schedule_on: master, resources: ${resource_presets.medium} }

k3s_app_overrides:
  output_dir: ${storage.seaweedfs_mount}/data_miner_output
  device: auto
```

---

## Node Configuration Options

| Option | Description | Effect |
|--------|-------------|--------|
| `image: base` | Use lightweight image | Download/extract workers only |
| `image: gpu` | Use ML image | All workers including filter/dedup/detect |
| `no_storage: true` | Don't contribute storage | Gets FUSE mount, no volume server |
| `storage_only: true` | Storage node only | No image, no workers, only SeaweedFS |
| `disk_limit_mb: N` | SeaweedFS disk limit | Limits storage contribution per node |
| `data_dir: /path` | Custom SeaweedFS data path | Per-node override of `seaweedfs.data_dir` |:
  postgres:  { kind: StatefulSet, image: postgres:16, port: 5432 }
  loki:      { kind: StatefulSet, image: grafana/loki:2.9.0, port: 3100 }
  grafana:   { kind: Deployment, image: grafana/grafana:10.0.0, node_port: 30300 }
  adminer:   { kind: Deployment, image: adminer:latest, node_port: 30080 }

k3s_app_overrides:
  output_dir: ${storage.seaweedfs_mount}/data_miner_output
  device: auto
```

---

## Provision Actions

```bash
pyinfra k3s_setup/inventory.py k3s_setup/provision.py --data action=<ACTION>
```

| Action | Runs on | What it does |
|---|---|---|
| `k3s` | all nodes | Install K3s master/agent |
| `nvidia` | master | Write containerd config, generate CDI spec, restart K3s |
| `storage` | master | Create DB dirs with correct UIDs |
| `seaweedfs` | all nodes | Install fuse3, create dirs, sysctl tuning |
| `labels` | master | kubectl label nodes + install NVIDIA device plugin |
| `docker` | all nodes | Build image, copy tar, import to K3s containerd |
| `all` | varies | Run all above in order |
| `clean` | all nodes | Full cleanup: FUSE unmount + SeaweedFS data + images + K3s |
| `clean_seaweedfs` | all nodes | Unmount FUSE, remove SeaweedFS data |
| `status` | all nodes | Show K3s service status |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| All manifests generated from `cluster_config.yaml` | Single source of truth, no drift between config and manifests |
| `DATABASE_URL` & `LOKI_URL` via ConfigMap `configMapKeyRef` | Workers reference config, not hardcoded values |
| `HF_TOKEN` via K8s Secret | Sensitive credential, injected to GPU workers only |
| `python:3.12-slim` (not nvidia/cuda) | PyTorch bundles CUDA runtime; NVIDIA Container Toolkit on host handles GPU |
| `imagePullPolicy: Never` | Local images only, distributed via tar + k3s ctr import |
| No `nvidia.com/gpu` resource limits | GPU time-sharing via nvidia as default runtime |
| SeaweedFS FUSE mount | Shared storage across all nodes at `/swdfs_mnt/swshared` |
| Infrastructure on master node only | Stateful services with hostPath persistence |
| Per-node SeaweedFS volume StatefulSets | Allows different `disk_limit_mb` per node for storage allocation |
| `storage_only` node flag | Nodes with this flag only run SeaweedFS, workers use nodeAffinity to avoid them |

---

## Issues Encountered & Resolved

| Issue | Root Cause | Resolution |
|---|---|---|
| Download workers crashing (CrashLoopBackOff) | Liveness probe checking `/tmp/healthy` but workers never create it | Removed liveness probes from download/extract workers |
| Filter worker failing with `'NoneType' object has no attribute 'replace'` | Transformers 5.x bug with SigLIP2 AutoProcessor | Use `SiglipImageProcessor` + `GemmaTokenizerFast` directly instead of `AutoProcessor` |
| `blocked_hashtag_patterns` error "Is a directory: '.'" | Empty string `''` in run_config coerced to `Path('.')` | Removed override from run_config, uses default from `default.yaml` |
| Grafana Loki datasource not showing labels | Loki datasource URL was empty | Updated via Grafana API to correct URL |
| pavanjci went `NotReady` after containerd config | Missing CNI `bin_dir`/`conf_dir` in config.toml.tmpl | Added proper CNI paths alongside nvidia runtime config |
| GPU pods stuck `Pending` with `nvidia.com/gpu` limits | Only 1 physical GPU, multiple pods requesting 1 GPU each | Removed limits; using nvidia as default runtime for time-sharing |

---

## File Reference

| File | Purpose |
|---|---|
| `k3s_setup/cluster_config.yaml` | Single source of truth — cluster, storage, workers, infrastructure |
| `k3s_setup/cluster.py` | Shared config loader (imported by all scripts) |
| `k3s_setup/provision.py` | Unified pyinfra script — all SSH/node-level operations |
| `k3s_setup/inventory.py` | Pyinfra inventory (reads from cluster_config.yaml) |
| `k3s_setup/generate_manifests.py` | Generates ALL K8s manifests (workers, infrastructure, seaweedfs, configmap) |
| `k3s_setup/orchestrate.py` | K8s orchestrator (kubectl only, no SSH) |
| `k3s_setup/Dockerfile` | Container image definition |
| `k3s_setup/setup_ssh_keys.py` | SSH key setup utility |
| `run_configs/glass_door.yaml` | Project run config (prompts, thresholds, search queries) |
| `.env` | Contains `HF_TOKEN` |

---

## CLI Commands

```bash
# Check pipeline status
kubectl exec -n data-miner deploy/monitor-worker -- python -m data_miner.cli status

# View worker logs
kubectl logs -n data-miner -l app=download-worker --tail=50
kubectl logs -n data-miner deploy/filter-worker --tail=50

# Scale workers
kubectl -n data-miner scale statefulset download-worker --replicas=6
kubectl -n data-miner scale deployment extract-worker --replicas=4

# Regenerate manifests after config change
python k3s_setup/generate_manifests.py --run-config run_configs/glass_door.yaml
kubectl apply -f k3s_setup/manifests/workers/
kubectl apply -f k3s_setup/manifests/infrastructure/

# Rebuild and redeploy image
pyinfra k3s_setup/inventory.py k3s_setup/provision.py -y --data action=docker
kubectl rollout restart deployment -n data-miner
kubectl rollout restart statefulset -n data-miner

# Full teardown (preserves data)
python k3s_setup/orchestrate.py teardown

# Full teardown (wipes all data)
python k3s_setup/orchestrate.py teardown --wipe-data
```

---

## Notes

- **Loki labels available:** `application`, `logger`, `severity`, `worker`
- **Lock file:** `provision.py` uses `/tmp/data-miner_build.lock`. Delete if builds get stuck.
- **Image namespace:** Must import with `--namespace k8s.io` for K3s to see the image.
- **containerd config:** `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl`
- **CDI spec:** `/etc/cdi/nvidia.yaml` on pavanjci
