# K3s Data Miner Deployment — Status & Handoff

> **Last updated:** 2026-02-13
> **Status:** DEPLOYED & RUNNING

---

## Cluster Overview

| Node | IP | Role | GPU | Status |
|---|---|---|---|---|
| pavanjci | 10.96.122.9 | Master (control-plane) | GPU=1 | Ready |
| manthana | 10.96.122.14 | Worker | none | Ready |
| arsenal | 10.96.122.132 | Worker | none | Ready |

- **K3s version:** v1.31.0+k3s1
- **SSH user:** `pavanmv` (password in `.env` as `MASTER_PASSWORD`)
- **Inventory:** `k3s_setup/inventory.py`

---

## What Has Been Completed ✅

### 1. Prerequisites (Pyinfra)
- Master node labeled: `node-role.kubernetes.io/master=true`, `gpu=true`
- NVIDIA device plugin installed
- Storage directories created: `/data/data_miner_db/{postgres,loki,grafana}`
- **Script:** `k3s_setup/data_miner_prep.py`

### 2. Docker Image Build & Import
- `Dockerfile` uses `python:3.12-slim` + `pip install uv` + `uv pip install --system`
- Image `data-miner:latest` built and imported to all 3 nodes via `k3s ctr images import --namespace k8s.io`
- **Script:** `k3s_setup/docker_build.py` (supports `FORCE=true` env var for rebuild)
- **Tarball cache:** `/tmp/data-miner.tar` (delete to force rebuild)

### 3. All K3s Manifests Created
```
k3s_setup/manifests/
├── namespace.yaml                          # data-miner namespace
├── config/
│   └── configmap.yaml                      # Application config (config.yaml)
├── infrastructure/
│   ├── postgres-statefulset.yaml           # PostgreSQL (master node, hostPath)
│   ├── postgres-service.yaml               # Headless service
│   ├── loki-statefulset.yaml               # Loki logging (master node)
│   ├── loki-service.yaml
│   ├── grafana-deployment.yaml             # Grafana (master node)
│   └── grafana-service.yaml                # NodePort :30300
└── workers/
    ├── download-statefulset.yaml           # 3 replicas (1 per node), spread across nodes
    ├── extract-deployment.yaml             # 2 replicas on master
    ├── filter-deployment.yaml              # 1 replica, GPU, master
    ├── dedup-deployment.yaml               # 1 replica, GPU, master
    ├── detect-deployment.yaml              # 1 replica, GPU, master
    └── monitor-deployment.yaml             # 1 replica
```

### 4. Deployment Script
- **Script:** `k3s_setup/deploy_manifests.sh`
- Applies manifests in order: namespace → config → infrastructure → workers

### 5. Manifest Fixes (from deployment session)
- Changed `imagePullPolicy` from `IfNotPresent` → `Never` in all 6 worker manifests (local images, not from a registry)
- Reduced download-worker replicas from 9 → 3 (1 per node via `topologySpreadConstraints`)
- Removed `nvidia.com/gpu: 1` resource limits from filter/dedup/detect manifests to enable GPU time-sharing (nvidia is the default runtime, all containers get GPU access)
- Added proper `requests`/`limits` for memory and CPU to GPU workers instead

### 6. NVIDIA GPU Configuration
- Created `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl` with:
  - `nvidia` set as `default_runtime_name`
  - Proper CNI `bin_dir`/`conf_dir` paths (missing these caused pavanjci to go `NotReady`)
  - `BinaryName = /usr/bin/nvidia-container-runtime`
- Generated CDI spec at `/etc/cdi/nvidia.yaml`
- After restart: **pavanjci reports GPU=1**

### 7. Code Bug Fix
- Fixed `NameError: DetectionResult is not defined` in `data_miner/models/detector_models.py`
- Root cause: Circular import via `models/__init__.py` → type annotations evaluated at class definition time
- Fix: Added `from __future__ import annotations` to defer annotation evaluation
- Rebuilt Docker image and imported to all 3 nodes

### 8. Namespace & Secrets Created
- Namespace `data-miner` created
- HF token secret created from `.env`

### 9. All Manifests Deployed
- All infrastructure (postgres, loki, grafana) deployed and running
- All workers deployed and running

### 10. Database Initialized & Populated
- Ran `kubectl exec ... -- data-miner init-db` to create schema
- Ran `kubectl exec ... -- data-miner populate` to seed project data
- **Result:** 187 videos added to `glass_door` project

---

## Current Pod Distribution (ALL RUNNING ✅)

| Pod | Node | Status | Activity |
|---|---|---|---|
| postgres-0 | pavanjci | Running | Database active |
| loki-0 | pavanjci | Running | Logs ingestion active |
| grafana | pavanjci | Running | Dashboard available at :30300 |
| monitor-worker | pavanjci | Running | Monitoring project stage |
| dedup-worker | pavanjci | Running | Waiting for filtered frames |
| detect-worker | pavanjci | Running | Waiting for filtered frames |
| filter-worker | pavanjci | Running | Waiting for extracted frames |
| extract-worker (x2) | pavanjci | Running | Ready to extract |
| download-worker-0 | arsenal | Running | **Downloading videos** |
| download-worker-1 | manthana | Running | **Downloading videos** |
| download-worker-2 | pavanjci | Running | **Downloading videos** |

---

## What Remains To Do ❌

> **All core deployment steps are complete.** Below are optional/operational items.

### Optional: Service Verification
- Grafana: `http://pavanjci:30300`
- Postgres logs: `kubectl logs postgres-0 -n data-miner`
- Worker logs: `kubectl logs -n data-miner -l app=download-worker --tail=20`

### Optional: Functional Test
```bash
kubectl exec -it -n data-miner deploy/monitor-worker -- python -m data_miner.cli status
```

### Optional: Scaling
```bash
# Scale download workers (e.g., more per node)
kubectl -n data-miner scale statefulset download-worker --replicas=6

# Scale extract workers
kubectl -n data-miner scale deployment extract-worker --replicas=4
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `DATABASE_URL` & `LOKI_URL` hardcoded in manifests | Internal cluster DNS, not sensitive |
| `HF_TOKEN` via K8s Secret | Sensitive credential, 12-factor best practice |
| `python:3.12-slim` (not nvidia/cuda) | PyTorch bundles CUDA runtime; NVIDIA Container Toolkit on host handles GPU |
| `uv pip install --system` in Dockerfile | Installs to system Python, no venv needed in container |
| `ENTRYPOINT ["python", "-m"]` | Workers run as `python -m data_miner.workers.<name>` |
| SeaweedFS mounted via `hostPath` | `/swdfs_mnt/swshared` assumed pre-mounted on all nodes |
| `imagePullPolicy: Never` | Local images only, not from a registry |
| No `nvidia.com/gpu` resource limits | GPU time-sharing via nvidia as default runtime; all GPU-node pods get access |

---

## Deployment Issues Encountered & Resolved

| Issue | Root Cause | Resolution |
|---|---|---|
| pavanjci went `NotReady` after containerd config | Missing CNI `bin_dir`/`conf_dir` in `config.toml.tmpl` | Added proper CNI paths alongside nvidia runtime config |
| `NameError: DetectionResult is not defined` | Circular import in `models/__init__.py` causing type annotations to be evaluated at class definition | Added `from __future__ import annotations` to `detector_models.py` |
| GPU pods stuck `Pending` with `nvidia.com/gpu` limits | Only 1 physical GPU, 3 pods each requesting 1 GPU = impossible | Removed `nvidia.com/gpu` limits; using nvidia as default runtime for time-sharing |
| `ImagePullBackOff` on workers | `imagePullPolicy: IfNotPresent` tried to pull from registry | Changed to `imagePullPolicy: Never` for all workers |
| 9 download workers couldn't schedule evenly | 9 replicas with `maxSkew: 1` + `DoNotSchedule` across 3 nodes | Reduced to 3 replicas (1 per node) |

---

## File Reference

| File | Purpose |
|---|---|
| `k3s_setup/inventory.py` | Pyinfra inventory (all 3 nodes, SSH config, K3s token) |
| `k3s_setup/data_miner_prep.py` | Node preparation (labels, NVIDIA plugin, storage dirs) |
| `k3s_setup/docker_build.py` | Build, save, copy, import Docker image |
| `k3s_setup/Dockerfile` | Container image definition |
| `k3s_setup/deploy_manifests.sh` | Apply all manifests in order |
| `k3s_setup/seaweedfs_prep.py` | SeaweedFS setup (separate concern) |
| `.env` | Contains `HF_TOKEN`, `MASTER_PASSWORD`, `WORKER_PASSWORD` |
| `k3s_setup/DATA_MINER_K8S_MIGRATION_PLAN.md` | Original migration plan |

---

## Known Issues / Notes

- **Sudo password:** Pyinfra prompts for sudo on remote nodes. Password is in `.env` as `MASTER_PASSWORD`.
- **Docker save is slow:** The image is ~4.8GB. `docker save` takes several minutes.
- **Lock file:** `docker_build.py` uses `/tmp/data_miner_build.lock`. Delete it if builds get stuck.
- **Image namespace:** Must import with `--namespace k8s.io` for K3s to see the image.
- **containerd config:** Located at `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl` — must include both nvidia runtime AND correct CNI paths.
- **CDI spec:** Generated at `/etc/cdi/nvidia.yaml` on pavanjci.
