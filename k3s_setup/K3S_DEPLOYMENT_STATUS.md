# K3s Data Miner Deployment — Status & Handoff

> **Last updated:** 2026-02-13  
> **Purpose:** Continue K3s deployment in a new chat session.

---

## Cluster Overview

| Node | IP | Role | GPU |
|---|---|---|---|
| pavanjci | 10.96.122.9 | Master (control-plane) | ✅ |
| manthana | 10.96.122.14 | Worker | ❌ |
| arsenal | 10.96.122.132 | Worker | ❌ |

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
    ├── download-statefulset.yaml           # 9 replicas, spread across nodes
    ├── extract-deployment.yaml             # 2 replicas on master
    ├── filter-deployment.yaml              # 1 replica, GPU, master
    ├── dedup-deployment.yaml               # 1 replica, GPU, master
    ├── detect-deployment.yaml              # 1 replica, GPU, master
    └── monitor-deployment.yaml             # 1 replica
```

### 4. Deployment Script
- **Script:** `k3s_setup/deploy_manifests.sh`
- Applies manifests in order: namespace → config → infrastructure → workers

---

## What Remains To Do ❌

### Step 1: Create Namespace & Secret
```bash
# Create namespace first
kubectl apply -f k3s_setup/manifests/namespace.yaml

# Create HF token secret from .env
export $(grep HF_TOKEN .env | xargs)
kubectl create secret generic hf-secret \
  --from-literal=token=$HF_TOKEN \
  -n data-miner --dry-run=client -o yaml | kubectl apply -f -
```

### Step 2: Deploy All Manifests
```bash
./k3s_setup/deploy_manifests.sh
```

### Step 3: Verify Pods
```bash
kubectl get pods -n data-miner -o wide
```
Expected: postgres-0, loki-0, grafana, 9 download-workers, 2 extract-workers, filter/dedup/detect/monitor workers all `Running`.

### Step 4: Service Verification
- Grafana: `http://pavanjci:30300`
- Postgres logs: `kubectl logs postgres-0 -n data-miner`
- Worker logs: `kubectl logs -n data-miner -l app=download-worker --tail=20`

### Step 5: Functional Test
```bash
kubectl exec -it -n data-miner deploy/monitor-worker -- python -m data_miner.cli status
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

---

## Known Issues / Notes

- **Sudo password:** Pyinfra prompts for sudo on remote nodes. Password is in `.env` as `MASTER_PASSWORD`.
- **Docker save is slow:** The image is ~4.8GB. `docker save` takes several minutes.
- **Lock file:** `docker_build.py` uses `/tmp/data_miner_build.lock`. Delete it if builds get stuck.
- **Image namespace:** Must import with `--namespace k8s.io` for K3s to see the image.
