# Data Miner K8s Migration Plan

## Overview

This document provides a comprehensive plan to migrate the Data Miner video processing pipeline to Kubernetes (K3s). It is intended to be fed to Claude Code for implementation.

---

## Current State

| Component | Status |
|-----------|--------|
| K3s Cluster | ✅ Running (3 nodes: pavanjci, manthana, arsenal) |
| SeaweedFS | ✅ Running in K3s, exposed via FUSE mount at `/swdfs_mnt/swshared` |
| PostgreSQL | ❌ Not in K3s (needs migration) |
| Loki/Grafana | ❌ Not in K3s (needs migration) |
| Data Miner | ❌ Not in K3s (needs migration) |
| NVIDIA Plugin | ❌ Not installed (needed for GPU workers) |

## Cluster Information

- **Master Node**: `pavanjci`
- **Worker Nodes**: `manthana`, `arsenal`
- **Total Nodes**: 3

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              K3s Cluster                                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Namespace: data-miner                            │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     INFRASTRUCTURE                              │  │  │
│  │  │                                                                 │  │  │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │  │  │
│  │  │  │ PostgreSQL  │  │    Loki     │  │  Grafana    │             │  │  │
│  │  │  │ StatefulSet │  │ StatefulSet │  │ Deployment  │             │  │  │
│  │  │  │ (1 replica) │  │ (1 replica) │  │ (1 replica) │             │  │  │
│  │  │  │  master     │  │   master    │  │   master    │             │  │  │
│  │  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘             │  │  │
│  │  │         │                │                                      │  │  │
│  │  │         ▼                ▼                                      │  │  │
│  │  │  ┌─────────────┐  ┌─────────────┐                              │  │  │
│  │  │  │   Service   │  │   Service   │                              │  │  │
│  │  │  │  postgres   │  │    loki     │                              │  │  │
│  │  │  │ :5432       │  │   :3100     │                              │  │  │
│  │  │  └─────────────┘  └─────────────┘                              │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                        WORKERS                                  │  │  │
│  │  │                                                                 │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │           StatefulSet: download-worker                    │  │  │  │
│  │  │  │           replicas: 9 (3 per node via TopologySpread)     │  │  │  │
│  │  │  │                                                           │  │  │  │
│  │  │  │   pavanjci        manthana         arsenal                │  │  │  │
│  │  │  │  ┌─┐ ┌─┐ ┌─┐    ┌─┐ ┌─┐ ┌─┐     ┌─┐ ┌─┐ ┌─┐             │  │  │  │
│  │  │  │  │0│ │1│ │2│    │3│ │4│ │5│     │6│ │7│ │8│             │  │  │  │
│  │  │  │  └─┘ └─┘ └─┘    └─┘ └─┘ └─┘     └─┘ └─┘ └─┘             │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────┘  │  │  │
│  │  │                                                                 │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │           Master-Only Workers (nodeSelector: pavanjci)    │  │  │  │
│  │  │  │                                                           │  │  │  │
│  │  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │  │  │
│  │  │  │  │ extract  │ │  filter  │ │  dedup   │ │  detect  │     │  │  │  │
│  │  │  │  │ Deploy   │ │  Deploy  │ │  Deploy  │ │  Deploy  │     │  │  │  │
│  │  │  │  │ (2)      │ │  (1+GPU) │ │  (1+GPU) │ │  (1+GPU) │     │  │  │  │
│  │  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │  │  │  │
│  │  │  │                                                           │  │  │  │
│  │  │  │  ┌──────────┐                                             │  │  │  │
│  │  │  │  │ monitor  │                                             │  │  │  │
│  │  │  │  │ Deploy(1)│                                             │  │  │  │
│  │  │  │  └──────────┘                                             │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      CONFIGURATION                              │  │  │
│  │  │                                                                 │  │  │
│  │  │  ┌─────────────┐  ┌─────────────┐                              │  │  │
│  │  │  │  ConfigMap  │  │   Secret    │                              │  │  │
│  │  │  │ data-miner  │  │  db-creds   │                              │  │  │
│  │  │  │   config    │  │             │                              │  │  │
│  │  │  └─────────────┘  └─────────────┘                              │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                        STORAGE                                  │  │  │
│  │  │                                                                 │  │  │
│  │  │  SeaweedFS FUSE: /swdfs_mnt/swshared (all nodes, hostPath)     │  │  │
│  │  │  Persistent Data: /data/data_miner_db (master only, hostPath)  │  │  │
│  │  │                                                                 │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Resource Summary

| Resource | Name | Replicas | Node Placement | Notes |
|----------|------|----------|----------------|-------|
| Namespace | `data-miner` | - | - | Isolates all resources |
| StatefulSet | `postgres` | 1 | master (pavanjci) | Persistent storage at `/data/data_miner_db/postgres` |
| StatefulSet | `loki` | 1 | master (pavanjci) | Persistent storage at `/data/data_miner_db/loki` |
| Deployment | `grafana` | 1 | master (pavanjci) | Persistent storage at `/data/data_miner_db/grafana` |
| StatefulSet | `download-worker` | 9 | 3 per node | TopologySpreadConstraints |
| Deployment | `extract-worker` | 2 | master (pavanjci) | nodeSelector |
| Deployment | `filter-worker` | 1 | master (pavanjci) | nodeSelector + GPU |
| Deployment | `dedup-worker` | 1 | master (pavanjci) | nodeSelector + GPU |
| Deployment | `detect-worker` | 1 | master (pavanjci) | nodeSelector + GPU |
| Deployment | `monitor-worker` | 1 | master (pavanjci) | nodeSelector |
| Service | `postgres` | - | - | ClusterIP :5432 |
| Service | `loki` | - | - | ClusterIP :3100 |
| Service | `grafana` | - | - | NodePort :3000→30300 |
| ConfigMap | `data-miner-config` | - | - | YAML configuration |

---

## Storage Layout

### Persistent Data on Master Node (hostPath)

```
/data/data_miner_db/
├── postgres/          # PostgreSQL data directory
├── loki/              # Loki chunks and index
└── grafana/           # Grafana dashboards and settings
```

**Important**: Create this directory on master node before deployment:
```bash
ssh pavanjci "sudo mkdir -p /data/data_miner_db/{postgres,loki,grafana}"
ssh pavanjci "sudo chown -R 999:999 /data/data_miner_db/postgres"   # postgres user
ssh pavanjci "sudo chown -R 10001:10001 /data/data_miner_db/loki"   # loki user
ssh pavanjci "sudo chown -R 472:472 /data/data_miner_db/grafana"    # grafana user
```

### SeaweedFS Shared Storage (all nodes)

```
/swdfs_mnt/swshared/
└── data_miner_output/
    ├── videos/                    # Downloaded videos
    │   └── {video_id}.mp4
    ├── frames_raw/                # Extracted frames
    │   └── {video_id}/
    │       └── frame_00001.jpg
    ├── logs/                      # Worker logs
    └── projects/
        └── glass_door/
            ├── frames_filtered/   # Filtered frames
            │   └── {video_id}/
            ├── frames_dedup/      # Deduplicated frames
            └── detections/        # Detection outputs
```

---

## Data Flow

```
1. POPULATE (CLI or Job)
   ┌─────────┐
   │ populate│──────▶ PostgreSQL (videos table: PENDING)
   │   job   │
   └─────────┘

2. DOWNLOAD (all nodes, 3 pods per node)
   PostgreSQL ◀──claim──▶ download-worker ──write──▶ /swdfs_mnt/.../videos/
   (PENDING → DOWNLOADING → DOWNLOADED)

3. EXTRACT (master only, 2 pods)
   /swdfs_mnt/.../videos/ ──read──▶ extract-worker ──write──▶ frames_raw/
   (DOWNLOADED → EXTRACTING → EXTRACTED)

4. FILTER (master only, 1 pod with GPU)
   frames_raw/ ──read──▶ filter-worker (SigLIP2) ──write──▶ frames_filtered/
   (project_videos: PENDING → FILTERING → FILTERED)

5. DEDUP (master only, 1 pod with GPU)
   frames_filtered/ ──read──▶ dedup-worker (DINOv3+FAISS) ──write──▶ frames_dedup/
   (projects: DEDUP_READY → DEDUPING → DETECT_READY)

6. DETECT (master only, 1 pod with GPU)
   frames_dedup/ ──read──▶ detect-worker (GroundingDINO) ──write──▶ detections/
   (projects: DETECT_READY → DETECTING → COMPLETE)
```

---

## Implementation Phases

### Phase 1: Prerequisites

**1.1 Label master node**
```bash
kubectl label node pavanjci node-role.kubernetes.io/master=true
kubectl label node pavanjci gpu=true
```

**1.2 Install NVIDIA device plugin**
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**1.3 Create persistent storage directories on master**
```bash
ssh pavanjci "sudo mkdir -p /data/data_miner_db/{postgres,loki,grafana}"
ssh pavanjci "sudo chown -R 999:999 /data/data_miner_db/postgres"
ssh pavanjci "sudo chown -R 10001:10001 /data/data_miner_db/loki"
ssh pavanjci "sudo chown -R 472:472 /data/data_miner_db/grafana"
```

**1.4 Verify SeaweedFS mount on all nodes**
```bash
for node in pavanjci manthana arsenal; do
  echo "=== $node ==="
  ssh $node "ls -la /swdfs_mnt/swshared"
done
```

### Phase 2: Docker Image

**2.1 Create Dockerfile**

Create `Dockerfile` in the data_miner project root:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY data_miner/ ./data_miner/
COPY settings/ ./settings/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 dataminer
USER dataminer

# Health check file location
ENV HEALTH_FILE=/tmp/healthy

# Default command (override in K8s)
ENTRYPOINT ["python", "-m"]
CMD ["data_miner.workers.download"]
```

**2.2 Build and import to K3s**

Option A: Build locally and import
```bash
docker build -t data-miner:latest .
docker save data-miner:latest | ssh pavanjci "sudo k3s ctr images import -"
docker save data-miner:latest | ssh manthana "sudo k3s ctr images import -"
docker save data-miner:latest | ssh arsenal "sudo k3s ctr images import -"
```

Option B: Use a registry (if available)
```bash
docker build -t your-registry/data-miner:latest .
docker push your-registry/data-miner:latest
```

### Phase 3: Kubernetes Manifests

Create the following directory structure:

```
k8s/
├── namespace.yaml
├── infrastructure/
│   ├── postgres-statefulset.yaml
│   ├── postgres-service.yaml
│   ├── loki-statefulset.yaml
│   ├── loki-service.yaml
│   ├── grafana-deployment.yaml
│   └── grafana-service.yaml
├── config/
│   └── configmap.yaml
└── workers/
    ├── download-statefulset.yaml
    ├── extract-deployment.yaml
    ├── filter-deployment.yaml
    ├── dedup-deployment.yaml
    ├── detect-deployment.yaml
    └── monitor-deployment.yaml
```

---

## Kubernetes Manifest Specifications

### namespace.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: data-miner
```

### infrastructure/postgres-statefulset.yaml

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: data-miner
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: postgres
          image: postgres:16
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_USER
              value: "postgres"
            - name: POSTGRES_PASSWORD
              value: "postgres"
            - name: POSTGRES_DB
              value: "data_miner"
            - name: PGDATA
              value: "/var/lib/postgresql/data/pgdata"
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
          livenessProbe:
            exec:
              command: ["pg_isready", "-U", "postgres", "-d", "data_miner"]
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "postgres", "-d", "data_miner"]
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
      volumes:
        - name: postgres-data
          hostPath:
            path: /data/data_miner_db/postgres
            type: DirectoryOrCreate
```

### infrastructure/postgres-service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: data-miner
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  clusterIP: None  # Headless for StatefulSet
```

### infrastructure/loki-statefulset.yaml

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: loki
  namespace: data-miner
spec:
  serviceName: loki
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: loki
          image: grafana/loki:2.9.0
          ports:
            - containerPort: 3100
          args:
            - "-config.file=/etc/loki/local-config.yaml"
          volumeMounts:
            - name: loki-data
              mountPath: /loki
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
      volumes:
        - name: loki-data
          hostPath:
            path: /data/data_miner_db/loki
            type: DirectoryOrCreate
```

### infrastructure/loki-service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: data-miner
spec:
  selector:
    app: loki
  ports:
    - port: 3100
      targetPort: 3100
```

### infrastructure/grafana-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: data-miner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: grafana
          image: grafana/grafana:10.0.0
          ports:
            - containerPort: 3000
          env:
            - name: GF_AUTH_ANONYMOUS_ENABLED
              value: "true"
            - name: GF_AUTH_ANONYMOUS_ORG_ROLE
              value: "Admin"
          volumeMounts:
            - name: grafana-data
              mountPath: /var/lib/grafana
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
      volumes:
        - name: grafana-data
          hostPath:
            path: /data/data_miner_db/grafana
            type: DirectoryOrCreate
```

### infrastructure/grafana-service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: data-miner
spec:
  type: NodePort
  selector:
    app: grafana
  ports:
    - port: 3000
      targetPort: 3000
      nodePort: 30300
```

### config/configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: data-miner-config
  namespace: data-miner
data:
  config.yaml: |
    # =============================================================================
    # Data Miner - K8s Configuration
    # =============================================================================
    
    project_name: "glass_door"
    output_dir: "/swdfs_mnt/swshared/data_miner_output"
    project_output_dir: "${output_dir}/projects/${project_name}"
    device: "auto"
    
    input:
      search_enabled: true
      search_queries:
        - "glass door installation"
        - "sliding glass door"
        - "french door installation"
        - "patio door replacement"
      max_results_per_query: 50
      urls: []
      url_file: null
    
    # K8s service DNS names
    database:
      url: "postgresql://postgres:postgres@postgres.data-miner.svc.cluster.local:5432/data_miner"
    
    logging:
      level: "INFO"
      loki_url: "http://loki.data-miner.svc.cluster.local:3100/loki/api/v1/push"
      log_dir: "/swdfs_mnt/swshared/data_miner_output/logs"
    
    # Not used in K8s (managed by K8s replicas)
    supervisor:
      download_workers: 1
      extract_workers: 1
      filter_workers: 1
      dedup_workers: 1
      detect_workers: 1
    
    monitor:
      poll_interval: 10
      stale_threshold_minutes: 2
      long_running_threshold_minutes: 30
      cleanup_extracted_videos: false
    
    backup:
      enabled: false
      remote_dest: ""
      delete_after_backup: false
      poll_interval: 300
      verification_timeout: 1800
    
    download:
      output_dir: "${output_dir}/videos"
      format: "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
      max_resolution: 1080
      timeout: 300
      sleep_interval: 30
      max_sleep_interval: 60
      sleep_requests: 10
      blocked_hashtag_patterns: ""
    
    extract:
      output_dir: "${output_dir}/frames_raw"
      strategy: "interval"
      interval_frames: 30
      interval_seconds: 1.0
      max_frames_per_video: 5000
      image_format: "jpg"
      quality: 95
    
    filter:
      output_dir: "${project_output_dir}/frames_filtered"
      device: "${device}"
      model_id: "siglip2-so400m"
      batch_size: 32
      threshold: 0.25
      margin_threshold: 0.05
      positive_prompts:
        - "a glass door"
        - "a french door"
        - "a patio door"
        - "a photo of a glass door"
        - "a glass entrance door"
        - "a glass entrance door with handles"
        - "commercial double glass doors"
        - "a sliding glass door entrance"
        - "a storefront entrance with a glass door"
        - "a building entrance with a glass door"
        - "framed glass door clearly visible"
        - "a push bar on a glass door"
      negative_prompts:
        - "a glass wall"
        - "a fixed glass window"
        - "a large display window"
        - "a glass curtain wall"
        - "a mirror"
        - "a reflective surface"
        - "a shower door"
        - "a glass partition"
        - "a skylight"
        - "a glass ceiling"
        - "a split screen video"
        - "text overlay on screen"
        - "a video game screenshot"
        - "blurry out of focus image"
        - "dark grainy night shot"
    
    dedup:
      output_dir: "${project_output_dir}/frames_dedup"
      device: "${device}"
      threshold: 0.90
      batch_size: 64
      model_type: "dino"
      dino_model_id: "dinov3-base"
      dino_embedding_stage: "pooler"
      k_neighbors: 50
    
    detect:
      output_dir: "${project_output_dir}/detections"
      device: "${device}"
      detector: "grounding_dino"
      threshold: 0.3
      confidence_threshold: 0.3
      batch_size: 16
      save_visualizations: true
```

### workers/download-statefulset.yaml

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: download-worker
  namespace: data-miner
spec:
  serviceName: download-worker
  replicas: 9  # 3 nodes × 3 workers per node
  selector:
    matchLabels:
      app: download-worker
  template:
    metadata:
      labels:
        app: download-worker
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: download-worker
      containers:
        - name: download-worker
          image: data-miner:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "data_miner.workers.download"]
          env:
            - name: DATA_MINER_CONFIG
              value: "/config/config.yaml"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
          volumeMounts:
            - name: config
              mountPath: /config
            - name: seaweedfs
              mountPath: /swdfs_mnt/swshared
          livenessProbe:
            exec:
              command: ["test", "-f", "/tmp/healthy"]
            initialDelaySeconds: 60
            periodSeconds: 30
            failureThreshold: 3
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
      volumes:
        - name: config
          configMap:
            name: data-miner-config
        - name: seaweedfs
          hostPath:
            path: /swdfs_mnt/swshared
            type: Directory
```

### workers/extract-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: extract-worker
  namespace: data-miner
spec:
  replicas: 2
  selector:
    matchLabels:
      app: extract-worker
  template:
    metadata:
      labels:
        app: extract-worker
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: extract-worker
          image: data-miner:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "data_miner.workers.extract"]
          env:
            - name: DATA_MINER_CONFIG
              value: "/config/config.yaml"
          volumeMounts:
            - name: config
              mountPath: /config
            - name: seaweedfs
              mountPath: /swdfs_mnt/swshared
          livenessProbe:
            exec:
              command: ["test", "-f", "/tmp/healthy"]
            initialDelaySeconds: 60
            periodSeconds: 30
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
      volumes:
        - name: config
          configMap:
            name: data-miner-config
        - name: seaweedfs
          hostPath:
            path: /swdfs_mnt/swshared
            type: Directory
```

### workers/filter-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: filter-worker
  namespace: data-miner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: filter-worker
  template:
    metadata:
      labels:
        app: filter-worker
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: filter-worker
          image: data-miner:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "data_miner.workers.filter"]
          env:
            - name: DATA_MINER_CONFIG
              value: "/config/config.yaml"
          volumeMounts:
            - name: config
              mountPath: /config
            - name: seaweedfs
              mountPath: /swdfs_mnt/swshared
          livenessProbe:
            exec:
              command: ["test", "-f", "/tmp/healthy"]
            initialDelaySeconds: 120
            periodSeconds: 30
          resources:
            requests:
              memory: "4Gi"
              cpu: "1000m"
              nvidia.com/gpu: 1
            limits:
              memory: "16Gi"
              cpu: "4000m"
              nvidia.com/gpu: 1
      volumes:
        - name: config
          configMap:
            name: data-miner-config
        - name: seaweedfs
          hostPath:
            path: /swdfs_mnt/swshared
            type: Directory
```

### workers/dedup-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dedup-worker
  namespace: data-miner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dedup-worker
  template:
    metadata:
      labels:
        app: dedup-worker
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: dedup-worker
          image: data-miner:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "data_miner.workers.dedup"]
          env:
            - name: DATA_MINER_CONFIG
              value: "/config/config.yaml"
          volumeMounts:
            - name: config
              mountPath: /config
            - name: seaweedfs
              mountPath: /swdfs_mnt/swshared
          livenessProbe:
            exec:
              command: ["test", "-f", "/tmp/healthy"]
            initialDelaySeconds: 120
            periodSeconds: 30
          resources:
            requests:
              memory: "8Gi"
              cpu: "1000m"
              nvidia.com/gpu: 1
            limits:
              memory: "32Gi"
              cpu: "4000m"
              nvidia.com/gpu: 1
      volumes:
        - name: config
          configMap:
            name: data-miner-config
        - name: seaweedfs
          hostPath:
            path: /swdfs_mnt/swshared
            type: Directory
```

### workers/detect-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: detect-worker
  namespace: data-miner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: detect-worker
  template:
    metadata:
      labels:
        app: detect-worker
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: detect-worker
          image: data-miner:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "data_miner.workers.detect"]
          env:
            - name: DATA_MINER_CONFIG
              value: "/config/config.yaml"
          volumeMounts:
            - name: config
              mountPath: /config
            - name: seaweedfs
              mountPath: /swdfs_mnt/swshared
          livenessProbe:
            exec:
              command: ["test", "-f", "/tmp/healthy"]
            initialDelaySeconds: 120
            periodSeconds: 30
          resources:
            requests:
              memory: "4Gi"
              cpu: "1000m"
              nvidia.com/gpu: 1
            limits:
              memory: "16Gi"
              cpu: "4000m"
              nvidia.com/gpu: 1
      volumes:
        - name: config
          configMap:
            name: data-miner-config
        - name: seaweedfs
          hostPath:
            path: /swdfs_mnt/swshared
            type: Directory
```

### workers/monitor-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: monitor-worker
  namespace: data-miner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: monitor-worker
  template:
    metadata:
      labels:
        app: monitor-worker
    spec:
      nodeSelector:
        kubernetes.io/hostname: pavanjci
      containers:
        - name: monitor-worker
          image: data-miner:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "data_miner.workers.monitor"]
          env:
            - name: DATA_MINER_CONFIG
              value: "/config/config.yaml"
          volumeMounts:
            - name: config
              mountPath: /config
            - name: seaweedfs
              mountPath: /swdfs_mnt/swshared
          livenessProbe:
            exec:
              command: ["test", "-f", "/tmp/healthy"]
            initialDelaySeconds: 30
            periodSeconds: 30
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
      volumes:
        - name: config
          configMap:
            name: data-miner-config
        - name: seaweedfs
          hostPath:
            path: /swdfs_mnt/swshared
            type: Directory
```

---

## Code Changes Required

### 1. Add Health Check to Base Worker

File: `data_miner/workers/base.py`

Add this method to `_BaseWorker` class and call it in the main loop:

```python
import time
from pathlib import Path

class _BaseWorker:
    # ... existing code ...
    
    def _write_health_file(self):
        """Write health check file for K8s liveness probe."""
        Path("/tmp/healthy").write_text(str(time.time()))
    
    def run(self):
        # ... existing loop ...
        while True:
            self._write_health_file()  # Add this line at start of loop
            # ... rest of loop ...
```

### 2. Config Loading Update (Optional)

File: `data_miner/config/loader.py`

Ensure `DATA_MINER_CONFIG` environment variable is respected:

```python
import os

def get_config_path() -> Path:
    """Get config path from env var or default."""
    env_path = os.environ.get("DATA_MINER_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "default.yaml"
```

---

## Deployment Commands

### Initial Setup

```bash
# 1. Label master node
kubectl label node pavanjci node-role.kubernetes.io/master=true
kubectl label node pavanjci gpu=true

# 2. Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# 3. Create storage directories on master
ssh pavanjci "sudo mkdir -p /data/data_miner_db/{postgres,loki,grafana}"
ssh pavanjci "sudo chown -R 999:999 /data/data_miner_db/postgres"
ssh pavanjci "sudo chown -R 10001:10001 /data/data_miner_db/loki"
ssh pavanjci "sudo chown -R 472:472 /data/data_miner_db/grafana"

# 4. Build and distribute Docker image
cd /path/to/data_miner
docker build -t data-miner:latest .
docker save data-miner:latest | ssh pavanjci "sudo k3s ctr images import -"
docker save data-miner:latest | ssh manthana "sudo k3s ctr images import -"
docker save data-miner:latest | ssh arsenal "sudo k3s ctr images import -"
```

### Deploy to K8s

```bash
# Apply in order
kubectl apply -f k8s/namespace.yaml

# Infrastructure first
kubectl apply -f k8s/infrastructure/

# Wait for postgres to be ready
kubectl -n data-miner wait --for=condition=ready pod -l app=postgres --timeout=120s

# Config
kubectl apply -f k8s/config/

# Workers
kubectl apply -f k8s/workers/
```

### Verify Deployment

```bash
# Check all pods
kubectl -n data-miner get pods -o wide

# Check download worker distribution (should be 3 per node)
kubectl -n data-miner get pods -l app=download-worker -o wide

# Check logs
kubectl -n data-miner logs -f deployment/monitor-worker
kubectl -n data-miner logs -f statefulset/download-worker

# Check services
kubectl -n data-miner get svc
```

### Initialize Database

```bash
# Run init-db as a one-time job or exec into a pod
kubectl -n data-miner exec -it deployment/monitor-worker -- \
  python -m data_miner.cli init-db
```

### Populate Videos

```bash
# Run populate command
kubectl -n data-miner exec -it deployment/monitor-worker -- \
  python -m data_miner.cli populate --config /config/config.yaml
```

### Scaling

```bash
# Scale download workers (e.g., 4 per node = 12 total)
kubectl -n data-miner scale statefulset download-worker --replicas=12

# Scale extract workers
kubectl -n data-miner scale deployment extract-worker --replicas=4
```

### Monitoring

```bash
# Access Grafana (NodePort)
# Open browser: http://pavanjci:30300

# Check status via CLI
kubectl -n data-miner exec -it deployment/monitor-worker -- \
  python -m data_miner.cli status --project glass_door
```

---

## Troubleshooting

### Pod not starting

```bash
# Check pod events
kubectl -n data-miner describe pod <pod-name>

# Check logs
kubectl -n data-miner logs <pod-name>
```

### Database connection issues

```bash
# Test from within cluster
kubectl -n data-miner exec -it deployment/monitor-worker -- \
  python -c "from data_miner.db.connection import engine; print(engine.url)"

# Check postgres service
kubectl -n data-miner get endpoints postgres
```

### SeaweedFS mount issues

```bash
# Verify mount on node
ssh <node> "ls -la /swdfs_mnt/swshared"

# Check from pod
kubectl -n data-miner exec -it <pod-name> -- ls -la /swdfs_mnt/swshared
```

### GPU not available

```bash
# Check NVIDIA plugin
kubectl get pods -n kube-system | grep nvidia

# Check node GPU resources
kubectl describe node pavanjci | grep nvidia
```

---

## Summary

This plan migrates Data Miner to K8s with:

1. **Native K8s workers** - One process per container (industry best practice)
2. **TopologySpreadConstraints** - Exactly 3 download workers per node
3. **nodeSelector** - Processing workers run only on master (pavanjci)
4. **GPU support** - Filter, dedup, detect workers request GPU resources
5. **Persistent storage** - PostgreSQL, Loki, Grafana data on `/data/data_miner_db`
6. **Shared storage** - SeaweedFS FUSE mount at `/swdfs_mnt/swshared`
7. **K8s-native service discovery** - `postgres.data-miner.svc.cluster.local`
8. **Health checks** - Liveness probes for automatic restart
