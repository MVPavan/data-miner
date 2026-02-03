# K3s Complete Tutorial
## From Zero to Production-Ready Cluster

This tutorial will take you from zero Kubernetes knowledge to running your SeaweedFS distributed filesystem cluster. We'll learn by doing.

---

# PART 1: UNDERSTANDING KUBERNETES CONCEPTS

Before touching the keyboard, let's understand what we're building.

## 1.1 What is Kubernetes?

Kubernetes is a container orchestrator. It answers: "I have containers, how do I run them across multiple machines reliably?"

```
Without Kubernetes:
┌─────────────────────────────────────────────────────────┐
│ You manually:                                           │
│ - SSH into each server                                  │
│ - Run docker commands                                   │
│ - Hope containers stay running                          │
│ - Manually restart crashed containers                   │
│ - Manually balance load                                 │
│ - Manually update containers one by one                 │
└─────────────────────────────────────────────────────────┘

With Kubernetes:
┌─────────────────────────────────────────────────────────┐
│ You declare: "I want 4 copies of this container"        │
│ Kubernetes:                                             │
│ - Schedules them across nodes                           │
│ - Restarts crashed ones                                 │
│ - Load balances traffic                                 │
│ - Performs rolling updates                              │
│ - Scales up/down on demand                              │
└─────────────────────────────────────────────────────────┘
```

## 1.2 K3s vs K8s

K3s IS Kubernetes, just packaged differently:

```
Full K8s:                          K3s:
┌─────────────────────┐            ┌─────────────────────┐
│ Multiple binaries   │            │ Single 50MB binary  │
│ Requires etcd       │            │ SQLite by default   │
│ 2GB+ RAM minimum    │            │ 512MB RAM minimum   │
│ Complex install     │            │ One-liner install   │
│ Same API            │ ─────────► │ Same API            │
│ Same kubectl        │            │ Same kubectl        │
│ Same YAML           │            │ Same YAML           │
└─────────────────────┘            └─────────────────────┘
```

Everything you learn about K3s applies to full Kubernetes.

## 1.3 Core Concepts (The Big Picture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLUSTER                                     │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     CONTROL PLANE (Master)                       │   │
│   │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │   │
│   │  │ API Server│ │ Scheduler │ │Controller │ │    etcd/      │   │   │
│   │  │           │ │           │ │  Manager  │ │   SQLite      │   │   │
│   │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    │ kubectl / API                       │
│                                    ▼                                     │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│   │   WORKER NODE   │  │   WORKER NODE   │  │   WORKER NODE   │        │
│   │                 │  │                 │  │                 │        │
│   │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │        │
│   │ │   kubelet   │ │  │ │   kubelet   │ │  │ │   kubelet   │ │        │
│   │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │        │
│   │                 │  │                 │  │                 │        │
│   │ ┌─────┐ ┌─────┐ │  │ ┌─────┐ ┌─────┐ │  │ ┌─────┐        │        │
│   │ │ Pod │ │ Pod │ │  │ │ Pod │ │ Pod │ │  │ │ Pod │        │        │
│   │ └─────┘ └─────┘ │  │ └─────┘ └─────┘ │  │ └─────┘        │        │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components:

| Component | What it does | Analogy |
|-----------|--------------|---------|
| **Cluster** | The whole system | The factory |
| **Node** | A machine (physical/virtual) | A workstation in the factory |
| **Control Plane** | Brain of the cluster | Factory management office |
| **kubelet** | Agent on each node | Foreman at each workstation |
| **Pod** | Smallest deployable unit (1+ containers) | A work order |

## 1.4 Kubernetes Objects (Things You Create)

### Pod
The smallest unit. Usually 1 container, sometimes multiple tightly-coupled containers.

```yaml
# You rarely create Pods directly, but understand them:
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: nginx
    image: nginx
```

### Deployment
"I want N copies of this Pod, keep them running."

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3  # Run 3 copies
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: nginx
        image: nginx
```

```
Deployment creates:
┌─────────────────────────────────────────────────────┐
│  Deployment: my-app                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │  ReplicaSet (manages pod count)               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│  │  │  Pod 1  │  │  Pod 2  │  │  Pod 3  │       │  │
│  │  └─────────┘  └─────────┘  └─────────┘       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### DaemonSet
"Run exactly ONE Pod on EVERY node." Perfect for your SeaweedFS volume/mount.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: seaweed-volume
spec:
  selector:
    matchLabels:
      app: seaweed-volume
  template:
    # ... pod template
```

```
DaemonSet:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Node 1     │  │   Node 2     │  │   Node 3     │
│  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │
│  │  Pod   │  │  │  │  Pod   │  │  │  │  Pod   │  │
│  └────────┘  │  │  └────────┘  │  │  └────────┘  │
└──────────────┘  └──────────────┘  └──────────────┘
       ▲                 ▲                 ▲
       └─────────────────┴─────────────────┘
                 Exactly 1 per node
```

### Service
"How do Pods find each other?" Services provide stable DNS names and load balancing.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app  # Find pods with this label
  ports:
  - port: 80
```

```
Without Service:                    With Service:
┌─────────────────────┐            ┌─────────────────────┐
│ Pod IPs change!     │            │ my-service:80       │
│ 10.42.0.5 → died    │            │        │            │
│ 10.42.0.9 → new one │            │   ┌────┴────┐       │
│ How to connect???   │            │   ▼    ▼   ▼       │
│                     │            │  Pod  Pod  Pod      │
└─────────────────────┘            └─────────────────────┘
                                   Stable DNS, load balanced
```

Service Types:
| Type | Use Case |
|------|----------|
| `ClusterIP` | Internal only (default) |
| `NodePort` | Expose on each node's IP:port |
| `LoadBalancer` | Cloud provider load balancer |

### ConfigMap & Secret
Configuration data for your pods.

```yaml
# ConfigMap - non-sensitive config
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
data:
  DATABASE_HOST: "db.example.com"
  LOG_LEVEL: "debug"

---
# Secret - sensitive data (base64 encoded)
apiVersion: v1
kind: Secret
metadata:
  name: my-secret
type: Opaque
data:
  password: cGFzc3dvcmQxMjM=  # base64 of "password123"
```

### Volumes & PersistentVolumes
How containers access storage.

```
Volume Types:
┌─────────────────────────────────────────────────────┐
│ emptyDir      - Temporary, dies with Pod            │
│ hostPath      - Mount host directory into Pod       │
│ configMap     - Mount ConfigMap as files            │
│ secret        - Mount Secret as files               │
│ persistentVolumeClaim - Request persistent storage  │
└─────────────────────────────────────────────────────┘
```

For your SeaweedFS mount, we'll use `hostPath` with `mountPropagation: Bidirectional`.

## 1.5 Labels and Selectors

How Kubernetes objects find each other:

```yaml
# Pod with labels
metadata:
  labels:
    app: nginx
    environment: production
    tier: frontend

# Service selects pods by labels
spec:
  selector:
    app: nginx
    tier: frontend
```

## 1.6 Namespaces

Logical separation within a cluster (like folders):

```
Cluster
├── namespace: default        ← Your stuff goes here by default
├── namespace: kube-system    ← K8s internal components
├── namespace: production     ← You can create your own
└── namespace: staging
```

## 1.7 The Declarative Model

Kubernetes is **declarative**, not imperative:

```
Imperative (what you're used to):
"Start 3 nginx containers"
"Stop container #2"
"Start a new container"

Declarative (Kubernetes way):
"I want 3 nginx containers running"
(Kubernetes figures out how to make it happen)
```

You write YAML describing DESIRED STATE, Kubernetes makes CURRENT STATE match it.

---

# Summary of Part 1

| Concept | What it is | Your Use Case |
|---------|------------|---------------|
| **Cluster** | Collection of nodes | Your 3 lab machines |
| **Node** | A machine | Each physical server |
| **Pod** | Container wrapper | Each SeaweedFS component |
| **Deployment** | Manages replicated Pods | yt-dlp workers (scale up/down) |
| **DaemonSet** | One Pod per node | SeaweedFS volume + mount |
| **Service** | Stable DNS + load balancing | `seaweed-filer:8888` |
| **ConfigMap** | Configuration | SeaweedFS settings |
| **hostPath** | Mount host directory | `/data/seaweed`, `/mnt/swdshared` |

---

Ready for Part 2? Let's install K3s on your machines!
