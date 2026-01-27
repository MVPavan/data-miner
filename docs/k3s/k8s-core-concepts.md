# Kubernetes Core Concepts

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CONTROL PLANE                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ API Server  │  │  Scheduler  │  │ Controller  │  │    etcd     │ │
│  │             │  │             │  │   Manager   │  │  (storage)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ manages
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CLUSTER ADD-ONS                                │
│         (run as pods, but provide cluster-wide services)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   CoreDNS   │  │ kube-proxy  │  │   Metrics   │                  │
│  │   (DNS)     │  │ (networking)│  │   Server    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ runs on
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            NODES                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Node                                                            ││
│  │  ┌──────────────────────────────────────────────────────────┐   ││
│  │  │                      kubelet                              │   ││
│  │  │  (talks to API server, manages pods on this node)         │   ││
│  │  └──────────────────────────────────────────────────────────┘   ││
│  │                            │                                    ││
│  │                            │ uses                               ││
│  │                            ▼                                    ││
│  │  ┌──────────────────────────────────────────────────────────┐   ││
│  │  │              Container Runtime (containerd)               │   ││
│  │  │  (actually pulls images, starts/stops containers)         │   ││
│  │  └──────────────────────────────────────────────────────────┘   ││
│  │                            │                                    ││
│  │                            │ runs                               ││
│  │                            ▼                                    ││
│  │  ┌──────────────────────────────────────────────────────────┐   ││
│  │  │                        Pods                               │   ││
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐               │   ││
│  │  │  │   pod-1   │ │   pod-2   │ │   pod-3   │               │   ││
│  │  │  │┌─────────┐│ │┌─────────┐│ │┌─────────┐│               │   ││
│  │  │  ││container││ ││container││ ││container││               │   ││
│  │  │  │└─────────┘│ │└─────────┘│ │└─────────┘│               │   ││
│  │  │  └───────────┘ └───────────┘ └───────────┘               │   ││
│  │  └──────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Layer | Component | What it does |
|-------|-----------|--------------|
| **Control Plane** | API Server | Single entry point, all components talk through it |
| | Scheduler | Decides which node runs each pod |
| | Controller Manager | Runs controllers (Deployment, StatefulSet, DaemonSet, etc.) |
| | etcd | Key-value store, holds all cluster state |
| **Add-ons** | CoreDNS | Translates service names → IPs (`master.seaweedfs` → `10.43.x.x`) |
| | kube-proxy | Network rules, routes service IPs to pod IPs |
| | Metrics Server | Optional, provides CPU/memory metrics for `kubectl top` |
| **Node** | kubelet | Agent on each node, manages pod lifecycle |
| | containerd | Container runtime, does the actual container work |
| | Pods | Your workloads |

---

## YAML Fields → K8s Components

```yaml
apiVersion: apps/v1              # → API Server: which API group handles this
kind: StatefulSet                # → Controller Manager: which controller manages this
metadata:
  name: master                   # → etcd: stored as unique key
  namespace: seaweedfs           # → etcd: partition/folder for storage
  labels:                        # → etcd: stored metadata for filtering
    app: seaweedfs
spec:
  serviceName: master            # → CoreDNS: creates DNS records
  replicas: 1                    # → Controller Manager: desired state to maintain
  selector:
    matchLabels:                 # → Controller Manager: how to find owned pods
      component: master
  template:                      # → kubelet: blueprint for creating pods
    metadata:
      labels:                    # → etcd + Controller: pod identification
        component: master
    spec:
      nodeSelector:              # → Scheduler: constraint for node selection
        kubernetes.io/hostname: pavanjci
      containers:                # → kubelet + containerd
      - name: master             # → containerd: container identifier
        image: seaweedfs:latest  # → containerd: what to pull/run
        args: [...]              # → containerd: process arguments
        ports:                   # → kube-proxy: network exposure info
        - containerPort: 9333
        volumeMounts:            # → kubelet: mount setup inside container
        - name: data
          mountPath: /data
      volumes:                   # → kubelet: volume provisioning
      - name: data
        hostPath:
          path: /data/seaweed/master
```

### Service YAML

```yaml
apiVersion: v1
kind: Service                    # → kube-proxy + CoreDNS
metadata:
  name: master                   # → CoreDNS: DNS name (master.seaweedfs.svc)
  namespace: seaweedfs
spec:
  clusterIP: None                # → kube-proxy: headless, no load balancer IP
  selector:                      # → kube-proxy: which pods to route to
    component: master
  ports:                         # → kube-proxy: port mapping rules
  - port: 9333
```

---

## Field to Component Mapping

| YAML Field | K8s Component | Purpose |
|------------|---------------|---------|
| `apiVersion` | API Server | Routes to correct API handler |
| `kind` | Controller Manager | Picks which controller manages it |
| `metadata.name` | etcd | Unique identifier, stored as key |
| `metadata.namespace` | etcd | Logical partition |
| `metadata.labels` | etcd | Stored for filtering/selection |
| `spec.replicas` | Controller Manager | Desired state to reconcile |
| `spec.selector` | Controller Manager | Links controller to pods |
| `spec.serviceName` | CoreDNS | Stable DNS for StatefulSet pods |
| `spec.template` | kubelet | Pod creation blueprint |
| `nodeSelector` | Scheduler | Node placement constraint |
| `containers` | kubelet + containerd | Actual workload definition |
| `image` | containerd | Container image to run |
| `ports` | kube-proxy | Network exposure |
| `volumeMounts` | kubelet | In-container mount points |
| `volumes` | kubelet | Volume source definitions |
| Service `selector` | kube-proxy | Pod discovery for routing |
| Service `clusterIP` | kube-proxy | Virtual IP assignment |

---

## The Flow: What Happens When You Apply

```
1. kubectl apply -f deployment.yaml
                │
                ▼
2. API Server: validates, stores in etcd
                │
                ▼
3. Controller Manager: sees new resource
   "I need 1 replica, currently 0, create pod"
                │
                ▼
4. Scheduler: sees unscheduled pod
   "nodeSelector says node-1, assign there"
                │
                ▼
5. kubelet (on node-1): sees pod assigned to it
   "Create container, mount volumes, start process"
                │
                ▼
6. kube-proxy: sees Service + pod labels match
   "Route traffic for service:port to this pod"
                │
                ▼
7. CoreDNS: sees Service
   "service.namespace.svc.cluster.local → pod IP"
```

---

## Labels and Names

### What's Required vs Optional

| Field | Required? | Purpose |
|-------|-----------|---------|
| `metadata.name` | Yes | K8s resource identifier |
| `metadata.namespace` | No (defaults to `default`) | Isolation |
| `spec.selector.matchLabels` | Yes | Links controller → pods |
| `template.metadata.labels` | Yes | Must match selector |
| `spec.serviceName` | StatefulSet only | Stable pod DNS |
| `app`, `component`, etc. | No | Human organization + filtering |

### Label Rules

- `selector.matchLabels` must be a **subset** of `template.metadata.labels`
- Label keys can be anything (`app`, `component`, `banana`, etc.)
- Labels are just key-value pairs for filtering

```yaml
# Valid - template has more labels than selector
selector:
  matchLabels:
    component: master
template:
  metadata:
    labels:
      component: master    # matches selector
      app: seaweedfs       # extra, ignored by selector
```

### Namespace vs App Label

| Aspect | Namespace | `app` label |
|--------|-----------|-------------|
| What it is | K8s built-in isolation | Just a text label |
| Enforced by | Kubernetes itself | Nothing (convention) |
| Scope | Resources isolated | No effect on visibility |
| Network | Services need full DNS to cross | No effect |
| Deletion | `kubectl delete ns X` removes all | Just a filter |

---

## K3s vs Full Kubernetes

| Full K8s | K3s |
|----------|-----|
| etcd | SQLite (or etcd optional) |
| Separate binaries | Single `k3s` binary |
| containerd separate | containerd bundled |
| kube-proxy separate | Built into k3s |

---

## Useful Commands

```bash
# View control plane components
kubectl get pods -n kube-system

# View CoreDNS
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Filter by labels
kubectl get pods -l app=seaweedfs
kubectl get pods -l component=master
kubectl logs -l component=mount

# Multiple labels (AND)
kubectl get pods -l app=seaweedfs,component=master

# All namespaces
kubectl get pods -A -l app=seaweedfs
```

---

## Tags

#kubernetes #k8s #infrastructure #containers #devops
