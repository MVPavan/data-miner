# K3s Tutorial: Your Questions Answered

## Question 1: YAML Config Structure

Yes! Every Kubernetes YAML has a standard structure with 4 required top-level fields:

```yaml
apiVersion: <api_group/version>  # Which API to use
kind: <ResourceType>              # What type of object
metadata:                         # Identity & organization
  name: <unique-name>
  namespace: <optional>
  labels: <optional>
  annotations: <optional>
spec:                             # Your desired state (varies by kind)
  # ... specific to each resource type
```

### Complete Structure Reference:

```yaml
# ─────────────────────────────────────────────────────────────
# FIELD 1: apiVersion
# ─────────────────────────────────────────────────────────────
# Format: <group>/<version> or just <version> for core resources
#
# Common values:
#   v1                  → Core resources (Pod, Service, ConfigMap, Secret, Namespace)
#   apps/v1             → Deployments, DaemonSets, StatefulSets, ReplicaSets
#   batch/v1            → Jobs, CronJobs
#   networking.k8s.io/v1 → Ingress, NetworkPolicy
#   storage.k8s.io/v1   → StorageClass, VolumeAttachment
#   rbac.authorization.k8s.io/v1 → Roles, ClusterRoles, Bindings

apiVersion: apps/v1

# ─────────────────────────────────────────────────────────────
# FIELD 2: kind
# ─────────────────────────────────────────────────────────────
# The type of resource (case-sensitive, PascalCase)

kind: Deployment

# ─────────────────────────────────────────────────────────────
# FIELD 3: metadata
# ─────────────────────────────────────────────────────────────
# Identity and organizational info

metadata:
  # REQUIRED: unique name within namespace
  name: my-application
  
  # OPTIONAL: which namespace (default: "default")
  namespace: production
  
  # OPTIONAL: key-value pairs for selection/organization
  labels:
    app: my-app
    environment: production
    team: backend
    version: v1.2.3
  
  # OPTIONAL: non-identifying metadata (for tools, humans)
  annotations:
    description: "Main application server"
    maintainer: "team@example.com"
    prometheus.io/scrape: "true"

# ─────────────────────────────────────────────────────────────
# FIELD 4: spec
# ─────────────────────────────────────────────────────────────
# Desired state - structure depends entirely on 'kind'

spec:
  # ... varies by resource type
```

### Example: Complete Deployment with All Common Fields

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
  namespace: default
  labels:
    app: web-server
    tier: frontend
spec:
  # ── Deployment-specific fields ──
  replicas: 3
  
  # How to find pods this deployment manages
  selector:
    matchLabels:
      app: web-server
  
  # Strategy for updates
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  # ── Pod Template (nested Pod spec) ──
  template:
    metadata:
      labels:
        app: web-server  # Must match selector above
    spec:
      # ── Container(s) ──
      containers:
      - name: nginx
        image: nginx:1.25
        
        # Resource limits
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        
        # Ports
        ports:
        - containerPort: 80
          name: http
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /healthz
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 3
        
        # Environment variables
        env:
        - name: LOG_LEVEL
          value: "info"
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database_host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: db_password
        
        # Volume mounts
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: data-volume
          mountPath: /data
      
      # ── Volumes ──
      volumes:
      - name: config-volume
        configMap:
          name: app-config
      - name: data-volume
        hostPath:
          path: /mnt/data
          type: DirectoryOrCreate
      
      # ── Pod-level settings ──
      restartPolicy: Always  # Always, OnFailure, Never
      
      # Node selection
      nodeSelector:
        disk: ssd
      
      # Tolerations (allow scheduling on tainted nodes)
      tolerations:
      - key: "node-role.kubernetes.io/master"
        operator: "Exists"
        effect: "NoSchedule"
      
      # Security context
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
```

### Quick Reference: spec Structure by Kind

```
Pod.spec:
├── containers[]
├── volumes[]
├── restartPolicy
├── nodeSelector
├── tolerations
├── securityContext
└── ...

Deployment.spec:
├── replicas
├── selector
├── strategy
└── template (contains Pod.spec)

DaemonSet.spec:
├── selector
├── updateStrategy
└── template (contains Pod.spec)

Service.spec:
├── selector
├── ports[]
├── type (ClusterIP/NodePort/LoadBalancer)
└── ...

ConfigMap:
├── data (key-value strings)
└── binaryData (key-value binary)

Secret:
├── type
├── data (base64 encoded)
└── stringData (plain text, converted to base64)
```

---

## Question 2: Tools/SDKs for Managing Configs

Yes! There are several approaches, from simple to complex:

### Tier 1: Built-in / Simple

| Tool | What it does | Complexity | Use when |
|------|--------------|------------|----------|
| **kubectl** | Direct YAML apply | ⭐ | Learning, simple deployments |
| **Kustomize** | Patch/overlay YAML (built into kubectl) | ⭐⭐ | Multiple environments |

### Tier 2: Package Managers

| Tool | What it does | Complexity | Use when |
|------|--------------|------------|----------|
| **Helm** | Package manager with templates | ⭐⭐⭐ | Using community charts, team standardization |

### Tier 3: Programming Languages (IaC)

| Tool | Language | Complexity | Use when |
|------|----------|------------|----------|
| **Pulumi** | Python/Go/TS/etc | ⭐⭐⭐⭐ | Complex infra, existing code |
| **CDK8s** | Python/TS/Go | ⭐⭐⭐⭐ | Type-safe K8s configs |
| **cdk8s+** | Python/TS | ⭐⭐⭐⭐ | Higher-level abstractions |

### Detailed Comparison:

#### 1. kubectl (Raw YAML)
```bash
# Just apply YAML files directly
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Or a whole directory
kubectl apply -f ./manifests/

# Or from URL
kubectl apply -f https://example.com/manifest.yaml
```
**Pros:** No extra tools, direct control
**Cons:** Repetition, no templating, manual env management

---

#### 2. Kustomize (Built into kubectl)

Lets you have a "base" and "overlays" for different environments:

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── overlays/
│   ├── dev/
│   │   └── kustomization.yaml    # patches for dev
│   └── prod/
│       └── kustomization.yaml    # patches for prod
```

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml

# overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
patchesStrategicMerge:
  - increase-replicas.yaml
namePrefix: prod-
```

```bash
# Apply dev
kubectl apply -k overlays/dev/

# Apply prod
kubectl apply -k overlays/prod/
```

**Pros:** Built-in, no templating language, composition over inheritance
**Cons:** Can get complex, limited logic

---

#### 3. Helm (Package Manager)

Think "apt/npm for Kubernetes":

```bash
# Install community charts
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-redis bitnami/redis

# Install with custom values
helm install my-app ./my-chart -f values-prod.yaml
```

Create your own chart:
```
my-chart/
├── Chart.yaml           # Metadata
├── values.yaml          # Default values
├── templates/
│   ├── deployment.yaml  # Go templates
│   ├── service.yaml
│   └── _helpers.tpl     # Template helpers
```

```yaml
# templates/deployment.yaml (Go templating)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-{{ .Chart.Name }}
spec:
  replicas: {{ .Values.replicas }}
  template:
    spec:
      containers:
      - name: app
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        {{- if .Values.resources }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        {{- end }}
```

```yaml
# values.yaml
replicas: 3
image:
  repository: nginx
  tag: latest
resources:
  limits:
    memory: 128Mi
```

**Pros:** Huge ecosystem, versioned releases, rollback support
**Cons:** Go templating is ugly, steep learning curve

---

#### 4. Pulumi (Python Example)

```python
import pulumi
import pulumi_kubernetes as k8s

# Create a deployment using Python!
app_labels = {"app": "nginx"}

deployment = k8s.apps.v1.Deployment(
    "nginx",
    spec=k8s.apps.v1.DeploymentSpecArgs(
        replicas=3,
        selector=k8s.meta.v1.LabelSelectorArgs(match_labels=app_labels),
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(labels=app_labels),
            spec=k8s.core.v1.PodSpecArgs(
                containers=[k8s.core.v1.ContainerArgs(
                    name="nginx",
                    image="nginx:1.25",
                    ports=[k8s.core.v1.ContainerPortArgs(container_port=80)],
                )],
            ),
        ),
    ),
)

# Export the deployment name
pulumi.export("deployment_name", deployment.metadata.name)
```

```bash
pulumi up  # Deploy
pulumi destroy  # Tear down
```

**Pros:** Real programming language, loops, conditionals, IDE support, type checking
**Cons:** More complex, state management, learning curve

---

#### 5. CDK8s (Python Example)

```python
from constructs import Construct
from cdk8s import App, Chart
from cdk8s_plus_25 import Deployment, Service, ContainerProps

class MyChart(Chart):
    def __init__(self, scope: Construct, id: str):
        super().__init__(scope, id)

        # Much simpler API than raw K8s
        deployment = Deployment(
            self, "nginx",
            replicas=3,
            containers=[ContainerProps(
                name="nginx",
                image="nginx:1.25",
                port=80,
            )]
        )

        # Automatically creates service pointing to deployment
        deployment.expose_via_service(port=80)

app = App()
MyChart(app, "my-app")
app.synth()  # Generates YAML files
```

```bash
cdk8s synth  # Generate YAML
kubectl apply -f dist/
```

**Pros:** Type-safe, IDE autocomplete, generates standard YAML
**Cons:** Extra build step, another tool to learn

---

### My Recommendation for You:

```
Learning (now):     Raw YAML + kubectl
                    ↓
Small projects:     Kustomize (built-in, no extra tools)
                    ↓
Team/Production:    Helm (if using community charts)
                    or Pulumi/CDK8s (if you prefer Python)
```

For your SeaweedFS project: **Start with raw YAML**, then maybe Kustomize if you want dev/prod variants.

---

## Question 3: DaemonSet and ALL Workload Types

A **DaemonSet** ensures exactly ONE pod runs on EVERY node (or a subset). When a new node joins, it automatically gets a pod. When a node leaves, the pod is garbage collected.

### Complete List of Kubernetes Workload Resources:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KUBERNETES WORKLOAD TYPES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         PODS (Base Unit)                            │   │
│  │  • Smallest deployable unit                                         │   │
│  │  • Usually created by controllers, not directly                     │   │
│  │  • Contains 1+ containers sharing network/storage                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│          ┌─────────────────────────┼─────────────────────────┐             │
│          │                         │                         │             │
│          ▼                         ▼                         ▼             │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐          │
│  │  Deployment   │      │  DaemonSet    │      │  StatefulSet  │          │
│  │               │      │               │      │               │          │
│  │ • N replicas  │      │ • 1 per node  │      │ • Stateful    │          │
│  │ • Stateless   │      │ • System-wide │      │ • Ordered     │          │
│  │ • Scale up/   │      │ • Logs/agents │      │ • Stable IDs  │          │
│  │   down freely │      │ • Storage     │      │ • Databases   │          │
│  └───────────────┘      └───────────────┘      └───────────────┘          │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────┐                                                         │
│  │  ReplicaSet   │  (Usually managed by Deployment, not used directly)    │
│  └───────────────┘                                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         BATCH WORKLOADS                              │   │
│  │                                                                      │   │
│  │  ┌───────────────┐                ┌───────────────┐                 │   │
│  │  │     Job       │                │   CronJob     │                 │   │
│  │  │               │                │               │                 │   │
│  │  │ • Run to      │                │ • Scheduled   │                 │   │
│  │  │   completion  │                │ • Like cron   │                 │   │
│  │  │ • Retries     │                │ • Creates     │                 │   │
│  │  │ • Parallelism │                │   Jobs        │                 │   │
│  │  └───────────────┘                └───────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown:

#### 1. Pod
```yaml
apiVersion: v1
kind: Pod
```
- **What:** 1+ containers that share network namespace and storage
- **Use:** Almost never directly; use controllers instead
- **Example:** Testing a single container quickly

---

#### 2. ReplicaSet
```yaml
apiVersion: apps/v1
kind: ReplicaSet
```
- **What:** Ensures N identical pods are running
- **Use:** Almost never directly; Deployment manages these
- **Example:** (Managed by Deployment)

---

#### 3. Deployment ⭐ (Most Common)
```yaml
apiVersion: apps/v1
kind: Deployment
```
- **What:** Declarative updates for ReplicaSets and Pods
- **Use:** Stateless applications, web servers, APIs, workers
- **Features:**
  - Scale replicas up/down
  - Rolling updates (zero downtime)
  - Rollback to previous versions
  - Self-healing (restarts failed pods)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yt-dlp-worker
spec:
  replicas: 10  # Run 10 workers
  selector:
    matchLabels:
      app: yt-dlp-worker
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Can have 2 extra during update
      maxUnavailable: 1  # Can have 1 less during update
  template:
    metadata:
      labels:
        app: yt-dlp-worker
    spec:
      containers:
      - name: worker
        image: your-yt-dlp-image:v1
```

**Your use case:** yt-dlp download workers

---

#### 4. DaemonSet ⭐ (For Your SeaweedFS)
```yaml
apiVersion: apps/v1
kind: DaemonSet
```
- **What:** Ensures exactly 1 pod runs on every node (or selected nodes)
- **Use:** Node-level services, storage, monitoring agents, log collectors
- **Features:**
  - Auto-deploys to new nodes
  - Auto-removes from deleted nodes
  - Can target specific nodes with selectors

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
    metadata:
      labels:
        app: seaweed-volume
    spec:
      containers:
      - name: volume
        image: chrislusf/seaweedfs
        args: ["volume", "-mserver=seaweed-master:9333", "-dir=/data"]
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        hostPath:
          path: /data/seaweed
```

**Visual:**
```
Node 1          Node 2          Node 3          (New) Node 4
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Pod auto │    │ Pod auto │    │ Pod auto │    │ Pod auto │
│ created  │    │ created  │    │ created  │    │ created  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ▲               ▲               ▲               ▲
     └───────────────┴───────────────┴───────────────┘
                    DaemonSet ensures this
```

**Your use case:** SeaweedFS volume servers and FUSE mounts (one per node)

---

#### 5. StatefulSet
```yaml
apiVersion: apps/v1
kind: StatefulSet
```
- **What:** Like Deployment but for stateful apps needing stable identity
- **Use:** Databases, distributed systems needing stable network IDs
- **Features:**
  - Stable, persistent hostname: `pod-0`, `pod-1`, `pod-2`
  - Ordered deployment/scaling (0 before 1 before 2)
  - Ordered rolling updates
  - Stable persistent storage per pod

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres  # Required: headless service name
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:  # Each pod gets its own PVC
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

**Visual:**
```
StatefulSet: postgres
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  postgres-0    │  │  postgres-1    │  │  postgres-2    │
│  (created 1st) │  │  (created 2nd) │  │  (created 3rd) │
│                │  │                │  │                │
│  PVC: data-0   │  │  PVC: data-1   │  │  PVC: data-2   │
│  (persistent)  │  │  (persistent)  │  │  (persistent)  │
└────────────────┘  └────────────────┘  └────────────────┘
```

**Your use case:** Could use for SeaweedFS master if you want HA (but single master is fine for lab)

---

#### 6. Job
```yaml
apiVersion: batch/v1
kind: Job
```
- **What:** Run pod(s) to completion, then stop
- **Use:** Batch processing, migrations, one-time tasks
- **Features:**
  - Runs until success (or max retries)
  - Can run multiple pods in parallel
  - Tracks completions

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: process-videos
spec:
  completions: 100      # Need 100 successful completions
  parallelism: 10       # Run 10 pods at a time
  backoffLimit: 3       # Retry failed pods 3 times
  template:
    spec:
      restartPolicy: Never  # Required for Jobs
      containers:
      - name: processor
        image: video-processor
```

**Your use case:** Batch download jobs (alternative to long-running workers)

---

#### 7. CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
```
- **What:** Scheduled Jobs (like cron)
- **Use:** Periodic tasks, backups, cleanups, reports
- **Features:**
  - Standard cron syntax
  - Creates Jobs on schedule
  - Concurrency policies

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  concurrencyPolicy: Forbid  # Don't run if previous still running
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: backup-tool
```

**Your use case:** Scheduled cleanup of old downloads, periodic stats collection

---

### Summary Table: When to Use What

| Workload | Replicas | Identity | Storage | Lifetime | Your Use Case |
|----------|----------|----------|---------|----------|---------------|
| **Deployment** | N (scalable) | Random names | Shared/None | Long-running | yt-dlp workers |
| **DaemonSet** | 1 per node | Per-node | Per-node | Long-running | SeaweedFS volume + mount |
| **StatefulSet** | N (ordered) | Stable (pod-0, pod-1) | Per-pod persistent | Long-running | Databases |
| **Job** | N (to completion) | Random | Temporary | Until done | Batch processing |
| **CronJob** | Creates Jobs | Random | Temporary | Scheduled | Nightly backups |

---

### Your SeaweedFS Architecture Mapped to K8s:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Your K3s Cluster                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Deployment (replicas: 1)           Deployment (replicas: 1)        │
│  ┌─────────────────────┐            ┌─────────────────────┐         │
│  │  seaweed-master     │            │  seaweed-filer      │         │
│  │  (single instance)  │◄───────────│  (single instance)  │         │
│  └─────────────────────┘            └─────────────────────┘         │
│            ▲                                   ▲                     │
│            │                                   │                     │
│  ┌─────────┴───────────────────────────────────┴─────────┐          │
│  │                                                        │          │
│  │  DaemonSet: seaweed-volume (1 per node)               │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │          │
│  │  │ volume-1 │  │ volume-2 │  │ volume-3 │            │          │
│  │  │ Node 1   │  │ Node 2   │  │ Node 3   │            │          │
│  │  └──────────┘  └──────────┘  └──────────┘            │          │
│  │                                                        │          │
│  │  DaemonSet: seaweed-mount (1 per node)                │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │          │
│  │  │ mount-1  │  │ mount-2  │  │ mount-3  │            │          │
│  │  │ /mnt/swd │  │ /mnt/swd │  │ /mnt/swd │            │          │
│  │  └──────────┘  └──────────┘  └──────────┘            │          │
│  └────────────────────────────────────────────────────────┘          │
│                                                                      │
│  Deployment: yt-dlp-workers (replicas: 10, scalable)                │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐...               │
│  │ w-1  ││ w-2  ││ w-3  ││ w-4  ││ w-5  ││ w-6  │                  │
│  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘                   │
│  (spread across nodes, can scale to 50+)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

Ready for Part 2 (Installation) now?
