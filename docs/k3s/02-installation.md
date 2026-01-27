# K3s Tutorial - Part 2: Installation

Let's get K3s running on your machines. This is the fun part!

---

## 2.1 Prerequisites

### Your Machines Need:

| Requirement | Minimum | Your Lab (probably fine) |
|-------------|---------|-------------------------|
| OS | Ubuntu 20.04+, Debian 10+ | ✓ |
| RAM | 512 MB (server), 256 MB (agent) | ✓ |
| CPU | 1 core | ✓ |
| Disk | 1 GB free | ✓ |

### Network Requirements:

| Port | Purpose | Used By |
|------|---------|---------|
| 6443 | Kubernetes API | kubectl, agents |
| 8472 | Flannel VXLAN (pod networking) | All nodes |
| 10250 | Kubelet metrics | Master |
| 2379-2380 | etcd (if HA setup) | Masters only |

### Before You Start:

```bash
# On ALL nodes, run these checks:

# 1. Check hostname is unique per machine
hostname

# 2. Ensure hostnames are set properly (not "localhost")
sudo hostnamectl set-hostname master-1   # On master
sudo hostnamectl set-hostname worker-1   # On worker 1
sudo hostnamectl set-hostname worker-2   # On worker 2

# 3. Check connectivity between nodes
ping <other-node-ip>

# 4. Ensure no firewall blocks (or open ports)
sudo ufw status
# If active, either disable or allow ports:
sudo ufw allow 6443/tcp
sudo ufw allow 8472/udp
sudo ufw allow 10250/tcp

# 5. Disable swap (Kubernetes doesn't like swap)
sudo swapoff -a
# Make permanent: comment out swap line in /etc/fstab
sudo sed -i '/ swap / s/^/#/' /etc/fstab
```

---

## 2.2 Cluster Architecture

For your lab, we'll do:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your K3s Cluster                          │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    MASTER NODE                           │   │
│   │                 (also runs workloads)                    │   │
│   │                                                          │   │
│   │   IP: 192.168.1.10 (replace with your actual IP)        │   │
│   │   Hostname: master-1                                     │   │
│   │                                                          │   │
│   │   Runs: K3s server (API, scheduler, controller)         │   │
│   │         + Can also run pods                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│               ┌──────────────┼──────────────┐                   │
│               │              │              │                   │
│               ▼              ▼              ▼                   │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│   │  WORKER NODE  │  │  WORKER NODE  │  │  WORKER NODE  │      │
│   │   worker-1    │  │   worker-2    │  │   worker-3    │      │
│   │ 192.168.1.11  │  │ 192.168.1.12  │  │ 192.168.1.13  │      │
│   │               │  │               │  │               │      │
│   │ Runs: K3s     │  │ Runs: K3s     │  │ Runs: K3s     │      │
│   │       agent   │  │       agent   │  │       agent   │      │
│   └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** In K3s, the master can also run workloads (unlike some K8s setups). Good for small clusters!

---

## 2.3 Install K3s Server (Master Node)

SSH into your master node and run:

```bash
# One-liner installation!
curl -sfL https://get.k3s.io | sh -

# That's it. Seriously.
```

### What Just Happened?

```
Downloaded:   /usr/local/bin/k3s (single binary, ~60MB)
Created:      /etc/rancher/k3s/k3s.yaml (kubeconfig)
Started:      k3s.service (systemd)
Installed:    kubectl, crictl, ctr (bundled)
```

### Verify It's Running:

```bash
# Check service status
sudo systemctl status k3s

# You should see:
#   Active: active (running)

# Check K3s version
k3s --version

# Check nodes (just master for now)
sudo kubectl get nodes

# Output:
# NAME       STATUS   ROLES                  AGE   VERSION
# master-1   Ready    control-plane,master   30s   v1.28.x+k3s1
```

### Get the Token (Needed for Workers)

```bash
# This token lets workers join the cluster
sudo cat /var/lib/rancher/k3s/server/node-token

# Output looks like:
# K10abc123...very-long-string...xyz789::server:abc123...
```

**Save this token!** You'll need it for each worker node.

### Get Master IP

```bash
# Get the IP address workers will connect to
hostname -I | awk '{print $1}'

# Or check manually
ip addr show
```

---

## 2.4 Install K3s Agent (Worker Nodes)

SSH into each worker node and run:

```bash
# Replace with YOUR values:
#   K3S_URL    = https://<master-ip>:6443
#   K3S_TOKEN  = <token from master>

curl -sfL https://get.k3s.io | K3S_URL=https://192.168.1.10:6443 K3S_TOKEN=<your-token> sh -

# Example with real values:
curl -sfL https://get.k3s.io | K3S_URL=https://192.168.1.10:6443 K3S_TOKEN=K10abc123xyz... sh -
```

### Verify on Worker:

```bash
# Check agent is running
sudo systemctl status k3s-agent

# Should show: Active: active (running)
```

### Verify on Master:

```bash
# Back on master, check all nodes
sudo kubectl get nodes

# Output:
# NAME       STATUS   ROLES                  AGE   VERSION
# master-1   Ready    control-plane,master   5m    v1.28.x+k3s1
# worker-1   Ready    <none>                 30s   v1.28.x+k3s1
# worker-2   Ready    <none>                 25s   v1.28.x+k3s1
```

### Repeat for Each Worker

```bash
# Worker 2
curl -sfL https://get.k3s.io | K3S_URL=https://192.168.1.10:6443 K3S_TOKEN=<token> sh -

# Worker 3
curl -sfL https://get.k3s.io | K3S_URL=https://192.168.1.10:6443 K3S_TOKEN=<token> sh -
```

---

## 2.5 Configure kubectl Access

By default, you need `sudo` to run kubectl on the master. Let's fix that:

### Option A: Copy kubeconfig to your user (Recommended)

```bash
# On master node:
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config

# Now kubectl works without sudo!
kubectl get nodes
```

### Option B: Set KUBECONFIG environment variable

```bash
# Add to ~/.bashrc
echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc
source ~/.bashrc

# Still needs sudo for this file, or fix permissions:
sudo chmod 644 /etc/rancher/k3s/k3s.yaml  # Less secure
```

### Option C: Access from Your Local Machine

```bash
# On master, copy the kubeconfig
sudo cat /etc/rancher/k3s/k3s.yaml

# On your local machine:
mkdir -p ~/.kube

# Paste the content, BUT change this line:
#   server: https://127.0.0.1:6443
# To:
#   server: https://<master-ip>:6443

# Save as ~/.kube/config
# Now you can run kubectl from your laptop!
```

---

## 2.6 Verify Your Cluster

### Check Nodes

```bash
kubectl get nodes -o wide

# Output:
# NAME       STATUS   ROLES                  AGE   VERSION        INTERNAL-IP     OS-IMAGE             KERNEL-VERSION
# master-1   Ready    control-plane,master   10m   v1.28.x+k3s1   192.168.1.10    Ubuntu 22.04.3 LTS   5.15.0-xx
# worker-1   Ready    <none>                 5m    v1.28.x+k3s1   192.168.1.11    Ubuntu 22.04.3 LTS   5.15.0-xx
# worker-2   Ready    <none>                 5m    v1.28.x+k3s1   192.168.1.12    Ubuntu 22.04.3 LTS   5.15.0-xx
```

### Check System Pods

```bash
kubectl get pods -A

# Output (K3s comes with these pre-installed):
# NAMESPACE     NAME                                      READY   STATUS    RESTARTS   AGE
# kube-system   coredns-597584b69b-xxxxx                  1/1     Running   0          10m
# kube-system   local-path-provisioner-79f67d76f8-xxxxx  1/1     Running   0          10m
# kube-system   metrics-server-5c8978b444-xxxxx          1/1     Running   0          10m
# kube-system   traefik-7d5f6474df-xxxxx                 1/1     Running   0          10m
```

What are these?

| Pod | Purpose |
|-----|---------|
| `coredns` | DNS for service discovery |
| `local-path-provisioner` | Dynamic storage provisioning |
| `metrics-server` | Resource metrics (CPU, memory) |
| `traefik` | Ingress controller (HTTP routing) |

### Check Cluster Info

```bash
kubectl cluster-info

# Output:
# Kubernetes control plane is running at https://192.168.1.10:6443
# CoreDNS is running at https://192.168.1.10:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
# Metrics-server is running at https://192.168.1.10:6443/api/v1/namespaces/kube-system/services/https:metrics-server:https/proxy
```

---

## 2.7 Your First Deployment (Test!)

Let's deploy something to verify everything works:

```bash
# Create a simple nginx deployment
kubectl create deployment nginx --image=nginx

# Check it's running
kubectl get pods

# Output:
# NAME                     READY   STATUS    RESTARTS   AGE
# nginx-77b4fdf86c-xxxxx   1/1     Running   0          30s

# Expose it as a service
kubectl expose deployment nginx --port=80 --type=NodePort

# Get the NodePort
kubectl get svc nginx

# Output:
# NAME    TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
# nginx   NodePort   10.43.xxx.xx   <none>        80:31234/TCP   5s
#                                                    ↑
#                                            This port (31234)

# Access it from any node's IP!
curl http://192.168.1.10:31234
curl http://192.168.1.11:31234
curl http://192.168.1.12:31234

# All should return nginx welcome page HTML!
```

### Check Where the Pod is Running

```bash
kubectl get pods -o wide

# Output:
# NAME                     READY   STATUS    NODE       
# nginx-77b4fdf86c-xxxxx   1/1     Running   worker-1   

# It landed on worker-1, but accessible from ANY node IP!
```

### Clean Up Test Deployment

```bash
kubectl delete deployment nginx
kubectl delete service nginx
```

---

## 2.8 Essential kubectl Commands

### The Ones You'll Use Daily

```bash
# ─────────────────────────────────────────────────────────
# VIEWING RESOURCES
# ─────────────────────────────────────────────────────────

kubectl get <resource>              # List resources
kubectl get pods                    # List pods
kubectl get pods -o wide            # More details (node, IP)
kubectl get pods -A                 # All namespaces
kubectl get deploy,svc,pods         # Multiple types

kubectl describe <resource> <name>  # Detailed info
kubectl describe pod nginx-xxxxx    # Pod details, events, errors

kubectl logs <pod>                  # Container logs
kubectl logs -f <pod>               # Follow logs (tail -f)
kubectl logs <pod> -c <container>   # Specific container in pod

# ─────────────────────────────────────────────────────────
# CREATING RESOURCES
# ─────────────────────────────────────────────────────────

kubectl apply -f file.yaml          # Create/update from file
kubectl apply -f ./manifests/       # Apply whole directory
kubectl apply -f https://url/x.yaml # Apply from URL

kubectl create deployment x --image=y  # Quick create (imperative)

# ─────────────────────────────────────────────────────────
# MODIFYING RESOURCES
# ─────────────────────────────────────────────────────────

kubectl edit deployment nginx       # Edit live (opens vim)
kubectl scale deployment nginx --replicas=5  # Scale up/down
kubectl set image deployment/nginx nginx=nginx:1.25  # Update image

# ─────────────────────────────────────────────────────────
# DELETING RESOURCES
# ─────────────────────────────────────────────────────────

kubectl delete -f file.yaml         # Delete what's in file
kubectl delete pod nginx-xxxxx      # Delete specific pod
kubectl delete deployment nginx     # Delete deployment (and pods)

# ─────────────────────────────────────────────────────────
# DEBUGGING
# ─────────────────────────────────────────────────────────

kubectl exec -it <pod> -- bash      # Shell into pod
kubectl exec -it <pod> -- sh        # If bash not available
kubectl exec <pod> -- ls /app       # Run command in pod

kubectl port-forward pod/nginx 8080:80    # Access pod locally
kubectl port-forward svc/nginx 8080:80    # Access service locally

kubectl top nodes                   # Node CPU/memory usage
kubectl top pods                    # Pod CPU/memory usage

# ─────────────────────────────────────────────────────────
# USEFUL FLAGS
# ─────────────────────────────────────────────────────────

-n <namespace>                      # Specific namespace
-A                                  # All namespaces
-o wide                             # More columns
-o yaml                             # Output as YAML
-o json                             # Output as JSON
--watch                             # Watch for changes
-l app=nginx                        # Filter by label
```

### Shortcuts (Aliases)

```bash
# Add to ~/.bashrc for convenience
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get svc'
alias kgd='kubectl get deployments'
alias kga='kubectl get all'
alias kaf='kubectl apply -f'
alias kdel='kubectl delete'
alias klog='kubectl logs -f'
alias kexec='kubectl exec -it'

# Now you can type:
k get pods
kgp -A
kaf myapp.yaml
```

---

## 2.9 Useful K3s-Specific Commands

```bash
# ─────────────────────────────────────────────────────────
# K3S SERVICE MANAGEMENT
# ─────────────────────────────────────────────────────────

# On master:
sudo systemctl status k3s
sudo systemctl restart k3s
sudo systemctl stop k3s
sudo journalctl -u k3s -f           # View K3s logs

# On workers:
sudo systemctl status k3s-agent
sudo systemctl restart k3s-agent
sudo journalctl -u k3s-agent -f     # View agent logs

# ─────────────────────────────────────────────────────────
# K3S UTILITIES
# ─────────────────────────────────────────────────────────

# K3s bundles these tools:
k3s kubectl get nodes               # Same as kubectl
k3s crictl ps                       # List containers (low-level)
k3s crictl images                   # List images
k3s ctr containers list             # containerd CLI

# Check K3s configuration
cat /etc/rancher/k3s/k3s.yaml       # Kubeconfig
cat /var/lib/rancher/k3s/server/node-token  # Join token

# ─────────────────────────────────────────────────────────
# UNINSTALL (if needed)
# ─────────────────────────────────────────────────────────

# On master:
/usr/local/bin/k3s-uninstall.sh

# On workers:
/usr/local/bin/k3s-agent-uninstall.sh
```

---

## 2.10 Summary: What You Have Now

```
✅ K3s master running on master-1
✅ K3s agents running on worker-1, worker-2, ...
✅ All nodes can see each other
✅ kubectl configured and working
✅ Cluster ready for deployments!

Your cluster:
┌─────────────────────────────────────────────────────────────┐
│  master-1 (control-plane)                                   │
│  ├── API Server (port 6443)                                 │
│  ├── Scheduler                                              │
│  ├── Controller Manager                                     │
│  ├── CoreDNS (service discovery)                           │
│  ├── Traefik (ingress)                                     │
│  └── Metrics Server                                        │
│                                                             │
│  worker-1 (agent)                                          │
│  ├── kubelet                                               │
│  └── Ready to run pods                                     │
│                                                             │
│  worker-2 (agent)                                          │
│  ├── kubelet                                               │
│  └── Ready to run pods                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

**Part 3:** We'll deploy your SeaweedFS cluster on this K3s setup!
- Create the YAML manifests
- Deploy master, filer, volume servers
- Set up FUSE mounts with privileged containers
- Test distributed storage

Ready to proceed?
