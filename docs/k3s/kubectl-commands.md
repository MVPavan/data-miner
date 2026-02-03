# kubectl Commands Reference

## Basic Syntax

```bash
kubectl [command] [TYPE] [NAME] [flags]
```

---

## Cluster Info

```bash
# Cluster info
kubectl cluster-info

# List all nodes
kubectl get nodes
kubectl get nodes -o wide

# Node details
kubectl describe node <node-name>

# Check component status
kubectl get componentstatuses
```

---

## Namespaces

```bash
# List namespaces
kubectl get namespaces
kubectl get ns

# Create namespace
kubectl create namespace <name>

# Delete namespace (deletes everything inside)
kubectl delete namespace <name>

# Set default namespace for session
kubectl config set-context --current --namespace=<name>
```

---

## Pods

```bash
# List pods
kubectl get pods
kubectl get pods -n <namespace>
kubectl get pods -A                      # All namespaces
kubectl get pods -o wide                 # Show node, IP
kubectl get pods -w                      # Watch mode
kubectl get pods -l app=seaweedfs        # Filter by label

# Pod details
kubectl describe pod <pod-name> -n <namespace>

# Pod logs
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --tail=50
kubectl logs <pod-name> -n <namespace> -f              # Follow
kubectl logs -l component=mount -n <namespace>         # By label

# Execute command in pod
kubectl exec -it <pod-name> -n <namespace> -- sh
kubectl exec -it <pod-name> -n <namespace> -- bash
kubectl exec <pod-name> -n <namespace> -- ls -la /data

# Delete pod (will recreate if managed by controller)
kubectl delete pod <pod-name> -n <namespace>
kubectl delete pods -l component=mount -n <namespace>  # By label

# Run debug pod
kubectl run debug --rm -it --image=busybox -n <namespace> -- sh
```

---

## Deployments

```bash
# List deployments
kubectl get deployments -n <namespace>
kubectl get deploy -n <namespace>

# Create deployment
kubectl create deployment <name> --image=<image>

# Scale deployment
kubectl scale deployment <name> --replicas=3 -n <namespace>

# Update image
kubectl set image deployment/<name> <container>=<new-image> -n <namespace>

# Rollout status
kubectl rollout status deployment/<name> -n <namespace>

# Rollout history
kubectl rollout history deployment/<name> -n <namespace>

# Rollback
kubectl rollout undo deployment/<name> -n <namespace>

# Restart deployment
kubectl rollout restart deployment/<name> -n <namespace>

# Delete deployment
kubectl delete deployment <name> -n <namespace>
```

---

## StatefulSets

```bash
# List statefulsets
kubectl get statefulsets -n <namespace>
kubectl get sts -n <namespace>

# Scale
kubectl scale statefulset <name> --replicas=3 -n <namespace>

# Restart
kubectl rollout restart statefulset <name> -n <namespace>

# Delete
kubectl delete statefulset <name> -n <namespace>
```

---

## DaemonSets

```bash
# List daemonsets
kubectl get daemonsets -n <namespace>
kubectl get ds -n <namespace>

# Restart all pods in daemonset
kubectl rollout restart daemonset <name> -n <namespace>

# Delete
kubectl delete daemonset <name> -n <namespace>
```

---

## Services

```bash
# List services
kubectl get services -n <namespace>
kubectl get svc -n <namespace>

# Service details
kubectl describe svc <name> -n <namespace>

# Expose deployment as service
kubectl expose deployment <name> --port=80 --target-port=8080 -n <namespace>

# Port forward (access locally)
kubectl port-forward svc/<name> <local-port>:<service-port> -n <namespace>
kubectl port-forward pod/<pod-name> <local-port>:<pod-port> -n <namespace>

# Delete service
kubectl delete svc <name> -n <namespace>
```

---

## ConfigMaps & Secrets

```bash
# List
kubectl get configmaps -n <namespace>
kubectl get secrets -n <namespace>

# Create configmap
kubectl create configmap <name> --from-file=<path> -n <namespace>
kubectl create configmap <name> --from-literal=key=value -n <namespace>

# Create secret
kubectl create secret generic <name> --from-literal=password=secret -n <namespace>

# View secret (base64 decoded)
kubectl get secret <name> -n <namespace> -o jsonpath='{.data.password}' | base64 -d

# Delete
kubectl delete configmap <name> -n <namespace>
kubectl delete secret <name> -n <namespace>
```

---

## Apply & Delete Resources

```bash
# Apply YAML
kubectl apply -f <file.yaml>
kubectl apply -f <directory>/
kubectl apply -f https://raw.githubusercontent.com/...

# Delete from YAML
kubectl delete -f <file.yaml>

# Create (fails if exists)
kubectl create -f <file.yaml>

# Replace (must exist)
kubectl replace -f <file.yaml>

# Dry run (preview changes)
kubectl apply -f <file.yaml> --dry-run=client
kubectl apply -f <file.yaml> --dry-run=server
```

---

## Resource Inspection

```bash
# Get YAML of existing resource
kubectl get <type> <name> -n <namespace> -o yaml
kubectl get pod <name> -n <namespace> -o yaml

# Get JSON
kubectl get <type> <name> -n <namespace> -o json

# Get specific field
kubectl get pod <name> -n <namespace> -o jsonpath='{.status.podIP}'
kubectl get pods -n <namespace> -o jsonpath='{.items[*].metadata.name}'

# Get all resources
kubectl get all -n <namespace>

# Get events
kubectl get events -n <namespace>
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

---

## Labels & Selectors

```bash
# Add label
kubectl label pod <name> env=prod -n <namespace>

# Remove label
kubectl label pod <name> env- -n <namespace>

# Filter by label
kubectl get pods -l app=seaweedfs -n <namespace>
kubectl get pods -l 'app in (nginx, redis)' -n <namespace>
kubectl get pods -l app!=nginx -n <namespace>

# Show labels
kubectl get pods --show-labels -n <namespace>
```

---

## Resource Management

```bash
# Top (requires metrics-server)
kubectl top nodes
kubectl top pods -n <namespace>

# Resource usage
kubectl describe node <name> | grep -A5 "Allocated resources"

# Edit resource directly
kubectl edit deployment <name> -n <namespace>
kubectl edit svc <name> -n <namespace>

# Patch resource
kubectl patch svc <name> -n <namespace> -p '{"spec": {"type": "NodePort"}}'
```

---

## Troubleshooting

```bash
# Pod not starting - check events
kubectl describe pod <name> -n <namespace>
kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name>

# Check logs
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --previous    # Previous container

# Debug with ephemeral container
kubectl debug <pod-name> -it --image=busybox -n <namespace>

# Check DNS resolution
kubectl run dns-test --rm -it --image=busybox -- nslookup <service-name>.<namespace>.svc.cluster.local

# Check connectivity
kubectl run curl-test --rm -it --image=curlimages/curl -- curl http://<service>:<port>
```

---

## Context & Config

```bash
# View config
kubectl config view

# List contexts
kubectl config get-contexts

# Current context
kubectl config current-context

# Switch context
kubectl config use-context <context-name>

# Set default namespace
kubectl config set-context --current --namespace=<namespace>
```

---

## Common Shortcuts

| Short | Full |
|-------|------|
| `po` | pods |
| `svc` | services |
| `deploy` | deployments |
| `ds` | daemonsets |
| `sts` | statefulsets |
| `ns` | namespaces |
| `cm` | configmaps |
| `pv` | persistentvolumes |
| `pvc` | persistentvolumeclaims |

---

## SeaweedFS Specific Commands

```bash
# Check all SeaweedFS pods
kubectl get pods -n seaweedfs -o wide

# Check logs
kubectl logs -l component=master -n seaweedfs
kubectl logs -l component=filer -n seaweedfs
kubectl logs -l component=volume -n seaweedfs
kubectl logs -l component=mount -n seaweedfs

# Restart components
kubectl delete pods -l component=mount -n seaweedfs
kubectl rollout restart statefulset master -n seaweedfs
kubectl rollout restart statefulset filer -n seaweedfs
kubectl rollout restart daemonset volume -n seaweedfs
kubectl rollout restart daemonset mount -n seaweedfs

# Port forward for UI
kubectl port-forward svc/master -n seaweedfs 9333:9333
kubectl port-forward svc/filer -n seaweedfs 8888:8888

# Exec into pods
kubectl exec -it master-0 -n seaweedfs -- sh
kubectl exec -it filer-0 -n seaweedfs -- sh

# Check services
kubectl get svc -n seaweedfs

# Full cleanup
kubectl delete namespace seaweedfs
```

---

## Useful Aliases (add to ~/.bashrc or ~/.zshrc)

```bash
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgpa='kubectl get pods -A'
alias kgs='kubectl get svc'
alias kgn='kubectl get nodes'
alias kd='kubectl describe'
alias kl='kubectl logs'
alias klf='kubectl logs -f'
alias kex='kubectl exec -it'
alias kaf='kubectl apply -f'
alias kdf='kubectl delete -f'
alias kns='kubectl config set-context --current --namespace'
```

---

## Tags

#kubernetes #kubectl #k8s #devops #commands
