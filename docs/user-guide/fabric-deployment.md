# Fabric Deployment Guide

This guide explains how to use Fabric to deploy and manage the Data Miner across distributed machines.

## Prerequisites

1. **Install Fabric** (included in project dependencies):
   ```bash
   uv sync
   ```

2. **Configure SSH access** to all machines (master and workers)

3. **Update `docker_configs/cluster.yaml`** with your machine details

## Configuration

Edit `docker_configs/cluster.yaml`:

```yaml
cluster:
  master:
    host: "192.168.1.100"
    user: "admin"
    key_file: "~/.ssh/id_rsa"
    local_disk: "/data/seaweed"

  workers:
    - host: "192.168.1.101"
      user: "worker"
      key_file: "~/.ssh/id_rsa"
      local_disk: "/data/seaweed"

storage:
  seaweed_data_dir: "/data/seaweed"      # All machines
  persistent_data_dir: "/data/data_miner" # Master only
```

## Deployment Modes

### Standalone (Single Machine)

For development or testing on one machine:

```bash
# Deploy standalone stack
fab deploy-standalone
```

### Docker Swarm (Distributed)

For production with multiple machines:

```bash
# 1. Initialize swarm on master
fab swarm-init

# 2. Join all workers to swarm
fab swarm-join

# 3. Deploy the stack
fab swarm-deploy

# Optional: Use a registry for images
# Optional: Use a registry for images
fab swarm-deploy --registry=myregistry.com:5000
```

### SeaweedFS Only (Testing Distributed Storage)

Deploy just SeaweedFS to test distributed storage before full deployment:

```bash
# After swarm-init and swarm-join
fab swarm-deploy-seaweed

# This deploys only:
# - seaweed-master (manager node)
# - seaweed-filer (manager node)  
# - seaweed-volume (all nodes, global)
# - seaweed-mount (all nodes, global)

# Check services
docker service ls | grep dm-seaweed

# Remove when done
docker stack rm dm-seaweed
```

## Common Tasks

### Check Status

```bash
fab status              # Swarm and service status
fab swarm-status        # Detailed swarm info
```

### View Logs

```bash
fab logs                           # Download service logs
fab logs --service=master-workers  # Master workers logs
fab logs --lines=100               # Last 100 lines
```

### Scale Services

```bash
fab swarm-scale --service=download --replicas=30
```

### Stop/Restart

```bash
fab stop                # Stop the swarm stack
fab swarm-down          # Same as stop
```

## Setup Tasks

### Install Docker on Remote Machines

```bash
# Run on a specific host
fab -H 192.168.1.101 install-docker
fab -H 192.168.1.101 install-nvidia-docker
```

### Sync Project Files

```bash
fab -H 192.168.1.100 sync-project
```

## Database Tasks

```bash
fab init-db          # Initialize database schema
fab populate         # Populate with data sources
fab pipeline-status  # Check pipeline progress
```

## Swarm Management

### Build Images on All Nodes

For registry-less deployments:

```bash
fab swarm-build-all
```

### Leave Swarm

```bash
fab swarm-leave-all  # Remove all nodes from swarm
```

## Task Reference

| Task | Description |
|------|-------------|
| `deploy-standalone` | Deploy single-machine stack |
| `prepare-storage` | Create storage dirs on all machines |
| `swarm-init` | Initialize Docker Swarm |
| `swarm-join` | Join workers to swarm |
| `swarm-deploy` | Deploy full stack to swarm |
| `swarm-deploy-seaweed` | Deploy SeaweedFS only (testing) |
| `swarm-scale` | Scale a service |
| `swarm-status` | Check swarm status |
| `swarm-logs` | View service logs |
| `swarm-down` | Remove swarm stack |
| `swarm-leave-all` | Leave swarm on all nodes |
| `swarm-build-all` | Build image on all nodes |
| `status` | Alias for swarm-status |
| `logs` | View service logs |
| `stop` | Alias for swarm-down |
| `init-db` | Initialize database |
| `populate` | Run populate command |
| `pipeline-status` | Check pipeline status |
| `install-docker` | Install Docker |
| `install-nvidia-docker` | Install NVIDIA toolkit |
| `sync-project` | Sync project to remote |

## Troubleshooting

### SSH Connection Issues

Ensure your SSH key is added:
```bash
ssh-add ~/.ssh/id_rsa
```

### Swarm Join Failures

If a worker fails to join:
```bash
# On the worker machine
docker swarm leave --force

# Then retry
fab swarm-join
```

### Image Not Found

For registry-less deployments, build on all nodes:
```bash
fab swarm-build-all
```
