# SeaweedFS Distributed Storage Setup Guide

This guide documents the **verified working configuration** for deploying a distributed SeaweedFS cluster across multiple nodes without using Docker Swarm overlay networks.

## Architecture

- **Master Node** (10.96.122.9): Runs Master, Filer, Volume, and Mount.
- **Worker Nodes** (10.96.122.14, 10.96.122.132): Run Volume servers connected to Master.
- **Host Access**: Achieved via native `weed` binary mount (hybrid approach).

---

## 1. Prerequisites (SSH Keys)

Ensure you have passwordless SSH access to all workers. The verified project key is `~/.ssh/id_rsa_data_miner`.

### Manual Key Setup (If needed)
If automated setup fails, manually verify keys are authorized:
```bash
# Append public key to worker's authorized_keys
cat ~/.ssh/id_rsa_data_miner.pub | ssh -i ~/.ssh/id_rsa user@WORKER "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# Verify access
ssh -i ~/.ssh/id_rsa_data_miner user@WORKER date
```

---

## 2. Master Node Setup (10.96.122.9)

### `docker-compose.seaweed-master.yml`
Key configuration details:
- **Services**: `master`, `filer`, `volume`, `mount`.
- **Clean IP Binding**: `master` command uses `-ip=seaweed-master` (container hostname).
- **Persistence**: `filer` maps `/data/seaweed/filer` and uses `-dir=/data`.

### Deployment
```bash
# Create directories with correct permissions (user 1000:1000)
sudo mkdir -p /data/seaweed/filer /data/seaweed/master /data/seaweed/volume /mnt/seaweed
sudo chown -R 1000:1000 /data/seaweed

# Deploy
cd docker_configs
docker compose -f docker-compose.seaweed-master.yml up -d
```

---

## 3. Worker Node Setup

### `docker-compose.seaweed-worker.yml`
**CRITICAL FIX**: Workers must advertise their **Public Host IP** to the Master, but act as a server inside Docker.
- **Command**: `volume -ip.bind=0.0.0.0 -ip=${PUBLIC_IP} -publicUrl=${PUBLIC_IP}:8080 ...`
- **Explanation**: 
  - `-ip.bind=0.0.0.0`: Binds to all interfaces *inside* the container (so it starts).
  - `-ip=${PUBLIC_IP}`: Tells Master "I am reachable at [Host IP]".

### Deployment Step-by-Step
Repeat for each worker:

1.  **Copy Configuration**:
    ```bash
    scp -i ~/.ssh/id_rsa_data_miner docker_configs/docker-compose.seaweed-worker.yml user@WORKER_IP:/tmp/
    ```

2.  **Deploy (must use sudo -E)**:
    ```bash
    ssh -i ~/.ssh/id_rsa_data_miner user@WORKER_IP
    
    # Export Public IP (Critical)
    export PUBLIC_IP=$(hostname -I | awk '{print $1}')
    
    # Deploy
    cd /tmp
    sudo -E docker compose -f docker-compose.seaweed-worker.yml up -d
    ```

---

## 4. Validated Verification

### Cluster Topology
From Master:
```bash
curl -s http://localhost:9333/dir/status | python3 -m json.tool
```
*Expected*: `DataNodes` array contains 3 entries (Master + 2 Workers).

### Distributed Write/Verify
1.  **Write on Master**:
    ```bash
    docker exec dm_seaweed_mount sh -c "echo 'Test File' > /mnt/seaweed/test.txt"
    ```
2.  **Read on Worker**:
    ```bash
    ssh user@WORKER_IP "sudo docker exec dm_seaweed_mount cat /mnt/seaweed/test.txt"
    ```

---

## 5. Host Access Solution (Important)

Docker containers cannot reliable expose FUSE mounts to the host filesystem due to propagation limits.

**Solution**: Use the native `weed` binary on the host machine.

1.  **Get Binary** (Extract from container):
    ```bash
    docker cp dm_seaweed_master:/usr/bin/weed .
    chmod +x weed
    ```
2.  **Mount on Host**:
    ```bash
    # Run natively (no docker)
    sudo mkdir -p /mnt/seaweed
    sudo ./weed mount -dir=/mnt/seaweed -filer=localhost:8888 -filer.path=/ -allowOthers=true &
    ```
    *Result*: You can now browse `/mnt/seaweed` directly on the host machine.

