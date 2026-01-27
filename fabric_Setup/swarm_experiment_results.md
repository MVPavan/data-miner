# Swarm Experiment Results

We conducted a clean experiment to deploy SeaweedFS on Docker Swarm using the automated `fab` tooling.

## 1. Experiment Setup (Pure Fabric Workflow)
- **Action**: Deep cleanup using new `fab reset-cluster` task.
- **Tooling**: 
  - `fab reset-cluster` (Wipe stacks, data, leave swarm)
  - `fab prepare-storage`
  - `fab swarm-init`
  - `fab swarm-join`
  - `fab swarm-deploy-seaweed`
- **Configuration**: Verified `generate_compose.py` with Swarm fixes.

## 2. Successes
- **Deployment**: Stack `dm-seaweed` successfully deployed.
- **Cluster Status**: All 3 Nodes (Master + 2 Workers) joined successfully.
- **Service Health**:
  - **Master**: ✅ Running (1/1)
  - **Filer**: ✅ Running (1/1)
  - **Volume**: ✅ Running (3/3 - Full Cluster)
- **Data Persistence**: Verified via native mount. Wrote `Deep Clean Test` to `/mnt/seaweed/cleaned.txt`.
- **Fixes Applied**:
  - **Overlay Binding**: Added `-ip.bind=0.0.0.0` to Master/Filer to fix "cannot assign requested address" crash loop.
  - **Port Exposure**: Kept ports 9333/8888 exposed in Swarm to allow Host Access.

## 3. Issues & Findings

### A. Mount Service Failure (Critical)
The `seaweed-mount` service (0/3) fails to run reliably in Swarm mode.
- **Cause**: Docker Swarm ignores `privileged: true`. While `cap_add: [SYS_ADMIN]` is supported, propagation of FUSE mounts within overlay networking + sidecar patterns is notoriously unstable.
- **Workaround**: **Native Host Mount**.
  - We verified that running `weed mount` directly on the host works perfectly.

### B. Worker Join Flakiness
- **Status**: **RESOLVED**. The `fab reset-cluster` + `fab swarm-join` sequence correctly handled the re-join process, resulting in a stable 3-node cluster without manual intervention.

## 4. Final Recommendation
For production stability, stick to the **Hybrid Approach**:
1.  **Core Services**: Run Master, Filer, Volume in **Docker Swarm** (using the fixed configuration).
2.  **Mounts**: Run `weed mount` as a **native systemd service** on hosts that need access (avoid containerizing the mount).

## 5. Reproduction Steps (Command Log)

Use this exact sequence to reproduce the clean swarm deployment:

### Step 1: Deep Cleanup
```bash
# Removes old stacks, leaves swarm on ALL nodes, wipes data directories
uv run fab reset-cluster
```

### Step 2: Prepare Environment
```bash
# Creates /data/seaweed and /mnt/swdshared on ALL nodes
uv run fab prepare-storage
```

### Step 3: Initialize Cluster
```bash
# Initializes Swarm on Master
uv run fab swarm-init

# Joins all Workers to the Swarm automatically
uv run fab swarm-join
```

### Step 4: Deploy SeaweedFS Stack
```bash
# generates docker-compose.seaweed.yml and deploys stack 'dm-seaweed'
uv run fab swarm-deploy-seaweed
```

### Step 5: Verify Deployment
```bash
# Check Master/Filer/Volume services
uv run fab swarm-status
```

### Step 6: Verify Host Access (Manual)
```bash
# Installs weed binary and starts native mount on ALL nodes (background process)
uv run fab setup-native-mount

# Test Write
echo "Fabric Test" > /mnt/seaweed/test.txt
cat /mnt/seaweed/test.txt
```
