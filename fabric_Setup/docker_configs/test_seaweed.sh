#!/bin/bash
# SeaweedFS Distributed Storage Test Script
#
# Tests that SeaweedFS is working correctly across all machines in the cluster.
# Run this from the master node after deploying all SeaweedFS containers.
#
# Usage: ./test_seaweed.sh
#
# What it tests:
# 1. SeaweedFS services are running
# 2. Write a file from master
# 3. Read the file from all workers via SSH
# 4. Write from a worker, read from master

set -e

# Configuration - update these to match your cluster
MASTER_IP="10.96.122.9"
WORKER1_IP="10.96.122.14"
WORKER1_USER="pavan"
WORKER2_IP="10.96.122.132"
WORKER2_USER="pavanmv"
MOUNT_PATH="/mnt/swdshared"
SSH_KEY="$HOME/.ssh/id_rsa_data_miner"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "SeaweedFS Distributed Storage Test"
echo "=============================================="
echo ""

# Test 1: Check local services
echo -e "${YELLOW}[Test 1] Checking SeaweedFS services on master...${NC}"
if docker ps | grep -q dm_seaweed_master; then
    echo -e "  ${GREEN}✓ seaweed-master is running${NC}"
else
    echo -e "  ${RED}✗ seaweed-master is NOT running${NC}"
    exit 1
fi

if docker ps | grep -q dm_seaweed_filer; then
    echo -e "  ${GREEN}✓ seaweed-filer is running${NC}"
else
    echo -e "  ${RED}✗ seaweed-filer is NOT running${NC}"
    exit 1
fi

if docker ps | grep -q dm_seaweed_mount; then
    echo -e "  ${GREEN}✓ seaweed-mount is running${NC}"
else
    echo -e "  ${RED}✗ seaweed-mount is NOT running${NC}"
    exit 1
fi
echo ""

# Test 2: Check mount is accessible
echo -e "${YELLOW}[Test 2] Checking FUSE mount on master...${NC}"
if docker exec dm_seaweed_mount ls $MOUNT_PATH >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Mount $MOUNT_PATH is accessible${NC}"
else
    echo -e "  ${RED}✗ Mount $MOUNT_PATH is NOT accessible${NC}"
    exit 1
fi
echo ""

# Test 3: Write test file from master
TEST_FILE="test_$(date +%s).txt"
TEST_CONTENT="Hello from master at $(date)"

echo -e "${YELLOW}[Test 3] Writing test file from master...${NC}"
docker exec dm_seaweed_mount sh -c "echo '$TEST_CONTENT' > $MOUNT_PATH/$TEST_FILE"
echo -e "  ${GREEN}✓ Wrote: $MOUNT_PATH/$TEST_FILE${NC}"
echo -e "  Content: $TEST_CONTENT"
echo ""

# Wait for replication
sleep 2

# Test 4: Read from workers
echo -e "${YELLOW}[Test 4] Reading test file from workers...${NC}"

# Worker 1
echo "  Checking Worker 1 ($WORKER1_IP)..."
if ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${WORKER1_USER}@${WORKER1_IP} \
    "docker exec dm_seaweed_mount cat $MOUNT_PATH/$TEST_FILE" 2>/dev/null | grep -q "Hello from master"; then
    echo -e "  ${GREEN}✓ Worker 1 can read the file${NC}"
else
    echo -e "  ${RED}✗ Worker 1 cannot read the file${NC}"
fi

# Worker 2
echo "  Checking Worker 2 ($WORKER2_IP)..."
if ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${WORKER2_USER}@${WORKER2_IP} \
    "docker exec dm_seaweed_mount cat $MOUNT_PATH/$TEST_FILE" 2>/dev/null | grep -q "Hello from master"; then
    echo -e "  ${GREEN}✓ Worker 2 can read the file${NC}"
else
    echo -e "  ${RED}✗ Worker 2 cannot read the file${NC}"
fi
echo ""

# Test 5: Write from worker, read from master
WORKER_FILE="worker_test_$(date +%s).txt"
WORKER_CONTENT="Hello from Worker 1 at $(date)"

echo -e "${YELLOW}[Test 5] Writing from Worker 1, reading from master...${NC}"
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${WORKER1_USER}@${WORKER1_IP} \
    "docker exec dm_seaweed_mount sh -c \"echo '$WORKER_CONTENT' > $MOUNT_PATH/$WORKER_FILE\"" 2>/dev/null
echo -e "  ${GREEN}✓ Worker 1 wrote: $MOUNT_PATH/$WORKER_FILE${NC}"

sleep 2

if docker exec dm_seaweed_mount cat $MOUNT_PATH/$WORKER_FILE | grep -q "Hello from Worker 1"; then
    echo -e "  ${GREEN}✓ Master can read file written by Worker 1${NC}"
else
    echo -e "  ${RED}✗ Master cannot read file written by Worker 1${NC}"
fi
echo ""

# Test 6: Check SeaweedFS cluster status
echo -e "${YELLOW}[Test 6] SeaweedFS Cluster Status...${NC}"
echo "  Volume servers registered:"
curl -s "http://${MASTER_IP}:9333/dir/status" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for dc in data.get('Topology', {}).get('DataCenters', []):
        for rack in dc.get('Racks', []):
            for node in rack.get('DataNodes', []):
                print(f\"    - {node.get('Url', 'unknown')} (volumes: {node.get('VolumeCount', 0)})\" )
except:
    print('    Could not parse cluster status')
" || echo "    Could not fetch cluster status"
echo ""

# Cleanup
echo -e "${YELLOW}[Cleanup] Removing test files...${NC}"
docker exec dm_seaweed_mount rm -f $MOUNT_PATH/$TEST_FILE $MOUNT_PATH/$WORKER_FILE 2>/dev/null || true
echo -e "  ${GREEN}✓ Cleaned up test files${NC}"
echo ""

echo "=============================================="
echo -e "${GREEN}All tests completed!${NC}"
echo "=============================================="
