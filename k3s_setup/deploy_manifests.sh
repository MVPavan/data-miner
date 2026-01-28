#!/bin/bash
set -e

echo "=== Deploying Data Miner to K3s ==="

# 1. Namespace
echo "Applying namespace..."
kubectl apply -f k3s_setup/manifests/namespace.yaml

# 2. Config & Secrets
echo "Applying config..."
kubectl apply -f k3s_setup/manifests/config/

# 3. Infrastructure
echo "Applying infrastructure (Postgres, Loki, Grafana)..."
kubectl apply -f k3s_setup/manifests/infrastructure/

# 4. Workers
echo "Applying workers..."
kubectl apply -f k3s_setup/manifests/workers/

echo "=== Deployment Applied ==="
echo "Check status: kubectl get pods -n data-miner -o wide"
