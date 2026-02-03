# Quickstart

This guide walks through running a complete video mining pipeline.

---

## Prerequisites

- Data Miner [installed](installation.md)
- PostgreSQL running (via Docker Compose or local)
- Database initialized (`data-miner init-db`)

---

## Step 1: Create Configuration

Create a config file `config.yaml`:

```yaml
project_name: "glass_doors_demo"
output_dir: "./output"

input:
  search_queries:
    - "glass door installation tutorial"
  max_results_per_query: 10  # Start small for testing

# Reduced workers for demo
supervisor:
  download_workers: 2
  extract_workers: 1
  filter_workers: 1
  dedup_workers: 1
  detect_workers: 1

filter:
  threshold: 0.25
  positive_prompts:
    - "a glass door"
    - "a sliding glass door"
  negative_prompts:
    - "a window"
    - "a mirror"
```

---

## Step 2: Initialize Database

```bash
# Create tables (first time only)
data-miner init-db

# Verify
data-miner status
```

---

## Step 3: Populate Videos

```bash
# Search YouTube and add videos to database
data-miner populate --config config.yaml

# Check status
data-miner status --project glass_doors_demo
```

Expected output:
```
Project: glass_doors_demo
  Stage: POPULATING
  Videos: 10 total
    PENDING: 10
```

---

## Step 4: Setup Workers

```bash
# Generate supervisor config
data-miner workers setup --config config.yaml

# Verify config was created
cat /etc/supervisor/conf.d/data_miner.conf
```

---

## Step 5: Start Pipeline

```bash
# Start all workers
data-miner workers start

# Monitor progress
watch -n 5 "data-miner status --project glass_doors_demo"
```

---

## Step 6: Monitor Progress

```bash
# Check worker status
data-miner workers status

# Check pipeline status
data-miner status --project glass_doors_demo
```

As the pipeline progresses, you'll see:

1. **POPULATING** → Videos being downloaded/extracted
2. **FILTERING** → Frames being filtered
3. **DEDUP_READY** → All videos filtered, cross-dedup starting
4. **DETECT_READY** → Dedup complete, detection starting
5. **COMPLETE** → Pipeline finished

---

## Step 7: View Results

```bash
# Output directory structure
tree output/projects/glass_doors_demo/
```

```
output/projects/glass_doors_demo/
├── frames_filtered/     # Frames that passed filter
│   └── {video_id}/
├── frames_dedup/        # Unique frames (flat)
└── detections/
    ├── annotations.json # COCO-format annotations
    └── visualizations/  # Bounding box images
```

---

## Common Workflows

### Re-run Deduplication

```bash
data-miner force-dedup glass_doors_demo
data-miner workers restart
```

### Re-run Detection

```bash
data-miner force-detect glass_doors_demo
data-miner workers restart
```

### Stop Pipeline

```bash
data-miner workers stop
```

### Delete and Start Over

```bash
data-miner delete-project glass_doors_demo --files --yes
data-miner init-db --force
```

---

## Troubleshooting

### Workers Not Starting

```bash
# Check supervisor logs
sudo tail -f /var/log/supervisor/supervisord.log

# Check worker logs
tail -f output/logs/download_*.log
```

### Videos Stuck

```bash
# Check for stale locks (monitor worker handles this automatically)
data-miner status --project glass_doors_demo
```

### GPU Memory Issues

Reduce batch size in config:

```yaml
filter:
  batch_size: 16  # Default: 32

dedup:
  batch_size: 32  # Default: 64
```

---

## Next Steps

- [Configuration](configuration.md) - All config options
- [Architecture Overview](../architecture/overview.md) - System design
