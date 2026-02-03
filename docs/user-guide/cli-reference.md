# CLI Reference

All commands are available via the `data-miner` CLI.

```bash
data-miner --help
```

---

## Core Commands

### `init-db`

Initialize database tables.

```bash
data-miner init-db
data-miner init-db --force  # Drop and recreate tables
```

---

### `populate`

Add videos to the database from config sources (search queries, URLs, files).

```bash
# Use config file
data-miner populate --config config.yaml

# Dry run (show what would be added)
data-miner populate --config config.yaml --dry-run
```

---

### `add-video`

Add a single video URL.

```bash
data-miner add-video "https://youtube.com/watch?v=..." \
    --project my_project \
    --source-type url
```

**Options:**
- `--project` - Project name (default: from config)
- `--source-type` - `url`, `search`, or `file`
- `--source-info` - Additional metadata

---

### `status`

Show pipeline status.

```bash
# All projects
data-miner status

# Specific project
data-miner status --project my_project
```

---

## Worker Management

### `workers setup`

Generate supervisor configuration.

```bash
data-miner workers setup --config config.yaml
```

This creates `/etc/supervisor/conf.d/data_miner.conf` with worker definitions.

---

### `workers start`

Start all workers.

```bash
data-miner workers start
```

---

### `workers stop`

Stop all workers.

```bash
data-miner workers stop
```

---

### `workers restart`

Restart all workers.

```bash
data-miner workers restart
```

---

### `workers status`

Show supervisor worker status.

```bash
data-miner workers status
```

---

## Maintenance Commands

### `delete-project`

Delete a project and optionally its files.

```bash
# Delete project (keep files)
data-miner delete-project my_project

# Delete project and files
data-miner delete-project my_project --files

# Also delete orphaned videos
data-miner delete-project my_project --files --orphans

# Skip confirmation
data-miner delete-project my_project --yes
```

---

### `delete-videos`

Delete project-videos with optional filters.

```bash
# Delete all FAILED videos
data-miner delete-videos --project my_project --pv-status FAILED

# Delete videos and files
data-miner delete-videos --project my_project --pv-status FAILED --files
```

---

### `cleanup-orphans`

Remove orphaned videos not linked to any project.

```bash
data-miner cleanup-orphans
data-miner cleanup-orphans --files  # Also delete files
```

---

### `force-dedup`

Force project back to DEDUP_READY stage (re-run cross-dedup).

```bash
data-miner force-dedup my_project
```

---

### `force-detect`

Force project back to DETECT_READY stage (re-run detection).

```bash
data-miner force-detect my_project
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATA_MINER_CONFIG` | Path to config file |
| `DATABASE_URL` | PostgreSQL connection string |
| `HF_TOKEN` | HuggingFace token for private models |
| `DATA_MINER_DEBUG` | Set to `1` to disable heartbeat (dev only) |

---

## Next Steps

- [Quickstart](quickstart.md) - End-to-end tutorial
- [Configuration](configuration.md) - Config options
