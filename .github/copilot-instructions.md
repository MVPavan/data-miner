# Data Miner - Copilot Instructions

## Architecture Overview

PostgreSQL-backed, supervisor-managed **video processing pipeline** for generating CV datasets from YouTube.

### Pipeline Flow
```
Download → Extract (Central/Video table)  →  Filter → Cross-Dedup → Detect (Per-Project/ProjectVideo table)
```

- **Central stages** (download/extract): Operate on `videos` table, shared across projects
- **Project stages** (filter/dedup/detect): Operate on `project_videos` table, per-project processing
- **Concurrency**: PostgreSQL `FOR UPDATE SKIP LOCKED` + heartbeat-based lock expiration

### Key Tables
- `projects`: Project metadata + aggregated counts
- `videos`: Central video state (download/extract stages)
- `project_videos`: Per-project video processing state (filter stage)

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `data_miner/workers/` | Long-running supervisor workers - extend `BaseVideoWorker` or `BaseProjectVideosWorker` |
| `data_miner/modules/` | Core processing logic (stateless functions) - downloader, frame_extractor, frame_filter, deduplicator |
| `data_miner/models/` | ML model wrappers (SigLIP, DINOv3, detectors) |
| `data_miner/config/` | Pydantic configs + OmegaConf loader; `constants.py` for enums/status definitions |
| `data_miner/db/` | SQLModel models + database operations with locking |

## Development Patterns

### Adding a New Worker
1. Create worker in `workers/` extending appropriate base:
   - `BaseVideoWorker`: Per-video processing (download/extract)
   - `BaseProjectVideosWorker`: Per-project-video processing (filter)
   - `BaseProjectStageWorker`: Project-level operations (cross-dedup/detect)
2. Set `stage_name = StageName.X` class attribute
3. Implement `process()` returning dict of fields to update
4. Worker handles claiming, heartbeat, release automatically

Example pattern from [filter.py](data_miner/workers/filter.py):
```python
class FilterWorker(BaseProjectVideosWorker):
    stage_name = StageName.FILTER
    
    def process(self, project_video: ProjectVideo, video: Video) -> dict:
        # Return fields to update on completion
        return {"filtered_dir": str(output_dir), "passed_frames": count}
```

### Configuration System
- OmegaConf YAML with Pydantic validation
- Configs loaded via `DATA_MINER_CONFIG` env var or `--config` flag
- Stage-specific accessors: `get_filter_config()`, `get_download_config()`, etc.
- All stage configs in [config/config.py](data_miner/config/config.py)

### Database Operations Pattern
Use `claim_next_*` / `release_*` pattern from [operations.py](data_miner/db/operations.py):
```python
video = claim_next_video(session, project_id, input_status, in_progress_status, worker_id)
# ... process ...
release_video(session, video_id, worker_id, output_status, **updates)
```

### Status Enums
All statuses defined in [constants.py](data_miner/config/constants.py):
- `VideoStatus`: PENDING → DOWNLOADING → DOWNLOADED → EXTRACTING → EXTRACTED
- `ProjectVideoStatus`: PENDING → FILTERING → FILTERED/FILTERED_EMPTY
- `ProjectStatus`: POPULATING → FILTERING → DEDUP_READY → DETECTING → COMPLETE

## Essential Commands

```bash
# Setup
uv sync                           # Create venv and install
docker compose up -d              # Start PostgreSQL + Grafana + Loki
data-miner init-db                # Create tables

# Running pipeline
data-miner populate --config run_configs/run.yaml    # Add videos from config
data-miner workers setup --config run_configs/run.yaml  # Generate supervisor config
data-miner workers start                             # Start all workers
data-miner status                                    # Check pipeline progress

# Debug mode (disable heartbeat expiration)
DATA_MINER_DEBUG=1 python -m data_miner.workers.filter --config run.yaml
```

## Conventions

- **Logging**: Use `from data_miner.logging import get_logger; logger = get_logger(__name__)`
- **Config access**: Import stage-specific getters from `data_miner.config`
- **ML models**: Lazy-load in `__init__`, use `init_hf_auth()` for private HF models
- **Device handling**: Use `device: "auto"` config, resolved via `utils/device.py`
- **Frame formats**: Support both `.jpg` and `.png` via `glob("*.jpg") + glob("*.png")`

## Testing

```bash
pytest                      # Run tests
pytest --cov=data_miner     # With coverage
```

Scripts in `scripts/` for manual testing (e.g., `test_filter.py`, `test_dedup.py`).
