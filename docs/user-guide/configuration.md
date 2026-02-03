# Configuration

Data Miner uses YAML configuration files with OmegaConf for variable interpolation and Pydantic for validation.

---

## Configuration Loading

The config system supports three modes:

1. **Default config** - Built-in defaults from `data_miner/config/default.yaml`
2. **User config** - Override with `--config path/to/config.yaml`
3. **Environment variable** - Set `DATA_MINER_CONFIG=/path/to/config.yaml`

User configs are **merged** with defaults, so you only need to specify overrides.

---

## Minimal Config Example

```yaml
# config.yaml
project_name: "glass_doors"
output_dir: "./output"

input:
  search_queries:
    - "glass door installation"
    - "sliding glass door"
  max_results_per_query: 50

filter:
  positive_prompts:
    - "a glass door"
    - "a sliding door"
```

---

## Full Configuration Reference

### Project Settings

```yaml
project_name: "my_project"
output_dir: "./output"
project_output_dir: "${output_dir}/projects/${project_name}"
device: "auto"  # auto, cuda, cuda:0, cpu
```

> **Variable Interpolation**: Use `${section.key}` to reference other config values.

---

### Input Sources

```yaml
input:
  # YouTube search
  search_enabled: true
  search_queries:
    - "glass door installation"
  max_results_per_query: 50
  
  # Direct URLs
  urls:
    - "https://www.youtube.com/watch?v=abc123"
  
  # URL file (one URL per line)
  url_file: "urls.txt"
```

---

### Database

```yaml
database:
  url: "postgresql://postgres:postgres@localhost:5432/data_miner"
```

---

### Supervisor (Worker Counts)

```yaml
supervisor:
  download_workers: 3    # Parallel downloaders
  extract_workers: 2     # Frame extractors
  filter_workers: 1      # ML filter workers (GPU-bound)
  dedup_workers: 1       # Deduplication workers
  detect_workers: 1      # Detection workers
```

Set any worker count to `0` to disable that stage.

---

### Download Stage

```yaml
download:
  output_dir: "${output_dir}/videos"
  format: "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
  max_resolution: 1080
  timeout: 300
  
  # Rate limiting (avoid YouTube blocks)
  sleep_interval: 30         # Min seconds between downloads
  max_sleep_interval: 60     # Max seconds (randomized)
  sleep_requests: 10         # Seconds between API requests
  
  # Hashtag blocklist file
  blocked_hashtag_patterns: "blocked_hashtags.txt"
```

---

### Extract Stage

```yaml
extract:
  output_dir: "${output_dir}/frames_raw"
  strategy: "interval"       # interval, time, keyframe
  interval_frames: 30        # Every N frames
  interval_seconds: 1.0      # Every N seconds (for time strategy)
  max_frames_per_video: 5000
  image_format: "jpg"        # jpg, png, webp
  quality: 95                # JPEG/WebP quality (1-100)
```

---

### Filter Stage (SigLIP2)

```yaml
filter:
  output_dir: "${project_output_dir}/frames_filtered"
  device: "${device}"
  model_id: "siglip2-so400m"   # siglip2-so400m, siglip2-giant
  batch_size: 32
  
  # Thresholds
  threshold: 0.25              # Min positive match score
  margin_threshold: 0.05       # Positive must beat negative by this
  
  positive_prompts:
    - "a glass door"
    - "a sliding door"
  
  negative_prompts:
    - "a glass wall"
    - "a mirror"
```

---

### Dedup Stage (FAISS)

```yaml
dedup:
  output_dir: "${project_output_dir}/frames_dedup"
  device: "${device}"
  model_type: "dino"           # dino, siglip
  dino_model_id: "dinov3-base" # dinov2-base, dinov3-base, etc.
  threshold: 0.90              # Similarity threshold
  batch_size: 64
  k_neighbors: 50              # FAISS KNN search depth
```

---

### Detect Stage

```yaml
detect:
  output_dir: "${project_output_dir}/detections"
  device: "${device}"
  detector: "grounding_dino"   # grounding_dino, owlv2, florence2
  threshold: 0.3
  confidence_threshold: 0.3
  batch_size: 16
  save_visualizations: true
```

---

### Monitor Settings

The monitor worker handles:

- **Project stage transitions** (e.g., FILTERING → DEDUP_READY)
- **Stale lock recovery** (resets locks from crashed workers)
- **Frame count aggregation**

```yaml
monitor:
  poll_interval: 10                   # Seconds between checks
  stale_threshold_minutes: 2          # Reset stale locks after N minutes
  long_running_threshold_minutes: 30  # Warn about old locks
  cleanup_extracted_videos: false     # Delete videos after extraction
```

---

### Backup Settings

The backup worker syncs `frames_raw/` to a remote destination after videos are extracted.

```yaml
backup:
  enabled: false                    # Enable backup worker
  remote_dest: "user@host:/path"    # SSH destination or local path
  delete_after_backup: false        # Delete local frames after verified backup
  poll_interval: 300                # Seconds between backup checks
  verification_timeout: 1800        # Seconds for rsync verification
```

> **Note**: Backup uses rsync over SSH. Ensure SSH keys are configured for passwordless access.

---

### Logging (Grafana + Loki)

```yaml
logging:
  level: "INFO"                                        # DEBUG, INFO, WARNING, ERROR
  loki_url: "http://localhost:3100/loki/api/v1/push"   # Loki push endpoint
  log_dir: "output/logs"                               # Local log directory
```

Logs are automatically sent to:

1. **Console** - Always enabled
2. **File** - If `LOG_FILE` env var is set
3. **Loki** - If `python-logging-loki` is installed and Loki is running

**Access logs in Grafana:**

1. Open `http://localhost:3000`
2. Add Loki data source: `http://loki:3100`
3. Use LogQL queries: `{application="data_miner"}`

---

## Model ID Reference

| Stage | Model ID | Full HuggingFace Path |
|-------|----------|----------------------|
| Filter | `siglip2-so400m` | `google/siglip2-so400m-patch14-384` |
| Filter | `siglip2-giant` | `google/siglip2-giant-opt-patch16-384` |
| Dedup | `dinov3-base` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Dedup | `dinov2-large` | `facebook/dinov2-large` |
| Detect | `grounding_dino` | `IDEA-Research/grounding-dino-base` |
| Detect | `florence2` | `microsoft/Florence-2-large` |

---

## Next Steps

- [CLI Reference](cli-reference.md) - Available commands
- [Quickstart](quickstart.md) - Run the pipeline
