# Video Miner v3

High-performance video mining pipeline for generating large-scale computer vision datasets from YouTube videos.

![Pipeline Architecture](docs/architecture.png)

## Features

- **YouTube Search**: Search and discover videos by keyword with yt-dlp
- **Video Registry**: YAML-based tracking of videos and processing status
- **Video Download**: Download highest quality videos from YouTube
- **Frame Extraction**: Configurable sampling strategies (interval, time-based, keyframe)
- **Frame Filtering**: SigLIP2-based semantic filtering with text prompts
- **Deduplication**: DINOv2/v3 or SigLIP2 embedding-based deduplication
- **Object Detection**: Open-set detection with multiple backends

## Installation

```bash
cd video_miner_async
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

## Quick Start

### 1. Search for Videos

```bash
# Search YouTube and add to registry
video-miner search "glass door" --max-results 100

# View registry status
video-miner registry status

# List videos in registry
video-miner registry list --status pending
```

### 2. Run Pipeline

```bash
# Full pipeline with URLs
video-miner run \
    -u "https://youtube.com/watch?v=VIDEO_ID" \
    -c "glass door" \
    -c "sliding door" \
    -o ./output

# Memory-efficient mode (SigLIP2 for both filtering and dedup)
video-miner run \
    -u "https://youtube.com/watch?v=VIDEO_ID" \
    -c "glass door" \
    --use-siglip-dedup
```

### 3. Individual Stages

```bash
# Download videos only
video-miner download -u "https://youtube.com/..." -o ./videos

# Filter frames by class
video-miner filter -i ./frames -c "glass door" -o ./filtered

# Deduplicate frames
video-miner deduplicate -i ./filtered -o ./unique --threshold 0.90

# Run detection
video-miner detect -i ./unique -p "glass door" --detector florence2
```

## CLI Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `run` | Run full pipeline on URLs |
| `search` | Search YouTube and add to registry |
| `registry status` | Show registry statistics |
| `registry list` | List videos in registry |
| `registry export` | Export URLs to file |
| `download` | Download videos only |
| `filter` | Filter frames by similarity |
| `deduplicate` | Remove duplicate frames |
| `detect` | Run object detection |

### Key Options

```
video-miner run [OPTIONS]

  -u, --urls TEXT              YouTube URLs to process
  -f, --url-file PATH          File with URLs (one per line)
  -c, --classes TEXT           Target classes/captions (required)
  -o, --output-dir PATH        Output directory [default: ./output]

Model Selection:
  --filter-model [siglip2-so400m|siglip2-giant]
                               SigLIP2 filter model
  --detector [dino-x|moondream3|florence2|grounding-dino]
                               Detection model [default: moondream3]
  --use-siglip-dedup           Use SigLIP2 for dedup (memory-efficient)

Thresholds:
  --filter-threshold FLOAT     Filter threshold [default: 0.25]
  --dedup-threshold FLOAT      Dedup threshold [default: 0.90]
  --detection-threshold FLOAT  Detection confidence [default: 0.3]

Processing:
  --sampling [interval|time|keyframe]  Frame sampling strategy
  --interval INTEGER           Frame interval [default: 30]
  --batch-size INTEGER         Batch size [default: 16]
  --device TEXT                auto, cuda, cpu [default: auto]
  --stages TEXT                Stages to run (comma-separated)
```

## Model Configuration

All model IDs are centralized in `constants.py`:

### Filtering (SigLIP2)

| Model | ID | Size |
|-------|----|------|
| **so400m** (default) | `google/siglip2-so400m-patch14-384` | ~2GB |
| giant | `google/siglip2-giant-opt-patch16-384` | ~4GB |

### Deduplication

| Model | ID | Use Case |
|-------|----|----------|
| **DINOv2-base** (default) | `facebook/dinov2-base` | Best quality |
| DINOv2-large | `facebook/dinov2-large` | Higher accuracy |
| SigLIP2 | (reuses filter model) | Memory-efficient |

### Detection

| Detector | ID | Notes |
|----------|----|-------|
| **Moondream3** (default) | `moondream/moondream3-preview` | VQA + detection |
| Florence-2 | `microsoft/Florence-2-large` | Multi-task |
| Grounding DINO | `IDEA-Research/grounding-dino-base` | Stable |

## Video Registry

The registry (`video_registry.yaml`) tracks all videos:

```yaml
metadata:
  total_videos: 50
  keywords_searched: ["glass door", "sliding door"]

videos:
  dQw4w9WgXcQ:
    url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    title: "Glass Door Installation"
    source_keyword: "glass door"
    status: complete  # pending|downloaded|filtered|complete|failed
    stages:
      download: {completed: true, path: "./videos/..."}
      filter: {passed_frames: 45}
      detection: {total_detections: 25}
```

### Registry Commands

```bash
# Search and add videos
video-miner search "glass door" -n 100

# Check status
video-miner registry status

# List pending videos
video-miner registry list -s pending -n 50

# Export URLs
video-miner registry export -s pending -o pending_urls.txt
```

## Output Structure

```
output/
├── video_registry.yaml        # Video tracking registry
├── videos/                    # Downloaded videos
├── frames_raw/{video_id}/     # Extracted frames
├── frames_filtered/{video_id}/ # Filtered frames
├── frames_deduplicated/       # Unique frames
├── detections/
│   ├── annotations.json       # COCO-format annotations
│   └── visualizations/        # Bounding box visualizations
└── pipeline_result.json       # Summary statistics
```

## Python API

```python
from video_miner_async.config import PipelineConfig, DetectionConfig, DetectorType
from video_miner_async.pipeline import VideoPipeline
from video_miner_async.registry import VideoRegistry
from video_miner_async.search import search_youtube

# Search videos
videos = search_youtube("glass door", max_results=50)

# Configure pipeline
config = PipelineConfig(
    urls=["https://youtube.com/watch?v=..."],
    classes=["glass door", "sliding door"],
    detection=DetectionConfig(detector=DetectorType.MOONDREAM3),
)

# Run pipeline
pipeline = VideoPipeline(config)
result = pipeline.run()
print(f"Found {result.detections_found} detections")

# Use registry
registry = VideoRegistry.load("video_registry.yaml")
pending = registry.get_pending()
print(f"{len(pending)} videos pending")
```

## Project Structure

```
video_miner_async/
├── config.py          # Pydantic configuration models
├── constants.py       # Centralized model IDs and defaults
├── pipeline.py        # Main pipeline orchestration
├── registry.py        # Video registry (Pydantic models)
├── search.py          # YouTube search via yt-dlp
├── cli.py             # Click CLI interface
├── models/
│   ├── base.py        # BaseModel class, shared utilities
│   ├── siglip_model.py
│   ├── dinov3_model.py
│   └── detector_models.py
├── modules/
│   ├── downloader.py
│   ├── frame_extractor.py
│   ├── frame_filter.py
│   ├── deduplicator.py
│   └── detector.py
└── utils/
    ├── device.py      # CUDA/CPU device management
    ├── io.py          # File I/O, video ID extraction
    └── validators.py  # Input validation
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, 4-8GB VRAM)
- ~10GB disk space for models

### Dependencies

- `torch`, `transformers` - Deep learning
- `yt-dlp` - Video download
- `opencv-python-headless` - Frame extraction
- `scikit-learn` - Similarity computation
- `click`, `rich` - CLI
- `pydantic` - Configuration
- `pyyaml` - Registry persistence

## License

MIT
