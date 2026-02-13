# CVAT + FiftyOne Annotation Workflow

Upload YOLO datasets to CVAT for annotation, download back to YOLO format.

## Setup

```bash
# 1. Install dependencies
pip install fiftyone omegaconf pydantic typer tqdm requests

# 2. Start CVAT
git clone https://github.com/cvat-ai/cvat && cd cvat && docker compose up -d
```

## Usage

```bash
# Create config from template
python -m scripts.annotation.annotate init job.yaml

# Edit job.yaml with your settings, then:
python -m scripts.annotation.annotate upload job.yaml --launch
# [Annotate in CVAT web UI]
python -m scripts.annotation.annotate download job.yaml

# Check status
python -m scripts.annotation.annotate status job.yaml

# List all annotation runs
python -m scripts.annotation.annotate list job.yaml

# Show detailed info
python -m scripts.annotation.annotate info job.yaml

# Cleanup failed/orphaned tasks
python -m scripts.annotation.annotate cleanup job.yaml --force
```

## Config Structure

```yaml
anno_key: "my_job"

cvat:
  url: "http://localhost:8080"
  username: ""  # or use FIFTYONE_CVAT_USERNAME env
  password: ""  # or use FIFTYONE_CVAT_PASSWORD env

dataset:
  dir: "/path/to/yolo/dataset"
  split: "train"
  label_field: "detections"

task:
  project_name: null
  task_size: 100
  classes: null  # simple mode: auto-inferred
  
  # Advanced: full label schema with attributes
  label_schema:
    detections:
      type: "detections"
      classes: ["person", "car"]
      attributes:
        occluded:
          type: "checkbox"
          default: false
  
  # Edit restrictions
  allow_additions: true
  allow_deletions: true
  allow_label_edits: true
  allow_spatial_edits: true

export:
  output_dir: "./output"
  include_confidence: true
  cleanup: false
```

## Features

- **Retry logic**: Automatic retry with exponential backoff for transient failures
- **Connection validation**: Tests CVAT connectivity before upload/download
- **Progress tracking**: tqdm progress bars for long operations
- **Label schema**: Full support for CVAT label schemas with attributes (select, radio, checkbox, text)
- **Edit restrictions**: Control what annotators can add, delete, or modify
- **Cleanup command**: Remove orphaned CVAT tasks from failed runs
  include_confidence: true
  cleanup: false
```
