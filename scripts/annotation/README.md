# CVAT + FiftyOne Annotation Workflow

Upload YOLO datasets to CVAT for annotation, download back to YOLO format.

## Setup

```bash
# 1. Install dependencies
pip install fiftyone omegaconf pydantic typer

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

export:
  output_dir: "./output"
  include_confidence: true
  cleanup: false
```
