# Detection Metrics

Unified object detection evaluation metrics toolkit with CLI support.

## Installation

```bash
cd detection_metrics
pip install -e .
```

## Usage

### With YAML Config (Recommended)

```bash
detection-metrics evaluate --config my_eval.yaml
```

### With CLI Arguments

```bash
detection-metrics evaluate \
    --gt /path/to/annotations.json \
    --predictions /path/to/pred.json \
    --classes 1 2 3 4 6 7 8
```

## Features

- **Two-stage evaluation**: PyCocoTools (mAP50/95) + Detailed analysis (PR/F1/confusion matrix)
- **Multiple formats**: COCO JSON and YOLO TXT
- **Rich logging**: Beautiful console output with progress bars
- **YAML configs**: OmegaConf-based config with smart defaults
