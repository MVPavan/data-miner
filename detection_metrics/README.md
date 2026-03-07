# Detection Metrics

Unified object detection evaluation & dataset conversion toolkit.

## Installation

```bash
cd detection_metrics
pip install -e .

# Optional extras
pip install -e ".[view]"        # fiftyone
pip install -e ".[dev]"         # pytest
```

> Requires Python ≥ 3.10

---

## Features

- **Two-stage evaluation** — PyCocoTools (mAP50 / mAP50-95) + detailed analysis (per-class AP, PR curves, F1 curves, confusion matrix)
- **Dataset conversion** — convert between 5 annotation formats with a single command
- **Dataset analysis** — class distribution, bbox size stats, area buckets
- **Pydantic configs** — typed config models (`EvaluateConfig`, `AnalysisConfig`, `OutputConfig`) with YAML/OmegaConf support
- **Rich CLI** — progress bars, coloured tables, structured logging (loguru + rich)
- **Caching** — skip re-processing with `.eval_cache/`

---

## CLI

```
detection-metrics [OPTIONS] COMMAND [ARGS]
```

| Command | Description |
|---|---|
| `evaluate` | Evaluate predictions against ground truth (mAP, PR, F1, confusion) |
| `analyze` | Generate detailed reports & visualisations from cached results |
| `analyze-dataset` | Analyse a YOLO dataset — class distribution, bbox stats, area buckets |
| `convert` | Convert a dataset between annotation formats |
| `detect-format` | Auto-detect the annotation format of a dataset directory |
| `version` | Show version information |

### Evaluate

```bash
# All settings in YAML config (see example_config.yaml)
detection-metrics evaluate -c example_config.yaml

# Override specific values via dotlist
detection-metrics evaluate -c example_config.yaml evaluate.iou_threshold=0.7
```

### Analyze

```bash
detection-metrics analyze -c example_config.yaml
detection-metrics analyze -c example_config.yaml analyze.vis_conf_threshold=0.6
```

### Convert datasets

```bash
# Auto-detect source format, convert to COCO
detection-metrics convert \
    -s /data/my_yolo_dataset \
    -t /data/my_coco_dataset \
    --target-format coco

# Explicit formats, specific splits, copy images
detection-metrics convert \
    -s /data/input \
    -t /data/output \
    --source-format yolo_v5a \
    --target-format roboflow \
    --splits train,valid \
    --copy-images

# Detect format
detection-metrics detect-format /data/my_dataset
# → yolo_v5a
```

**Supported formats:**

| Format | Layout |
|---|---|
| `coco` | `annotations/instances_*.json` + `images/{split}/` |
| `darknet` | Flat `images/` + `labels/` + `train.txt` / `valid.txt` |
| `roboflow` | `{split}/_annotations.coco.json` |
| `yolo_v5a` | `{split}/images/` + `{split}/labels/` (split-first) |
| `yolo_v5b` | `images/{split}/` + `labels/{split}/` (modality-first) |

### Analyse a dataset

```bash
detection-metrics analyze-dataset -d /data/my_yolo_dataset
detection-metrics analyze-dataset -d /data/my_yolo_dataset --split valid
```

---

## Python API

### Pipeline (orchestrator)

```python
from detection_metrics import DetectionMetrics, EvaluateConfig, OutputConfig, AnalysisConfig

dm = DetectionMetrics(
    gt_path="path/to/coco_gt.json",
    eval_config=EvaluateConfig(iou_threshold=0.5, conf_threshold=0.001),
    output_config=OutputConfig(path="./results", use_cache=True, overwrite=True),
    analysis_config=AnalysisConfig(conf_thresholds=[0.3, 0.5, 0.7]),
)

dm.add_predictions("model_a", "path/to/predictions.json")
results = dm.evaluate()     # → Dict[str, MetricsResult]
dm.analyze()                # TP/FP analysis, precision targets
dm.visualize()              # PR curves, F1 curves, confusion matrix (PDF)
```

### Dataset conversion

```python
from detection_metrics import convert, ConvertConfig, DatasetFormat, Split

cfg = ConvertConfig(
    source="/data/yolo_dataset",
    target="/data/coco_dataset",
    target_format=DatasetFormat.COCO,
    splits=[Split.TRAIN, Split.VALID, Split.TEST],
)
bundle = convert(cfg)
print(bundle.summary())
```

### Format detection

```python
from detection_metrics import detect_format

fmt = detect_format("/data/my_dataset")  # → DatasetFormat.YOLO_V5A
```

### Low-level evaluators

```python
from detection_metrics import PyCocoEvaluator, DetailedEvaluator

# Stage 1: COCO mAP
coco_eval = PyCocoEvaluator(gt_path, pred_path)
coco_result = coco_eval.run()  # → PyCocoResult (map_50, map_50_95, per-class, per-size)

# Stage 2: Detailed analysis
detailed = DetailedEvaluator(
    ground_truths=gt_data,
    predictions=pred_data,
    iou_threshold=0.5,
    conf_threshold=0.001,
)
result = detailed.run()  # → EvaluationResult (per-class AP, PR/F1 curves, confusion)
```

---

## Package structure

```
detection_metrics/src/detection_metrics/
├── __init__.py            # Public API exports
├── pipeline.py            # DetectionMetrics orchestrator
├── evaluator.py           # PyCocoEvaluator + DetailedEvaluator
├── converter.py           # Eval-focused YOLO↔COCO bridges
├── convert_dataset.py     # Full 5-format dataset converter
├── data_loader.py         # COCO-centric data loading
├── report.py              # ReportGenerator (console summaries & tables)
├── visualizer.py          # PR curves, F1 curves, confusion matrix plots
├── dataset_analysis.py    # Dataset statistics & distribution analysis
├── remote_view.py         # FiftyOne dataset viewer integration
├── cache.py               # Result caching
├── logging.py             # Loguru + Rich logging setup
├── utils.py               # Shared constants & helpers
├── cli.py                 # Click CLI entry point
└── configs/
    ├── config.py          # Pydantic config models
    └── default.yaml       # Default config template
```

---

## Config models

All function signatures use Pydantic models instead of loose arguments:

| Model | Key fields |
|---|---|
| `EvaluateConfig` | `iou_threshold`, `conf_threshold`, `class_ids`, `class_names` |
| `AnalysisConfig` | `conf_thresholds`, `precision_targets`, `vis_conf_threshold` |
| `OutputConfig` | `path`, `use_cache`, `overwrite` |
| `ConvertConfig` | `source`, `target`, `source_format`, `target_format`, `splits`, `copy_images` |

---

## License

MIT
