"""Detection Metrics - Unified object detection evaluation toolkit."""

from detection_metrics.evaluator import (
    DetailedEvaluator,
    PyCocoEvaluator,
    EvaluationResult,
    PyCocoResult,
    ClassMetrics,
)
from detection_metrics.data_loader import DataLoader
from detection_metrics.converter import (
    yolo_to_coco,
    ensure_coco_gt,
    predictions_to_coco,
    load_coco_gt,
)
from detection_metrics.convert_dataset import (
    Annotation,
    ConvertConfig,
    DatasetBundle,
    DatasetFormat,
    ImageEntry,
    Split,
    convert,
    detect_format,
)
from detection_metrics.pipeline import DetectionMetrics, MetricsResult
from detection_metrics.configs.config import (
    AnalysisConfig,
    EvaluateConfig,
    FullConfig,
    OutputConfig,
    load_config,
)

__version__ = "0.2.0"

__all__ = [
    # Pipeline (orchestrator)
    "DetectionMetrics",
    "MetricsResult",
    # Evaluators
    "DetailedEvaluator",
    "PyCocoEvaluator",
    # Results
    "EvaluationResult",
    "PyCocoResult",
    "ClassMetrics",
    # Data
    "DataLoader",
    # Eval converter
    "yolo_to_coco",
    "ensure_coco_gt",
    "predictions_to_coco",
    "load_coco_gt",
    # Dataset converter
    "Annotation",
    "ConvertConfig",
    "DatasetBundle",
    "DatasetFormat",
    "ImageEntry",
    "Split",
    "convert",
    "detect_format",
    # Config
    "AnalysisConfig",
    "EvaluateConfig",
    "FullConfig",
    "OutputConfig",
    "load_config",
]
