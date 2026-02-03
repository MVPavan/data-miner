"""Detection Metrics - Unified object detection evaluation toolkit."""

from detection_metrics.evaluator import DetailedEvaluator, PyCocoEvaluator
from detection_metrics.data_loader import DataLoader
from detection_metrics.configs.config import (
    FullConfig,
    EvaluateConfig,
    PredictConfig,
    AnalysisConfig,
    OutputConfig,
    load_config,
)

__version__ = "0.1.0"

__all__ = [
    "DetailedEvaluator",
    "PyCocoEvaluator",
    "DataLoader",
    "FullConfig",
    "EvaluateConfig",
    "PredictConfig",
    "AnalysisConfig",
    "OutputConfig",
    "load_config",
]
