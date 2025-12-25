"""Pydantic config models and OmegaConf loader for detection_metrics."""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from omegaconf import OmegaConf


# Path to default config
DEFAULTS_PATH = Path(__file__).parent / "default.yaml"


# --- Pydantic Models ---

class PredictionEntry(BaseModel):
    """Single prediction file entry."""
    path: Path
    name: str

    @field_validator("path", mode="before")
    @classmethod
    def convert_path(cls, v):
        if v is None:
            return v
        return Path(v)


class DatasetConfig(BaseModel):
    """Dataset configuration for evaluation."""
    gt_path: Optional[Path] = None
    predictions: List[PredictionEntry] = Field(default_factory=list)

    @field_validator("gt_path", mode="before")
    @classmethod
    def convert_gt_path(cls, v):
        if v is None:
            return v
        return Path(v)

    @field_validator("predictions", mode="before")
    @classmethod
    def convert_predictions(cls, v):
        if v is None:
            return []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(PredictionEntry(**item))
            elif isinstance(item, PredictionEntry):
                result.append(item)
        return result


class PredictConfig(BaseModel):
    """Configuration for prediction generation."""
    model: Optional[str] = None
    checkpoint: Optional[Path] = None
    images: Optional[Path] = None
    output: Path = Path("./predictions.json")
    threshold: float = 0.001
    device: str = "cuda"

    @field_validator("checkpoint", "images", "output", mode="before")
    @classmethod
    def convert_paths(cls, v):
        if v is None:
            return v
        return Path(v)


class EvaluateConfig(BaseModel):
    """Configuration for evaluation."""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    iou_threshold: float = 0.5
    conf_threshold: float = 0.001
    classes: List[int] = Field(default_factory=list)

    @field_validator("dataset", mode="before")
    @classmethod
    def convert_dataset(cls, v):
        if v is None:
            return DatasetConfig()
        if isinstance(v, dict):
            return DatasetConfig(**v)
        return v


class AnalysisConfig(BaseModel):
    """Configuration for analysis."""
    precision_targets: List[float] = Field(default_factory=lambda: [0.8, 0.85, 0.9])
    tpfp_conf_thresholds: List[float] = Field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6])


class OutputConfig(BaseModel):
    """Configuration for output."""
    path: Path = Path("./results")
    overwrite: bool = False
    use_cache: bool = True
    log_file: Optional[Path] = None

    @field_validator("path", "log_file", mode="before")
    @classmethod
    def convert_paths(cls, v):
        if v is None:
            return v
        return Path(v)


class FullConfig(BaseModel):
    """Root configuration containing all sub-configs."""
    predict: Optional[PredictConfig] = None
    evaluate: Optional[EvaluateConfig] = None
    analyze: Optional[AnalysisConfig] = None
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("predict", mode="before")
    @classmethod
    def convert_predict(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return PredictConfig(**v)
        return v

    @field_validator("evaluate", mode="before")
    @classmethod
    def convert_evaluate(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return EvaluateConfig(**v)
        return v

    @field_validator("analyze", mode="before")
    @classmethod
    def convert_analyze(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return AnalysisConfig(**v)
        return v

    @field_validator("output", mode="before")
    @classmethod
    def convert_output(cls, v):
        if v is None:
            return OutputConfig()
        if isinstance(v, dict):
            return OutputConfig(**v)
        return v


# --- Config Loader ---

def load_config(user_config_path: Optional[Path] = None) -> FullConfig:
    """
    Load and validate config: OmegaConf merge → Pydantic validation.
    
    Args:
        user_config_path: Optional path to user's YAML config file.
                         User only needs to specify overrides.
    
    Returns:
        FullConfig: Validated configuration object.
    """
    # Load defaults
    default_cfg = OmegaConf.load(DEFAULTS_PATH)
    
    # Merge with user config if provided
    if user_config_path:
        user_cfg = OmegaConf.load(user_config_path)
        merged = OmegaConf.merge(default_cfg, user_cfg)
    else:
        merged = default_cfg
    
    # Convert to dict and validate with Pydantic
    config_dict = OmegaConf.to_container(merged, resolve=True)
    return FullConfig.model_validate(config_dict)


def load_config_from_dict(config_dict: dict) -> FullConfig:
    """
    Load config from a dictionary (useful for CLI args).
    
    Args:
        config_dict: Configuration dictionary.
    
    Returns:
        FullConfig: Validated configuration object.
    """
    # Load defaults
    default_cfg = OmegaConf.load(DEFAULTS_PATH)
    user_cfg = OmegaConf.create(config_dict)
    merged = OmegaConf.merge(default_cfg, user_cfg)
    
    config_dict = OmegaConf.to_container(merged, resolve=True)
    return FullConfig.model_validate(config_dict)
