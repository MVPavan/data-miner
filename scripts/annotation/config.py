"""
CVAT + FiftyOne Annotation Configuration.

All settings loaded from YAML with OmegaConf, validated with Pydantic.
Environment variables override YAML values for credentials.
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator


class UnexpectedHandling(str, Enum):
    """How to handle unexpected labels from CVAT."""
    prompt = "prompt"
    ignore = "ignore"
    keep = "keep"


class CVATConfig(BaseModel):
    """CVAT server configuration."""
    url: str = Field("http://localhost:8080", description="CVAT server URL")
    username: Optional[str] = Field(None, description="CVAT username")
    password: Optional[str] = Field(None, description="CVAT password")
    
    @field_validator('url')
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip('/')
    
    @model_validator(mode='after')
    def apply_env_overrides(self) -> 'CVATConfig':
        """Environment variables override YAML."""
        if url := os.getenv("FIFTYONE_CVAT_URL"):
            self.url = url.rstrip('/')
        if user := os.getenv("FIFTYONE_CVAT_USERNAME"):
            self.username = user
        if pwd := os.getenv("FIFTYONE_CVAT_PASSWORD"):
            self.password = pwd
        return self
    
    def validate_credentials(self):
        if not self.username or not self.password:
            raise ValueError("CVAT credentials required. Set in YAML or env vars.")
    
    def to_kwargs(self) -> dict:
        """Get kwargs for FiftyOne annotate/load_annotations."""
        return {"url": self.url, "username": self.username, "password": self.password, "backend": "cvat"}


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    dir: Path = Field(..., description="Path to YOLO dataset")
    name: Optional[str] = Field(None, description="FiftyOne dataset name")
    split: Optional[str] = Field(None, description="Split to use (train/val/test)")
    label_field: str = Field("detections", description="Field for detections")
    
    @field_validator('dir')
    @classmethod
    def validate_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Dataset dir not found: {v}")
        return v


class TaskConfig(BaseModel):
    """CVAT task configuration."""
    project_name: Optional[str] = Field(None, description="CVAT project name")
    task_size: int = Field(100, ge=1, le=10000, description="Images per task")
    segment_size: Optional[int] = Field(None, ge=1, description="Images per job")
    task_assignee: Optional[str] = Field(None, description="Task assignee")
    job_assignees: Optional[List[str]] = Field(None, description="Job assignees")
    classes: Optional[List[str]] = Field(None, description="Class names")


class ExportConfig(BaseModel):
    """Export configuration."""
    output_dir: Path = Field(Path("./output"), description="Output directory")
    split: str = Field("annotated", description="Export split name")
    include_confidence: bool = Field(True, description="Include confidence")
    unexpected: UnexpectedHandling = Field(UnexpectedHandling.keep, description="Handle unexpected labels")
    cleanup: bool = Field(False, description="Delete CVAT tasks after download")


class AnnotationConfig(BaseModel):
    """Complete annotation job configuration."""
    anno_key: str = Field(..., description="Unique annotation run identifier")
    cvat: CVATConfig = Field(default_factory=CVATConfig)
    dataset: DatasetConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    
    @property
    def dataset_name(self) -> str:
        return self.dataset.name or self.anno_key


def load_config(config_path: Path) -> AnnotationConfig:
    """Load and validate configuration from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return AnnotationConfig(**data)


# Sample config template
SAMPLE_CONFIG = """# CVAT Annotation Job Configuration
anno_key: "my_annotation_job"

cvat:
  url: "http://localhost:8080"
  username: ""  # or set FIFTYONE_CVAT_USERNAME
  password: ""  # or set FIFTYONE_CVAT_PASSWORD

dataset:
  dir: "/path/to/yolo/dataset"
  name: null  # defaults to anno_key
  split: "train"  # train/val/test or null for all
  label_field: "detections"

task:
  project_name: null
  task_size: 100
  segment_size: null
  task_assignee: null
  job_assignees: null
  classes: null  # auto-inferred if null

export:
  output_dir: "./output"
  split: "annotated"
  include_confidence: true
  unexpected: "keep"  # keep/ignore/prompt
  cleanup: false
"""


def create_sample_config(path: Path):
    """Create sample config file."""
    path.write_text(SAMPLE_CONFIG)
    print(f"Created: {path}")
