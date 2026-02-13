"""
CVAT + FiftyOne Annotation Configuration.

All settings loaded from YAML with OmegaConf, validated with Pydantic.
Environment variables override YAML values for credentials.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
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

    @field_validator("url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "CVATConfig":
        """Environment variables override YAML."""
        if url := os.getenv("FIFTYONE_CVAT_URL"):
            self.url = url.rstrip("/")
        if user := os.getenv("FIFTYONE_CVAT_USERNAME"):
            self.username = user
        if pwd := os.getenv("FIFTYONE_CVAT_PASSWORD"):
            self.password = pwd
        return self

    def validate_credentials(self):
        if not self.username or not self.password:
            raise ValueError("CVAT credentials required. Set in YAML or env vars.")

    def test_connection(self, timeout: int = 10) -> bool:
        """Test CVAT server connectivity.

        Returns True if connection successful, raises ValueError otherwise.
        """
        try:
            response = requests.get(
                f"{self.url}/api/server/about",
                auth=(self.username, self.password),
                timeout=timeout,
            )
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                raise ValueError("CVAT authentication failed. Check username/password.")
            else:
                raise ValueError(f"CVAT server error: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Cannot connect to CVAT at {self.url}. Is the server running?"
            )
        except requests.exceptions.Timeout:
            raise ValueError(f"CVAT connection timeout after {timeout}s")

    def to_kwargs(self) -> dict:
        """Get kwargs for FiftyOne annotate() - includes backend."""
        return {
            "url": self.url,
            "username": self.username,
            "password": self.password,
            "backend": "cvat",
        }

    def to_credentials(self) -> dict:
        """Get credentials only for load_annotation_results() - no backend."""
        return {
            "url": self.url,
            "username": self.username,
            "password": self.password,
        }


class DatasetConfig(BaseModel):
    """Dataset configuration (follows remote_view.py pattern)."""

    name: str = Field(..., description="FiftyOne dataset name")
    images_dir: Path = Field(..., description="Path to images directory")
    labels_dir: Optional[Path] = Field(None, description="Path to YOLO labels directory")
    label_field: str = Field("ground_truth", description="Field for detections")
    overwrite: bool = Field(False, description="Overwrite existing dataset")

    @field_validator("images_dir")
    @classmethod
    def validate_images_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Images dir not found: {v}")
        return v

    @field_validator("labels_dir")
    @classmethod
    def validate_labels_dir(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"Labels dir not found: {v}")
        return v


class QualityMapping(BaseModel):
    """Map confidence scores to quality attributes for CVAT."""

    enabled: bool = Field(False, description="Enable confidence-to-quality mapping")
    attribute_name: str = Field("quality", description="Attribute name in CVAT")
    
    # Threshold values (upper bounds, exclusive)
    # Default: bad<0.4, partial<0.6, loose<0.8, good>=0.8
    bad_threshold: float = Field(0.4, ge=0.0, le=1.0, description="Below this = bad")
    partial_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Below this = partial")
    loose_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Below this = loose")
    # >= loose_threshold = good
    
    # Attribute values
    values: List[str] = Field(
        ["bad", "partial", "loose", "good"],
        description="Quality values (ordered low to high)",
    )
    default: str = Field("good", description="Default value when no confidence")

    def confidence_to_quality(self, confidence: Optional[float]) -> str:
        """Map confidence score to quality attribute value."""
        if confidence is None:
            return self.default
        if confidence < self.bad_threshold:
            return self.values[0]  # bad
        elif confidence < self.partial_threshold:
            return self.values[1]  # partial
        elif confidence < self.loose_threshold:
            return self.values[2]  # loose
        else:
            return self.values[3]  # good

    def get_attribute_schema(self) -> dict:
        """Get CVAT attribute schema for this quality mapping."""
        return {
            self.attribute_name: {
                "type": "select",
                "values": self.values,
                "default": self.default,
            }
        }


class AnnotateConfig(BaseModel):
    """Configuration for FiftyOne's annotate() method."""

    project_name: Optional[str] = Field(None, description="CVAT project name")
    task_size: Optional[int] = Field(None, ge=1, le=10000, description="Images per task")
    segment_size: Optional[int] = Field(None, ge=1, description="Images per job")
    task_assignee: Optional[str] = Field(None, description="Task assignee")
    job_assignees: Optional[List[str]] = Field(None, description="Job assignees")
    classes: Optional[List[str]] = Field(None, description="Class names (simple mode)")

    # Full label schema for advanced use (pass-through to FiftyOne)
    # See: https://docs.voxel51.com/integrations/cvat.html#label-schema
    label_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Full label schema dict with types, classes, attributes per field",
    )

    # Quality attribute mapping from confidence
    quality_mapping: QualityMapping = Field(
        default_factory=QualityMapping,
        description="Map confidence to quality attribute",
    )

    # Restriction flags
    allow_additions: bool = Field(True, description="Allow new labels")
    allow_deletions: bool = Field(True, description="Allow label deletion")
    allow_label_edits: bool = Field(True, description="Allow label class changes")
    allow_spatial_edits: bool = Field(True, description="Allow bbox/polygon changes")

    # CVAT-specific settings
    image_quality: int = Field(75, ge=0, le=100, description="Upload image quality")
    use_cache: bool = Field(True, description="Use cache for faster upload")

    def to_annotate_kwargs(
        self, cvat: CVATConfig, label_field: str, launch: bool = False
    ) -> dict:
        """Build validated kwargs dict for dataset.annotate()."""
        kwargs = cvat.to_kwargs()
        kwargs.update(
            {
                "launch_editor": launch,
                "image_quality": self.image_quality,
                "use_cache": self.use_cache,
                "allow_additions": self.allow_additions,
                "allow_deletions": self.allow_deletions,
                "allow_label_edits": self.allow_label_edits,
                "allow_spatial_edits": self.allow_spatial_edits,
            }
        )

        # Use label_schema if provided (advanced), else simple mode
        if self.label_schema:
            kwargs["label_schema"] = self.label_schema
        else:
            kwargs["label_field"] = label_field
            # When quality mapping is enabled, use quality values as classes
            if self.quality_mapping.enabled:
                kwargs["classes"] = self.quality_mapping.values
            elif self.classes:
                kwargs["classes"] = self.classes

        # Optional settings
        if self.project_name:
            kwargs["project_name"] = self.project_name
        if self.task_size:
            kwargs["task_size"] = self.task_size
        if self.segment_size:
            kwargs["segment_size"] = self.segment_size
        if self.task_assignee:
            kwargs["task_assignee"] = self.task_assignee
        if self.job_assignees:
            kwargs["job_assignees"] = self.job_assignees

        return kwargs


class ExportConfig(BaseModel):
    """Export configuration."""

    output_dir: Path = Field(Path("./output"), description="Output directory")
    split: str = Field("annotated", description="Export split name")
    include_confidence: bool = Field(True, description="Include confidence")
    unexpected: UnexpectedHandling = Field(
        UnexpectedHandling.keep, description="Handle unexpected labels"
    )
    cleanup: bool = Field(False, description="Delete CVAT tasks after download")


class AnnotationConfig(BaseModel):
    """Complete annotation job configuration."""

    anno_key: str = Field(..., description="Unique annotation run identifier")
    cvat: CVATConfig = Field(default_factory=CVATConfig)
    dataset: DatasetConfig
    annotate: AnnotateConfig = Field(default_factory=AnnotateConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


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
  username: ""  # or set FIFTYONE_CVAT_USERNAME env var
  password: ""  # or set FIFTYONE_CVAT_PASSWORD env var

dataset:
  name: "my_dataset"
  images_dir: "/path/to/images"
  labels_dir: "/path/to/yolo/labels"  # optional, null for images-only
  label_field: "ground_truth"
  overwrite: false

annotate:
  project_name: null
  task_size: 100
  segment_size: null
  task_assignee: null
  job_assignees: null
  classes: null  # auto-inferred if null (simple mode)
  
  # Full label schema (advanced mode) - overrides 'classes' if provided
  # See: https://docs.voxel51.com/integrations/cvat.html#label-schema
  label_schema: null
  
  # Restriction flags (for editing existing labels)
  allow_additions: true
  allow_deletions: true
  allow_label_edits: true
  allow_spatial_edits: true
  
  # CVAT upload settings
  image_quality: 75
  use_cache: true

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
