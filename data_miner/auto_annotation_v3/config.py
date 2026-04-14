"""Pydantic + YAML configuration system for auto_annotation_v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Model server configs
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """Config for one LitServe model server (GDINO, Falcon, SAM3, OWLv2)."""

    model_config = ConfigDict(extra="forbid")

    port: int
    gpu: str
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    model_id: str


class VLMConfig(BaseModel):
    """Config for the vLLM-hosted VLM (Qwen3.5 etc.)."""

    model_config = ConfigDict(extra="forbid")

    url: str = "http://localhost:8955/v1"
    model: str = "Qwen/Qwen3.5-27B-FP8"
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 2
    timeout: float = 120.0


class ServersConfig(BaseModel):
    """All model server endpoints."""

    model_config = ConfigDict(extra="forbid")

    grounding_dino: ServerConfig
    falcon: ServerConfig
    sam3: ServerConfig
    owlvit2: ServerConfig
    vlm: VLMConfig = Field(default_factory=VLMConfig)


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------


class ClassConfig(BaseModel):
    """One target class with its COCO-style id, tier, and detection prompt."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    tier: int = Field(ge=1, le=4)
    prompt: str


# ---------------------------------------------------------------------------
# Auto-accept rules
# ---------------------------------------------------------------------------


class AutoAcceptConfig(BaseModel):
    """Conditions under which candidates are auto-accepted without VLM evaluation."""

    model_config = ConfigDict(extra="forbid")

    min_model_agreement: int = 2
    min_score: float = 0.3
    applies_to: str = "tier_1_only"


# ---------------------------------------------------------------------------
# Evaluation groups
# ---------------------------------------------------------------------------


class EvaluationGroupConfig(BaseModel):
    """Classes grouped together for a single VLM classification call."""

    model_config = ConfigDict(extra="forbid")

    classes: list[str]
    requires_crops: bool = True
    description: str | None = None
    annotation_rules: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Co-existence rules
# ---------------------------------------------------------------------------


class CoExistenceConfig(BaseModel):
    """Rules governing which classes can/cannot overlap."""

    model_config = ConfigDict(extra="forbid")

    globally_exempt: list[str] = Field(default_factory=list)
    confusion_pairs: list[list[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Refinement config
# ---------------------------------------------------------------------------


class RefinementConfig(BaseModel):
    """SAM refinement thresholds and per-class strategies."""

    model_config = ConfigDict(extra="forbid")

    class_rules: dict[str, dict[str, Any]] = Field(default_factory=dict)
    auto_accept_iou: float = 0.3
    reject_iou: float = 0.1


# ---------------------------------------------------------------------------
# Filter config (ported from v2)
# ---------------------------------------------------------------------------


class FilterConfig(BaseModel):
    """Programmatic bounding-box filtering thresholds."""

    model_config = ConfigDict(extra="forbid")

    min_area: float = 0.0005
    max_area: float = 0.95
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_edge_distance: float = 0.0
    iou_dedup_threshold: float = 0.7
    max_per_class: int = 30


# ---------------------------------------------------------------------------
# Redis config
# ---------------------------------------------------------------------------


class RedisConfig(BaseModel):
    """Redis connection and stream topology."""

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    consumer_group: str = "annotation_v3"
    streams: dict[str, str] = Field(
        default_factory=lambda: {
            "detect": "stream:detect",
            "evaluate": "stream:evaluate",
            "refine": "stream:refine",
            "done": "stream:done",
        }
    )


# ---------------------------------------------------------------------------
# Worker counts
# ---------------------------------------------------------------------------


class WorkersConfig(BaseModel):
    """Number of concurrent worker processes per stage."""

    model_config = ConfigDict(extra="forbid")

    detect_count: int = 4
    evaluate_count: int = 6
    refine_count: int = 2


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    """Root output directory structure for one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    job_dir: str = "output/auto_annotation_v3"
    checkpoint_dir: str = "checkpoints"
    labels_dir: str = "labels"
    traces_dir: str = "traces"
    review_dir: str = "review"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AutoAnnotationV3Config(BaseModel):
    """Full pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    servers: ServersConfig
    classes: list[ClassConfig]
    auto_accept: AutoAcceptConfig = Field(default_factory=AutoAcceptConfig)
    evaluation_groups: dict[str, EvaluationGroupConfig] = Field(default_factory=dict)
    co_existence: CoExistenceConfig = Field(default_factory=CoExistenceConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompts_dir: str = "prompts"

    def class_by_name(self, name: str) -> ClassConfig | None:
        """Look up a class by its name."""
        for c in self.classes:
            if c.name == name:
                return c
        return None

    def class_by_id(self, class_id: int) -> ClassConfig | None:
        """Look up a class by its integer id."""
        for c in self.classes:
            if c.id == class_id:
                return c
        return None

    def tier_1_names(self) -> list[str]:
        return [c.name for c in self.classes if c.tier == 1]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    return Path(__file__).parent / "default.yaml"


def load_config(path: str | Path | None = None) -> AutoAnnotationV3Config:
    """Load config from default.yaml, optionally merged with a custom YAML file.

    The custom file is shallow-merged at the top level; nested keys override
    only the keys they specify (lists replace entirely, dicts merge one level).
    """
    base_path = default_config_path()
    with base_path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if path is not None:
        with Path(path).open() as f:
            overrides: dict[str, Any] = yaml.safe_load(f) or {}
        # Deep-merge one level: top-level dicts are merged, everything else replaced.
        for key, value in overrides.items():
            if (
                isinstance(value, dict)
                and isinstance(data.get(key), dict)
            ):
                data[key] = {**data[key], **value}
            else:
                data[key] = value

    return AutoAnnotationV3Config.model_validate(data)


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------


def compute_config_hash(config: AutoAnnotationV3Config, prompts_dir: str | Path) -> str:
    """Stable hash of the full config + all active prompt files.

    Used to detect config/prompt changes between runs so downstream stages
    can be invalidated selectively.
    """
    hasher = hashlib.sha256()

    # Hash the serialised config (excluding the hash itself, obviously).
    config_json = config.model_dump_json(indent=None)
    hasher.update(config_json.encode())

    # Hash every prompt file under the active version directory.
    prompts_root = Path(prompts_dir)
    active_link = prompts_root / "active"
    if active_link.is_symlink() or active_link.is_dir():
        active_dir = active_link.resolve()
        for prompt_file in sorted(active_dir.rglob("*.yaml")):
            hasher.update(prompt_file.read_bytes())
    else:
        # Fall back: hash all yaml files under prompts_dir.
        for prompt_file in sorted(prompts_root.rglob("*.yaml")):
            hasher.update(prompt_file.read_bytes())

    return hasher.hexdigest()[:16]
