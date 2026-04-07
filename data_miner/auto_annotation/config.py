from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


StageKind = Literal["proposal", "consensus", "refinement", "verification", "escalation"]


class ClassPackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    synonyms: list[str] = Field(default_factory=list)
    negatives: list[str] = Field(default_factory=list)
    prompt_variants: list[str] = Field(default_factory=list)

    def names(self) -> list[str]:
        names = [self.name, *self.synonyms]
        return list(dict.fromkeys(name.strip() for name in names if name.strip()))


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    enabled: bool = True
    model_id: str | None = None
    device: str = "auto"
    params: dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: StageKind
    enabled: bool = True
    implementation: str | None = None
    models: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class LimitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_candidates_per_class: int = 12
    max_retry_rounds: int = 1
    auto_accept_quality: float = 0.8
    auto_accept_uncertainty: float = 0.25
    reject_quality: float = 0.5
    reject_uncertainty: float = 0.6


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_labels: bool = True
    save_sidecars: bool = True
    save_review_queue: bool = True
    label_dirname: str = "labels"
    sidecar_dirname: str = "sidecars"
    review_dirname: str = "review"


class AutoAnnotationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ClassPackConfig]
    models: dict[str, ModelConfig]
    stages: list[StageConfig]
    limits: LimitConfig = Field(default_factory=LimitConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_references(self) -> "AutoAnnotationConfig":
        model_names = {name for name, cfg in self.models.items() if cfg.enabled}
        for stage in self.stages:
            missing = [name for name in stage.models if name not in model_names]
            if missing:
                raise ValueError(f"Stage '{stage.name}' references unknown models: {missing}")
        return self


def default_config_path() -> Path:
    return Path(__file__).parent / "config" / "default.yaml"


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AutoAnnotationConfig:
    base = OmegaConf.load(default_config_path())
    if config_path:
        base = OmegaConf.merge(base, OmegaConf.load(Path(config_path)))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    data = OmegaConf.to_container(base, resolve=True)
    return AutoAnnotationConfig.model_validate(data)