"""OmegaConf + Pydantic configuration system for auto_annotation_v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------------


class ClassPackConfig(BaseModel):
    """One target class with synonyms, negatives, and prompt variants."""

    model_config = ConfigDict(extra="forbid")

    name: str
    synonyms: list[str] = Field(default_factory=list)
    negatives: list[str] = Field(default_factory=list)
    prompt_variants: list[str] = Field(default_factory=list)

    def all_names(self) -> list[str]:
        names = [self.name, *self.synonyms]
        return list(dict.fromkeys(n.strip() for n in names if n.strip()))


# ---------------------------------------------------------------------------
# Detection / segmentation model config
# ---------------------------------------------------------------------------


class DetectionModelConfig(BaseModel):
    """Config for a detection/segmentation model (Falcon, DINO, SAM, etc.)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["falcon", "grounding_dino", "sam"]
    enabled: bool = True
    model_id: str | None = None
    device: str = "auto"
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# VLM config
# ---------------------------------------------------------------------------


class VLMConfig(BaseModel):
    """Config for the VLM reasoning model (Qwen3.5, Gemma4, etc.)."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai-compatible"] = "openai-compatible"
    model_name: str = "Qwen/Qwen3.5-27B-FP8"
    base_url: str = "http://localhost:8955/v1"
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 2
    timeout: float = 120.0


# ---------------------------------------------------------------------------
# Filter config
# ---------------------------------------------------------------------------


class FilterConfig(BaseModel):
    """Programmatic bbox filtering thresholds."""

    model_config = ConfigDict(extra="forbid")

    min_area: float = 0.0005
    max_area: float = 0.95
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_edge_distance: float = 0.0
    iou_dedup_threshold: float = 0.7
    max_candidates_per_class: int = 30


# ---------------------------------------------------------------------------
# Reasoning config
# ---------------------------------------------------------------------------


class ReasoningPassConfig(BaseModel):
    """Custom pass/fail criteria for VLM reasoning."""

    model_config = ConfigDict(extra="forbid")

    accept_confidence_threshold: float = 0.75
    reject_confidence_threshold: float = 0.4
    image_inputs: list[Literal["original", "annotated", "crop"]] = Field(
        default_factory=lambda: ["annotated", "crop"]
    )


class ReasoningConfig(BaseModel):
    """VLM reasoning stage configuration."""

    model_config = ConfigDict(extra="forbid")

    screening: ReasoningPassConfig = Field(default_factory=ReasoningPassConfig)
    detailed: ReasoningPassConfig = Field(
        default_factory=lambda: ReasoningPassConfig(
            image_inputs=["original", "annotated", "crop"]
        )
    )
    max_concurrent_calls: int = 4


# ---------------------------------------------------------------------------
# Refinement config
# ---------------------------------------------------------------------------


class RefinementConfig(BaseModel):
    """VLM refinement stage configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    refinement_models: list[str] = Field(default_factory=lambda: ["sam"])
    max_refinement_rounds: int = 1


# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_labels: bool = True
    save_traces: bool = True
    save_review_queue: bool = True
    label_dirname: str = "labels"
    trace_dirname: str = "traces"
    review_dirname: str = "review"


# ---------------------------------------------------------------------------
# Stage enable/disable
# ---------------------------------------------------------------------------


class StageFlags(BaseModel):
    """Which pipeline stages to run."""

    model_config = ConfigDict(extra="forbid")

    proposal: bool = True
    filtering: bool = True
    vlm_reasoning: bool = True
    vlm_refinement: bool = True
    vlm_validation: bool = True
    finalize: bool = True


# ---------------------------------------------------------------------------
# Runtime config (CLI-overridable params)
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Parameters also settable via CLI args. CLI values override YAML."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "output/auto_annotation_v2"
    log_level: str = "INFO"
    image: str | None = None
    image_dir: str | None = None
    force_redo: str | None = None  # comma-separated stage names or "all"


# ---------------------------------------------------------------------------
# Proposal stage config
# ---------------------------------------------------------------------------


class ProposalConfig(BaseModel):
    """Which detection models to use for proposal stage."""

    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(
        default_factory=lambda: ["falcon", "grounding_dino", "sam"]
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AutoAnnotationV2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ClassPackConfig]
    detection_models: dict[str, DetectionModelConfig]
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    proposal: ProposalConfig = Field(default_factory=ProposalConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    stages: StageFlags = Field(default_factory=StageFlags)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_model_references(self) -> AutoAnnotationV2Config:
        enabled_models = {
            name for name, cfg in self.detection_models.items() if cfg.enabled
        }
        for model_name in self.proposal.models:
            if model_name not in enabled_models:
                raise ValueError(
                    f"Proposal references unknown or disabled model: '{model_name}'. "
                    f"Available: {sorted(enabled_models)}"
                )
        for model_name in self.refinement.refinement_models:
            if model_name not in self.detection_models:
                raise ValueError(
                    f"Refinement references unknown model: '{model_name}'. "
                    f"Available: {sorted(self.detection_models)}"
                )
        return self


# ---------------------------------------------------------------------------
# Config loading (OmegaConf merge → Pydantic validation)
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    return Path(__file__).parent / "default.yaml"


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AutoAnnotationV2Config:
    """Load config from default.yaml, optionally merge custom YAML + CLI overrides."""
    base = OmegaConf.load(default_config_path())
    if config_path:
        custom = OmegaConf.load(Path(config_path))
        base = OmegaConf.merge(base, custom)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    data = OmegaConf.to_container(base, resolve=True)
    return AutoAnnotationV2Config.model_validate(data)
