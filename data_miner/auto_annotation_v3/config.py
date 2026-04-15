"""Pydantic + OmegaConf configuration system for auto_annotation_v3.

Loads `configs/default.yaml` as the base config and deep-merges user overrides
(from a user-provided YAML file and/or CLI dotlist) via OmegaConf. The merged
dict is then validated by Pydantic for type safety.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import DetectorName

# ---------------------------------------------------------------------------
# Model server configs
# ---------------------------------------------------------------------------


class DetectorConfig(BaseModel):
    """Config for one LitServe detector server.

    Keyed by :class:`DetectorName` in ``servers.detectors``. The ``script``
    field is read by ``launch_all.py`` and is the only knowledge the launcher
    needs about a detector — no separate ``serve_config.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    port: int
    gpu: str
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    model_id: str
    script: str   # e.g. "serve_gdino.py" — relative to servers/ dir


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
    """All model server endpoints (detectors + VLM).

    Detectors are keyed by :class:`DetectorName` — adding a new detector is a
    YAML entry whose key matches the enum value. Unknown keys raise at
    config-load time (not at runtime).
    """

    model_config = ConfigDict(extra="forbid")

    detectors: dict[DetectorName, DetectorConfig] = Field(default_factory=dict)
    vlm: VLMConfig = Field(default_factory=VLMConfig)

    @field_validator("detectors", mode="before")
    @classmethod
    def _validate_detector_keys(cls, v):
        if not isinstance(v, dict):
            return v
        valid = {m.value for m in DetectorName}
        for key in v.keys():
            key_s = key.value if hasattr(key, "value") else str(key)
            if key_s not in valid:
                raise ValueError(
                    f"Unknown detector '{key_s}'. Must be one of: {sorted(valid)}. "
                    f"Add a new entry to DetectorName in contracts.py if this is new."
                )
        return v

    def enabled_detectors(self) -> dict[DetectorName, DetectorConfig]:
        return {n: c for n, c in self.detectors.items() if c.enabled}


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
    min_score: float = 0.0
    """Global score floor; usually 0 because per-model floors are incomparable.
    Used as a fallback when ``per_model_score`` does not list a model.
    """
    per_model_score: dict[str, float] = Field(default_factory=dict)
    """Per-detector score floor. Keys are ``source_model`` strings as set by the
    detector workers (e.g. ``grounding_dino``, ``owlvit2``, ``sam3``, ``falcon``).
    A candidate qualifies for auto-accept only if its score >= the floor for
    its own model. Models absent from the map fall back to ``min_score``.
    """
    tiers: list[int] = Field(default_factory=lambda: [1])
    """Tiers eligible for auto-accept. Empty list disables auto-accept entirely.

    Example: ``tiers: [1]`` → only tier-1 classes auto-accept.
    """


# ---------------------------------------------------------------------------
# Evaluation groups
# ---------------------------------------------------------------------------


class EvaluationGroupConfig(BaseModel):
    """Classes grouped together for a single VLM classification call."""

    model_config = ConfigDict(extra="forbid")

    classes: list[str]
    requires_crops: bool = False
    """When True, per-candidate evaluate sends a close-up crop in addition to
    the bbox-highlighted overview. Default off — overview alone is enough for
    most groups; turn on only where small detail matters (luggage, electronics).
    """
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


class EvaluateConfig(BaseModel):
    """Confidence thresholds for the evaluate stage's three-way routing.

    Replaces the hard-coded 0.3 / 0.5 cascade in the old verdict resolver.
    Pure confidence-based — bbox_quality / object_complete are no longer
    routing inputs (they remain on the verdict for telemetry only).
    """

    model_config = ConfigDict(extra="forbid")

    reject_below: float = 0.3
    accept_above: float = 0.5
    concurrency: int = 8
    """Max in-flight per-candidate VLM requests per worker. Bounds vLLM
    scheduler load; vLLM does its own continuous batching across the
    in-flight requests so each one runs as a small, isolated forward pass.
    """


class MergeRulesConfig(BaseModel):
    """Geometric sanity gates applied after a refine prompt produces a merged bbox."""

    model_config = ConfigDict(extra="forbid")

    max_area_ratio: float = 3.0
    """Hard cap on (merged_area / original_area)."""
    max_gap_diag_frac: float = 0.02
    """Maximum allowed gap (as fraction of image diagonal) between original
    bbox and proposed load bbox before merge is considered geometrically
    implausible. Overlapping (IoU > 0) always passes the gap gate.
    """
    aspect_ratio_range: list[float] = Field(default_factory=lambda: [0.25, 4.0])


class RefinePromptConfig(BaseModel):
    """One prompt for the per-class refine inner loop."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    """Free-form rule shown to the VLM. Should explicitly say when to
    ``action: skip`` (no extension needed) vs ``action: propose`` (return
    a target_region + load_vocab).
    """
    load_vocab: list[str] = Field(default_factory=list)
    """Default load-vocabulary noun phrases queried against the SAM3
    presence head on the proposed load bbox. The VLM may override per-call.
    """
    presence_threshold: float = 0.5


class RefineRuleConfig(BaseModel):
    """Per-class refine rule: one or more sequential prompts + merge sanity."""

    model_config = ConfigDict(extra="forbid")

    prompts: list[RefinePromptConfig] = Field(default_factory=list)
    merge_rules: MergeRulesConfig = Field(default_factory=MergeRulesConfig)


class RefineRulesConfig(BaseModel):
    """Top-level container — keys are class names, values are per-class rules.

    A candidate enters the refine stage iff its (post-relabel) ``class_name``
    is a key in this map AND its evaluate verdict is not ``reject``. Pure
    class-match trigger; no `strategy` enum, no VLM-signal driven routing.
    """

    model_config = ConfigDict(extra="forbid")

    classes: dict[str, RefineRuleConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filter config (ported from v2)
# ---------------------------------------------------------------------------


class IouDedupConfig(BaseModel):
    """IoU-based clustering + tiebreak cascade for cross-model dedup.

    Replaces the old flat ``iou_dedup_threshold`` float. Used by
    ``utils.cluster_and_collapse`` to build per-class IoU clusters across all
    detector models, attach agreement metadata, and pick one representative
    per cluster via the ``tiebreak_by`` cascade.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.7
    tiebreak_by: list[Literal["agreement", "model_priority", "score"]] = Field(
        default_factory=lambda: ["agreement", "model_priority", "score"]
    )
    """Discriminator cascade applied in order; first one that distinguishes
    the cluster members picks the survivor.
    """
    model_priority: list[str] = Field(
        default_factory=lambda: ["sam3", "falcon", "grounding_dino"]
    )
    """Model order from most-trusted (lowest index) to least. Earlier wins
    when ``model_priority`` is the active discriminator.
    """


class FilterConfig(BaseModel):
    """Programmatic bounding-box filtering thresholds."""

    model_config = ConfigDict(extra="forbid")

    min_area: float = 0.0005
    max_area: float = 0.95
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_edge_distance: float = 0.0
    iou_dedup: IouDedupConfig = Field(default_factory=IouDedupConfig)
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
            "finalize": "stream:finalize",
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
    finalize_count: int = 2


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
# Runtime config (per-job inputs formerly passed as CLI flags)
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Per-job runtime inputs (input source, job id, logging).

    These fields typically change from run to run and are provided by the user
    via their YAML config or as CLI dotlist overrides.
    """

    model_config = ConfigDict(extra="forbid")

    image_dir: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    job_id: str | None = None
    log_level: str = "INFO"
    log_file: str | None = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AutoAnnotationV3Config(BaseModel):
    """Full pipeline configuration.

    Class selection model:
        - ``class_registry`` — the full catalog of known classes (id, name,
          tier, prompt). Lives in ``configs/default.yaml``; rarely changed.
        - ``detect_classes`` — a per-job list of class names selecting which
          subset of the registry to actually run. Empty list → all classes.

    Consumers should use the ``.classes`` property to iterate the active
    subset (it's automatically filtered against ``detect_classes``).
    """

    model_config = ConfigDict(extra="forbid")

    servers: ServersConfig
    class_registry: list[ClassConfig] = Field(default_factory=list)
    detect_classes: list[str] = Field(default_factory=list)
    auto_accept: AutoAcceptConfig = Field(default_factory=AutoAcceptConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    evaluation_groups: dict[str, EvaluationGroupConfig] = Field(default_factory=dict)
    co_existence: CoExistenceConfig = Field(default_factory=CoExistenceConfig)
    refine_rules: RefineRulesConfig = Field(default_factory=RefineRulesConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompts_dir: str = "prompts"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # ------------------------------------------------------------------
    # Active class/group filtering (computed at access time)
    # ------------------------------------------------------------------

    @property
    def classes(self) -> list[ClassConfig]:
        """Active classes — filtered to ``detect_classes`` if non-empty."""
        if not self.detect_classes:
            return self.class_registry
        keep = set(self.detect_classes)
        return [c for c in self.class_registry if c.name in keep]

    @property
    def active_evaluation_groups(self) -> dict[str, EvaluationGroupConfig]:
        """Evaluation groups with each group's ``classes`` list filtered to
        the active class set. Groups that would end up empty are dropped.
        """
        active_names = {c.name for c in self.classes}
        out: dict[str, EvaluationGroupConfig] = {}
        for name, grp in self.evaluation_groups.items():
            kept = [c for c in grp.classes if c in active_names]
            if kept:
                out[name] = grp.model_copy(update={"classes": kept})
        return out

    def class_by_name(self, name: str) -> ClassConfig | None:
        for c in self.class_registry:
            if c.name == name:
                return c
        return None

    def class_by_id(self, class_id: int) -> ClassConfig | None:
        for c in self.class_registry:
            if c.id == class_id:
                return c
        return None

    def tier_names(self, tier: int) -> list[str]:
        """Names of active classes at a given tier."""
        return [c.name for c in self.classes if c.tier == tier]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    """Path to the packaged default config."""
    return Path(__file__).parent / "configs" / "default.yaml"


def load_config(
    user_config: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AutoAnnotationV3Config:
    """Load config via OmegaConf: defaults → user YAML → CLI dotlist.

    Args:
        user_config: Path to a user YAML that overrides individual keys. Only
            keys present in this file are overridden; everything else comes
            from ``configs/default.yaml``. Deep-merged via OmegaConf.
        overrides: List of OmegaConf dotlist strings (e.g.
            ``["runtime.log_level=DEBUG", "workers.detect_count=2"]``).

    Returns:
        Validated :class:`AutoAnnotationV3Config` instance.

    Example:
        >>> cfg = load_config("my_job.yaml",
        ...                    overrides=["runtime.log_level=DEBUG"])
    """
    base: DictConfig = OmegaConf.load(default_config_path())

    if user_config is not None:
        user_cfg: DictConfig = OmegaConf.load(Path(user_config))
        base = OmegaConf.merge(base, user_cfg)  # type: ignore[assignment]

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        base = OmegaConf.merge(base, cli_cfg)  # type: ignore[assignment]

    # Resolve any interpolations, then convert to plain dict for Pydantic.
    OmegaConf.resolve(base)
    data: dict[str, Any] = OmegaConf.to_container(base, resolve=True)  # type: ignore[assignment]

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
