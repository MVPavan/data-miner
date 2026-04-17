"""Pydantic configuration models — mirrors YAML config structure."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import ClassTier, DetectorName, Stage, STAGE_ORDER, TiebreakField

__all__ = [
    "DetectorConfig",
    "VLMConfig",
    "ServersConfig",
    "ClassConfig",
    "AutoAcceptConfig",
    "EvaluateConfig",
    "EvaluationGroupConfig",
    "CoExistenceConfig",
    "MergeRulesConfig",
    "RefinePromptConfig",
    "RefineRuleConfig",
    "RefineRulesConfig",
    "IouDedupConfig",
    "FilterConfig",
    "DatabaseConfig",
    "WorkersConfig",
    "OutputConfig",
    "RuntimeConfig",
    "AutoAnnotationV4Config",
]


# ---------------------------------------------------------------------------
# Model server configs
# ---------------------------------------------------------------------------


class DetectorConfig(BaseModel):
    """Config for one LitServe detector server.

    Keyed by :class:`DetectorName` in ``servers.detectors``. The ``script``
    field is read by the launcher and is the only knowledge it needs about a
    detector — no separate ``serve_config.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    port: int
    gpu: str
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    model_id: str
    script: str  # e.g. "serve_gdino.py" — relative to servers/ dir
    options: dict[str, Any] = Field(default_factory=dict)
    """Model-specific extras forwarded to the server as ``--key value`` CLI
    args by the launcher. Keep the keys flat and value types simple
    (bool/int/float/str).
    """


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
    def _validate_detector_keys(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        valid = {m.value for m in DetectorName}
        for key in v:
            key_s = key.value if hasattr(key, "value") else str(key)
            if key_s not in valid:
                raise ValueError(
                    f"Unknown detector '{key_s}'. Must be one of: {sorted(valid)}. "
                    f"Add a new entry to DetectorName in enums.py if this is new."
                )
        return v

    def enabled_detectors(self) -> dict[DetectorName, DetectorConfig]:
        """Return only the detectors whose ``enabled`` flag is True."""
        return {n: c for n, c in self.detectors.items() if c.enabled}


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------


class ClassConfig(BaseModel):
    """One target class with its COCO-style id, tier, prompts, and metadata.

    In v4 the class name is the dict key in ``class_registry``, not a field
    on this model.  ``prompts`` is a list (multiple prompts per class).
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    tier: ClassTier = Field(ge=1, le=3)
    prompts: list[str]
    synonyms: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Auto-accept rules
# ---------------------------------------------------------------------------


class AutoAcceptConfig(BaseModel):
    """Conditions under which candidates are auto-accepted without VLM evaluation."""

    model_config = ConfigDict(extra="forbid")

    min_model_agreement: int = 2
    min_score: float = 0.0
    """Global score floor used as routing fallback. Per-model floors live in
    ``filtering.per_model_score`` and are applied earlier.
    """
    tiers: list[int] = Field(default_factory=lambda: [1])
    """Tiers eligible for auto-accept. Empty list disables auto-accept entirely."""


# ---------------------------------------------------------------------------
# Evaluate stage
# ---------------------------------------------------------------------------


class EvaluateConfig(BaseModel):
    """Confidence thresholds for the evaluate stage's three-way routing.

    Pure confidence-based — reject_below / accept_above define the tri-state
    boundaries.
    """

    model_config = ConfigDict(extra="forbid")

    reject_below: float = 0.3
    accept_above: float = 0.5
    concurrency: int = 8
    """Max in-flight per-candidate VLM requests per worker."""


# ---------------------------------------------------------------------------
# Evaluation groups
# ---------------------------------------------------------------------------


class EvaluationGroupConfig(BaseModel):
    """Classes grouped together for a single VLM classification call.

    In v4, ``description`` is renamed to ``disambiguation`` to better convey
    its purpose as inter-class distinction hints.
    """

    model_config = ConfigDict(extra="forbid")

    classes: list[str]
    requires_crops: bool = False
    """When True, per-candidate evaluate sends a close-up crop in addition to
    the bbox-highlighted overview.
    """
    disambiguation: str | None = None
    annotation_rules: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Co-existence rules
# ---------------------------------------------------------------------------


class CoExistenceConfig(BaseModel):
    """Tag-driven rules governing which classes can/cannot overlap.

    v4 replaces v3's ``globally_exempt`` / ``confusion_pairs`` with a
    tag-based system: ``overlap_exempt_tags`` / ``confusion_tags`` plus
    ``extra_confusion_pairs`` for manual overrides.
    """

    model_config = ConfigDict(extra="forbid")

    overlap_exempt_tags: list[str] = Field(default_factory=list)
    confusion_tags: list[str] = Field(default_factory=list)
    extra_confusion_pairs: list[list[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Refinement config
# ---------------------------------------------------------------------------


class MergeRulesConfig(BaseModel):
    """Geometric sanity gates applied after a refine prompt produces a merged bbox."""

    model_config = ConfigDict(extra="forbid")

    max_area_ratio: float = 3.0
    """Hard cap on (merged_area / original_area)."""
    max_gap_diag_frac: float = 0.02
    """Maximum allowed gap (as fraction of image diagonal) between original
    bbox and proposed load bbox before merge is considered geometrically
    implausible.
    """
    aspect_ratio_range: list[float] = Field(default_factory=lambda: [0.25, 4.0])


class RefinePromptConfig(BaseModel):
    """One prompt for the per-class refine inner loop."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    """Free-form rule shown to the VLM. Should explicitly say when to
    ``action: skip`` vs ``action: propose``.
    """
    load_vocab: list[str] = Field(default_factory=list)
    """Default load-vocabulary noun phrases queried against the SAM3
    presence head on the proposed load bbox.
    """
    presence_threshold: float = 0.5


class RefineRuleConfig(BaseModel):
    """Per-class refine rule: one or more sequential prompts + merge sanity."""

    model_config = ConfigDict(extra="forbid")

    prompts: list[RefinePromptConfig] = Field(default_factory=list)
    merge_rules: MergeRulesConfig = Field(default_factory=MergeRulesConfig)


class RefineRulesConfig(BaseModel):
    """Top-level container — keys are class names, values are per-class rules.

    A candidate enters the refine stage iff its ``class_name`` is a key in
    this map AND its evaluate verdict is not ``reject``.
    """

    model_config = ConfigDict(extra="forbid")

    classes: dict[str, RefineRuleConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filter config
# ---------------------------------------------------------------------------


class IouDedupConfig(BaseModel):
    """IoU-based clustering + tiebreak cascade for cross-model dedup.

    Used to build per-class IoU clusters across all detector models, attach
    agreement metadata, and pick one representative per cluster via the
    ``tiebreak_by`` cascade.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.7
    tiebreak_by: list[TiebreakField] = Field(
        default_factory=lambda: [
            TiebreakField.AGREEMENT,
            TiebreakField.MODEL_PRIORITY,
            TiebreakField.SCORE,
        ]
    )
    """Discriminator cascade applied in order; first one that distinguishes
    the cluster members picks the survivor.
    """
    model_priority: list[str] = Field(
        default_factory=lambda: ["sam3", "falcon", "grounding_dino"]
    )
    """Model order from most-trusted (lowest index) to least."""


class FilterConfig(BaseModel):
    """Programmatic bounding-box filtering thresholds."""

    model_config = ConfigDict(extra="forbid")

    min_area: float = 0.0005
    max_area: float = 0.95
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_edge_distance: float = 0.0
    per_model_score: dict[str, float] = Field(default_factory=dict)
    """Per-detector score floor. Keys are ``source_model`` strings. Candidates
    whose score is below their model's floor are dropped before dedup.
    """
    iou_dedup: IouDedupConfig = Field(default_factory=IouDedupConfig)
    max_per_class: int = 30


# ---------------------------------------------------------------------------
# Database config (replaces RedisConfig from v3)
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    """SQLite-based pipeline state database (replaces Redis from v3)."""

    model_config = ConfigDict(extra="forbid")

    filename: str = "pipeline.db"
    lock_ttl: int = 300
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Worker counts
# ---------------------------------------------------------------------------


class WorkersConfig(BaseModel):
    """Number of concurrent worker processes per stage."""

    model_config = ConfigDict(extra="forbid")

    detect_per_model: int = 2
    """Workers per enabled detector model. Total detect workers =
    detect_per_model × number of enabled detectors."""
    detect_merge: int = 2
    """Workers for the detect merge stage (combine per-model proposals,
    filter, dedup, route)."""
    evaluate_count: int = 6
    refine_count: int = 2
    finalize_count: int = 2


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    """Root output directory structure for one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    job_dir: str = "output/auto_annotation_v4"
    checkpoint_dir: str = "checkpoints"
    labels_dir: str = "labels"
    traces_dir: str = "traces"
    review_dir: str = "review"


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Per-job runtime inputs (input source, job id, logging, stage control).

    These fields typically change from run to run and are provided by the user
    via their YAML config or as CLI dotlist overrides.  Phase-2 stage control
    fields (``stages``, ``force_stages``, ``force_rerun``) are defined now
    with safe defaults.
    """

    model_config = ConfigDict(extra="forbid")

    image_dir: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    job_id: str | None = None
    log_level: str = "INFO"
    log_file: str | None = None
    stages: list[Stage] = Field(default_factory=lambda: list(STAGE_ORDER))
    """Ordered list of stages to execute. Defaults to all stages."""
    force_stages: list[Stage] = Field(default_factory=list)
    """Stages to force-rerun even if checkpoints exist."""
    force_rerun: bool = False
    """When True, ignore all checkpoints and rerun the entire pipeline."""
    detect_models: list[DetectorName] = Field(default_factory=list)
    """Subset of enabled detectors to run. Empty list means all enabled.
    Allows targeting specific models for incremental re-runs."""
    force_detect_models: list[DetectorName] = Field(default_factory=list)
    """Detectors whose proposals should be deleted and re-run, even if
    cached. Triggers downstream invalidation (detect stage + later)."""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AutoAnnotationV4Config(BaseModel):
    """Full pipeline configuration for auto_annotation_v4.

    Class selection model:
        - ``class_registry`` — the full catalog of known classes keyed by
          canonical name. Each value is a :class:`ClassConfig` (id, tier,
          prompts, synonyms, tags, description).
        - ``detect_classes`` — a per-job list of class names selecting which
          subset of the registry to actually run. Empty list means all classes.

    Consumers should use the ``.classes`` property to iterate the active
    subset (it's automatically filtered against ``detect_classes``).
    """

    model_config = ConfigDict(extra="forbid")

    servers: ServersConfig = Field(default_factory=ServersConfig)
    class_registry: dict[str, ClassConfig] = Field(default_factory=dict)
    detect_classes: list[str] = Field(default_factory=list)
    auto_accept: AutoAcceptConfig = Field(default_factory=AutoAcceptConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    evaluation_groups: dict[str, EvaluationGroupConfig] = Field(default_factory=dict)
    co_existence: CoExistenceConfig = Field(default_factory=CoExistenceConfig)
    refine_rules: RefineRulesConfig = Field(default_factory=RefineRulesConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompts_dir: str = "prompts"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # ------------------------------------------------------------------
    # Active class/group filtering (computed at access time)
    # ------------------------------------------------------------------

    @property
    def classes(self) -> dict[str, ClassConfig]:
        """Active classes — filtered to ``detect_classes`` if non-empty."""
        if not self.detect_classes:
            return self.class_registry
        keep = set(self.detect_classes)
        return {name: cfg for name, cfg in self.class_registry.items() if name in keep}

    @property
    def active_evaluation_groups(self) -> dict[str, EvaluationGroupConfig]:
        """Evaluation groups with each group's ``classes`` list filtered to
        the active class set. Groups that would end up empty are dropped.
        """
        active_names = set(self.classes)
        out: dict[str, EvaluationGroupConfig] = {}
        for name, grp in self.evaluation_groups.items():
            kept = [c for c in grp.classes if c in active_names]
            if kept:
                out[name] = grp.model_copy(update={"classes": kept})
        return out

    def class_by_name(self, name: str) -> ClassConfig | None:
        """Look up a class config by its canonical name."""
        return self.class_registry.get(name)

    def class_by_id(self, class_id: int) -> ClassConfig | None:
        """Look up a class config by its COCO-style integer id."""
        for cfg in self.class_registry.values():
            if cfg.id == class_id:
                return cfg
        return None

    def tier_names(self, tier: int) -> list[str]:
        """Names of active classes at a given tier."""
        return [name for name, cfg in self.classes.items() if cfg.tier == tier]
