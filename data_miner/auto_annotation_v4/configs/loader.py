"""Config loader for auto_annotation_v4.

Loads split YAML config files from the ``configs/`` directory and deep-merges
them via OmegaConf, then validates the merged dict against the Pydantic
:class:`AutoAnnotationV4Config` model.

Public API:
    load_config        — build a validated config from base + user + CLI overrides
    default_config_path — path to the packaged default.yaml
    compute_config_hash — stable SHA-256 fingerprint of config + prompt files
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .settings import AutoAnnotationV4Config

__all__ = [
    "load_config",
    "default_config_path",
    "compute_config_hash",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).parent


def default_config_path() -> Path:
    """Path to the packaged default config."""
    return _CONFIGS_DIR / "default.yaml"


def _load_base_config() -> DictConfig:
    """Merge the split base config files into one DictConfig.

    Load order (each layer deep-merges onto the previous):
      1. servers.yaml      — detector + VLM server definitions
      2. class_config.yaml — class registry, eval groups, co-existence, refine rules
      3. database.yaml     — SQLite pipeline state database (replaces Redis from v3)
      4. runtime.yaml      — per-job runtime defaults (stages, logging, inputs)
      5. default.yaml      — pipeline settings (filtering, workers, output, prompts)

    The user sees a single flat config namespace — the split is an
    implementation detail for maintainability.
    """
    base = OmegaConf.create({})
    for filename in (
        "servers.yaml",
        "class_config.yaml",
        "database.yaml",
        "runtime.yaml",
        "default.yaml",
    ):
        path = _CONFIGS_DIR / filename
        if path.exists():
            layer: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]
            base = OmegaConf.merge(base, layer)  # type: ignore[assignment]
    return base  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    user_config: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AutoAnnotationV4Config:
    """Load config via OmegaConf: defaults -> user YAML -> CLI dotlist.

    Args:
        user_config: Path to a user YAML that overrides individual keys. Only
            keys present in this file are overridden; everything else comes
            from the base config files (``servers.yaml``, ``class_config.yaml``,
            ``database.yaml``, ``runtime.yaml``, ``default.yaml``).
            Deep-merged via OmegaConf.
        overrides: List of OmegaConf dotlist strings (e.g.
            ``["runtime.log_level=DEBUG", "workers.detect_per_model=4"]``).

    Returns:
        Validated :class:`AutoAnnotationV4Config` instance.

    Example:
        >>> cfg = load_config("my_job.yaml",
        ...                    overrides=["runtime.log_level=DEBUG"])
    """
    base = _load_base_config()

    if user_config is not None:
        user_cfg: DictConfig = OmegaConf.load(Path(user_config))  # type: ignore[assignment]
        base = OmegaConf.merge(base, user_cfg)  # type: ignore[assignment]

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        base = OmegaConf.merge(base, cli_cfg)  # type: ignore[assignment]

    # Resolve any interpolations, then convert to plain dict for Pydantic.
    OmegaConf.resolve(base)
    data: dict[str, Any] = OmegaConf.to_container(base, resolve=True)  # type: ignore[assignment]

    return AutoAnnotationV4Config.model_validate(data)


def compute_config_hash(
    config: AutoAnnotationV4Config,
    prompts_dir: str | Path,
) -> str:
    """Stable hash of the full config + all active prompt files.

    Used to detect config/prompt changes between runs so downstream stages
    can be invalidated selectively.

    Returns:
        First 16 characters of the SHA-256 hex digest.
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
