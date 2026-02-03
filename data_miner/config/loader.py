"""
Config Loader - Central config loading with stage-specific access.

Usage:
    from data_miner.config import get_filter_config, get_download_config
    
    config = get_filter_config()  # Returns FilterConfig
    print(config.positive_prompts)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load .env file (looks in cwd and parent directories)
load_dotenv()


def init_hf_auth() -> bool:
    """
    Initialize HuggingFace authentication using HF_TOKEN env var.
    Call this before loading private models.
    
    Returns:
        True if logged in, False if no token available
    """
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            return True
        except Exception as e:
            # Can't use get_logger here - circular import
            import warnings
            warnings.warn(f"HF login failed: {e}")
    return False

from .config import (
    DatabaseConfig,
    DownloadConfig,
    ExtractionConfig,
    FilterConfig,
    DeduplicationConfig,
    DetectionConfig,
    LoggingConfig,
    SupervisorConfig,
    MonitorConfig,
)
from .constants import StageName

# Environment variable for config override
CONFIG_PATH_ENV = "DATA_MINER_CONFIG"

# Default config path (relative to project root)
DEFAULT_CONFIG = Path(__file__).parent / "default.yaml"

@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load and merge YAML config (cached).
    
    Priority:
    1. Explicit config_path argument
    2. DATA_MINER_CONFIG environment variable
    3. Default config/default.yaml
    
    Returns merged config as dict with resolved interpolations.
    """
    # Determine path
    if config_path:
        path = Path(config_path)
    elif os.getenv(CONFIG_PATH_ENV):
        path = Path(os.getenv(CONFIG_PATH_ENV))
    else:
        path = DEFAULT_CONFIG
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load with OmegaConf (supports merging, interpolation)
    cfg = OmegaConf.load(path)
    
    # If custom config, merge with defaults
    if path != DEFAULT_CONFIG and DEFAULT_CONFIG.exists():
        base = OmegaConf.load(DEFAULT_CONFIG)
        cfg = OmegaConf.merge(base, cfg)
    
    return OmegaConf.to_container(cfg, resolve=True)


# Backward compatibility alias
_load_yaml = load_config


def reload_config(config_path: str | None = None) -> None:
    """Clear cache and reload config."""
    _load_yaml.cache_clear()
    _load_yaml(config_path)


# =============================================================================
# Stage-specific config getters
# =============================================================================

def get_database_config(config_path: str | None = None) -> DatabaseConfig:
    """Get database configuration."""
    yaml = _load_yaml(config_path)
    return DatabaseConfig(**yaml.get("database", {}))


def get_download_config(config_path: str | None = None) -> DownloadConfig:
    """Get download stage configuration."""
    yaml = _load_yaml(config_path)
    data = yaml.get(StageName.DOWNLOAD.value, {})
    # Merge with global output_dir if not specified
    if "output_dir" not in data:
        data["output_dir"] = f"{yaml.get('output_dir', './output')}/videos"
    
    # Resolve blocked_hashtag_patterns relative to config directory
    if data.get("blocked_hashtag_patterns"):
        pattern_path = Path(data["blocked_hashtag_patterns"]).resolve()
        if pattern_path.is_file():
            data["blocked_hashtag_patterns"] = pattern_path
        else:
            config_dir = Path(__file__).parent
            data["blocked_hashtag_patterns"] = config_dir / Path(data["blocked_hashtag_patterns"]).name
    
    return DownloadConfig(**data)


def get_extraction_config(config_path: str | None = None) -> ExtractionConfig:
    """Get extraction stage configuration."""
    yaml = _load_yaml(config_path)
    data = yaml.get(StageName.EXTRACT.value, {})
    if "output_dir" not in data:
        data["output_dir"] = f"{yaml.get('output_dir', './output')}/frames_raw"
    return ExtractionConfig(**data)


def get_filter_config(config_path: str | None = None) -> FilterConfig:
    """Get filter stage configuration."""
    from .constants import SIGLIP2_MODELS, SIGLIP2_DEFAULT
    
    yaml = _load_yaml(config_path)
    data = yaml.get(StageName.FILTER.value, {})
    if "output_dir" not in data:
        data["output_dir"] = f"{yaml.get('output_dir', './output')}/frames_filtered"
    
    # Map short model_id to full HF path
    model_id = data.get("model_id", SIGLIP2_DEFAULT)
    if model_id in SIGLIP2_MODELS:
        data["model_id"] = SIGLIP2_MODELS[model_id]
    elif not model_id.startswith(("google/", "huggingface/")):
        raise ValueError(f"Unknown filter model_id: {model_id}. Valid: {list(SIGLIP2_MODELS.keys())}")
    
    return FilterConfig(**data)


def get_deduplication_config(config_path: str | None = None) -> DeduplicationConfig:
    """Get deduplication stage configuration."""
    from .constants import DINO_MODELS, DINO_DEFAULT
    
    yaml = _load_yaml(config_path)
    data = yaml.get(StageName.DEDUP.value, {})
    if "output_dir" not in data:
        data["output_dir"] = f"{yaml.get('output_dir', './output')}/frames_dedup"
    
    # Map short dino_model_id to full HF path
    model_id = data.get("dino_model_id", DINO_DEFAULT)
    if model_id in DINO_MODELS:
        data["dino_model_id"] = DINO_MODELS[model_id]
    elif not model_id.startswith(("facebook/", "huggingface/")):
        raise ValueError(f"Unknown dino_model_id: {model_id}. Valid: {list(DINO_MODELS.keys())}")
    
    return DeduplicationConfig(**data)


def get_detection_config(config_path: str | None = None) -> DetectionConfig:
    """Get detection stage configuration."""
    yaml = _load_yaml(config_path)
    data = yaml.get(StageName.DETECT.value, {})
    if "output_dir" not in data:
        data["output_dir"] = f"{yaml.get('output_dir', './output')}/detections"
    return DetectionConfig(**data)


# =============================================================================
# Global settings getters
# =============================================================================

def get_project_name(config_path: str | None = None) -> str:
    """Get project name."""
    yaml = _load_yaml(config_path)
    return yaml.get("project_name", "default")


def get_output_dir(config_path: str | None = None) -> Path:
    """Get global output directory."""
    yaml = _load_yaml(config_path)
    return Path(yaml.get("output_dir", "./output"))


def get_logging_config(config_path: str | None = None) -> LoggingConfig:
    """Get logging configuration."""
    yaml = _load_yaml(config_path)
    return LoggingConfig(**yaml.get("logging", {}))


def get_supervisor_config(config_path: str | None = None) -> SupervisorConfig:
    """Get supervisor worker counts."""
    yaml = _load_yaml(config_path)
    return SupervisorConfig(**yaml.get("supervisor", {}))


def get_monitor_config(config_path: str | None = None) -> MonitorConfig:
    """Get monitor worker configuration."""
    yaml = _load_yaml(config_path)
    return MonitorConfig(**yaml.get("monitor", {}))


def get_backup_config(config_path: str | None = None) -> "BackupConfig":
    """Get backup worker configuration."""
    from .config import BackupConfig
    yaml = _load_yaml(config_path)
    return BackupConfig(**yaml.get("backup", {}))

