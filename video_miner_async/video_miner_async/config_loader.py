"""
Configuration Loader Module

OmegaConf-based configuration with layered merge support.
Supports variable interpolation and environment variables.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf

from .config import StageName

logger = logging.getLogger(__name__)

# Default config location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config(
    user_config: Optional[Union[str, Path]] = None,
    overrides: Optional[dict] = None,
    resolve: bool = True,
) -> DictConfig:
    """
    Load and merge configuration from YAML files.
    
    Merge order (later overrides earlier):
    1. config/default.yaml (base defaults)
    2. user_config (user overrides)
    3. overrides dict (CLI/programmatic overrides)
    
    Args:
        user_config: Path to user config file (e.g., pipeline.yaml)
        overrides: Dictionary of additional overrides
        resolve: Whether to resolve interpolations immediately
        
    Returns:
        Merged DictConfig with resolved interpolations
        
    Example:
        >>> config = load_config("pipeline.yaml")
        >>> print(config.project.name)
        'glass_doors_v1'
        
        >>> config = load_config(overrides={"project.name": "custom"})
    """
    # 1. Load default configuration
    if DEFAULT_CONFIG_PATH.exists():
        default_config = OmegaConf.load(DEFAULT_CONFIG_PATH)
        logger.debug(f"Loaded default config from {DEFAULT_CONFIG_PATH}")
    else:
        logger.warning(f"Default config not found at {DEFAULT_CONFIG_PATH}")
        default_config = OmegaConf.create({})
    
    # 2. Load user configuration if provided
    if user_config is not None:
        user_path = Path(user_config)
        if user_path.exists():
            user_cfg = OmegaConf.load(user_path)
            logger.info(f"Loaded user config from {user_path}")
        else:
            raise FileNotFoundError(f"User config not found: {user_path}")
    else:
        user_cfg = OmegaConf.create({})
    
    # 3. Create overrides config
    if overrides:
        override_cfg = OmegaConf.create(overrides)
    else:
        override_cfg = OmegaConf.create({})
    
    # 4. Merge configs (later overrides earlier)
    merged = OmegaConf.merge(default_config, user_cfg, override_cfg)
    
    # 5. Enable environment variable resolution
    OmegaConf.register_new_resolver("env", _env_resolver, replace=True)
    
    # 6. Resolve interpolations if requested
    if resolve:
        OmegaConf.resolve(merged)
    
    logger.info(f"Configuration loaded: project={merged.get('project', {}).get('name', 'unnamed')}")
    
    return merged


def _env_resolver(var_name: str, default: str = "") -> str:
    """
    Resolve environment variables with optional default.
    
    Usage in YAML:
        ${oc.env:VAR_NAME}           # Required, errors if not set
        ${oc.env:VAR_NAME,default}   # Optional with default
    """
    return os.environ.get(var_name, default)


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, path)
    logger.info(f"Configuration saved to {path}")


def validate_config(config: DictConfig) -> tuple[bool, list[str]]:
    """
    Validate configuration for required fields and valid choices.
    Only validates stage-specific fields if that stage is active.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Get active stages
    stages = set(config.get("stages", []))
    has_pipeline_stages = len(stages) > 0
    
    # Check input sources
    has_urls = config.get("input", {}).get("urls") and len(config.input.urls) > 0
    has_url_file = config.get("input", {}).get("url_file")
    has_registry = config.get("input", {}).get("from_registry", False)
    has_search = config.get("search", {}).get("enabled", False)
    
    # Check if stages have their own input_dir configured (for standalone runs)
    has_stage_input = any([
        config.get("extraction", {}).get("input_dir"),
        config.get("filter", {}).get("input_dir"),
        config.get("deduplication", {}).get("input_dir"),
        config.get("detection", {}).get("input_dir"),
    ])
    
    # For download stage, we need URLs/registry/search
    # For other stages, input_dir is sufficient
    needs_download_input = StageName.DOWNLOAD.value in stages
    if needs_download_input and not any([has_urls, has_url_file, has_registry, has_search]):
        errors.append("No input source specified for download stage. Provide urls, url_file, from_registry, or search.")
    
    # For non-download stages without download, we need input_dir
    non_download_stages = stages - {StageName.DOWNLOAD.value}
    if non_download_stages and StageName.DOWNLOAD.value not in stages and not has_stage_input:
        errors.append("No input_dir specified. Required when running stages without download.")
    
    # Classes are required for filter and detect stages
    needs_classes = StageName.FILTER.value in stages or StageName.DETECTION.value in stages
    if needs_classes and (not config.get("classes") or len(config.classes) == 0):
        errors.append("No classes specified. Required for filter/detect stages.")
    
    # Validate filter config only if filter stage is active
    if StageName.FILTER.value in stages:
        valid_filter_models = {"siglip2-so400m", "siglip2-giant"}
        filter_model = config.get("filter", {}).get("model", "siglip2-so400m")
        if filter_model not in valid_filter_models:
            errors.append(f"Invalid filter.model: {filter_model}. Choices: {valid_filter_models}")
        
        threshold = config.get("filter", {}).get("threshold", 0.25)
        if not (0.0 <= threshold <= 1.0):
            errors.append(f"filter.threshold must be between 0.0 and 1.0, got {threshold}")
    
    # Validate deduplication config only if deduplicate stage is active
    if StageName.DEDUPLICATION.value in stages:
        valid_dedup_models = {"dinov3", "siglip"}
        dedup_model = config.get("deduplication", {}).get("model", "dinov3")
        if dedup_model not in valid_dedup_models:
            errors.append(f"Invalid deduplication.model: {dedup_model}. Choices: {valid_dedup_models}")
        
        # Validate DINO variant if using dinov3
        if dedup_model == "dinov3":
            valid_dino_variants = {
                "dinov3-small", "dinov3-base", "dinov3-large", "dinov3-huge", "dinov3-giant",
                "dinov2-base", "dinov2-large"
            }
            dino_variant = config.get("deduplication", {}).get("dino_variant", "dinov2-base")
            if dino_variant not in valid_dino_variants:
                errors.append(f"Invalid deduplication.dino_variant: {dino_variant}. Choices: {valid_dino_variants}")
        
        threshold = config.get("deduplication", {}).get("threshold", 0.90)
        if not (0.0 <= threshold <= 1.0):
            errors.append(f"deduplication.threshold must be between 0.0 and 1.0, got {threshold}")
    
    # Validate detection config only if detect stage is active
    if StageName.DETECTION.value in stages:
        valid_detect_models = {"moondream3", "florence2", "grounding-dino", "dino-x"}
        detect_model = config.get("detection", {}).get("model", "moondream3")
        if detect_model not in valid_detect_models:
            errors.append(f"Invalid detection.model: {detect_model}. Choices: {valid_detect_models}")
        
        threshold = config.get("detection", {}).get("threshold", 0.3)
        if not (0.0 <= threshold <= 1.0):
            errors.append(f"detection.threshold must be between 0.0 and 1.0, got {threshold}")
    
    # Validate search config if enabled
    if has_search:
        keywords = config.get("search", {}).get("keywords", [])
        keywords_file = config.get("search", {}).get("keywords_file")
        if (not keywords or len(keywords) == 0) and not keywords_file:
            errors.append("Search enabled but no keywords or keywords_file specified.")
    
    return len(errors) == 0, errors


def print_config(config: DictConfig, show_defaults: bool = False) -> str:
    """
    Convert config to readable YAML string.
    
    Args:
        config: Configuration to print
        show_defaults: Include default values (verbose)
        
    Returns:
        YAML string representation
    """
    return OmegaConf.to_yaml(config)


def config_to_dict(config: DictConfig) -> dict:
    """Convert OmegaConf config to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)
