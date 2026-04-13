"""Logging setup for auto_annotation_v2."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOGGER_NAMESPACE = "data_miner.auto_annotation_v2"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(
    level: str | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Configure logging for auto_annotation_v2.

    Args:
        level: Log level. Falls back to AUTO_ANNOTATION_V2_LOG_LEVEL env var, then INFO.
        output_dir: If provided, also writes logs to {output_dir}/pipeline.log.
    """
    level_name = (level or os.getenv("AUTO_ANNOTATION_V2_LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger(_LOGGER_NAMESPACE)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(console)

    # Add file handler if output_dir given and not already attached
    if output_dir is not None:
        log_path = Path(output_dir) / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename) == log_path.resolve()
            for h in logger.handlers
        )
        if not has_file_handler:
            fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            logger.addHandler(fh)

    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name.split('.')[-1]}")
