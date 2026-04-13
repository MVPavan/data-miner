from __future__ import annotations

import logging
import os


_LOGGER_NAMESPACE = "data_miner.auto_annotation"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str | None = None) -> None:
    level_name = (level or os.getenv("AUTO_ANNOTATION_LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger(_LOGGER_NAMESPACE)
    if not root.handlers:
        configure_logging()
    return logging.getLogger(name)