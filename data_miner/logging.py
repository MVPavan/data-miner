"""
Logging setup - Single point of change for all logging configuration.

Usage in any module:
    from data_miner.logging import get_logger
    logger = get_logger(__name__)

Captures:
- Application logs (via get_logger)
- System stdout/stderr
- Uncaught exceptions
- Python warnings
"""

import logging
import logging.handlers
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, TextIO

# Environment variables for configuration
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
LOG_FILE = os.getenv("LOG_FILE", "")  # Optional file path
WORKER_ID = os.getenv("WORKER_ID", "main")

# Flags
_loki_available = False
_handlers_configured = False

# Default format
DEFAULT_FORMAT = "%(asctime)s | {worker} | %(levelname)s | %(name)s | %(message)s"


class StreamToLogger:
    """Redirect stdout/stderr to logger."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level
        self._linebuf = ""
    
    def write(self, message: str) -> int:
        if message and message.strip():
            self._logger.log(self._level, message.strip())
        return len(message)
    
    def flush(self) -> None:
        pass
    
    def fileno(self) -> int:
        return -1


def _setup_handlers() -> None:
    """Setup all logging handlers (called once on first logger request)."""
    global _loki_available, _handlers_configured
    
    if _handlers_configured:
        return
    
    root_logger = logging.getLogger("data_miner")
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    formatter = logging.Formatter(DEFAULT_FORMAT.format(worker=WORKER_ID))
    
    # 1. Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.__stdout__)  # Use original stdout
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. File handler (if LOG_FILE is set)
    if LOG_FILE:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 3. Loki handler (if available)
    try:
        import logging_loki
        loki_handler = logging_loki.LokiHandler(
            url=LOKI_URL,
            tags={"application": "data_miner", "worker": WORKER_ID},
            version="1",
        )
        root_logger.addHandler(loki_handler)
        _loki_available = True
    except ImportError:
        pass
    except Exception as e:
        root_logger.warning(f"Failed to connect to Loki: {e}")
    
    # 4. Capture uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    
    sys.excepthook = exception_handler
    
    _handlers_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    _setup_handlers()
    
    if name is None:
        return logging.getLogger("data_miner")
    
    if not name.startswith("data_miner"):
        name = f"data_miner.{name}"
    
    return logging.getLogger(name)


# Convenience exports
logger = get_logger()

# to remove progress bar form download logger
# sed -i -e 's/\r/\n/g' -e '/\[download\]/d' -e '/^[[:space:]]*$/d' download_02.log