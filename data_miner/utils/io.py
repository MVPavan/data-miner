"""
I/O Utilities

File operations, JSON handling, and path management.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from ..config import YOUTUBE_DOMAINS

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        The same path for chaining
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to serialize (must be JSON-serializable)
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    
    logger.debug(f"Saved JSON to {path}")


def load_json(path: Path) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {path}")
    return data


def get_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    
    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID string or None if not found
    """
    # Try standard watch URL
    parsed = urlparse(url)
    
    # youtu.be short format
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("/")[0].split("?")[0]
    
    # youtube.com formats
    if parsed.netloc in YOUTUBE_DOMAINS:
        # /watch?v=VIDEO_ID
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        
        # /embed/VIDEO_ID, /v/VIDEO_ID, /shorts/VIDEO_ID
        path_parts = parsed.path.split("/")
        if len(path_parts) >= 3 and path_parts[1] in ("embed", "v", "shorts"):
            return path_parts[2].split("?")[0]
    
    # Fallback: try regex for video ID pattern (11 chars, alphanumeric + _-)
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})(?:[?&/]|$)", url)
    if match:
        return match.group(1)
    
    return None


def get_safe_filename(name: str, max_length: int = 100) -> str:
    """
    Convert string to safe filename by removing/replacing invalid characters.
    
    Args:
        name: Original name
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Replace problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = re.sub(r'\s+', "_", safe)
    safe = re.sub(r'_+', "_", safe)
    safe = safe.strip("_")
    
    # Truncate if needed
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("_")
    
    return safe or "unnamed"
