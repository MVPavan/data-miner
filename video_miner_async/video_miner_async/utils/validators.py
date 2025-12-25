"""
Input Validation Utilities

Validators for URLs, paths, and other inputs.
"""

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ..constants import YOUTUBE_DOMAINS

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def validate_youtube_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate YouTube URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if not url:
        return False, "URL is empty"
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"
    
    # Check scheme
    if parsed.scheme not in ("http", "https"):
        return False, f"Invalid URL scheme: {parsed.scheme}"
    
    # Check domain
    if parsed.netloc not in YOUTUBE_DOMAINS:
        return False, f"Not a YouTube domain: {parsed.netloc}"
    
    # Check for video ID presence
    from .io import get_video_id
    video_id = get_video_id(url)
    if not video_id:
        return False, "Could not extract video ID from URL"
    
    # Validate video ID format (11 chars, alphanumeric + _-)
    if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
        return False, f"Invalid video ID format: {video_id}"
    
    return True, None


def validate_image_path(path: Path) -> tuple[bool, Optional[str]]:
    """
    Validate image file path.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)
    
    if not path.exists():
        return False, f"File does not exist: {path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {path}"
    
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return False, f"Invalid image extension: {path.suffix}"
    
    # Check file is readable and not empty
    try:
        size = path.stat().st_size
        if size == 0:
            return False, f"File is empty: {path}"
    except OSError as e:
        return False, f"Cannot access file: {e}"
    
    return True, None


def validate_classes(classes: list[str]) -> tuple[bool, Optional[str]]:
    """
    Validate class/caption list for filtering.
    
    Args:
        classes: List of class names or captions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not classes:
        return False, "No classes provided"
    
    for i, cls in enumerate(classes):
        if not isinstance(cls, str):
            return False, f"Class at index {i} is not a string"
        if not cls.strip():
            return False, f"Class at index {i} is empty"
        if len(cls) > 500:
            return False, f"Class at index {i} exceeds max length (500 chars)"
    
    return True, None
