"""
YouTube Search Module

Search YouTube for videos using yt-dlp and populate the registry.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

from .constants import (
    DEFAULT_SEARCH_MAX_RESULTS,
    SEARCH_MAX_LIMIT,
    SEARCH_TIMEOUT_SECONDS,
    YOUTUBE_BASE_URL,
)
from .utils.io import get_video_id

logger = logging.getLogger(__name__)


# =============================================================================
# Video Info Dataclass
# =============================================================================

@dataclass
class VideoInfo:
    """Information about a YouTube video from search."""
    video_id: str
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None
    duration_seconds: Optional[int] = None


def make_youtube_url(video_id: str) -> str:
    """Create a standard YouTube URL from video ID."""
    return f"{YOUTUBE_BASE_URL}{video_id}"


# =============================================================================
# YouTube Search
# =============================================================================

def search_youtube(
    keyword: str,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    sort_by: str = "relevance",  # relevance, date, viewCount, rating
) -> list[VideoInfo]:
    """
    Search YouTube for videos matching a keyword.
    
    Supports both regular search and hashtag search:
    - Keywords starting with '!#' use YouTube hashtag page
    - Regular keywords use yt-dlp ytsearch
    
    Args:
        keyword: Search keyword/phrase (prefix with '!#' for hashtag search)
        max_results: Maximum number of results (capped at SEARCH_MAX_LIMIT)
        sort_by: Sort order (relevance, date, viewCount, rating)
        
    Returns:
        List of VideoInfo objects
    """
    max_results = min(max_results, SEARCH_MAX_LIMIT)
    
    # Handle !# prefix (user convention for hashtags in safe files)
    if keyword.startswith("!#"):
        keyword = keyword[1:]
        
    # Detect if this is a hashtag search
    is_hashtag = keyword.startswith("#")
    
    if is_hashtag:
        # Hashtag search - use YouTube hashtag page
        hashtag = keyword.lstrip("#").strip()
        search_query = f"https://www.youtube.com/hashtag/{hashtag}"
        logger.info(f"Searching YouTube hashtag: #{hashtag} (max {max_results} results)")
    else:
        # Regular search - use yt-dlp ytsearch
        search_query = f"ytsearch{max_results}:{keyword}"
        logger.info(f"Searching YouTube for: '{keyword}' (max {max_results} results)")
    
    try:
        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(id)s|%(title)s|%(channel)s|%(duration)s",
            "--no-warnings",
        ]
        
        # For hashtag, limit results with --playlist-end
        if is_hashtag:
            cmd.extend(["--playlist-end", str(max_results)])
        
        cmd.append(search_query)
        
        # Run yt-dlp
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        
        if result.returncode != 0:
            logger.error(f"yt-dlp search failed: {result.stderr}")
            return []
        
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            
            parts = line.split("|")
            if len(parts) < 1:
                continue
            
            video_id = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else None
            channel = parts[2].strip() if len(parts) > 2 else None
            duration_str = parts[3].strip() if len(parts) > 3 else None
            
            # Parse duration
            duration = None
            if duration_str and duration_str.isdigit():
                duration = int(duration_str)
            
            if video_id:
                videos.append(VideoInfo(
                    video_id=video_id,
                    url=make_youtube_url(video_id),
                    title=title if title != "NA" else None,
                    channel=channel if channel != "NA" else None,
                    duration_seconds=duration,
                ))
        
        search_type = "hashtag" if is_hashtag else "keyword"
        logger.info(f"Found {len(videos)} videos for {search_type} '{keyword}'")
        return videos
        
    except subprocess.TimeoutExpired:
        logger.error("YouTube search timed out")
        return []
    except FileNotFoundError:
        logger.error("yt-dlp not found. Please install with: pip install yt-dlp")
        return []
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return []


def search_and_register(
    keyword: str,
    registry,  # VideoRegistry
    max_results: int = 50,
) -> tuple[int, int]:
    """
    Search YouTube and add results to registry.
    
    Args:
        keyword: Search keyword
        registry: VideoRegistry instance
        max_results: Maximum results to fetch
        
    Returns:
        Tuple of (total_found, newly_added)
    """
    videos = search_youtube(keyword, max_results)
    
    added = 0
    for video in videos:
        if registry.add_video(
            video_id=video.video_id,
            url=video.url,
            title=video.title,
            channel=video.channel,
            keyword=keyword,
        ):
            added += 1
    
    logger.info(f"Search complete: {len(videos)} found, {added} newly added to registry")
    return len(videos), added

# =============================================================================
# Batch Search
# =============================================================================

def search_multiple_keywords(
    keywords: list[str],
    registry,  # VideoRegistry
    max_per_keyword: int = 50,
) -> dict[str, tuple[int, int]]:
    """
    Search multiple keywords and add all to registry.
    
    Args:
        keywords: List of search keywords
        registry: VideoRegistry instance
        max_per_keyword: Max results per keyword
        
    Returns:
        Dict mapping keyword -> (total_found, newly_added)
    """
    results = {}
    
    for keyword in keywords:
        found, added = search_and_register(keyword, registry, max_per_keyword)
        results[keyword] = (found, added)
    
    return results


def execute_search_stage(cfg, registry) -> tuple[int, int]:
    """
    Execute the search stage based on configuration.
    
    Loads keywords from config and/or file, performs searches,
    and updates the registry.
    
    Args:
        cfg: Pipeline configuration object
        registry: VideoRegistry instance
        
    Returns:
        Tuple of (total_found, total_added)
    """
    if not cfg.get("search", {}).get("enabled", False):
        return 0, 0
        
    from pathlib import Path
    
    keywords = list(cfg.search.get("keywords", []))
    default_max = cfg.search.get("default_max_results", 50)
    
    # Load keywords from file if specified
    keywords_file = cfg.search.get("keywords_file")
    if keywords_file:
        kw_path = Path(keywords_file)
        if kw_path.exists():
            with open(kw_path) as f:
                # STRICT: starts with # is comment. !# passes through.
                file_keywords = [
                    line.strip() for line in f 
                    if line.strip() and not line.strip().startswith("#")
                ]
            keywords.extend(file_keywords)
            logger.info(f"Loaded {len(file_keywords)} keywords from {kw_path}")
        else:
            logger.warning(f"keywords_file not found: {kw_path}")
    
    if not keywords:
        logger.warning("Search enabled but no keywords specified.")
        return 0, 0
    
    logger.info(f"Searching YouTube ({len(keywords)} keywords)...")
    total_found = 0
    total_added = 0
    
    for kw_config in keywords:
        if isinstance(kw_config, str):
            query = kw_config
            max_results = default_max
        else:
            query = kw_config.get("query", kw_config) if hasattr(kw_config, "get") else str(kw_config)
            max_results = kw_config.get("max_results", default_max) if hasattr(kw_config, "get") else default_max
        
        logger.info(f"Searching: {query} (max {max_results})")
        found, added = search_and_register(query, registry, max_results)
        total_found += found
        total_added += added
    
    # Save registry
    registry.save()
    logger.info(f"Search complete: {total_found} found, {total_added} new videos added")
    logger.debug(f"Registry total: {len(registry.videos)}")
    
    return total_found, total_added
