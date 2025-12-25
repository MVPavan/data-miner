"""
YouTube Video Downloader Module

Downloads videos from YouTube using yt-dlp with configurable quality settings.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..config import DownloadConfig
from ..utils.io import ensure_dir, get_video_id
from ..utils.validators import validate_youtube_url

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a video download operation."""
    url: str
    video_id: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds


class YouTubeDownloader:
    """
    YouTube video downloader using yt-dlp.
    
    Supports:
        - Highest quality video download
        - Concurrent downloads with ThreadPoolExecutor
        - Progress tracking with callbacks
        - Configurable format and resolution
    
    Example:
        >>> config = DownloadConfig(output_dir=Path("./videos"))
        >>> downloader = YouTubeDownloader(config)
        >>> results = downloader.download_batch(["https://youtube.com/watch?v=..."])
    """
    
    def __init__(self, config: DownloadConfig):
        """
        Initialize the downloader.
        
        Args:
            config: Download configuration
        """
        self.config = config
        ensure_dir(config.output_dir)
        
        # Import yt-dlp lazily to speed up module import
        try:
            import yt_dlp
            self._yt_dlp = yt_dlp
        except ImportError:
            raise ImportError(
                "yt-dlp is required for video downloading. "
                "Install it with: pip install yt-dlp"
            )
    
    def _get_ydl_opts(self, output_template: str, progress_hook: Optional[Callable] = None) -> dict:
        """
        Build yt-dlp options dictionary.
        
        Args:
            output_template: Output file path template
            progress_hook: Optional progress callback
            
        Returns:
            yt-dlp options dict
        """
        opts = {
            "format": self.config.format,
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "socket_timeout": self.config.timeout,
            # Merge video and audio if needed
            "merge_output_format": "mp4",
            # Limit resolution and avoid AV1 codec (prefer H.264/H.265)
            "format_sort": [
                f"res:{self.config.max_resolution}" if self.config.max_resolution else "res:1080",
                "vcodec:h264",  # Prefer H.264
                "vcodec:h265",  # Then H.265
                "+acodec:aac",  # Prefer AAC audio
            ],
            # Retry options for resilient downloads
            "retries": 5,  # Retry on HTTP errors
            "fragment_retries": 10,  # Retry fragments (for DASH/HLS)
            "file_access_retries": 3,  # Retry on file access errors
            "extractor_retries": 3,  # Retry on extractor errors
        }
        
        # Rate limiting options (to avoid YouTube blocks)
        if self.config.sleep_interval > 0:
            opts["sleep_interval"] = self.config.sleep_interval
        if self.config.max_sleep_interval > 0:
            opts["max_sleep_interval"] = self.config.max_sleep_interval
        if self.config.sleep_requests > 0:
            opts["sleep_interval_requests"] = self.config.sleep_requests
        
        if progress_hook:
            opts["progress_hooks"] = [progress_hook]
        
        return opts
    
    def download_single(
        self,
        url: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> DownloadResult:
        """
        Download a single video.
        
        Args:
            url: YouTube video URL
            progress_callback: Optional callback for progress updates
            
        Returns:
            DownloadResult with success status and file path
        """
        # Validate URL
        is_valid, error = validate_youtube_url(url)
        if not is_valid:
            return DownloadResult(
                url=url,
                video_id="",
                success=False,
                error=f"Invalid URL: {error}",
            )
        
        video_id = get_video_id(url)
        if not video_id:
            return DownloadResult(
                url=url,
                video_id="",
                success=False,
                error="Could not extract video ID",
            )
        
        # Check if video already exists on disk (skip re-download)
        for ext in ["mp4", "webm", "mkv", "avi"]:
            existing_path = self.config.output_dir / f"{video_id}.{ext}"
            if existing_path.exists():
                logger.info(f"Video already exists, skipping download: {existing_path}")
                return DownloadResult(
                    url=url,
                    video_id=video_id,
                    success=True,
                    output_path=existing_path,
                    title=None,  # Metadata not available without download
                    duration=None,
                )
        
        # Build output path template
        output_template = str(self.config.output_dir / f"{video_id}.%(ext)s")
        
        try:
            # Create progress hook if callback provided
            def progress_hook(d):
                if progress_callback and d.get("status") == "downloading":
                    progress_callback({
                        "video_id": video_id,
                        "status": "downloading",
                        "downloaded_bytes": d.get("downloaded_bytes", 0),
                        "total_bytes": d.get("total_bytes") or d.get("total_bytes_estimate", 0),
                        "speed": d.get("speed", 0),
                        "eta": d.get("eta", 0),
                    })
            
            opts = self._get_ydl_opts(
                output_template,
                progress_hook if progress_callback else None,
            )
            
            with self._yt_dlp.YoutubeDL(opts) as ydl:
                # Extract info first to get metadata
                info = ydl.extract_info(url, download=True)
                
                if info is None:
                    return DownloadResult(
                        url=url,
                        video_id=video_id,
                        success=False,
                        error="Failed to extract video info",
                    )
                
                # Find the downloaded file
                output_path = self.config.output_dir / f"{video_id}.mp4"
                if not output_path.exists():
                    # Try other extensions
                    for ext in ["webm", "mkv", "avi"]:
                        alt_path = self.config.output_dir / f"{video_id}.{ext}"
                        if alt_path.exists():
                            output_path = alt_path
                            break
                
                return DownloadResult(
                    url=url,
                    video_id=video_id,
                    success=True,
                    output_path=output_path,
                    title=info.get("title"),
                    duration=info.get("duration"),
                )
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return DownloadResult(
                url=url,
                video_id=video_id,
                success=False,
                error=str(e),
            )
    
    # download_batch removed - supervisor handles parallelism via multiple workers


def gather_input_urls(cfg, registry) -> list[str]:
    """
    Gather URLs from configuration sources (list, file, registry).
    
    Implements strict checking:
    - Ignores lines starting with '#'
    - Reads from url_file
    - Reads from input.urls list
    - Reads from registry (pending status) if enabled
    
    Args:
        cfg: Pipeline configuration object
        registry: VideoRegistry instance (optional)
        
    Returns:
        List of unique URLs
    """
    urls = list(cfg.input.urls) if cfg.input.urls else []
    url_file = cfg.input.get("url_file")
    
    # 1. From Registry
    if cfg.input.get("from_registry", False) and registry:
        try:
            from ..registry import VideoStatus
            pending_urls = [v.url for v in registry.get_pending()]
            urls.extend(pending_urls)
            logger.info(f"Added {len(pending_urls)} pending URLs from registry")
        except ImportError:
            pass  # Registry logic might vary, skip if import fails
            
    # 2. From URL File
    if url_file:
        path = Path(url_file)
        if path.exists():
            with open(path) as f:
                # STRICT: starts with # is comment.
                file_urls = [
                    line.strip() for line in f 
                    if line.strip() and not line.strip().startswith("#")
                ]
            urls.extend(file_urls)
            logger.info(f"Loaded {len(file_urls)} URLs from {path}")
        else:
            logger.warning(f"url_file not found: {path}")
            
    # Deduplicate
    unique_urls = list(dict.fromkeys(urls))
    if len(unique_urls) < len(urls):
        logger.debug(f"Deduplicated total input URLs: {len(urls)} -> {len(unique_urls)}")
        
    return unique_urls
