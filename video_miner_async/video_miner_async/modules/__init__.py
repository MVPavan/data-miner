"""Video Miner v3 - Modules package."""

from .downloader import YouTubeDownloader
from .frame_extractor import FrameExtractor
from .frame_filter import FrameFilter
from .deduplicator import Deduplicator
from .detector import ObjectDetector

__all__ = [
    "YouTubeDownloader",
    "FrameExtractor",
    "FrameFilter",
    "Deduplicator",
    "ObjectDetector",
]
