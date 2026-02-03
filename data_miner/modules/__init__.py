"""Processing modules."""

from .downloader import YouTubeDownloader, DownloadResult
from .frame_extractor import FrameExtractor, ExtractionResult
from .frame_filter import FrameFilter, FilterResult
from .deduplicator import Deduplicator, DeduplicationResult
from .detector import ObjectDetector, DetectionBatchResult

__all__ = [
    "YouTubeDownloader",
    "DownloadResult",
    "FrameExtractor",
    "ExtractionResult",
    "FrameFilter",
    "FilterResult",
    "Deduplicator",
    "DeduplicationResult",
    "ObjectDetector",
    "DetectionBatchResult",
]
