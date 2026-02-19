"""Processing modules."""

from .downloader import YouTubeDownloader, DownloadResult
from .frame_extractor import FrameExtractor, ExtractionResult

try:
    from .frame_filter import FrameFilter, FilterResult
    from .deduplicator import Deduplicator, DeduplicationResult
    from .detector import ObjectDetector, DetectionBatchResult
    _ml_available = True
except ImportError:
    _ml_available = False

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
