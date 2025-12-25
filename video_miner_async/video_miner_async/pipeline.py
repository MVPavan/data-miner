"""
Pipeline Orchestrator

Main pipeline that coordinates all stages of video mining.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tqdm import tqdm

from .config import PipelineConfig, StageName
from .modules.downloader import YouTubeDownloader, DownloadResult
from .modules.frame_extractor import FrameExtractor, ExtractionResult
from .modules.frame_filter import FrameFilter, FilterResult
from .modules.deduplicator import Deduplicator, DeduplicationResult
from .modules.detector import ObjectDetector, DetectionBatchResult
from .utils.io import ensure_dir, save_json, get_video_id
from .utils.device import clear_gpu_cache
from .registry import VideoStatus

if TYPE_CHECKING:
    from .registry import VideoRegistry

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PipelineResult:
    """Result of the full pipeline execution."""
    
    # Stage results
    download_results: list[DownloadResult] = field(default_factory=list)
    extraction_results: list[ExtractionResult] = field(default_factory=list)
    filter_results: dict[str, FilterResult] = field(default_factory=dict)
    dedup_result: Optional[DeduplicationResult] = None
    detection_result: Optional[DetectionBatchResult] = None
    
    # Summary stats
    videos_downloaded: int = 0
    frames_extracted: int = 0
    frames_filtered: int = 0
    frames_deduplicated: int = 0
    detections_found: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "videos_downloaded": self.videos_downloaded,
                "frames_extracted": self.frames_extracted,
                "frames_filtered": self.frames_filtered,
                "frames_deduplicated": self.frames_deduplicated,
                "detections_found": self.detections_found,
            },
        }


class VideoPipeline:
    """
    Main video mining pipeline orchestrator.
    
    Coordinates all stages:
    1. Download videos from YouTube
    2. Extract frames with configurable sampling
    3. Filter frames using SigLIP similarity
    4. Deduplicate using DINOv3 embeddings
    5. Run open-set detection
    
    Example:
        >>> config = PipelineConfig(urls=["..."], classes=["glass door"])
        >>> pipeline = VideoPipeline(config)
        >>> result = pipeline.run()
    """
    
    def __init__(self, config: PipelineConfig, registry: Optional["VideoRegistry"] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            registry: Optional video registry for tracking progress
        """
        self.config = config
        self.registry = registry
        
        # Initialize components lazily
        self._downloader: Optional[YouTubeDownloader] = None
        self._extractor: Optional[FrameExtractor] = None
        self._filter: Optional[FrameFilter] = None
        self._deduplicator: Optional[Deduplicator] = None
        self._detector: Optional[ObjectDetector] = None
        
        # Ensure output directories
        ensure_dir(config.output_dir)
    
    @property
    def downloader(self) -> YouTubeDownloader:
        if self._downloader is None:
            self._downloader = YouTubeDownloader(self.config.download)
        return self._downloader
    
    @property
    def extractor(self) -> FrameExtractor:
        if self._extractor is None:
            self._extractor = FrameExtractor(self.config.extraction)
        return self._extractor
    
    @property
    def frame_filter(self) -> FrameFilter:
        if self._filter is None:
            self._filter = FrameFilter(self.config.filter, device_map=self.config.device_map)
        return self._filter
    
    @property
    def deduplicator(self) -> Deduplicator:
        if self._deduplicator is None:
            self._deduplicator = Deduplicator(
                self.config.deduplication, 
                device_map=self.config.device_map,
                use_fp16=self.config.use_fp16,
            )
        return self._deduplicator
    
    @property
    def detector(self) -> ObjectDetector:
        if self._detector is None:
            self._detector = ObjectDetector(self.config.detection, device_map=self.config.device_map)
        return self._detector
    
    def _filter_by_registry(
        self,
        items: list,
        get_video_id: callable,
        stage_name: str,
        force: bool = False,
    ) -> list:
        """Filter items by checking registry stage completion status.
        
        Args:
            items: List of items to filter
            get_video_id: Function to extract video_id from item
            stage_name: Name of stage for logging
            force: If True, skip registry check and return all items
        """
        # Skip registry check if force=True
        if force:
            logger.info(f"Force mode: rerunning {stage_name} for all {len(items)} videos")
            return items
        
        if not self.registry:
            return items
        
        stage_attr_map = {
            "extraction": lambda e: e.stages.extraction.completed,
            "filter": lambda e: e.stages.filter.completed,
            "deduplication": lambda e: e.stages.deduplication.completed,
            "detection": lambda e: e.stages.detection.completed,
        }
        check_completed = stage_attr_map.get(stage_name)
        if not check_completed:
            return items
        
        filtered = []
        skipped = 0
        for item in items:
            vid = get_video_id(item)
            entry = self.registry.get_video(vid)
            if entry and check_completed(entry):
                skipped += 1
            else:
                filtered.append(item)
        
        if skipped > 0:
            logger.info(f"Skipping {skipped} already {stage_name} videos")
        
        return filtered
    
    def run(self, show_progress: bool = True) -> PipelineResult:
        """
        Run the full pipeline.
        
        Args:
            show_progress: Show progress indicators
            
        Returns:
            PipelineResult with all results
        """
        result = PipelineResult()
        stages = self.config.stages
        
        console.print(Panel.fit(
            f"[bold blue]Video Mining Pipeline v3[/bold blue]\n"
            f"Stages: {', '.join(stages)}",
            title="Starting Pipeline",
        ))
        
        try:
            # Stage 1: Download
            if StageName.DOWNLOAD in stages:
                console.print("\n[bold cyan]Stage 1: Downloading Videos[/bold cyan]")
                result.download_results = self._run_download(show_progress)
                result.videos_downloaded = sum(1 for r in result.download_results if r.success)
                console.print(f"[green]✓ Downloaded {result.videos_downloaded} videos[/green]")
                
                # Update registry for each download
                self._update_registry_downloads(result.download_results)
                
                clear_gpu_cache()
            
            # Stage 2: Extract frames
            if StageName.EXTRACTION in stages:
                console.print("\n[bold cyan]Stage 2: Extracting Frames[/bold cyan]")
                result.extraction_results = self._run_extraction(
                    result.download_results,
                    show_progress,
                )
                result.frames_extracted = sum(r.frame_count for r in result.extraction_results)
                console.print(f"[green]✓ Extracted {result.frames_extracted} frames[/green]")
                
                # Update registry for each extraction
                self._update_registry_extractions(result.extraction_results)
                
                clear_gpu_cache()
            
            # Stage 3: Filter frames
            if StageName.FILTER in stages:
                console.print("\n[bold cyan]Stage 3: Filtering Frames[/bold cyan]")
                result.filter_results = self._run_filter(
                    result.extraction_results,
                    show_progress,
                )
                result.frames_filtered = sum(r.passed_frames for r in result.filter_results.values())
                console.print(f"[green]✓ {result.frames_filtered} frames passed filter[/green]")
                
                # Update registry for each filter result
                self._update_registry_filters(result.filter_results)
                
                # Unload filter model to free memory
                self.frame_filter.unload_model()
                clear_gpu_cache()
            
            # Stage 4: Deduplicate
            if StageName.DEDUPLICATION in stages:
                console.print("\n[bold cyan]Stage 4: Deduplicating Frames[/bold cyan]")
                result.dedup_result = self._run_deduplication(
                    result.filter_results,
                    show_progress,
                )
                result.frames_deduplicated = result.dedup_result.unique_frames
                console.print(f"[green]✓ {result.frames_deduplicated} unique frames[/green]")
                
                # Update registry for deduplication
                self._update_registry_deduplication(result.dedup_result, result.filter_results)
                
                # Unload dedup model
                self.deduplicator.unload_model()
                clear_gpu_cache()
            
            # Stage 5: Detect objects
            if StageName.DETECTION in stages:
                console.print("\n[bold cyan]Stage 5: Detecting Objects[/bold cyan]")
                result.detection_result = self._run_detection(
                    result.dedup_result,
                    show_progress,
                )
                result.detections_found = result.detection_result.total_detections
                console.print(f"[green]✓ Found {result.detections_found} detections[/green]")
                
                # Update registry for detection
                self._update_registry_detections(result.detection_result)
                
                # Unload detector
                self.detector.unload_model()
                clear_gpu_cache()
            
            # Mark all videos as complete if all stages ran
            self._update_registry_complete(stages)
            
            # Save summary
            summary_path = self.config.output_dir / "pipeline_result.json"
            save_json(result.to_dict(), summary_path)
            
            console.print(Panel.fit(
                f"[bold green]Pipeline Complete![/bold green]\n\n"
                f"Videos: {result.videos_downloaded}\n"
                f"Frames extracted: {result.frames_extracted}\n"
                f"Frames filtered: {result.frames_filtered}\n"
                f"Unique frames: {result.frames_deduplicated}\n"
                f"Detections: {result.detections_found}\n\n"
                f"Results saved to: {self.config.output_dir}",
                title="Summary",
            ))
            
        except Exception as e:
            console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
            logger.exception("Pipeline error")
            raise
        
        return result
    
    def _run_download(self, show_progress: bool) -> list[DownloadResult]:
        """Run download stage."""
        urls = self.config.get_urls()
        if not urls:
            logger.warning("No URLs provided")
            return []
        
        # Define callback for per-video registry update
        def on_video_downloaded(result: DownloadResult):
            if self.registry and result.success:
                entry = self.registry.get_or_create(result.video_id, result.url)
                entry.status = VideoStatus.DOWNLOADED
                entry.stages.download.completed = True
                entry.stages.download.path = str(result.output_path) if result.output_path else None
                entry.stages.download.size_mb = (result.output_path.stat().st_size / 1024 / 1024) if result.output_path and result.output_path.exists() else None
                entry.title = result.title or entry.title
                entry.duration_seconds = int(result.duration) if result.duration else entry.duration_seconds
                self.registry.save()
        
        return self.downloader.download_batch(
            urls, 
            show_progress=show_progress,
            on_complete=on_video_downloaded,
        )
    
    def _run_extraction(
        self,
        download_results: list[DownloadResult],
        show_progress: bool,
    ) -> list[ExtractionResult]:
        """Run frame extraction stage."""
        # If no download results, try to load from input_dir (standalone run)
        if not download_results and self.config.extraction.input_dir:
            input_dir = self.config.extraction.input_dir
            if input_dir.exists():
                video_paths = [
                    (f, f.stem) for f in input_dir.glob("*.mp4")
                ] + [
                    (f, f.stem) for f in input_dir.glob("*.webm")
                ] + [
                    (f, f.stem) for f in input_dir.glob("*.mkv")
                ]
                logger.info(f"Loading {len(video_paths)} videos from input_dir: {input_dir}")
            else:
                logger.warning(f"input_dir not found: {input_dir}")
                return []
        else:
            # Get successfully downloaded videos from results
            video_paths = [
                (r.output_path, r.video_id)
                for r in download_results
                if r.success and r.output_path and r.output_path.exists()
            ]
        
        if not video_paths:
            logger.warning("No videos to extract frames from")
            return []
        
        # Skip already extracted videos (unless force=True)
        video_paths = self._filter_by_registry(
            video_paths, lambda x: x[1], "extraction",
            force=self.config.extraction.force,
        )
        if not video_paths:
            logger.info("All videos already extracted")
            return []
        
        # Define callback for per-video registry update
        def on_video_extracted(result: ExtractionResult):
            if self.registry and result.success:
                entry = self.registry.get_video(result.video_id)
                if entry:
                    entry.status = VideoStatus.EXTRACTED
                    entry.stages.extraction.completed = True
                    entry.stages.extraction.total_frames = result.frame_count
                    entry.stages.extraction.output_dir = str(result.output_dir) if result.output_dir else None
                    self.registry.save()
        
        return self.extractor.extract_batch(
            video_paths, 
            show_progress=show_progress,
            on_complete=on_video_extracted,
            max_workers=self.config.extraction.max_workers,
        )
    
    def _run_filter(
        self,
        extraction_results: list[ExtractionResult],
        show_progress: bool,
    ) -> dict[str, FilterResult]:
        """Run frame filtering stage."""
        classes = self.config.classes
        if not classes:
            logger.warning("No classes provided for filtering, passing all frames")
            return {}
        
        # Build frame groups by video
        frame_groups = {}
        
        # If no extraction results, try to load from input_dir (standalone run)
        if not extraction_results and self.config.filter.input_dir:
            input_dir = self.config.filter.input_dir
            if input_dir.exists():
                # Group frames by subdirectory (video_id) or treat all as one video
                for subdir in input_dir.iterdir():
                    if subdir.is_dir():
                        frames = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                        if frames:
                            frame_groups[subdir.name] = frames
                # If no subdirs, treat all frames as one group
                if not frame_groups:
                    frames = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
                    if frames:
                        frame_groups["standalone"] = frames
                logger.info(f"Loading {sum(len(f) for f in frame_groups.values())} frames from input_dir: {input_dir}")
            else:
                logger.warning(f"input_dir not found: {input_dir}")
        else:
            for result in extraction_results:
                if result.success and result.output_paths:
                    frame_groups[result.video_id] = result.output_paths
        
        if not frame_groups:
            return {}
        
        # Skip already filtered videos (unless force=True)
        frame_groups_list = list(frame_groups.items())
        frame_groups_list = self._filter_by_registry(
            frame_groups_list, lambda x: x[0], "filter",
            force=self.config.filter.force,
        )
        frame_groups = dict(frame_groups_list)
        if not frame_groups:
            logger.info("All videos already filtered")
            return {}
        
        # Define callback for per-video registry update
        def on_video_filtered(video_id: str, result: FilterResult):
            if self.registry:
                entry = self.registry.get_video(video_id)
                if entry:
                    entry.status = VideoStatus.FILTERED
                    entry.stages.filter.completed = True
                    entry.stages.filter.input_frames = result.total_frames
                    entry.stages.filter.passed_frames = result.passed_frames
                    entry.stages.filter.output_dir = str(self.config.filter.output_dir / video_id)
                    self.registry.save()
        
        return self.frame_filter.filter_batch(
            frame_groups=frame_groups,
            classes=classes,
            show_progress=show_progress,
            on_complete=on_video_filtered,
        )
    
    def _run_deduplication(
        self,
        filter_results: dict[str, FilterResult],
        show_progress: bool,
    ) -> DeduplicationResult:
        """Run deduplication stage."""
        # Collect all filtered frame paths
        frame_groups = {}
        
        # If no filter results, try to load from input_dir (standalone run)
        if not filter_results and self.config.deduplication.input_dir:
            input_dir = self.config.deduplication.input_dir
            if input_dir.exists():
                # Group by subdirectories or treat all as one
                for subdir in input_dir.iterdir():
                    if subdir.is_dir():
                        frames = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                        if frames:
                            frame_groups[subdir.name] = frames
                if not frame_groups:
                    frames = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
                    if frames:
                        frame_groups["standalone"] = frames
                logger.info(f"Loading {sum(len(f) for f in frame_groups.values())} frames from input_dir: {input_dir}")
        else:
            for video_id, result in filter_results.items():
                paths = [f.output_path for f in result.filtered_frames]
                if paths:
                    frame_groups[video_id] = paths
        
        if not frame_groups:
            return DeduplicationResult(
                total_frames=0,
                unique_frames=0,
                duplicates_removed=0,
            )
        
        return self.deduplicator.deduplicate_cross_video(
            frame_groups=frame_groups,
            show_progress=show_progress,
        )
    
    def _run_detection(
        self,
        dedup_result: Optional[DeduplicationResult],
        show_progress: bool,
    ) -> DetectionBatchResult:
        """Run detection stage."""
        image_paths = []
        
        # If no dedup result, try to load from input_dir (standalone run)
        if (dedup_result is None or not dedup_result.unique_paths) and self.config.detection.input_dir:
            input_dir = self.config.detection.input_dir
            if input_dir.exists():
                image_paths = list(input_dir.glob("**/*.jpg")) + list(input_dir.glob("**/*.png"))
                logger.info(f"Loading {len(image_paths)} frames from input_dir: {input_dir}")
        elif dedup_result and dedup_result.unique_paths:
            image_paths = dedup_result.unique_paths
        
        if not image_paths:
            return DetectionBatchResult(
                total_frames=0,
                frames_with_detections=0,
                total_detections=0,
            )
        
        # Build detection prompt from classes
        prompt = ", ".join(self.config.classes) if self.config.classes else "objects"
        
        return self.detector.detect_batch(
            image_paths=image_paths,
            prompt=prompt,
            show_progress=show_progress,
        )
    
    # -------------------------------------------------------------------------
    # Registry Update Methods
    # -------------------------------------------------------------------------
    
    def _update_registry_downloads(self, download_results: list[DownloadResult]) -> None:
        """Update registry after download stage."""
        if not self.registry:
            return
        
        from .registry import VideoStatus
        
        for result in download_results:
            if not result.video_id:
                continue
            
            if result.success:
                self.registry.update_status(result.video_id, VideoStatus.DOWNLOADED)
                size_mb = None
                if result.output_path and result.output_path.exists():
                    size_mb = result.output_path.stat().st_size / (1024 * 1024)
                
                self.registry.update_download_stage(
                    video_id=result.video_id,
                    completed=True,
                    path=str(result.output_path) if result.output_path else None,
                    size_mb=size_mb,
                )
                # Update title if available
                if result.title and result.video_id in self.registry.videos:
                    self.registry.videos[result.video_id].title = result.title
            else:
                self.registry.update_status(result.video_id, VideoStatus.FAILED)
                self.registry.update_download_stage(
                    video_id=result.video_id,
                    completed=False,
                    error=result.error,
                )
        
        self.registry.save()
    
    def _update_registry_extractions(self, extraction_results: list[ExtractionResult]) -> None:
        """Update registry after extraction stage."""
        if not self.registry:
            return
        
        from .registry import VideoStatus
        
        for result in extraction_results:
            if result.success:
                self.registry.update_status(result.video_id, VideoStatus.EXTRACTED)
                self.registry.update_extraction_stage(
                    video_id=result.video_id,
                    completed=True,
                    total_frames=result.frame_count,
                    output_dir=str(result.output_dir) if result.output_dir else None,
                )
            else:
                self.registry.update_extraction_stage(
                    video_id=result.video_id,
                    completed=False,
                    error=result.error,
                )
        
        self.registry.save()
    
    def _update_registry_filters(self, filter_results: dict[str, FilterResult]) -> None:
        """Update registry after filter stage."""
        if not self.registry:
            return
        
        from .registry import VideoStatus
        
        for video_id, result in filter_results.items():
            self.registry.update_status(video_id, VideoStatus.FILTERED)
            self.registry.update_filter_stage(
                video_id=video_id,
                completed=True,
                input_frames=result.total_frames,
                passed_frames=result.passed_frames,
                output_dir=str(self.config.filter.output_dir),
            )
        
        self.registry.save()
    
    def _update_registry_deduplication(
        self, 
        dedup_result: DeduplicationResult,
        filter_results: dict[str, FilterResult],
    ) -> None:
        """Update registry after deduplication stage."""
        if not self.registry:
            return
        
        from .registry import VideoStatus
        
        # Dedup happens across all videos, update each one
        for video_id in filter_results.keys():
            self.registry.update_status(video_id, VideoStatus.DEDUPLICATED)
            # Calculate per-video contribution (approximate)
            video_frames = filter_results[video_id].passed_frames
            self.registry.update_deduplication_stage(
                video_id=video_id,
                completed=True,
                input_frames=video_frames,
                unique_frames=dedup_result.unique_frames,  # Total unique
                duplicates_removed=dedup_result.duplicates_removed,  # Total removed
                output_dir=str(self.config.deduplication.output_dir),
            )
        
        self.registry.save()
    
    def _update_registry_detections(self, detection_result: DetectionBatchResult) -> None:
        """Update registry after detection stage."""
        if not self.registry:
            return
        
        from .registry import VideoStatus
        
        # Detection is across all deduped frames, update all videos
        for video_id in self.registry.videos:
            video = self.registry.videos[video_id]
            if video.stages.deduplication.completed:
                self.registry.update_status(video_id, VideoStatus.DETECTED)
                self.registry.update_detection_stage(
                    video_id=video_id,
                    completed=True,
                    frames_processed=detection_result.total_frames,
                    total_detections=detection_result.total_detections,
                    output_dir=str(self.config.detection.output_dir),
                )
        
        self.registry.save()
    
    def _update_registry_complete(self, stages: list[str]) -> None:
        """Mark videos as complete if all stages ran."""
        if not self.registry:
            return
        
        from .registry import VideoStatus
        
        # Only mark complete if detect was the last stage run
        if StageName.DETECTION in stages:
            for video_id in self.registry.videos:
                video = self.registry.videos[video_id]
                if video.stages.detection.completed:
                    self.registry.update_status(video_id, VideoStatus.COMPLETE)
            
            self.registry.save()

