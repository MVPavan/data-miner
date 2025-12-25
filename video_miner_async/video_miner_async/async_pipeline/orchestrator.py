"""
Async Pipeline Orchestrator

Coordinates all pipeline stages with async queues.
Supports flexible stage selection - run any contiguous slice of stages.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..config import (
    PipelineConfig,
    StageName,
    STAGE_ORDER,
    DownloadConfig,
    ExtractionConfig,
    FilterConfig,
    DeduplicationConfig,
    DetectionConfig,
)
from ..modules.deduplicator import DeduplicationResult
from ..modules.detector import ObjectDetector, DetectionBatchResult
from ..utils.io import get_video_id

from .messages import StageMessage
from .metrics import PipelineMetrics
from .workers import (
    DownloadWorker,
    ExtractWorker,
    FilterWorker,
    DedupCollector,
)

logger = logging.getLogger(__name__)


class NullQueue:
    """
    A queue that discards all items.
    
    Used when a stage's output has no consumer (i.e., next stage not active).
    """
    async def put(self, item): pass
    async def join(self): pass
    def task_done(self): pass


class AsyncPipelineConfig(BaseModel):
    """Configuration for async pipeline."""
    
    # Worker counts
    download_workers: int = Field(default=2, ge=1, le=5)
    extract_workers: int = Field(default=2, ge=1, le=4)
    filter_workers: int = Field(default=1, ge=1, le=2)  # GPU-bound
    
    # Queue sizes (for backpressure)
    download_queue_size: int = Field(default=4, ge=1)
    extract_queue_size: int = Field(default=4, ge=1)
    filter_queue_size: int = Field(default=2, ge=1)
    
    # Thread pool
    max_thread_workers: int = Field(default=8, ge=4)
    
    # Cleanup
    cleanup_raw_frames: bool = Field(default=False)
    cleanup_videos: bool = Field(default=False)


@dataclass
class AsyncPipelineResult:
    """Result of async pipeline execution."""
    
    total_urls: int
    downloaded: int
    extracted: int
    filtered: int
    deduplicated: int
    detected: int
    failed: int
    dedup_result: Optional[DeduplicationResult] = None
    detection_result: Optional[DetectionBatchResult] = None
    metrics: Optional[PipelineMetrics] = None
    elapsed_seconds: float = 0.0


class AsyncPipelineOrchestrator:
    """
    Orchestrates async pipeline with stage-level parallelism.
    
    Supports flexible stage selection - run any contiguous slice:
    - [DOWNLOAD, EXTRACTION, FILTER, DEDUPLICATION, DETECTION] (all)
    - [FILTER, DEDUPLICATION] (just filter + dedup from existing frames)
    - [EXTRACTION] (just extract from existing videos)
    
    Stages:
    1. Download (N workers, network I/O)
    2. Extract (N workers, CPU decode)
    3. Filter (1 worker, GPU)
    4. Dedup Collector (accumulates)
    5. Cross-video Dedup (after all videos)
    6. Detection (optional)
    
    Example:
        >>> orchestrator = AsyncPipelineOrchestrator(
        ...     pipeline_config=config,
        ...     async_config=AsyncPipelineConfig(),
        ... )
        >>> result = await orchestrator.run(urls)
    """
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        async_config: Optional[AsyncPipelineConfig] = None,
        registry=None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            pipeline_config: Main pipeline configuration
            async_config: Async-specific configuration
            registry: Optional video registry
        """
        self.pipeline_config = pipeline_config
        self.async_config = async_config or AsyncPipelineConfig()
        self.registry = registry
        
        # Validate stages are contiguous
        self._validate_stages()
        
        # Will be initialized in run()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.metrics = PipelineMetrics()
        self.workers = []
        
        # Dict of queues keyed by stage (real Queue or NullQueue)
        self.queues: dict[StageName, asyncio.Queue | NullQueue] = {}
        
        # Special collector
        self.dedup_collector: Optional[DedupCollector] = None
    
    def _validate_stages(self) -> None:
        """Validate stages are contiguous in STAGE_ORDER."""
        stages = self.pipeline_config.stages
        if not stages:
            raise ValueError("No stages specified")
        
        indices = [STAGE_ORDER.index(s) for s in stages]
        expected = list(range(min(indices), max(indices) + 1))
        
        if indices != expected:
            raise ValueError(
                f"Stages must be contiguous. Got {[s.value for s in stages]}, "
                f"expected contiguous slice of {[s.value for s in STAGE_ORDER]}"
            )
    
    def _get_first_stage(self) -> StageName:
        """Get the first stage in the pipeline."""
        return self.pipeline_config.stages[0]
    
    def _create_queues(self) -> None:
        """Create async queues based on active stages."""
        stages = set(self.pipeline_config.stages)
        cfg = self.async_config
        
        # Create queue if stage is active, otherwise NullQueue
        self.queues = {
            StageName.DOWNLOAD: (
                asyncio.Queue(maxsize=cfg.download_workers * 2)
                if StageName.DOWNLOAD in stages else NullQueue()
            ),
            StageName.EXTRACTION: (
                asyncio.Queue(maxsize=cfg.download_queue_size)
                if StageName.EXTRACTION in stages else NullQueue()
            ),
            StageName.FILTER: (
                asyncio.Queue(maxsize=cfg.extract_queue_size)
                if StageName.FILTER in stages else NullQueue()
            ),
            StageName.DEDUPLICATION: (
                asyncio.Queue(maxsize=cfg.filter_queue_size)
                if StageName.DEDUPLICATION in stages else NullQueue()
            ),
        }
    
    def _create_workers(self) -> None:
        """Create workers only for active stages."""
        stages = set(self.pipeline_config.stages)
        cfg = self.async_config
        pcfg = self.pipeline_config
        
        # Download workers
        if StageName.DOWNLOAD in stages:
            for i in range(cfg.download_workers):
                self.workers.append(DownloadWorker(
                    name=f"download-{i+1}",
                    input_queue=self.queues[StageName.DOWNLOAD],
                    output_queue=self.queues[StageName.EXTRACTION],
                    executor=self.executor,
                    metrics=self.metrics,
                    registry=self.registry,
                    config=pcfg.download,
                ))
            self.metrics.add_stage("download")
        
        # Extract workers
        if StageName.EXTRACTION in stages:
            for i in range(cfg.extract_workers):
                self.workers.append(ExtractWorker(
                    name=f"extract-{i+1}",
                    input_queue=self.queues[StageName.EXTRACTION],
                    output_queue=self.queues[StageName.FILTER],
                    executor=self.executor,
                    metrics=self.metrics,
                    registry=self.registry,
                    config=pcfg.extraction,
                    cleanup_videos=cfg.cleanup_videos,
                ))
            self.metrics.add_stage("extract")
        
        # Filter worker(s) - typically 1 for GPU
        if StageName.FILTER in stages:
            for i in range(cfg.filter_workers):
                self.workers.append(FilterWorker(
                    name=f"filter-{i+1}",
                    input_queue=self.queues[StageName.FILTER],
                    output_queue=self.queues[StageName.DEDUPLICATION],
                    executor=self.executor,
                    metrics=self.metrics,
                    registry=self.registry,
                    config=pcfg.filter,
                    classes=pcfg.classes,
                    device_map=pcfg.device_map,
                    cleanup_raw_frames=cfg.cleanup_raw_frames,
                ))
            self.metrics.add_stage("filter")
        
        # Dedup collector (accumulates results)
        if StageName.DEDUPLICATION in stages:
            self.dedup_collector = DedupCollector(
                name="dedup-collector",
                input_queue=self.queues[StageName.DEDUPLICATION],
                output_queue=NullQueue(),  # Dedup is terminal
                executor=self.executor,
                metrics=self.metrics,
                config=pcfg.deduplication,
                device_map=pcfg.device_map,
                use_fp16=pcfg.use_fp16,
            )
            self.workers.append(self.dedup_collector)
            self.metrics.add_stage("dedup-collector")
    
    async def _url_producer(self, urls: list[str]) -> None:
        """
        Producer task that streams URLs into the queue.
        
        Uses bounded queue to limit memory usage for large URL lists (50k+).
        Only keeps a small buffer in memory at any time.
        """
        for url in urls:
            video_id = get_video_id(url) or url[-11:]  # Fallback to last 11 chars
            await self.queues[StageName.DOWNLOAD].put(StageMessage(
                video_id=video_id,
                input_path=url,  # Keep as URL string for download worker
                metadata={"url": url},
            ))
        logger.info(f"URL producer finished queuing {len(urls)} URLs")
    
    def _get_videos_from_registry(self, first_stage: StageName) -> list:
        """Get videos from registry based on first stage and force flag."""
        from ..registry import VideoStatus
        
        stage_config = getattr(self.pipeline_config, first_stage.value, None)
        force = stage_config.force if stage_config else False
        
        if force:
            # Get ALL videos
            return list(self.registry.videos.values())
        
        # Get videos at expected status for this stage
        status_map = {
            StageName.DOWNLOAD: VideoStatus.PENDING,
            StageName.EXTRACTION: VideoStatus.DOWNLOADED,
            StageName.FILTER: VideoStatus.EXTRACTED,
            StageName.DEDUPLICATION: VideoStatus.FILTERED,
            StageName.DETECTION: VideoStatus.DEDUPLICATED,
        }
        return self.registry.get_by_status(status_map.get(first_stage, VideoStatus.PENDING))
    
    async def _seed_from_registry(self, first_stage: StageName) -> int:
        """Seed first stage queue from registry."""
        videos = self._get_videos_from_registry(first_stage)
        count = 0
        
        for video in videos:
            if first_stage == StageName.DOWNLOAD:
                url = video.url
                await self.queues[StageName.DOWNLOAD].put(StageMessage(
                    video_id=video.video_id,
                    input_path=url,
                    metadata={"url": url},
                ))
            elif first_stage == StageName.EXTRACTION:
                video_path = video.stages.download.path
                if video_path and Path(video_path).exists():
                    await self.queues[StageName.EXTRACTION].put(StageMessage(
                        video_id=video.video_id,
                        input_path=Path(video_path),
                    ))
                    count += 1
            elif first_stage == StageName.FILTER:
                frames_dir = video.stages.extraction.output_dir
                if frames_dir and Path(frames_dir).exists():
                    await self.queues[StageName.FILTER].put(StageMessage(
                        video_id=video.video_id,
                        input_path=Path(frames_dir),
                    ))
                    count += 1
            elif first_stage == StageName.DEDUPLICATION:
                filtered_dir = video.stages.filter.output_dir
                if filtered_dir and Path(filtered_dir).exists():
                    await self.queues[StageName.DEDUPLICATION].put(StageMessage(
                        video_id=video.video_id,
                        input_path=Path(filtered_dir),
                    ))
                    count += 1
        
        if first_stage == StageName.DOWNLOAD:
            count = len(videos)
        
        logger.info(f"Seeded {count} items from registry for {first_stage.value} stage")
        return count
    
    async def _seed_from_disk(self, first_stage: StageName) -> int:
        """Seed first stage queue from disk input_dir config."""
        pcfg = self.pipeline_config
        count = 0
        
        if first_stage == StageName.EXTRACTION:
            input_dir = pcfg.extraction.input_dir or pcfg.download.output_dir
            for video_file in Path(input_dir).glob("*.mp4"):
                await self.queues[StageName.EXTRACTION].put(StageMessage(
                    video_id=video_file.stem,
                    input_path=video_file,
                ))
                count += 1
                
        elif first_stage == StageName.FILTER:
            input_dir = pcfg.filter.input_dir or pcfg.extraction.output_dir
            for folder in Path(input_dir).iterdir():
                if folder.is_dir():
                    await self.queues[StageName.FILTER].put(StageMessage(
                        video_id=folder.name,
                        input_path=folder,
                    ))
                    count += 1
                    
        elif first_stage == StageName.DEDUPLICATION:
            input_dir = pcfg.deduplication.input_dir or pcfg.filter.output_dir
            for folder in Path(input_dir).iterdir():
                if folder.is_dir():
                    await self.queues[StageName.DEDUPLICATION].put(StageMessage(
                        video_id=folder.name,
                        input_path=folder,
                    ))
                    count += 1
        
        logger.info(f"Seeded {count} items from disk for {first_stage.value} stage")
        return count
    
    async def _seed_first_stage(self, urls: list[str] = None) -> int:
        """Seed the first stage queue from appropriate source."""
        first_stage = self._get_first_stage()
        
        # If DOWNLOAD and URLs provided, use URL producer
        if first_stage == StageName.DOWNLOAD and urls:
            await self._url_producer(urls)
            return len(urls)
        
        # Try registry first if configured
        use_registry = getattr(self.pipeline_config, 'input', None)
        if use_registry and getattr(use_registry, 'from_registry', False) and self.registry:
            return await self._seed_from_registry(first_stage)
        
        # Fallback to disk
        return await self._seed_from_disk(first_stage)
    
    async def run(self, urls: list[str] = None) -> AsyncPipelineResult:
        """
        Run the async pipeline.
        
        Args:
            urls: List of YouTube URLs (only used if first stage is DOWNLOAD)
            
        Returns:
            AsyncPipelineResult with statistics
        """
        stages = self.pipeline_config.stages
        first_stage = self._get_first_stage()
        
        start_time = time.time()
        
        logger.info(f"Starting async pipeline with stages: {[s.value for s in stages]}")
        logger.info(f"First stage: {first_stage.value}")
        
        # Initialize
        self.executor = ThreadPoolExecutor(
            max_workers=self.async_config.max_thread_workers
        )
        self._create_queues()
        self._create_workers()
        
        if not self.workers:
            logger.warning("No workers created - check stages configuration")
            return AsyncPipelineResult(total_urls=0, downloaded=0, extracted=0, 
                                       filtered=0, deduplicated=0, detected=0, failed=0)
        
        # Start all worker tasks
        tasks = [asyncio.create_task(w.run_loop()) for w in self.workers]
        
        try:
            # Seed first stage
            total_items = await self._seed_first_stage(urls)
            self.metrics.total_videos = total_items
            
            logger.info(f"Processing {total_items} items")
            
            # Wait for active queues to drain
            for stage in stages:
                queue = self.queues.get(stage)
                if queue and not isinstance(queue, NullQueue):
                    await queue.join()
                    logger.info(f"{stage.value} queue drained")
            
        finally:
            # Signal all workers to stop
            for worker in self.workers:
                worker.stop()
            
            # Wait for workers to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cleanup executor
            self.executor.shutdown(wait=True)
        
        # Run cross-video deduplication
        dedup_result = None
        if self.dedup_collector:
            from ..utils.device import clear_gpu_cache
            clear_gpu_cache()
            num_videos, num_frames = self.dedup_collector.get_collected_count()
            logger.info(f"Running cross-video dedup on {num_frames} frames from {num_videos} videos")
            
            dedup_result = await asyncio.get_event_loop().run_in_executor(
                None, self.dedup_collector.run_deduplication
            )
            self.dedup_collector.cleanup()
        
        # Run detection if configured
        detection_result = None
        if "detect" in self.pipeline_config.stages and dedup_result:
            detection_result = await self._run_detection(dedup_result)
        
        elapsed = time.time() - start_time
        
        # Cleanup filter workers
        for worker in self.workers:
            if isinstance(worker, FilterWorker):
                worker.cleanup()
        
        # Build result
        result = AsyncPipelineResult(
            total_urls=len(urls),
            downloaded=self._get_stage_count("download"),
            extracted=self._get_stage_count("extract"),
            filtered=self._get_stage_count("filter"),
            deduplicated=dedup_result.unique_frames if dedup_result else 0,
            detected=detection_result.total_detections if detection_result else 0,
            failed=self.metrics.failed_videos,
            dedup_result=dedup_result,
            detection_result=detection_result,
            metrics=self.metrics,
            elapsed_seconds=elapsed,
        )
        
        logger.info(
            f"Pipeline complete in {elapsed:.1f}s: "
            f"{result.downloaded} downloaded, {result.filtered} filtered, "
            f"{result.deduplicated} unique frames"
        )
        
        return result
    
    async def _run_detection(
        self, dedup_result: DeduplicationResult
    ) -> Optional[DetectionBatchResult]:
        """Run detection on unique frames."""
        if not dedup_result.unique_paths:
            return None
        
        logger.info(f"Running detection on {len(dedup_result.unique_paths)} unique frames")
        
        detector = ObjectDetector(
            config=self.pipeline_config.detection,
            device_map=self.pipeline_config.device,
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            detector.detect_batch,
            dedup_result.unique_paths,
            ", ".join(self.pipeline_config.classes),
            True,
        )
        
        detector.unload_model()
        return result
    
    def _get_stage_count(self, stage_name: str) -> int:
        """Get processed count for a stage."""
        # Aggregate from workers with matching prefix
        total = 0
        for worker in self.workers:
            if worker.name.startswith(stage_name):
                total += worker.processed_count
        return total
