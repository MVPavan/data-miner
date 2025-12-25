"""
Base Stage Worker

Abstract base class for all async pipeline workers.
Provides common functionality: run loop, error handling, metrics, registry updates.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable, Any

from .messages import StageMessage
from .metrics import PipelineMetrics

logger = logging.getLogger(__name__)


class BaseStageWorker(ABC):
    """
    Abstract base class for pipeline stage workers.
    
    Subclasses only need to implement `process()` method.
    The base class handles:
    - Queue polling
    - Graceful shutdown
    - Error handling
    - Metrics recording
    - ThreadPool execution for blocking calls
    - Registry updates (optional)
    
    Example:
        >>> class MyWorker(BaseStageWorker):
        ...     def process(self, msg):
        ...         # Do work
        ...         return StageMessage(...)
    """
    
    def __init__(
        self,
        name: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        executor: ThreadPoolExecutor,
        metrics: Optional[PipelineMetrics] = None,
        registry: Optional[Any] = None,
        registry_save_interval: int = 5,
    ):
        """
        Initialize base worker.
        
        Args:
            name: Worker name for logging
            input_queue: Queue to read messages from
            output_queue: Queue to write results to
            executor: ThreadPoolExecutor for blocking operations
            metrics: Optional shared metrics object
            registry: Optional VideoRegistry for status updates
            registry_save_interval: Save registry every N videos processed
        """
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.executor = executor
        self.metrics = metrics
        self.registry = registry
        self.registry_save_interval = registry_save_interval
        
        self.shutdown = asyncio.Event()
        self.processed_count = 0
        self.failed_count = 0
    
    @abstractmethod
    def process(self, msg: StageMessage) -> Optional[StageMessage]:
        """
        Process a single message.
        
        This method runs in a thread pool, so blocking calls are OK.
        
        Args:
            msg: Input message from previous stage
            
        Returns:
            StageMessage to pass to next stage, or None to drop
        """
        pass
    
    def update_registry(self, video_id: str, success: bool, **kwargs) -> None:
        """
        Update registry after processing. Override in subclasses.
        
        Args:
            video_id: Video ID being processed
            success: Whether processing succeeded
            **kwargs: Additional data to store
        """
        pass  # Subclasses implement stage-specific updates
    
    async def run_loop(self) -> None:
        """
        Main worker loop. Do not override.
        
        Polls input queue, processes messages, handles errors.
        """
        logger.info(f"[{self.name}] Worker started")
        
        while not self.shutdown.is_set():
            try:
                # Wait for message with timeout (allows checking shutdown)
                msg = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            start_time = time.time()
            video_id = msg.video_id
            
            try:
                # Run blocking process() in thread pool
                result = await self._run_blocking(self.process, msg)
                duration = time.time() - start_time
                
                if result is not None:
                    await self.output_queue.put(result)
                    self.processed_count += 1
                    
                    # Update registry if available
                    if self.registry:
                        self.update_registry(video_id, success=True, result=result)
                        
                        # Periodic save
                        if self.processed_count % self.registry_save_interval == 0:
                            await self._run_blocking(self.registry.save)
                    
                    if self.metrics:
                        self.metrics.record_success(self.name, duration)
                    
                    logger.debug(
                        f"[{self.name}] Processed {video_id} in {duration:.2f}s"
                    )
                else:
                    # Update registry for failed/dropped
                    if self.registry:
                        self.update_registry(video_id, success=False)
                    
                    logger.debug(f"[{self.name}] Dropped {video_id} (no output)")
                    
            except Exception as e:
                self.failed_count += 1
                logger.error(f"[{self.name}] Failed on {video_id}: {e}")
                
                # Update registry for error
                if self.registry:
                    self.update_registry(video_id, success=False, error=str(e))
                
                if self.metrics:
                    self.metrics.record_failure(self.name, video_id, str(e))
                    
            finally:
                self.input_queue.task_done()
        
        # Final save on shutdown
        if self.registry:
            try:
                self.registry.save()
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to save registry on shutdown: {e}")
        
        logger.info(
            f"[{self.name}] Worker stopped. "
            f"Processed: {self.processed_count}, Failed: {self.failed_count}"
        )
    
    async def _run_blocking(self, func, *args):
        """Execute a blocking function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)
    
    def stop(self) -> None:
        """Signal the worker to stop."""
        self.shutdown.set()

