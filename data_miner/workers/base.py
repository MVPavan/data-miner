"""
Base worker classes - shared logic for all stage workers.

Architecture:
- BaseVideoWorker: For download/extract stages (works on Video table)
- BaseProjectWorker: For filter/dedup/detect stages (works on ProjectVideo table)
"""

import os
import signal
import time
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Union

from ..config import (
    VideoStatus, ProjectVideoStatus, StageName,
    VIDEO_INPUT_STATUS, VIDEO_IN_PROGRESS_STATUS, VIDEO_OUTPUT_STATUS,
    PROJECT_VIDEO_INPUT_STATUS, PROJECT_VIDEO_IN_PROGRESS_STATUS, PROJECT_VIDEO_OUTPUT_STATUS,
    get_project_name,
)
from ..db.connection import get_session
from ..db.models import Video, ProjectVideo
from ..db.operations import (
    claim_next_video, release_video, mark_video_failed, update_video_heartbeat,
    claim_next_project_video, release_project_video, mark_project_video_failed, update_project_video_heartbeat,
    get_project_by_name,
)
from ..logging import get_logger

logger = get_logger(__name__)


class _BaseWorker(ABC):
    """
    Base class with common functionality.
    """
    
    stage_name: StageName
    
    def __init__(self, poll_interval: float = 5.0, heartbeat_interval: float = 30.0):
        self.worker_id = f"{self.stage_name.value}-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.running = True
        
        # Heartbeat state
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._current_item_id: Optional[Union[str, int]] = None
        self._lost_lock = False
        self.debug_mode = bool(int(os.getenv("DATA_MINER_DEBUG", "0")))
        
        # Project context
        self.project_id: Optional[int] = None
        self._init_project()
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _init_project(self):
        """Resolve project_id from config at startup."""
        project_name = get_project_name()
        with get_session() as session:
            project = get_project_by_name(session, project_name)
            if project:
                self.project_id = project.project_id
                logger.info(f"[{self.worker_id}] Project: {project_name} (id={self.project_id})")
            else:
                logger.error(f"[{self.worker_id}] Project '{project_name}' not found!")
    
    def _handle_shutdown(self, signum, frame):
        logger.info(f"[{self.worker_id}] Received shutdown signal")
        self.running = False
        # Don't stop heartbeat here - let it continue until process() completes
    
    def _start_heartbeat(self, item_id: Union[str, int]):
        """Start heartbeat thread when processing starts."""
        self._current_item_id = item_id
        self._lost_lock = False
        self._heartbeat_stop.clear()
        
        # Skip heartbeat in debug mode (set DATA_MINER_DEBUG=1)
        if self.debug_mode:
            logger.info(f"[{self.worker_id}] Debug mode: heartbeat disabled")
            return
        
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
    
    def _stop_heartbeat(self):
        """Stop heartbeat thread when processing completes."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
        self._current_item_id = None
    
    @abstractmethod
    def _heartbeat_loop(self):
        """Subclasses implement heartbeat for their table."""
        pass
    
    @abstractmethod
    def run(self):
        """Main worker loop."""
        pass


class BaseVideoWorker(_BaseWorker):
    """
    Base worker for central stages (download, extract).
    Works on the Video table.
    """
    
    @property
    def input_status(self) -> VideoStatus:
        return VIDEO_INPUT_STATUS[self.stage_name]
    
    @property
    def in_progress_status(self) -> VideoStatus:
        return VIDEO_IN_PROGRESS_STATUS[self.stage_name]
    
    @property
    def output_status(self) -> VideoStatus:
        return VIDEO_OUTPUT_STATUS[self.stage_name]
    
    @abstractmethod
    def process(self, video: Video) -> dict:
        """Process a video. Returns dict of fields to update."""
        pass
    
    def _heartbeat_loop(self):
        """Update heartbeat every 30s. Exit if lock lost."""
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            try:
                with get_session() as session:
                    still_owner = update_video_heartbeat(
                        session, self._current_item_id, self.worker_id
                    )
                    if not still_owner:
                        logger.warning(f"[{self.worker_id}] Lost lock on {self._current_item_id}, exiting...")
                        self._lost_lock = True
                        self.running = False
                        os._exit(1)  # Supervisor will restart
            except Exception as e:
                logger.warning(f"[{self.worker_id}] Heartbeat error: {e}")
    
    def run(self):
        """Main worker loop for Video table."""
        logger.info(f"[{self.worker_id}] Starting worker")
        
        if not self.project_id:
            logger.error(f"[{self.worker_id}] No project_id, cannot start")
            return
        
        while self.running:
            try:
                with get_session() as session:
                    video = claim_next_video(
                        session, self.project_id, self.input_status, 
                        self.in_progress_status, self.worker_id,
                        debug_mode=self.debug_mode,
                    )
                    
                    if not video:
                        time.sleep(self.poll_interval)
                        continue
                    
                    logger.info(f"[{self.worker_id}] Processing {video.video_id}")
                    self._start_heartbeat(video.video_id)
                    
                    try:
                        updates = self.process(video)
                        success = release_video(
                            session, video.video_id, self.worker_id,
                            self.output_status, **updates
                        )
                        if success:
                            logger.info(f"[{self.worker_id}] Completed {video.video_id}")
                        else:
                            logger.warning(f"[{self.worker_id}] Lock lost for {video.video_id}")
                    except Exception as e:
                        logger.error(f"[{self.worker_id}] Failed {video.video_id}: {e}")
                        mark_video_failed(session, video.video_id, str(e))
                    finally:
                        self._stop_heartbeat()
                        
            except Exception as e:
                logger.error(f"[{self.worker_id}] Worker error: {e}")
                time.sleep(self.poll_interval)
        
        logger.info(f"[{self.worker_id}] Worker stopped")


class BaseProjectVideosWorker(_BaseWorker):
    """
    Base worker for project-specific stages (filter, dedup, detect).
    Works on the ProjectVideo table.
    """
    
    @property
    def input_status(self) -> ProjectVideoStatus:
        return PROJECT_VIDEO_INPUT_STATUS[self.stage_name]
    
    @property
    def in_progress_status(self) -> ProjectVideoStatus:
        return PROJECT_VIDEO_IN_PROGRESS_STATUS[self.stage_name]
    
    @property
    def output_status(self) -> ProjectVideoStatus:
        return PROJECT_VIDEO_OUTPUT_STATUS[self.stage_name]
    
    @abstractmethod
    def process(self, project_video: ProjectVideo, video: Video) -> dict:
        """Process a project_video. Returns dict of fields to update."""
        pass
    
    def _heartbeat_loop(self):
        """Update heartbeat every 30s. Exit if lock lost."""
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            try:
                with get_session() as session:
                    still_owner = update_project_video_heartbeat(
                        session, self._current_item_id, self.worker_id
                    )
                    if not still_owner:
                        logger.warning(f"[{self.worker_id}] Lost lock on pv:{self._current_item_id}, exiting...")
                        self._lost_lock = True
                        self.running = False
                        os._exit(1)
            except Exception as e:
                logger.warning(f"[{self.worker_id}] Heartbeat error: {e}")
    
    def run(self):
        """Main worker loop for ProjectVideo table."""
        logger.info(f"[{self.worker_id}] Starting worker")
        
        if not self.project_id:
            logger.error(f"[{self.worker_id}] No project_id, cannot start")
            return
        
        while self.running:
            try:
                with get_session() as session:
                    result = claim_next_project_video(
                        session, self.project_id, self.input_status,
                        self.in_progress_status, self.worker_id,
                        debug_mode=self.debug_mode,
                    )
                    
                    if not result:
                        time.sleep(self.poll_interval)
                        continue
                    
                    pv, video = result
                    logger.info(f"[{self.worker_id}] Processing pv:{pv.id} video:{video.video_id}")
                    self._start_heartbeat(pv.id)
                    
                    try:
                        updates = self.process(pv, video)
                        # Check if worker signaled a skip status (0 frames case)
                        final_status = updates.pop("_skip_status", None) or self.output_status
                        success = release_project_video(
                            session, pv.id, self.worker_id,
                            final_status, **updates
                        )
                        if success:
                            logger.info(f"[{self.worker_id}] Completed pv:{pv.id} status:{final_status.value}")
                        else:
                            logger.warning(f"[{self.worker_id}] Lock lost for pv:{pv.id}")
                    except Exception as e:
                        logger.error(f"[{self.worker_id}] Failed pv:{pv.id}: {e}")
                        mark_project_video_failed(session, pv.id, str(e))
                    finally:
                        self._stop_heartbeat()
                        
            except Exception as e:
                logger.error(f"[{self.worker_id}] Worker error: {e}")
                time.sleep(self.poll_interval)
        
        logger.info(f"[{self.worker_id}] Worker stopped")


class BaseProjectStageWorker(ABC):
    """
    Base class for project-level stage workers (cross-dedup, detection).
    
    Unlike per-video workers, these operate on entire projects.
    No heartbeat needed since project operations are atomic transitions.
    
    Subclasses must implement:
    - claim_project(): Claim and return a project to process
    - process(): Process the claimed project
    - complete_project(): Mark project as complete
    - fail_project(): Mark project as failed
    """
    
    worker_name: str = "project-stage"  # Override in subclass
    
    def __init__(self, poll_interval: int = 10):
        self.poll_interval = poll_interval
        self.worker_id = f"{self.worker_name}-{os.getpid()}"
        self._running = False
    
    @abstractmethod
    def claim_project(self, session) -> Optional[any]:
        """Claim a project to process. Returns Project if claimed, None otherwise."""
        pass
    
    @abstractmethod
    def process(self, project) -> dict:
        """Process the project. Returns result dict."""
        pass
    
    @abstractmethod
    def complete_project(self, session, project_id: int) -> bool:
        """Mark project as complete."""
        pass
    
    @abstractmethod
    def fail_project(self, session, project_id: int, error: str) -> bool:
        """Mark project as failed."""
        pass
    
    def run(self):
        """Main worker loop - shared by all project-level workers."""
        self._running = True
        logger.info(f"[{self.worker_id}] Starting {self.worker_name} worker")
        
        while self._running:
            try:
                with get_session() as session:
                    project = self.claim_project(session)
                    
                    if not project:
                        time.sleep(self.poll_interval)
                        continue
                    
                    logger.info(f"[{self.worker_id}] Processing project: {project.name}")
                    
                    try:
                        result = self.process(project)
                        self.complete_project(session, project.project_id, result)
                        logger.info(f"[{self.worker_id}] Completed {self.worker_name} for: {project.name}")
                    except Exception as e:
                        logger.error(f"[{self.worker_id}] Failed: {e}")
                        self.fail_project(session, project.project_id, str(e))
                        
            except Exception as e:
                logger.error(f"[{self.worker_id}] Worker error: {e}")
                time.sleep(self.poll_interval)
        
        logger.info(f"[{self.worker_id}] Worker stopped")
    
    def stop(self):
        """Stop the worker."""
        self._running = False
