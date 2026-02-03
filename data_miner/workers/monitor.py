"""
Project Monitor worker - handles project stage transitions and lock recovery.

Responsibilities:
- POPULATING → FILTERING (when any video starts filtering)
- FILTERING → DEDUP_READY (when all videos are filtered)
- Reset projects with PENDING videos (if past FILTERING stage)
- Reset stale video/project_video locks
- Warn about long-running locks (>30 mins)
- Cleanup extracted videos (delete source after extraction)
"""

import os
import signal
import time
from pathlib import Path

from sqlmodel import Session

from ..config import MonitorConfig, get_monitor_config
from ..db.connection import engine
from ..db.operations import (
    transition_populating_to_filtering,
    transition_filtering_to_dedup_ready,
    reset_projects_with_pending_videos,
    update_project_frame_counts,
    reset_stale_video_locks,
    reset_stale_project_video_locks,
    get_long_running_video_locks,
    get_long_running_project_video_locks,
    get_extracted_videos_for_cleanup,
    clear_video_path_db,
)
from ..logging import get_logger

logger = get_logger(__name__)


class ProjectMonitorWorker:
    """
    Monitors and transitions project stages, recovers stale locks.
    
    Responsibilities:
    - Check POPULATING projects and transition to FILTERING
    - Check FILTERING projects and transition to DEDUP_READY
    - Reset stale locks (heartbeat > 2 min)
    - Warn about long-running locks (> 30 min)
    - Cleanup extracted videos (if configured)
    
    Runs every poll_interval seconds.
    """
    
    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or get_monitor_config()
        self.poll_interval = self.config.poll_interval
        self.stale_threshold_minutes = self.config.stale_threshold_minutes
        self.long_running_threshold_minutes = self.config.long_running_threshold_minutes
        self.cleanup_extracted_videos = self.config.cleanup_extracted_videos
        self.worker_id = f"monitor-{os.getpid()}"
        self._running = False
        # Track warned locks to avoid spamming logs
        self._warned_video_locks: set[str] = set()
        self._warned_pv_locks: set[int] = set()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        logger.info(f"[{self.worker_id}] Received shutdown signal")
        self._running = False
    
    def run(self):
        """Main monitor loop."""
        self._running = True
        logger.info(f"[{self.worker_id}] Starting project monitor")
        
        while self._running:
            try:
                with Session(engine) as session:
                    # Stage transitions
                    count1 = transition_populating_to_filtering(session)
                    if count1 > 0:
                        logger.info(f"[{self.worker_id}] Transitioned {count1} projects: POPULATING → FILTERING")
                    
                    count2 = transition_filtering_to_dedup_ready(session)
                    if count2 > 0:
                        logger.info(f"[{self.worker_id}] Transitioned {count2} projects: FILTERING → DEDUP_READY")
                    
                    # Consistency: reset projects with PENDING videos that are past FILTERING
                    count3 = reset_projects_with_pending_videos(session)
                    if count3 > 0:
                        logger.info(f"[{self.worker_id}] Reset {count3} projects with PENDING videos back to POPULATING")
                    
                    # Update frame counts (only writes if changed)
                    count4 = update_project_frame_counts(session)
                    if count4 > 0:
                        logger.debug(f"[{self.worker_id}] Updated frame counts for {count4} projects")
                    
                    # Stale lock recovery
                    stale_videos = reset_stale_video_locks(session, self.stale_threshold_minutes)
                    if stale_videos > 0:
                        logger.warning(f"[{self.worker_id}] Reset {stale_videos} stale video locks")
                    
                    stale_pvs = reset_stale_project_video_locks(session, self.stale_threshold_minutes)
                    if stale_pvs > 0:
                        logger.warning(f"[{self.worker_id}] Reset {stale_pvs} stale project_video locks")
                    
                    # Long-running lock warnings (including debug sessions)
                    self._warn_long_running_locks(session)
                    
                    # Cleanup extracted videos (delete source files)
                    if self.cleanup_extracted_videos:
                        self._cleanup_extracted_videos(session)
                        
            except Exception as e:
                logger.error(f"[{self.worker_id}] Monitor error: {e}")
            
            time.sleep(self.poll_interval)
            # log monitor heartbeat every 10 minutes
            if int(time.time()) % 600 < self.poll_interval:
                logger.info(f"[{self.worker_id}] Monitor heartbeat at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info(f"[{self.worker_id}] Monitor stopped")
    
    def _warn_long_running_locks(self, session: Session):
        """Warn about locks that have been held for too long."""
        # Video locks
        long_videos = get_long_running_video_locks(session, self.long_running_threshold_minutes)
        for v in long_videos:
            video_id = v["video_id"]
            if video_id not in self._warned_video_locks:
                logger.warning(
                    f"[{self.worker_id}] LONG-RUNNING: video {video_id} locked by {v['locked_by']} "
                    f"for {v['minutes_locked']:.1f} mins (status: {v['status']})"
                )
                self._warned_video_locks.add(video_id)
        
        # Clean up warnings for released locks
        current_video_ids = {v["video_id"] for v in long_videos}
        self._warned_video_locks &= current_video_ids
        
        # Project video locks
        long_pvs = get_long_running_project_video_locks(session, self.long_running_threshold_minutes)
        for pv in long_pvs:
            pv_id = pv["id"]
            if pv_id not in self._warned_pv_locks:
                logger.warning(
                    f"[{self.worker_id}] LONG-RUNNING: pv:{pv_id} (video:{pv['video_id']}) "
                    f"locked by {pv['locked_by']} for {pv['minutes_locked']:.1f} mins (status: {pv['status']})"
                )
                self._warned_pv_locks.add(pv_id)
        
        # Clean up warnings for released locks
        current_pv_ids = {pv["id"] for pv in long_pvs}
        self._warned_pv_locks &= current_pv_ids
    
    def _cleanup_extracted_videos(self, session: Session):
        """Delete source video files after extraction."""
        videos = get_extracted_videos_for_cleanup(session)
        
        for video_id, video_path in videos:
            try:
                path = Path(video_path)
                if path.exists():
                    path.unlink()
                    logger.info(f"[{self.worker_id}] Deleted source video: {video_path}")
                
                # Clear path in DB after deletion (or if file already gone)
                clear_video_path_db(session, video_id)
            except Exception as e:
                logger.warning(f"[{self.worker_id}] Failed to delete video {video_id}: {e}")
    
    def stop(self):
        """Stop the monitor."""
        self._running = False


def main():
    """Entry point for project monitor."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = ProjectMonitorWorker()  # Uses config
    worker.run()


if __name__ == "__main__":
    main()
