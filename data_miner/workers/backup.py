"""
Backup worker - syncs frames_raw to destination after all project_videos are filtered.

Supports both local and remote (SSH) destinations with multi-phase verification.
"""

import os
import signal
import time
import subprocess
import shutil
from pathlib import Path

from sqlmodel import Session

from ..config import BackupConfig, get_backup_config
from ..db.connection import engine
from ..db.operations import get_videos_ready_for_backup, mark_video_backed_up
from ..logging import get_logger

logger = get_logger(__name__)


class BackupWorker:
    """
    Backup worker that syncs frames_raw folders to destination.
    
    Supports local paths and remote SSH destinations.
    Only syncs videos where all project_videos are in terminal status.
    """
    
    worker_name = "backup"
    
    def __init__(self, config: BackupConfig | None = None):
        self.config = config or get_backup_config()
        self.dest = self.config.remote_dest
        self.is_remote = self.config.is_remote
        self.poll_interval = self.config.poll_interval
        self.verification_timeout = self.config.verification_timeout
        self.delete_after_backup = self.config.delete_after_backup
        self.worker_id = f"backup-{os.getpid()}"
        self._running = False
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        logger.info(f"[{self.worker_id}] Received shutdown signal")
        self._running = False
    
    def run(self):
        """Main backup loop."""
        self._running = True
        mode = "remote (SSH)" if self.is_remote else "local"
        logger.info(f"[{self.worker_id}] Starting backup worker ({mode})")
        logger.info(f"[{self.worker_id}] Destination: {self.dest}")
        logger.info(f"[{self.worker_id}] Poll interval: {self.poll_interval}s")
        
        while self._running:
            try:
                with Session(engine) as session:
                    videos = get_videos_ready_for_backup(session)
                    
                    if videos:
                        logger.info(f"[{self.worker_id}] Found {len(videos)} videos ready for backup")
                    
                    for video in videos:
                        video_id = video["video_id"]
                        frames_dir = video["frames_dir"]
                        
                        logger.info(f"[{self.worker_id}] Backing up {video_id}: {frames_dir}")
                        
                        if self._backup_and_verify(frames_dir, video_id):
                            mark_video_backed_up(session, video_id)
                            logger.info(f"[{self.worker_id}] ✓ Backup complete: {video_id}")
                            
                            if self.delete_after_backup:
                                self._delete_source(frames_dir, video_id)
                        else:
                            logger.error(f"[{self.worker_id}] ✗ Backup failed: {video_id}")
            
            except Exception as e:
                logger.error(f"[{self.worker_id}] Error: {e}")
            
            time.sleep(self.poll_interval)
        
        logger.info(f"[{self.worker_id}] Backup worker stopped")
    
    def stop(self):
        """Stop the worker."""
        self._running = False
    
    def _backup_and_verify(self, frames_dir: str, video_id: str) -> bool:
        """
        Backup folder with multi-phase verification.
        
        Phase 0: Create destination directory
        Phase 1: rsync with checksum
        Phase 2: Verify file count
        Phase 3: rsync dry-run checksum verification
        """
        source_path = Path(frames_dir)
        if not source_path.exists():
            logger.error(f"[{self.worker_id}] Source not found: {frames_dir}")
            return False
        
        # Build destination path (preserve video_id folder name)
        dest_path = f"{self.dest.rstrip('/')}/{video_id}"
        
        # Phase 0: Create destination directory
        if not self._create_dest_directory(dest_path):
            logger.error(f"[{self.worker_id}] Phase 0 (create dir) failed: {video_id}")
            return False
        
        # Phase 1: Sync with checksum
        if not self._sync_folder(source_path, dest_path):
            logger.error(f"[{self.worker_id}] Phase 1 (sync) failed: {video_id}")
            return False
        
        # Phase 2: Verify file count
        if not self._verify_file_count(source_path, dest_path):
            logger.error(f"[{self.worker_id}] Phase 2 (file count) failed: {video_id}")
            return False
        
        # Phase 3: rsync dry-run checksum verification
        if not self._verify_checksums(source_path, dest_path):
            logger.error(f"[{self.worker_id}] Phase 3 (checksum) failed: {video_id}")
            return False
        
        logger.info(f"[{self.worker_id}] All verification phases passed: {video_id}")
        return True
    
    def _create_dest_directory(self, dest: str) -> bool:
        """Create destination directory if it doesn't exist."""
        try:
            if self.is_remote:
                # Remote: SSH mkdir -p
                remote_host, remote_path = dest.split(":", 1)
                result = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                     remote_host, f"mkdir -p '{remote_path}'"],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    logger.error(f"[{self.worker_id}] Remote mkdir failed: {result.stderr}")
                    return False
            else:
                # Local: pathlib mkdir
                Path(dest).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"[{self.worker_id}] Create directory error: {e}")
            return False
    
    def _sync_folder(self, source: Path, dest: str) -> bool:
        """Sync folder using rsync with checksum."""
        rsync_cmd = [
            "rsync",
            "-avz",
            "--checksum",
            "--progress",
            "--mkpath",
        ]
        
        # Add SSH options for remote destinations
        if self.is_remote:
            rsync_cmd.extend(["-e", "ssh -o BatchMode=yes -o ConnectTimeout=30"])
        
        rsync_cmd.extend([f"{source}/", f"{dest}/"])
        
        try:
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error(f"[{self.worker_id}] rsync timeout")
            return False
        except Exception as e:
            logger.error(f"[{self.worker_id}] rsync error: {e}")
            return False
    
    def _verify_file_count(self, source: Path, dest: str) -> bool:
        """Quick verification: compare file counts."""
        local_count = sum(1 for _ in source.rglob("*") if _.is_file())
        
        if self.is_remote:
            # Remote: SSH to count files
            remote_host, remote_path = dest.split(":", 1)
            try:
                result = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", remote_host,
                     f"find '{remote_path}' -type f | wc -l"],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    remote_count = int(result.stdout.strip())
                else:
                    return False
            except Exception as e:
                logger.error(f"[{self.worker_id}] File count error: {e}")
                return False
        else:
            # Local: direct count
            dest_path = Path(dest)
            if not dest_path.exists():
                return False
            remote_count = sum(1 for _ in dest_path.rglob("*") if _.is_file())
        
        if local_count == remote_count:
            logger.info(f"[{self.worker_id}] File count match: {local_count}")
            return True
        else:
            logger.error(f"[{self.worker_id}] File count mismatch: local={local_count}, dest={remote_count}")
            return False
    
    def _verify_checksums(self, source: Path, dest: str) -> bool:
        """Strict verification: rsync dry-run with checksums."""
        verify_cmd = [
            "rsync",
            "-avz",
            "--checksum",
            "--dry-run",
        ]
        
        if self.is_remote:
            verify_cmd.extend(["-e", "ssh -o BatchMode=yes -o ConnectTimeout=30"])
        
        verify_cmd.extend([f"{source}/", f"{dest}/"])
        
        try:
            result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=self.verification_timeout
            )
            
            if result.returncode == 0:
                output_lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
                file_diffs = [l for l in output_lines 
                             if not l.startswith("sending incremental")
                             and not l.startswith("sent ")
                             and not l.startswith("total size")
                             and l != "./"
                             and l]
                
                if not file_diffs:
                    logger.info(f"[{self.worker_id}] Checksum verification passed")
                    return True
                else:
                    logger.error(f"[{self.worker_id}] Found {len(file_diffs)} differences")
                    for diff in file_diffs[:5]:
                        logger.error(f"  - {diff}")
                    return False
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"[{self.worker_id}] Verification timeout")
            return False
        except Exception as e:
            logger.error(f"[{self.worker_id}] Verification error: {e}")
            return False
    
    def _delete_source(self, frames_dir: str, video_id: str):
        """Delete source folder after verified backup."""
        try:
            shutil.rmtree(frames_dir)
            logger.info(f"[{self.worker_id}] Deleted source: {frames_dir}")
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to delete source: {e}")


def main():
    """Entry point for backup worker."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = BackupWorker()
    worker.run()


if __name__ == "__main__":
    main()
