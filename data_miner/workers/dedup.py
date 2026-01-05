"""
Cross-Dedup worker - project-level deduplication across all videos.
"""

import os
from pathlib import Path

from sqlmodel import Session

from .base import BaseProjectStageWorker
from ..db.connection import get_session
from ..db.models import Project
from ..db.operations import (
    claim_project_for_cross_dedup,
    complete_project_cross_dedup,
    mark_project_failed,
    get_filtered_frame_dirs,
)
from ..modules.deduplicator import Deduplicator
from ..config import get_deduplication_config, DeduplicationConfig, init_hf_auth
from ..logging import get_logger

logger = get_logger(__name__)


class CrossDedupWorker(BaseProjectStageWorker):
    """
    Project-level worker that runs cross-video deduplication.
    
    Claims projects where all videos are done filtering, then runs
    FAISS-based deduplication across all frames from all videos.
    """
    
    worker_name = "cross-dedup"
    
    def __init__(self, config: DeduplicationConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config or get_deduplication_config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Authenticate with HuggingFace for private models
        init_hf_auth()
        
        self._dedup = Deduplicator(self.config, device_map=self.config.device)
    
    def claim_project(self, session: Session) -> Project | None:
        """Claim a project in DEDUP_READY stage."""
        return claim_project_for_cross_dedup(session, self.worker_id)
    
    def complete_project(self, session: Session, project_id: int, result: dict) -> bool:
        """Mark project cross-dedup complete with result data."""
        return complete_project_cross_dedup(
            session, 
            project_id,
            unique_frames=result.get("unique_frames", 0),
            dedup_dir=str(self.config.output_dir) if self.config.output_dir else None,
        )
    
    def fail_project(self, session: Session, project_id: int, error: str) -> bool:
        """Mark project as failed."""
        return mark_project_failed(session, project_id, error)
    
    def process(self, project: Project) -> dict:
        """Run cross-video deduplication on all filtered frames."""
        with get_session() as session:
            # Get filtered directories: {video_id: filtered_dir}
            filtered_dirs = get_filtered_frame_dirs(session, project.project_id)
        
        if not filtered_dirs:
            logger.warning(f"[{self.worker_id}] No filtered frames for project {project.name}")
            return {"unique_frames": 0}
        
        # Build frame_groups: {video_id: [frame_paths]}
        frame_groups: dict[str, list[Path]] = {}
        for video_id, dir_path in filtered_dirs.items():
            dir_path = Path(dir_path)
            if dir_path.exists():
                frames = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
                if frames:
                    frame_groups[video_id] = frames
        
        if not frame_groups:
            logger.warning(f"[{self.worker_id}] No frame files found in filtered dirs")
            return {"unique_frames": 0}
        
        total_frames = sum(len(frames) for frames in frame_groups.values())
        logger.info(
            f"[{self.worker_id}] Cross-deduplicating {total_frames} frames "
            f"from {len(frame_groups)} videos"
        )
        
        # Run two-phase cross-video deduplication
        result = self._dedup.deduplicate_cross_video(
            frame_groups=frame_groups,
            show_progress=False,
        )
        
        logger.info(
            f"[{self.worker_id}] Cross-dedup complete: {result.unique_frames}/{result.total_frames} unique "
            f"({result.dedup_rate:.1%} duplicates removed)"
        )
        
        return {
            "unique_frames": result.unique_frames,
            "total_frames": result.total_frames,
            "duplicates_removed": result.duplicates_removed,
        }


def main():
    """Entry point for cross-dedup worker."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = CrossDedupWorker()
    worker.run()


if __name__ == "__main__":
    main()
