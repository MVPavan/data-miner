"""
Detection worker - project-level detection on cross-dedup'd frames.
"""

import os
from pathlib import Path

from sqlmodel import Session

from .base import BaseProjectStageWorker
from ..db.connection import get_session
from ..db.models import Project
from ..db.operations import (
    claim_project_for_detection,
    complete_project_detection,
    mark_project_failed,
)
from ..modules.detector import ObjectDetector
from ..config import get_detection_config, get_deduplication_config, DetectionConfig, init_hf_auth
from ..logging import get_logger

logger = get_logger(__name__)


class ProjectDetectWorker(BaseProjectStageWorker):
    """
    Project-level worker that runs detection on cross-dedup'd frames.
    
    Claims projects in DETECT_READY stage and runs detection on
    all frames in the dedup output directory.
    """
    
    worker_name = "detect"
    
    def __init__(self, config: DetectionConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config or get_detection_config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dedup output dir to know where frames are
        self.dedup_config = get_deduplication_config()
        
        # Authenticate with HuggingFace for private models
        init_hf_auth()
        
        self._detector = ObjectDetector(self.config, device_map=self.config.device)
    
    def claim_project(self, session: Session) -> Project | None:
        """Claim a project in DETECT_READY stage."""
        return claim_project_for_detection(session, self.worker_id)
    
    def complete_project(self, session: Session, project_id: int, result: dict) -> bool:
        """Mark project detection complete with result data."""
        return complete_project_detection(
            session, 
            project_id,
            detect_dir=str(self.config.output_dir) if self.config.output_dir else None,
        )
    
    def fail_project(self, session: Session, project_id: int, error: str) -> bool:
        """Mark project as failed."""
        return mark_project_failed(session, project_id, error)
    
    def process(self, project: Project) -> dict:
        """Run detection on all cross-dedup'd frames."""
        # Frames are in the dedup output directory (flat)
        dedup_dir = self.dedup_config.output_dir
        
        if not dedup_dir.exists():
            logger.warning(f"[{self.worker_id}] Dedup dir not found: {dedup_dir}")
            return {"detection_count": 0}
        
        # Get all frames
        frame_paths = list(dedup_dir.glob("*.jpg")) + list(dedup_dir.glob("*.png"))
        
        if not frame_paths:
            logger.warning(f"[{self.worker_id}] No frames in dedup dir for project {project.name}")
            return {"detection_count": 0}
        
        logger.info(f"[{self.worker_id}] Running detection on {len(frame_paths)} frames")
        
        # Run detection
        result = self._detector.detect_batch(image_paths=frame_paths)
        
        logger.info(f"[{self.worker_id}] Detection complete: {result.total_detections} detections")
        
        return {
            "detection_count": result.total_detections,
            "frames_processed": len(frame_paths),
        }


def main():
    """Entry point for detection worker."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config YAML file path")
    args = parser.parse_args()
    
    if args.config:
        os.environ["DATA_MINER_CONFIG"] = args.config
    
    worker = ProjectDetectWorker()
    worker.run()


if __name__ == "__main__":
    main()
