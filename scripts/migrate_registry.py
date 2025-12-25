#!/usr/bin/env python3
"""
Migration script: video_registry YAML → PostgreSQL

Migrates VideoRegistry YAML files from video_miner_async output folders
to the new PostgreSQL database. Project name is derived from folder name.

Usage:
    # Migrate a single registry file (project name = parent folder)
    python migrate_registry.py /path/to/glass_doors/video_registry.yaml
    
    # Migrate all registries in an output directory
    python migrate_registry.py /path/to/output --all
    
    # Dry run - show what would be migrated
    python migrate_registry.py /path/to/output --all --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_miner.db.connection import get_session, create_tables
from data_miner.db.operations import add_video, get_or_create_project
from data_miner.db.models import Video, ProjectVideo
from data_miner.config import VideoStatus, ProjectVideoStatus, SourceType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Status mapping from v3 to new schema
V3_TO_VIDEO_STATUS = {
    "pending": VideoStatus.PENDING,
    "downloading": VideoStatus.DOWNLOADING,
    "downloaded": VideoStatus.DOWNLOADED,
    "extracting": VideoStatus.EXTRACTING,
    "extracted": VideoStatus.EXTRACTED,
    "filtering": VideoStatus.EXTRACTED,  # After extract, video is done
    "filtered": VideoStatus.EXTRACTED,
    "deduplicating": VideoStatus.EXTRACTED,
    "deduplicated": VideoStatus.EXTRACTED,
    "detecting": VideoStatus.EXTRACTED,
    "detected": VideoStatus.EXTRACTED,
    "complete": VideoStatus.EXTRACTED,
    "failed": VideoStatus.FAILED,
    "skipped": VideoStatus.FAILED,
}

V3_TO_PROJECT_STATUS = {
    "pending": ProjectVideoStatus.PENDING,
    "downloading": ProjectVideoStatus.PENDING,
    "downloaded": ProjectVideoStatus.PENDING,
    "extracting": ProjectVideoStatus.PENDING,
    "extracted": ProjectVideoStatus.PENDING,
    "filtering": ProjectVideoStatus.FILTERING,
    "filtered": ProjectVideoStatus.FILTERED,
    "deduplicating": ProjectVideoStatus.DEDUPLICATING,
    "deduplicated": ProjectVideoStatus.DEDUPLICATED,
    "detecting": ProjectVideoStatus.DETECTING,
    "detected": ProjectVideoStatus.DETECTED,
    "complete": ProjectVideoStatus.DETECTED,
    "failed": ProjectVideoStatus.FAILED,
    "skipped": ProjectVideoStatus.FAILED,
}


def load_registry(path: Path) -> dict:
    """Load VideoRegistry YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def migrate_registry(
    registry_path: Path,
    project_name: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate a single video_registry.yaml to PostgreSQL.
    
    Args:
        registry_path: Path to video_registry.yaml
        project_name: Project name (default: parent folder name)
        dry_run: If True, don't actually insert
        
    Returns:
        (videos_added, videos_skipped)
    """
    # Derive project name from parent folder
    if not project_name:
        project_name = registry_path.parent.name
    
    logger.info(f"Migrating: {registry_path}")
    logger.info(f"Project: {project_name}")
    
    # Load registry
    data = load_registry(registry_path)
    videos = data.get("videos", {})
    
    if not videos:
        logger.warning("No videos found in registry")
        return 0, 0
    
    logger.info(f"Found {len(videos)} videos")
    
    added = 0
    skipped = 0
    
    if dry_run:
        for vid_id, entry in list(videos.items())[:5]:
            status = entry.get("status", "pending")
            title = entry.get("title", "")[:50]
            logger.info(f"  {vid_id}: {status} - {title}")
        if len(videos) > 5:
            logger.info(f"  ... and {len(videos) - 5} more")
        return len(videos), 0
    
    with get_session() as session:
        # Get or create project
        project = get_or_create_project(session, project_name)
        
        for vid_id, entry in videos.items():
            try:
                url = entry.get("url", "")
                status_str = entry.get("status", "pending")
                stages = entry.get("stages", {})
                
                # Check if video already exists
                existing = session.get(Video, vid_id)
                if existing:
                    skipped += 1
                    continue
                
                # Map status
                video_status = V3_TO_VIDEO_STATUS.get(status_str, VideoStatus.PENDING)
                project_status = V3_TO_PROJECT_STATUS.get(status_str, ProjectVideoStatus.PENDING)
                
                # Create Video entry
                video = Video(
                    video_id=vid_id,
                    url=url,
                    title=entry.get("title"),
                    source_type=SourceType.SEARCH if entry.get("source_keyword") else SourceType.MANUAL,
                    source_info=entry.get("source_keyword"),
                    status=video_status,
                    video_path=stages.get("download", {}).get("path"),
                    frames_dir=stages.get("extraction", {}).get("output_dir"),
                    frame_count=stages.get("extraction", {}).get("total_frames"),
                )
                session.add(video)
                
                # Create ProjectVideo entry
                pv = ProjectVideo(
                    project_id=project.project_id,
                    video_id=vid_id,
                    status=project_status,
                    filtered_dir=stages.get("filter", {}).get("output_dir"),
                    passed_frames=stages.get("filter", {}).get("passed_frames"),
                    dedup_dir=stages.get("deduplication", {}).get("output_dir"),
                    unique_frames=stages.get("deduplication", {}).get("unique_frames"),
                    detection_dir=stages.get("detection", {}).get("output_dir"),
                    detection_count=stages.get("detection", {}).get("total_detections"),
                )
                session.add(pv)
                
                added += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate {vid_id}: {e}")
                skipped += 1
        
        session.commit()
    
    return added, skipped


def find_registries(output_dir: Path) -> list[Path]:
    """Find all video_registry.yaml files in output directory."""
    registries = []
    
    for path in output_dir.iterdir():
        if path.is_dir():
            registry = path / "video_registry.yaml"
            if registry.exists():
                registries.append(registry)
    
    return sorted(registries)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate video_registry YAML files to PostgreSQL"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to video_registry.yaml or output directory (with --all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all registries in the directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without inserting"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database tables before migration"
    )
    
    args = parser.parse_args()
    
    # Initialize DB if requested
    if args.init_db:
        logger.info("Initializing database tables...")
        create_tables()
    
    # Find registries to migrate
    if args.all:
        if not args.path.is_dir():
            logger.error(f"--all requires a directory: {args.path}")
            return 1
        registries = find_registries(args.path)
        if not registries:
            logger.error(f"No video_registry.yaml files found in {args.path}")
            return 1
        logger.info(f"Found {len(registries)} registries to migrate")
    else:
        if not args.path.exists():
            logger.error(f"File not found: {args.path}")
            return 1
        registries = [args.path]
    
    # Migrate each registry
    total_added = 0
    total_skipped = 0
    
    for registry in registries:
        added, skipped = migrate_registry(registry, dry_run=args.dry_run)
        total_added += added
        total_skipped += skipped
    
    # Summary
    print()
    if args.dry_run:
        print(f"DRY RUN: Would add {total_added} videos")
    else:
        print(f"✓ Added: {total_added} videos")
        print(f"  Skipped: {total_skipped} (already exist)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
