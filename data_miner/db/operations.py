# mypy: disable-error-code="call-overload"
"""
Database operations - claim, release, and CRUD functions.

Architecture:
- Video operations: claim_next_video, release_video (for download/extract)
- ProjectVideo operations: claim_next_project_video, release_project_video (for filter/dedup/detect)
"""

from datetime import datetime
from typing import Optional

from sqlmodel import Session, select, text, func

from .models import Video, Project, ProjectVideo
from ..config import (
    VideoStatus, ProjectVideoStatus, ProjectStatus, SourceType,
    FILTER_TERMINAL_STATUSES,
)


# =============================================================================
# Video + Project CRUD
# =============================================================================

def add_video(
    session: Session,
    video_id: str,
    url: str,
    project_id: int,
    source_type: SourceType,
    source_info: Optional[str] = None,
) -> tuple[Video, ProjectVideo, bool]:
    """
    Add a video to the database and link it to a project.
    Creates Video if not exists, always creates ProjectVideo entry.
    
    Returns:
        (Video, ProjectVideo, is_new_video) tuple - is_new_video is True if video was created
    """
    # Get or create video
    video = session.get(Video, video_id)
    is_new_video = video is None
    if not video:
        video = Video(
            video_id=video_id, 
            url=url, 
            source_type=source_type,
            source_info=source_info,
        )
        session.add(video)
    
    # Check if already linked to project
    stmt = select(ProjectVideo).where(
        ProjectVideo.project_id == project_id,
        ProjectVideo.video_id == video_id,
    )
    pv = session.exec(stmt).first()
    
    if not pv:
        pv = ProjectVideo(
            project_id=project_id,
            video_id=video_id,
        )
        session.add(pv)
    
    session.commit()
    session.refresh(video)
    session.refresh(pv)
    return video, pv, is_new_video


def get_or_create_project(
    session: Session,
    name: str,
    output_dir: Optional[str] = None,
) -> Project:
    """Get existing project or create new one."""
    stmt = select(Project).where(Project.name == name)
    project = session.exec(stmt).first()
    if not project:
        project = Project(name=name, output_dir=output_dir)
        session.add(project)
        session.commit()
        session.refresh(project)
    return project


def get_project_by_name(session: Session, name: str) -> Optional[Project]:
    """Get project by name."""
    stmt = select(Project).where(Project.name == name)
    return session.exec(stmt).first()


# =============================================================================
# Video Operations (Central - download/extract stages)
# =============================================================================

def claim_next_video(
    session: Session,
    project_id: int,
    status: VideoStatus,
    in_progress_status: VideoStatus,
    worker_id: str,
    debug_mode: bool = False,
) -> Optional[Video]:
    """
    Atomically claim the next video for a project with given status.
    Uses FOR UPDATE SKIP LOCKED for concurrent worker safety.
    
    Args:
        session: Database session
        project_id: Project to scope the query
        status: Status to claim (e.g., PENDING for download)
        in_progress_status: Status to set when claimed (e.g., DOWNLOADING)
        worker_id: Unique worker identifier
        debug_mode: If True, set heartbeat_at=NULL (won't be auto-expired by monitor)
        
    Returns:
        Video if claimed, None if no work available
    """
    # In debug mode: heartbeat_at = NULL (never expires)
    # In production: heartbeat_at = NOW() (monitor will reset if stale)
    result = session.exec(
        text("""
            UPDATE videos 
            SET locked_by = :worker_id, 
                locked_at = NOW(), 
                heartbeat_at = CASE WHEN :debug_mode THEN NULL ELSE NOW() END,
                status = :in_progress_status
            WHERE video_id = (
                SELECT v.video_id FROM videos v
                JOIN project_videos pv ON v.video_id = pv.video_id
                WHERE pv.project_id = :project_id
                  AND v.status = :status
                  AND v.locked_by IS NULL
                ORDER BY v.created_at
                FOR UPDATE SKIP LOCKED 
                LIMIT 1
            )
            RETURNING *
        """).bindparams(
            worker_id=worker_id, 
            status=status.value, 
            in_progress_status=in_progress_status.value,
            project_id=project_id,
            debug_mode=debug_mode,
        )
    )
    row = result.fetchone()
    if row:
        session.commit()  # Persist the lock
        return Video.model_validate(row._mapping)
    return None


def update_video_heartbeat(session: Session, video_id: str, worker_id: str) -> bool:
    """
    Update heartbeat timestamp. Returns False if lock was lost.
    """
    result = session.exec(
        text("""
            UPDATE videos SET heartbeat_at = NOW() 
            WHERE video_id = :video_id AND locked_by = :worker_id
            RETURNING video_id
        """).bindparams(video_id=video_id, worker_id=worker_id)
    )
    session.commit()
    return result.fetchone() is not None


def release_video(
    session: Session,
    video_id: str,
    worker_id: str,
    new_status: VideoStatus,
    **updates,
) -> bool:
    """
    Release lock and update video status/fields.
    Returns False if lock was lost (another worker completed this).
    """
    video = session.get(Video, video_id)
    if not video or video.locked_by != worker_id:
        return False  # Lock lost
    
    video.status = new_status
    video.locked_by = None
    video.locked_at = None
    video.heartbeat_at = None
    video.updated_at = datetime.utcnow()
    
    for key, value in updates.items():
        if hasattr(video, key):
            setattr(video, key, value)
    
    session.add(video)
    session.commit()
    return True


def mark_video_failed(
    session: Session,
    video_id: str,
    error: str,
    cascade: bool = True,
) -> None:
    """
    Mark video as failed with error message.
    
    If cascade=True (default), also marks all linked project_videos as FAILED.
    """
    video = session.get(Video, video_id)
    if video:
        video.status = VideoStatus.FAILED
        video.error = error
        video.locked_by = None
        video.locked_at = None
        video.heartbeat_at = None
        video.updated_at = datetime.utcnow()
        session.add(video)
    
    # Cascade failure to all project_videos for this video
    if cascade:
        pv_ids = session.exec(
            select(ProjectVideo.id).where(ProjectVideo.video_id == video_id)
        ).all()
        for pv_id in pv_ids:
            mark_project_video_failed(session, pv_id, f"Video failed: {error}", commit=False)
    
    session.commit()


# =============================================================================
# ProjectVideo Operations (Per-project - filter/dedup/detect stages)
# =============================================================================

def claim_next_project_video(
    session: Session,
    project_id: int,
    status: ProjectVideoStatus,
    in_progress_status: ProjectVideoStatus,
    worker_id: str,
    debug_mode: bool = False,
) -> Optional[tuple[ProjectVideo, Video]]:
    """
    Atomically claim the next project_video with given status.
    Only claims if:
    - Underlying video is EXTRACTED
    - Project is in POPULATING or FILTERING stage
    
    Args:
        status: Status to look for (e.g., PENDING)
        in_progress_status: Status to set when claimed (e.g., FILTERING)
        debug_mode: If True, set heartbeat_at=NULL (won't be auto-expired by monitor)
    
    Returns:
        (ProjectVideo, Video) tuple if claimed, None if no work
    """
    # In debug mode: heartbeat_at = NULL (never expires)
    # In production: heartbeat_at = NOW() (monitor will reset if stale)
    result = session.exec(
        text("""
            UPDATE project_videos 
            SET locked_by = :worker_id, 
                locked_at = NOW(), 
                heartbeat_at = CASE WHEN :debug_mode THEN NULL ELSE NOW() END,
                status = :in_progress_status
            WHERE id = (
                SELECT pv.id FROM project_videos pv
                JOIN videos v ON pv.video_id = v.video_id
                JOIN projects p ON pv.project_id = p.project_id
                WHERE pv.project_id = :project_id
                  AND pv.status = :status
                  AND v.status = :video_extracted
                  AND p.project_stage IN ('POPULATING', 'FILTERING')
                  AND pv.locked_by IS NULL
                ORDER BY pv.created_at
                FOR UPDATE SKIP LOCKED 
                LIMIT 1
            )
            RETURNING *
        """).bindparams(
            worker_id=worker_id, 
            status=status.value,
            in_progress_status=in_progress_status.value,
            project_id=project_id,
            video_extracted=VideoStatus.EXTRACTED.value,
            debug_mode=debug_mode,
        )
    )
    row = result.fetchone()
    if row:
        session.commit()  # Persist the lock
        pv = ProjectVideo.model_validate(row._mapping)
        video = session.get(Video, pv.video_id)
        return pv, video
    return None


def update_project_video_heartbeat(session: Session, pv_id: int, worker_id: str) -> bool:
    """
    Update heartbeat timestamp. Returns False if lock was lost.
    """
    result = session.exec(
        text("""
            UPDATE project_videos SET heartbeat_at = NOW() 
            WHERE id = :id AND locked_by = :worker_id
            RETURNING id
        """).bindparams(id=pv_id, worker_id=worker_id)
    )
    session.commit()
    return result.fetchone() is not None


def release_project_video(
    session: Session,
    pv_id: int,
    worker_id: str,
    new_status: ProjectVideoStatus,
    **updates,
) -> bool:
    """
    Release lock and update project_video status/fields.
    Returns False if lock was lost.
    """
    pv = session.get(ProjectVideo, pv_id)
    if not pv or pv.locked_by != worker_id:
        return False
    
    pv.status = new_status
    pv.locked_by = None
    pv.locked_at = None
    pv.heartbeat_at = None
    pv.updated_at = datetime.utcnow()
    
    for key, value in updates.items():
        if hasattr(pv, key):
            setattr(pv, key, value)
    
    session.add(pv)
    session.commit()
    return True


def mark_project_video_failed(
    session: Session,
    pv_id: int,
    error: str,
    commit: bool = True,
) -> None:
    """Mark project_video as failed with error message."""
    pv = session.get(ProjectVideo, pv_id)
    if pv:
        pv.status = ProjectVideoStatus.FAILED
        pv.error = error
        pv.locked_by = None
        pv.locked_at = None
        pv.heartbeat_at = None
        pv.updated_at = datetime.utcnow()
        session.add(pv)
        if commit:
            session.commit()


# =============================================================================
# Project-Level Operations (Cross-Dedup and Detection)
# =============================================================================

def _claim_project_by_stage(
    session: Session,
    current_stage: ProjectStatus,
    new_stage: ProjectStatus,
) -> Optional[Project]:
    """
    Generic project claim: atomically transition from current_stage to new_stage.
    Uses FOR UPDATE SKIP LOCKED for concurrent safety.
    """
    result = session.exec(
        text("""
            UPDATE projects 
            SET project_stage = :new_stage
            WHERE project_id = (
                SELECT p.project_id FROM projects p
                WHERE p.project_stage = :current_stage
                ORDER BY p.created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
        """).bindparams(
            current_stage=current_stage.value,
            new_stage=new_stage.value,
        )
    )
    row = result.fetchone()
    if row:
        session.commit()
        return Project.model_validate(row._mapping)
    return None


def _update_project_stage(
    session: Session,
    project_id: int,
    stage: ProjectStatus,
) -> bool:
    """Generic project stage update."""
    result = session.exec(
        text("""
            UPDATE projects SET project_stage = :stage
            WHERE project_id = :pid
            RETURNING project_id
        """).bindparams(pid=project_id, stage=stage.value)
    )
    session.commit()
    return result.fetchone() is not None


# Public API (explicit function names for clarity)

def claim_project_for_cross_dedup(session: Session, worker_id: str) -> Optional[Project]:
    """Claim project in DEDUP_READY stage for cross-dedup."""
    return _claim_project_by_stage(session, ProjectStatus.DEDUP_READY, ProjectStatus.DEDUPING)


def claim_project_for_detection(session: Session, worker_id: str) -> Optional[Project]:
    """Claim project in DETECT_READY stage for detection."""
    return _claim_project_by_stage(session, ProjectStatus.DETECT_READY, ProjectStatus.DETECTING)


def complete_project_cross_dedup(
    session: Session, 
    project_id: int, 
    unique_frames: int = 0,
    dedup_dir: Optional[str] = None,
) -> bool:
    """Mark project as cross-dedup complete, ready for detection."""
    project = session.get(Project, project_id)
    if not project:
        return False
    project.project_stage = ProjectStatus.DETECT_READY
    project.unique_frames = unique_frames
    if dedup_dir:
        project.dedup_dir = dedup_dir
    session.add(project)
    session.commit()
    return True


def complete_project_detection(
    session: Session, 
    project_id: int,
    detect_dir: Optional[str] = None,
) -> bool:
    """Mark project as complete."""
    project = session.get(Project, project_id)
    if not project:
        return False
    project.project_stage = ProjectStatus.COMPLETE
    if detect_dir:
        project.detect_dir = detect_dir
    session.add(project)
    session.commit()
    return True


def mark_project_failed(session: Session, project_id: int, error: str = None) -> bool:
    """Mark project as failed."""
    return _update_project_stage(session, project_id, ProjectStatus.FAILED)


def get_filtered_frame_dirs(session: Session, project_id: int) -> dict[str, str]:
    """
    Get all filtered frame directories for a project.
    
    Returns:
        Dict mapping video_id to filtered_dir path
    """
    result = session.exec(
        text("""
            SELECT pv.video_id, pv.filtered_dir FROM project_videos pv
            WHERE pv.project_id = :pid
              AND pv.status = :status
              AND pv.filtered_dir IS NOT NULL
        """).bindparams(pid=project_id, status=ProjectVideoStatus.FILTERED.value)
    )
    return {row[0]: row[1] for row in result.fetchall()}


# =============================================================================
# Monitor Transition Operations
# =============================================================================

def transition_populating_to_filtering(session: Session) -> int:
    """
    Transition POPULATING projects to FILTERING when any video starts filtering.
    
    Condition: project has at least one project_video with status != PENDING
    
    Returns:
        Number of projects transitioned
    """
    result = session.exec(
        text("""
            UPDATE projects 
            SET project_stage = :new_stage
            WHERE project_stage = :current_stage
              AND EXISTS (
                  SELECT 1 FROM project_videos pv
                  WHERE pv.project_id = projects.project_id
                    AND pv.status != :pending_status
              )
        """).bindparams(
            current_stage=ProjectStatus.POPULATING.value,
            new_stage=ProjectStatus.FILTERING.value,
            pending_status=ProjectVideoStatus.PENDING.value,
        )
    )
    session.commit()
    return result.rowcount or 0


def transition_filtering_to_dedup_ready(session: Session) -> int:
    """
    Transition FILTERING projects to DEDUP_READY when all videos are filtered.
    
    Condition: all project_videos are in terminal filter states
    
    Returns:
        Number of projects transitioned
    """
    # Build status list for SQL
    terminal_statuses = tuple(s.value for s in FILTER_TERMINAL_STATUSES)
    
    result = session.exec(
        text("""
            UPDATE projects 
            SET project_stage = :new_stage
            WHERE project_stage = :current_stage
              AND NOT EXISTS (
                  SELECT 1 FROM project_videos pv
                  WHERE pv.project_id = projects.project_id
                    AND pv.status NOT IN :terminal_statuses
              )
              AND EXISTS (
                  SELECT 1 FROM project_videos pv
                  WHERE pv.project_id = projects.project_id
              )
        """).bindparams(
            current_stage=ProjectStatus.FILTERING.value,
            new_stage=ProjectStatus.DEDUP_READY.value,
            terminal_statuses=terminal_statuses,
        )
    )
    session.commit()
    return result.rowcount or 0


def reset_projects_with_pending_videos(session: Session) -> int:
    """
    Reset projects that have PENDING project_videos but are past FILTERING stage.
    
    This handles the case where new videos are added to a project that's already
    at DEDUP_READY or later. Resetting to POPULATING allows filter workers to
    process the new videos, then normal flow resumes.
    """
    result = session.exec(
        text("""
            UPDATE projects 
            SET project_stage = 'POPULATING'::projectstatus
            WHERE project_stage NOT IN ('POPULATING', 'FILTERING', 'FAILED')
              AND EXISTS (
                  SELECT 1 FROM project_videos pv
                  WHERE pv.project_id = projects.project_id
                    AND pv.status = 'PENDING'
              )
            RETURNING project_id, name
        """)
    )
    rows = result.fetchall()
    session.commit()
    return len(rows)


def update_project_frame_counts(session: Session) -> int:
    """
    Update frame counts and video counts for all active projects.
    Only writes if counts have changed (read is cheap, write is expensive).
    Skips COMPLETE projects.
    
    Updates:
    - extracted_frames, filtered_frames (frame counts)
    - total_videos, videos_pending, videos_downloaded, videos_extracted, videos_failed
    
    Count logic (cumulative where applicable):
    - videos_pending: PENDING or DOWNLOADING
    - videos_downloaded: DOWNLOADED, EXTRACTING, or EXTRACTED (cumulative - includes extracted)
    - videos_extracted: EXTRACTED only (terminal success state)
    - videos_failed: FAILED only
    
    Returns number of projects updated.
    """
    result = session.exec(
        text("""
            UPDATE projects p SET
                extracted_frames = counts.extracted,
                filtered_frames = counts.filtered,
                total_videos = counts.total,
                videos_pending = counts.pending,
                videos_downloaded = counts.downloaded,
                videos_extracted = counts.extr,
                videos_failed = counts.failed
            FROM (
                SELECT 
                    pv.project_id,
                    COALESCE(SUM(v.frame_count), 0)::INTEGER as extracted,
                    COALESCE(SUM(pv.passed_frames), 0)::INTEGER as filtered,
                    COUNT(*)::INTEGER as total,
                    COUNT(*) FILTER (WHERE v.status IN ('PENDING', 'DOWNLOADING'))::INTEGER as pending,
                    COUNT(*) FILTER (WHERE v.status IN ('DOWNLOADED', 'EXTRACTING', 'EXTRACTED'))::INTEGER as downloaded,
                    COUNT(*) FILTER (WHERE v.status = 'EXTRACTED')::INTEGER as extr,
                    COUNT(*) FILTER (WHERE v.status = 'FAILED')::INTEGER as failed
                FROM project_videos pv
                JOIN videos v ON pv.video_id = v.video_id
                GROUP BY pv.project_id
            ) counts
            WHERE p.project_id = counts.project_id
              AND p.project_stage != 'COMPLETE'
              AND (p.extracted_frames != counts.extracted 
                   OR p.filtered_frames != counts.filtered
                   OR p.total_videos != counts.total
                   OR p.videos_pending != counts.pending
                   OR p.videos_downloaded != counts.downloaded
                   OR p.videos_extracted != counts.extr
                   OR p.videos_failed != counts.failed)
            RETURNING p.project_id
        """)
    )
    rows = result.fetchall()
    session.commit()
    return len(rows)


def get_extracted_videos_for_cleanup(session: Session) -> list[tuple[str, str]]:
    """
    Get videos that are EXTRACTED and still have video_path.
    Returns list of (video_id, video_path) tuples.
    """
    result = session.exec(
        text("""
            SELECT video_id, video_path FROM videos
            WHERE status = 'EXTRACTED'
              AND video_path IS NOT NULL
        """)
    )
    return [(row[0], row[1]) for row in result.fetchall()]


def clear_video_path_db(session: Session, video_id: str) -> bool:
    """Clear video_path after video file has been deleted."""
    result = session.exec(
        text("""
            UPDATE videos SET video_path = NULL
            WHERE video_id = :video_id
        """).bindparams(video_id=video_id)
    )
    session.commit()
    return (result.rowcount or 0) > 0


# =============================================================================
# Backup Operations (called by backup worker)
# =============================================================================

def get_videos_ready_for_backup(session: Session) -> list:
    """
    Find videos ready for backup.
    
    A video is ready when:
    - status = EXTRACTED
    - frames_dir is not null
    - backed_up = False
    - ALL project_videos for this video are in terminal status (FILTERED, FILTERED_EMPTY, FAILED)
    
    Returns list of Video objects.
    """
    from .models import Video
    
    result = session.exec(
        text("""
            SELECT v.video_id, v.frames_dir
            FROM videos v
            WHERE v.status = 'EXTRACTED'
              AND v.frames_dir IS NOT NULL
              AND v.backed_up = FALSE
              AND NOT EXISTS (
                  SELECT 1 FROM project_videos pv
                  WHERE pv.video_id = v.video_id
                    AND pv.status NOT IN ('FILTERED', 'FILTERED_EMPTY', 'FAILED')
              )
              AND EXISTS (
                  SELECT 1 FROM project_videos pv
                  WHERE pv.video_id = v.video_id
              )
        """)
    )
    return [{"video_id": row[0], "frames_dir": row[1]} for row in result.fetchall()]


def mark_video_backed_up(session: Session, video_id: str) -> bool:
    """Mark video as backed up with timestamp."""
    result = session.exec(
        text("""
            UPDATE videos 
            SET backed_up = TRUE, backed_up_at = NOW()
            WHERE video_id = :video_id
        """).bindparams(video_id=video_id)
    )
    session.commit()
    return (result.rowcount or 0) > 0


# =============================================================================
# Stale Lock Recovery (called by monitor)
# =============================================================================

def reset_stale_video_locks(session: Session, stale_threshold_minutes: int = 2) -> int:
    """
    Reset stale videos back to input status.
    Only affects rows with heartbeat_at set (skips debug locks with NULL heartbeat).
    
    DOWNLOADING → PENDING
    EXTRACTING → DOWNLOADED
    """
    result = session.exec(
        text("""
            UPDATE videos 
            SET status = (CASE 
                    WHEN status = 'DOWNLOADING' THEN 'PENDING'
                    WHEN status = 'EXTRACTING' THEN 'DOWNLOADED'
                END)::videostatus,
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL
            WHERE locked_by IS NOT NULL
              AND heartbeat_at IS NOT NULL
              AND heartbeat_at < NOW() - INTERVAL ':threshold minutes'
              AND status IN ('DOWNLOADING', 'EXTRACTING')
            RETURNING video_id, status
        """).bindparams(threshold=stale_threshold_minutes)
    )
    rows = result.fetchall()
    session.commit()
    return len(rows)


def reset_stale_project_video_locks(session: Session, stale_threshold_minutes: int = 2) -> int:
    """
    Reset stale project_videos back to input status.
    Only affects rows with heartbeat_at set (skips debug locks with NULL heartbeat).
    
    FILTERING → PENDING
    """
    result = session.exec(
        text("""
            UPDATE project_videos 
            SET status = 'PENDING'::projectvideostatus,
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL
            WHERE locked_by IS NOT NULL
              AND heartbeat_at IS NOT NULL
              AND heartbeat_at < NOW() - INTERVAL ':threshold minutes'
              AND status = 'FILTERING'
            RETURNING id
        """).bindparams(threshold=stale_threshold_minutes)
    )
    rows = result.fetchall()
    session.commit()
    return len(rows)


def get_long_running_video_locks(session: Session, threshold_minutes: int = 30) -> list[dict]:
    """
    Get videos that have been locked for longer than threshold.
    For observability - shows what's taking a long time.
    """
    result = session.exec(
        text("""
            SELECT video_id, status, locked_by, locked_at,
                   EXTRACT(EPOCH FROM (NOW() - locked_at))/60 as minutes_locked
            FROM videos
            WHERE locked_by IS NOT NULL
              AND locked_at < NOW() - INTERVAL ':threshold minutes'
            ORDER BY locked_at
        """).bindparams(threshold=threshold_minutes)
    )
    return [dict(row._mapping) for row in result.fetchall()]


def get_long_running_project_video_locks(session: Session, threshold_minutes: int = 30) -> list[dict]:
    """
    Get project_videos that have been locked for longer than threshold.
    For observability - shows what's taking a long time.
    """
    result = session.exec(
        text("""
            SELECT pv.id, pv.video_id, pv.status, pv.locked_by, pv.locked_at,
                   EXTRACT(EPOCH FROM (NOW() - pv.locked_at))/60 as minutes_locked
            FROM project_videos pv
            WHERE pv.locked_by IS NOT NULL
              AND pv.locked_at < NOW() - INTERVAL ':threshold minutes'
            ORDER BY pv.locked_at
        """).bindparams(threshold=threshold_minutes)
    )
    return [dict(row._mapping) for row in result.fetchall()]


# =============================================================================
# Status Reporting
# =============================================================================

def get_video_status_counts(session: Session) -> list[tuple[str, int]]:
    """Get counts of videos by status (central stages)."""
    stmt = select(Video.status, func.count()).group_by(Video.status)
    return session.exec(stmt).all()


def get_project_status_counts(session: Session, project_id: int) -> dict:
    """
    Get both ProjectVideo and Video status counts for a project.
    
    Returns:
        {
            "project_video": {FILTERED: n, FILTERED_EMPTY: n, FAILED: n, ...},
            "video": {PENDING: n, DOWNLOADED: n, EXTRACTED: n, ...}
        }
    """
    # 1. ProjectVideo status counts (PENDING, FILTERING, FILTERED, FILTERED_EMPTY, FAILED)
    pv_stmt = (
        select(ProjectVideo.status, func.count())
        .where(ProjectVideo.project_id == project_id)
        .group_by(ProjectVideo.status)
    )
    pv_counts = {str(status): count for status, count in session.exec(pv_stmt).all()}
    
    # 2. Video status counts (for videos in this project)
    v_stmt = (
        select(Video.status, func.count())
        .join(ProjectVideo, Video.video_id == ProjectVideo.video_id)
        .where(ProjectVideo.project_id == project_id)
        .group_by(Video.status)
    )
    v_counts = {str(status): count for status, count in session.exec(v_stmt).all()}
    
    return {
        "project_video": pv_counts,
        "video": v_counts,
    }


def count_videos_by_project(session: Session, project_id: int) -> int:
    """Count videos linked to a project."""
    stmt = select(func.count()).select_from(ProjectVideo).where(ProjectVideo.project_id == project_id)
    return session.exec(stmt).one()


# =============================================================================
# Delete Operations
# =============================================================================

def delete_project(
    session: Session,
    project_name: str,
) -> tuple[bool, int, list[dict]]:
    """
    Delete a project and its project_videos from the database.
    Videos themselves are NOT deleted (may be used by other projects).
    
    Returns:
        (success, pv_count, paths) - paths contains file paths for cleanup
    """
    project = get_project_by_name(session, project_name)
    if not project:
        return False, 0, []
    
    # Get project_videos
    stmt = select(ProjectVideo).where(ProjectVideo.project_id == project.project_id)
    pvs = session.exec(stmt).all()
    pv_count = len(pvs)
    
    # Collect paths for file cleanup
    paths = []
    for pv in pvs:
        paths.append({
            "filtered_dir": pv.filtered_dir,
            "dedup_dir": pv.dedup_dir,
            "detection_dir": pv.detection_dir,
        })
    
    # Delete project_videos first (must be flushed before deleting project)
    for pv in pvs:
        session.delete(pv)
    session.flush()  # Ensure project_videos are deleted before project
    
    # Delete project
    session.delete(project)
    session.commit()
    
    return True, pv_count, paths


def delete_project_videos_by_filter(
    session: Session,
    project_id: int,
    status: Optional[ProjectVideoStatus] = None,
) -> tuple[int, list[dict]]:
    """
    Delete project_videos matching filter criteria.
    
    Returns:
        (deleted_count, paths) - paths for file cleanup
    """
    stmt = select(ProjectVideo).where(ProjectVideo.project_id == project_id)
    
    if status:
        stmt = stmt.where(ProjectVideo.status == status)
    
    pvs = session.exec(stmt).all()
    
    if not pvs:
        return 0, []
    
    paths = []
    for pv in pvs:
        paths.append({
            "filtered_dir": pv.filtered_dir,
            "dedup_dir": pv.dedup_dir,
            "detection_dir": pv.detection_dir,
        })
    
    for pv in pvs:
        session.delete(pv)
    session.commit()
    
    return len(pvs), paths


def delete_orphaned_videos(session: Session) -> tuple[int, list[dict]]:
    """
    Delete videos that are not referenced by any project_videos.
    
    Returns:
        (deleted_count, paths) - paths for file cleanup
    """
    # Find videos with no project_video references
    orphaned = session.exec(
        text("""
            SELECT v.video_id, v.video_path, v.frames_dir FROM videos v
            LEFT JOIN project_videos pv ON v.video_id = pv.video_id
            WHERE pv.id IS NULL
        """)
    ).all()
    
    if not orphaned:
        return 0, []
    
    # Collect paths for file cleanup
    paths = []
    video_ids = []
    for row in orphaned:
        video_ids.append(row._mapping["video_id"])
        paths.append({
            "video_path": row._mapping.get("video_path"),
            "frames_dir": row._mapping.get("frames_dir"),
        })
    
    # Delete orphaned videos
    session.exec(
        text("DELETE FROM videos WHERE video_id = ANY(:ids)").bindparams(ids=video_ids)
    )
    session.commit()
    
    return len(video_ids), paths

