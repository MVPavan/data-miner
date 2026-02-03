# mypy: disable-error-code="call-overload"
"""Database helper utilities for migrations and maintenance."""

import subprocess
from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session

from ..db.connection import engine


def add_backup_columns() -> bool:
    """
    Add backed_up and backed_up_at columns to videos table if they don't exist.
    
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    with Session(engine) as session:
        try:
            session.exec(text("""
                ALTER TABLE videos 
                ADD COLUMN IF NOT EXISTS backed_up BOOLEAN DEFAULT FALSE;
            """))
            session.exec(text("""
                ALTER TABLE videos 
                ADD COLUMN IF NOT EXISTS backed_up_at TIMESTAMP;
            """))
            session.commit()
            return True
        except Exception as e:
            print(f"Error adding backup columns: {e}")
            session.rollback()
            return False


def check_column_exists(table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    with Session(engine) as session:
        result = session.exec(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = :table AND column_name = :column
            )
        """).bindparams(table=table, column=column))
        return result.scalar() or False

def fix_legacy_paths() -> int:
    """
    Fix legacy paths that have project_name embedded.
    
    Old format:
        video_path: output/project_name/videos/video_id.mp4
        frames_dir: output/project_name/frames_raw/video_id
    
    New format:
        video_path: output/videos/video_id.mp4
        frames_dir: output/frames_raw/video_id
    
    Returns count of videos updated.
    """
    import re
    
    with Session(engine) as session:
        # Get all videos with paths
        result = session.exec(text("""
            SELECT video_id, video_path, frames_dir FROM videos
            WHERE video_path IS NOT NULL OR frames_dir IS NOT NULL
        """))
        videos = result.fetchall()
    
    print(f"Checking {len(videos)} videos for legacy paths...")
    
    # Pattern: output/something/videos/... or output/something/frames_raw/...
    video_pattern = re.compile(r"^(output)/[^/]+/(videos/.+)$")
    frames_pattern = re.compile(r"^(output)/[^/]+/(frames_raw/.+)$")
    
    updated = 0
    with Session(engine) as session:
        for video_id, video_path, frames_dir in videos:
            new_video_path = video_path
            new_frames_dir = frames_dir
            needs_update = False
            
            if video_path:
                match = video_pattern.match(video_path)
                if match:
                    new_video_path = f"{match.group(1)}/{match.group(2)}"
                    needs_update = True
            
            if frames_dir:
                match = frames_pattern.match(frames_dir)
                if match:
                    new_frames_dir = f"{match.group(1)}/{match.group(2)}"
                    needs_update = True
            
            if needs_update:
                session.exec(text("""
                    UPDATE videos 
                    SET video_path = :video_path, frames_dir = :frames_dir
                    WHERE video_id = :video_id
                """).bindparams(
                    video_path=new_video_path,
                    frames_dir=new_frames_dir,
                    video_id=video_id
                ))
                print(f"  {video_id}: fixed paths")
                updated += 1
        
        session.commit()
    
    print(f"Updated {updated} videos with legacy paths")
    return updated


def fix_legacy_project_video_paths() -> int:
    """
    Fix legacy project_videos paths that are missing 'projects/' folder.
    
    Old format:
        filtered_dir: output/project_name/frames_filtered/video_id
    
    New format:
        filtered_dir: output/projects/project_name/frames_filtered/video_id
    
    Returns count of project_videos updated.
    """
    import re
    
    with Session(engine) as session:
        result = session.exec(text("""
            SELECT id, filtered_dir FROM project_videos
            WHERE filtered_dir IS NOT NULL
        """))
        pvs = result.fetchall()
    
    print(f"Checking {len(pvs)} project_videos for legacy paths...")
    
    # Pattern: output/something/frames_filtered/... (missing 'projects/')
    # Should become: output/projects/something/frames_filtered/...
    pattern = re.compile(r"^(output)/([^/]+)/(frames_filtered/.+)$")
    
    updated = 0
    with Session(engine) as session:
        for pv_id, filtered_dir in pvs:
            if filtered_dir:
                match = pattern.match(filtered_dir)
                if match:
                    # Check it's not already output/projects/...
                    if match.group(2) != "projects":
                        new_path = f"{match.group(1)}/projects/{match.group(2)}/{match.group(3)}"
                        session.exec(text("""
                            UPDATE project_videos 
                            SET filtered_dir = :filtered_dir
                            WHERE id = :pv_id
                        """).bindparams(filtered_dir=new_path, pv_id=pv_id))
                        print(f"  {pv_id}: {filtered_dir} -> {new_path}")
                        updated += 1
        
        session.commit()
    
    print(f"Updated {updated} project_videos with legacy paths")
    return updated


# =============================================================================
# Helper Functions (DRY)
# =============================================================================

def _get_remote_video_ids(remote_dest: str) -> set[str]:
    """Get all video_ids from remote destination via SSH."""
    remote_host, remote_path = remote_dest.split(":", 1)
    
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", remote_host,
             f"ls -1 '{remote_path}'"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            video_ids = set(line.strip() for line in result.stdout.strip().split("\n") if line.strip())
            print(f"Found {len(video_ids)} folders on remote")
            return video_ids
        else:
            print(f"Error listing remote: {result.stderr}")
            return set()
    except Exception as e:
        print(f"Error connecting to remote: {e}")
        return set()

def _reset_video_extract_dir(session: Session, video_id: str, frames_dir: Path):
    session.exec(text("""
        UPDATE videos SET frames_dir = :frames_dir
        WHERE video_id = :video_id
    """).bindparams(frames_dir=str(frames_dir), video_id=video_id))
        
def _reset_project_video_filtered_empty(session: Session, pv_id: int):
    """Reset project_video filter status."""
    result = session.exec(text("""
        UPDATE project_videos SET status = 'FILTERED_EMPTY', error = NULL
        WHERE id = :pv_id
    """).bindparams(pv_id=pv_id))
    session.flush()
    if result.rowcount == 0:
        print(f"    WARNING: No rows updated for pv_id={pv_id}")

def _reset_project_video_pending(session: Session, pv_id: int):
    """Reset project_video to PENDING status."""
    result = session.exec(text("""
        UPDATE project_videos SET status = 'PENDING', error = NULL
        WHERE id = :pv_id
    """).bindparams(pv_id=pv_id))
    session.flush()
    if result.rowcount == 0:
        print(f"    WARNING: No rows updated for pv_id={pv_id}")


def _reset_video_for_redownload(session: Session, video_id: str):
    """Reset video to PENDING for re-download."""
    session.exec(text("""
        UPDATE videos 
        SET status = 'PENDING', frames_dir = NULL, video_path = NULL, frame_count = NULL
        WHERE video_id = :video_id
    """).bindparams(video_id=video_id))


def _rsync_from_remote(remote_src: str, local_dest: Path) -> bool:
    """Rsync folder from remote to local."""
    local_dest.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["rsync", "-avz", "--checksum",
             "-e", "ssh -o BatchMode=yes",
             remote_src, f"{local_dest}/"],
            capture_output=True, text=True, timeout=3600
        )
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# Backup Recovery Functions
# =============================================================================

def sync_backup_status_from_remote(remote_dest: str) -> int:
    """
    Sync backed_up status from remote folder list to DB.
    
    Lists all video_id folders on remote and marks matching videos
    as backed_up=True in the database.
    """
    remote_video_ids = _get_remote_video_ids(remote_dest)
    if not remote_video_ids:
        return 0
    
    updated = 0
    with Session(engine) as session:
        for video_id in remote_video_ids:
            result = session.exec(text("""
                UPDATE videos 
                SET backed_up = TRUE, backed_up_at = NOW()
                WHERE video_id = :video_id AND backed_up = FALSE
            """).bindparams(video_id=video_id))
            if result.rowcount and result.rowcount > 0:
                updated += 1
        session.commit()
    
    print(f"Updated {updated} videos as backed_up=True")
    return updated


def restore_premature_backups(
    remote_dest: str, 
    local_base: str, 
    projects_base: str,  # Path to projects folder (e.g., /mnt/data/output/projects)
    error_pattern: str = "Frames dir not found"
) -> int:
    """
    Restore videos that were backed up before filtering completed.
    
    Finds project_videos with FAILED status and path-related errors,
    restores them from remote, and resets status to PENDING.
    """
    remote_host, remote_path = remote_dest.split(":", 1)
    local_base_path = Path(local_base)
    projects_base_path = Path(projects_base)
    
    # Get all remote video IDs upfront
    remote_video_ids = _get_remote_video_ids(remote_dest)
    failed_videos = {}
    # Find failed project_videos with path errors
    with Session(engine) as session:
        result = session.exec(text("""
            SELECT pv.id, pv.video_id, pv.error, v.frames_dir, p.name
            FROM project_videos pv
            JOIN videos v ON v.video_id = pv.video_id
            JOIN projects p ON p.project_id = pv.project_id
            WHERE pv.status = 'FAILED'
              AND pv.error LIKE :pattern
        """).bindparams(pattern=f"%{error_pattern}%"))
        
        failed_videos = {
            row[1]: {"pv_id": row[0], "video_id": row[1], "error": row[2], "frames_dir": row[3], "project_name": row[4]}
            for row in result.fetchall()
        }
    
    print(f"Found {len(failed_videos)} failed videos with path errors")
    from collections import Counter
    project_counter = Counter(v["project_name"] for v in failed_videos.values())
    for project_name, count in project_counter.items():
        print(f"  Project {project_name}: {count} videos")
    
    restored = 0
    reset_for_download = 0
    marked_filtered_empty = 0
    marked_filtered_pending = 0

    failed_filtered_not_empty = []
    failed_not_filtered = []
    failed_filtered_empty = []
    exists_locally = []
    exists_in_remote = []
    exists_nowhere = []
    
    for vid, video in failed_videos.items():
        video_id = video["video_id"]
        assert video_id == vid, "Video ID mismatch"
        project_name = video["project_name"]
        local_dest = local_base_path / video_id
        filtered_dir = projects_base_path / project_name / "frames_filtered" / video_id
        if filtered_dir.exists():
            if any(filtered_dir.iterdir()):
                failed_filtered_not_empty.append(video_id)
            else:
                failed_filtered_empty.append(video_id)
        else:
            failed_not_filtered.append(video_id)
        
        if local_dest.exists():
            exists_locally.append(video_id)
        
        elif video_id in remote_video_ids:
            exists_in_remote.append(video_id)
        else:
            exists_nowhere.append(video_id) 
    print(f"Summary before restoration:")
    print(f"  Total failed: {len(failed_videos)}")
    print(f"  Failed not filtered: {len(failed_not_filtered)}")
    print(f"  Failed filtered not empty: {len(failed_filtered_not_empty)}")
    print(f"  Failed filtered empty: {len(failed_filtered_empty)}")
    print(f"  Exists locally: {len(exists_locally)}")
    print(f"  Exists in remote: {len(exists_in_remote)}")
    print(f"  Exists nowhere: {len(exists_nowhere)}")
    print(f"Exists local + remote == {len(exists_locally) + len(exists_in_remote)} (should be {len(failed_videos)})")
    print(f"Not filtered and not in local frames_raw: ", set(failed_not_filtered) - set(exists_locally))
    print("Starting restoration process...")

    with Session(engine) as session:
        # case 1: Update empty filtered dirs
        for vid in failed_filtered_empty:
            print(f"  {vid}: Filtered dir exists, marking as FILTERED_EMPTY")
            _reset_project_video_filtered_empty(session, pv_id=failed_videos[vid]["pv_id"])
            marked_filtered_empty += 1
        
        # case 2: update not filtered and exists locally/remotely
        for vid in failed_not_filtered:
            if vid in exists_locally:
                print(f"  {vid}: Already exists locally, resetting to PENDING")
                _reset_project_video_pending(session, pv_id=failed_videos[vid]["pv_id"])
                marked_filtered_pending += 1
            elif vid in exists_in_remote:
                print(f"  {vid}: Exists in remote, will restore ")
                remote_src = f"{remote_host}:{remote_path.rstrip('/')}/{vid}/"
                local_dest = local_base_path / vid
                if _rsync_from_remote(remote_src, local_dest):
                    _reset_video_extract_dir(session, vid, local_dest)
                    _reset_project_video_pending(session, pv_id=failed_videos[vid]["pv_id"])
                    print(f"  {vid}: ✓ Restored from remote")
                    restored += 1
                else:
                    print(f"  {vid}: ✗ rsync failed")
            else:
                print(f"  {vid}: Neither exists, resetting for re-download")
                _reset_video_for_redownload(session, vid)
                _reset_project_video_pending(session, pv_id=failed_videos[vid]["pv_id"])
                reset_for_download += 1        
        print(f"DEBUG: About to commit. pending={marked_filtered_pending}, empty={marked_filtered_empty}")
        session.commit()
        print("DEBUG: Commit complete")
    
    print(f"Restored: {restored}, Filtered pending: {marked_filtered_pending}, Filtered empty: {marked_filtered_empty}, Reset for re-download: {reset_for_download}")
    return restored


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Restore premature backups")
    parser.add_argument("remote_dest", type=str, help="user@host:/path/to/backup")
    parser.add_argument("local_base", type=str, help="Local frames_raw path")
    parser.add_argument("projects_base", type=str, help="Local projects path (for frames_filtered check)")
    parser.add_argument("--error_pattern", type=str, default="Frames dir not found")
    
    args = parser.parse_args()
    restore_premature_backups(args.remote_dest, args.local_base, args.projects_base, args.error_pattern)