"""
CLI entry point for Data Miner.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from omegaconf import OmegaConf

from .logging import get_logger

logger = get_logger(__name__)

# Default config path
DEFAULT_CONFIG = Path(__file__).parent.parent / "settings" / "default.yaml"


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load configuration with OmegaConf merging.
    
    - Loads default.yaml first
    - If config_path provided, merges it on top (overriding defaults)
    - Returns merged config as dict
    """
    # Load default config
    if not DEFAULT_CONFIG.exists():
        click.echo(f"Default config not found: {DEFAULT_CONFIG}", err=True)
        sys.exit(1)
    
    base_cfg = OmegaConf.load(DEFAULT_CONFIG)
    
    # Merge custom config if provided
    if config_path:
        if not config_path.exists():
            click.echo(f"Config not found: {config_path}", err=True)
            sys.exit(1)
        custom_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base_cfg, custom_cfg)
    else:
        cfg = base_cfg
    
    return OmegaConf.to_container(cfg, resolve=True)


@click.group()
def main():
    """Data Miner - PostgreSQL-based video pipeline."""
    pass


@main.command("init-db")
@click.option("--force", is_flag=True, help="Drop and recreate all tables (WARNING: deletes all data)")
def init_db(force: bool):
    """Initialize database tables."""
    from .db.connection import create_tables, engine
    from sqlmodel import SQLModel
    
    if force:
        click.confirm(
            "This will DROP ALL TABLES and delete all data. Continue?",
            abort=True
        )
        # Import models to register them
        from .db import models  # noqa: F401
        SQLModel.metadata.drop_all(engine)
        click.echo("Dropped all tables.")
    
    create_tables()
    click.echo("Database tables created.")


@main.command("populate")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config YAML file")
@click.option("--dry-run", is_flag=True, help="Show what would be added without inserting")
def populate_cmd(config: Optional[str], dry_run: bool):
    """Populate database with videos from config sources (search, URLs, files)."""
    import subprocess
    import json
    from .db.connection import get_session
    from .db.operations import add_video as db_add_video, get_or_create_project
    from .config import SourceType, YOUTUBE_BASE_URL
    from .utils.io import get_video_id
    
    cfg = load_config(Path(config) if config else None)
    
    project_name = cfg.get("project_name", "default")
    input_cfg = cfg.get("input", {})
    
    # Collect all videos to add
    videos_to_add = []  # [(video_id, url, source_type, source_info), ...]
    
    # 1. YouTube Search Queries (if enabled)
    search_enabled = input_cfg.get("search_enabled", True)
    search_queries = input_cfg.get("search_queries", [])
    max_results = input_cfg.get("max_results_per_query", 50)
    
    if search_enabled and search_queries:
        for query in search_queries:
            click.echo(f"Searching: {query}...")
            try:
                # Use yt-dlp to search (returns video URLs)
                result = subprocess.run(
                    ["yt-dlp", "--flat-playlist", "-j", f"ytsearch{max_results}:{query}"],
                    capture_output=True, text=True, timeout=120
                )
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        vid_id = entry.get("id")
                        if vid_id:
                            url = f"{YOUTUBE_BASE_URL}{vid_id}"
                            videos_to_add.append((vid_id, url, SourceType.SEARCH, query))
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                click.echo(f"Search failed: {e}", err=True)
    elif not search_enabled and search_queries:
        click.echo("YouTube search disabled in config (search_enabled: false)")
    
    # 2. Direct URLs from config
    for url in input_cfg.get("urls", []):
        vid_id = get_video_id(url)
        if vid_id:
            videos_to_add.append((vid_id, url, SourceType.MANUAL, "config"))
    
    # 3. URLs from file
    url_file = input_cfg.get("url_file")
    if url_file:
        url_path = Path(url_file)
        if url_path.exists():
            with open(url_path) as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith("#"):
                        vid_id = get_video_id(url)
                        if vid_id:
                            videos_to_add.append((vid_id, url, SourceType.FILE, str(url_path)))
    
    # Dedupe by video_id
    seen = set()
    unique_videos = []
    for v in videos_to_add:
        if v[0] not in seen:
            seen.add(v[0])
            unique_videos.append(v)
    
    click.echo(f"\nFound {len(unique_videos)} unique videos")
    
    if dry_run:
        for vid_id, url, src_type, src_info in unique_videos[:10]:
            click.echo(f"  {vid_id} ({src_type.value}): {src_info}")
        if len(unique_videos) > 10:
            click.echo(f"  ... and {len(unique_videos) - 10} more")
        return
    
    # Insert into DB
    added = 0
    new_videos = 0
    existing_videos = 0
    with get_session() as session:
        project_output_dir = cfg.get("project_output_dir", f"./output/projects/{project_name}")
        project = get_or_create_project(session, project_name, output_dir=project_output_dir)
        
        for vid_id, url, src_type, src_info in unique_videos:
            try:
                _, _, is_new = db_add_video(
                    session,
                    video_id=vid_id,
                    url=url,
                    project_id=project.project_id,
                    source_type=src_type,
                    source_info=src_info,
                )
                added += 1
                if is_new:
                    new_videos += 1
                else:
                    existing_videos += 1
            except Exception as e:
                click.echo(f"Failed to add {vid_id}: {e}", err=True)
    
    click.echo(f"✓ Added {added} videos to project '{project_name}' ({new_videos} new, {existing_videos} existing)")


@main.command("add-video")
@click.argument("url")
@click.option("--project", required=True, help="Project name")
@click.option("--source-type", default="manual", type=click.Choice(["search", "manual", "file"]))
@click.option("--source-info", default=None, help="Keyword/description/path")
def add_video(url: str, project: str, source_type: str, source_info: str):
    """Add a video URL to process."""
    from .db.connection import get_session
    from .db.operations import add_video as db_add_video, get_or_create_project
    from .config import SourceType
    from .utils.io import get_video_id
    
    video_id = get_video_id(url)
    if not video_id:
        click.echo("Invalid YouTube URL", err=True)
        return
    
    with get_session() as session:
        proj = get_or_create_project(session, project)
        db_add_video(
            session, 
            video_id=video_id, 
            url=url, 
            project_id=proj.project_id,
            source_type=SourceType(source_type),
            source_info=source_info,
        )
    
    click.echo(f"Added video {video_id} to project {project}")


@main.command("status")
@click.option("--project", help="Show status for a specific project")
def status(project: Optional[str]):
    """Show pipeline status."""
    from .db.connection import get_session
    from .db.operations import get_video_status_counts, get_project_status_counts, get_project_by_name
    
    with get_session() as session:
        # Central video status
        video_results = get_video_status_counts(session)
        
        click.echo("\nCentral Video Status (download/extract):")
        for st, count in video_results:
            click.echo(f"  {st.value if hasattr(st, 'value') else st}: {count}")
        
        # Project-specific status
        if project:
            proj = get_project_by_name(session, project)
            if not proj:
                click.echo(f"\nProject '{project}' not found.", err=True)
                return
            
            click.echo(f"\nProject: {project}")
            click.echo(f"  Stage: {proj.project_stage.value}")
            click.echo(f"  Frames: {proj.extracted_frames} extracted → {proj.filtered_frames} filtered → {proj.unique_frames} unique")
            
            if proj.dedup_dir:
                click.echo(f"  Dedup Dir: {proj.dedup_dir}")
            if proj.detect_dir:
                click.echo(f"  Detect Dir: {proj.detect_dir}")
            
            project_results = get_project_status_counts(session, proj.project_id)
            click.echo(f"\n  Video Status (filter/dedup/detect):")
            for st, count in project_results:
                click.echo(f"    {st.value if hasattr(st, 'value') else st}: {count}")


@main.command("delete-project")
@click.argument("project_name")
@click.option("--files", is_flag=True, help="Also delete output files on disk")
@click.option("--orphans", is_flag=True, help="Also delete videos no longer used by any project")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete_project_cmd(project_name: str, files: bool, orphans: bool, yes: bool):
    """Delete a project and its videos from the database."""
    import shutil
    from .db.connection import get_session
    from .db.operations import delete_project, delete_orphaned_videos, get_project_by_name
    from .config import get_output_dir
    
    with get_session() as session:
        # Check project exists first for confirmation
        project = get_project_by_name(session, project_name)
        if not project:
            click.echo(f"Project '{project_name}' not found.", err=True)
            return
    
    if not yes:
        click.confirm(f"Delete project '{project_name}'?", abort=True)
    
    with get_session() as session:
        success, pv_count, paths = delete_project(session, project_name)
        
    if not success:
        click.echo(f"Failed to delete project.", err=True)
        return
    
    click.echo(f"✓ Deleted {pv_count} project-video links from database")
    
    # Delete orphaned videos if requested
    if orphans:
        with get_session() as session:
            orphan_count, video_paths = delete_orphaned_videos(session)
        if orphan_count > 0:
            click.echo(f"✓ Deleted {orphan_count} orphaned videos from database")
            if files:
                _cleanup_video_paths(video_paths)
    
    # Delete files if requested
    if files:
        _cleanup_project_paths(paths)
        # Also try project directory
        output_dir = get_output_dir()
        project_dir = output_dir.parent / project_name if project_name in str(output_dir) else output_dir / project_name
        if project_dir.exists():
            shutil.rmtree(project_dir)
            click.echo(f"✓ Deleted {project_dir}")
    
    click.echo(f"✓ Project '{project_name}' deleted")


@main.command("force-dedup")
@click.argument("project_name")
def force_dedup_cmd(project_name: str):
    """Force project to DEDUP_READY stage (re-runs cross-dedup)."""
    from sqlalchemy import text
    from .db.connection import get_session
    from .db.operations import get_project_by_name
    
    with get_session() as session:
        project = get_project_by_name(session, project_name)
        if not project:
            click.echo(f"Project '{project_name}' not found.", err=True)
            return
        
        session.exec(
            text("UPDATE projects SET project_stage = 'DEDUP_READY'::projectstatus WHERE project_id = :pid")
            .bindparams(pid=project.project_id)
        )
        session.commit()
        click.echo(f"✓ Project '{project_name}' set to DEDUP_READY. CrossDedupWorker will pick it up.")


@main.command("force-detect")
@click.argument("project_name")
def force_detect_cmd(project_name: str):
    """Force project to DETECT_READY stage (re-runs detection)."""
    from sqlalchemy import text
    from .db.connection import get_session
    from .db.operations import get_project_by_name
    
    with get_session() as session:
        project = get_project_by_name(session, project_name)
        if not project:
            click.echo(f"Project '{project_name}' not found.", err=True)
            return
        
        session.exec(
            text("UPDATE projects SET project_stage = 'DETECT_READY'::projectstatus WHERE project_id = :pid")
            .bindparams(pid=project.project_id)
        )
        session.commit()
        click.echo(f"✓ Project '{project_name}' set to DETECT_READY. ProjectDetectWorker will pick it up.")


@main.command("cleanup-orphans")
@click.option("--files", is_flag=True, help="Also delete video/frame files on disk")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def cleanup_orphans_cmd(files: bool, yes: bool):
    """Delete orphaned videos not referenced by any project."""
    from .db.connection import get_session
    from .db.operations import delete_orphaned_videos
    
    # First check how many orphans exist
    with get_session() as session:
        from sqlmodel import text
        count = session.exec(
            text("""
                SELECT COUNT(*) FROM videos v
                LEFT JOIN project_videos pv ON v.video_id = pv.video_id
                WHERE pv.id IS NULL
            """)
        ).scalar()
    
    if count == 0:
        click.echo("No orphaned videos found.")
        return
    
    if not yes:
        click.confirm(f"Delete {count} orphaned videos?", abort=True)
    
    with get_session() as session:
        deleted_count, video_paths = delete_orphaned_videos(session)
    
    click.echo(f"✓ Deleted {deleted_count} orphaned videos from database")
    
    if files:
        _cleanup_video_paths(video_paths)


@main.command("delete-videos")
@click.option("--status", "pv_status", help="Delete project-videos with this status (e.g., 'failed', 'pending')")
@click.option("--project", required=True, help="Project to delete videos from")
@click.option("--files", is_flag=True, help="Also delete output files on disk")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete_videos_cmd(pv_status: Optional[str], project: str, files: bool, yes: bool):
    """Delete project-videos from the database (with optional filters)."""
    from .db.connection import get_session
    from .db.operations import delete_project_videos_by_filter, get_project_by_name
    from .config import ProjectVideoStatus
    
    # Parse status
    status_enum = None
    if pv_status:
        try:
            status_enum = ProjectVideoStatus(pv_status)
        except ValueError:
            click.echo(f"Invalid status: {pv_status}", err=True)
            click.echo(f"Valid statuses: {[s.value for s in ProjectVideoStatus]}")
            return
    
    if not yes:
        msg = f"Delete project-videos from '{project}'?"
        if pv_status:
            msg = f"Delete project-videos with status '{pv_status}' from '{project}'?"
        click.confirm(msg, abort=True)
    
    with get_session() as session:
        proj = get_project_by_name(session, project)
        if not proj:
            click.echo(f"Project '{project}' not found.", err=True)
            return
        
        deleted_count, paths = delete_project_videos_by_filter(
            session, 
            project_id=proj.project_id,
            status=status_enum, 
        )
    
    if deleted_count == 0:
        click.echo("No project-videos match the criteria.")
        return
    
    click.echo(f"✓ Deleted {deleted_count} project-videos from database")
    
    if files:
        _cleanup_project_paths(paths)


def _cleanup_video_paths(video_paths: list[dict], skip_confirm: bool = False):
    """Helper to delete video/frame files from paths returned by db operations."""
    import shutil
    
    deleted_files = 0
    deleted_dirs = 0
    
    for paths in video_paths:
        if paths.get("video_path"):
            p = Path(paths["video_path"])
            if p.exists():
                p.unlink()
                deleted_files += 1
        
        for dir_key in ["frames_dir", "filtered_dir", "dedup_dir"]:
            if paths.get(dir_key):
                d = Path(paths[dir_key])
                if d.exists():
                    shutil.rmtree(d)
                    deleted_dirs += 1
    
    if deleted_files or deleted_dirs:
        click.echo(f"✓ Deleted {deleted_files} video files, {deleted_dirs} frame directories")


def _cleanup_project_paths(paths: list[dict]):
    """Helper to delete project-specific files (filtered, dedup, detection dirs)."""
    import shutil
    
    deleted_dirs = 0
    
    for p in paths:
        for dir_key in ["filtered_dir", "dedup_dir", "detection_dir"]:
            if p.get(dir_key):
                d = Path(p[dir_key])
                if d.exists():
                    shutil.rmtree(d)
                    deleted_dirs += 1
    
    if deleted_dirs:
        click.echo(f"✓ Deleted {deleted_dirs} directories")


# =============================================================================
# Workers Management
# =============================================================================

@main.group()
def workers():
    """Manage pipeline workers via supervisor."""
    pass


@workers.command("setup")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config YAML file")
def workers_setup(config: Optional[str]):
    """Generate and install supervisor configuration."""
    import shutil
    
    # Check if supervisor is installed and configured
    if not shutil.which("supervisord"):
        click.echo("Error: supervisor not installed.", err=True)
        click.echo("\nRun the following commands to install and configure supervisor:", err=True)
        click.echo("  sudo apt-get update", err=True)
        click.echo("  sudo apt-get install -y supervisor", err=True)
        click.echo("  sudo systemctl enable supervisor", err=True)
        click.echo("  sudo systemctl start supervisor", err=True)
        sys.exit(1)
    
    if not Path("/etc/supervisor/conf.d").exists():
        click.echo("Error: /etc/supervisor/conf.d directory not found.", err=True)
        click.echo("\nRun: sudo mkdir -p /etc/supervisor/conf.d", err=True)
        sys.exit(1)
    
    
    project_dir = Path(__file__).parent.parent.resolve()
    venv_dir = project_dir / ".venv"
    
    # error on missing config
    if not config:
        click.echo("Error: --config is required for workers setup", err=True)
        sys.exit(1)
    # config_path = Path(config).resolve() if config else project_dir / "settings" / "default.yaml"
    config_path = Path(config).resolve()
    if not (config_path.is_file() and (config_path.suffix in [".yaml", ".yml"])):
        click.echo(f"Error: not valid config file: {config_path}", err=True)
        sys.exit(1)
    cfg = load_config(config_path)

    # Extract settings from config
    sup = cfg.get("supervisor", {})
    log_cfg = cfg.get("logging", {})
    db_cfg = cfg.get("database", {})
    
    log_dir = log_cfg.get("log_dir", "/var/log/data_miner")
    loki_url = log_cfg.get("loki_url", "http://localhost:3100/loki/api/v1/push")
    log_level = log_cfg.get("level", "INFO")
    database_url = db_cfg.get("url", "postgresql://postgres:postgres@localhost:5432/data_miner")
    log_dir = Path(log_dir).resolve().as_posix()
    # HuggingFace token for private models (from env or .env)
    
    download_workers = sup.get("download_workers", 3)
    extract_workers = sup.get("extract_workers", 2)
    filter_workers = sup.get("filter_workers", 1)
    dedup_workers = sup.get("dedup_workers", 1)
    detect_workers = sup.get("detect_workers", 1)
    
    # Build program list (skip if worker count is 0)
    programs = []
    if download_workers > 0:
        programs.append("download_worker")
    if extract_workers > 0:
        programs.append("extract_worker")
    if filter_workers > 0:
        programs.append("filter_worker")
    if dedup_workers > 0:
        programs.append("dedup_worker")
    if detect_workers > 0:
        programs.append("detect_worker")
    programs.append("monitor_worker")  # Always include monitor
    
    programs_str = ",".join(programs)
    
    # Template for supervisor program blocks
    def program_block(
        name: str,
        module: str,
        numprocs: int,
        startsecs: int = 5,
        stopwaitsecs: int = 30,
        log_name: str = None,
        extra_env: str = "",
    ) -> str:
        log_name = log_name or name.replace("_worker", "")
        # For single-process workers, don't use process_num in log name
        if numprocs == 1:
            log_suffix = ".log"
            proc_name = "%(program_name)s"
        else:
            log_suffix = "_%(process_num)02d.log"
            proc_name = "%(program_name)s_%(process_num)02d"
        
        env = f'DATABASE_URL="{database_url}",LOKI_URL="{loki_url}",LOG_LEVEL="{log_level}",DATA_MINER_CONFIG="{config_path}"'
        if extra_env:
            env += f",{extra_env}"
        
        return f"""[program:{name}]
command={venv_dir}/bin/python -m data_miner.workers.{module}
directory={project_dir}
numprocs={numprocs}
process_name={proc_name}
autorestart=true
startsecs={startsecs}
stopwaitsecs={stopwaitsecs}
stdout_logfile={log_dir}/{log_name}{log_suffix}
stderr_logfile={log_dir}/{log_name}{log_suffix}
environment={env}

"""
    
    # Generate config
    supervisor_conf = f"""# Auto-generated by data-miner workers setup
[group:data_miner]
programs={programs_str}

"""
    
    # Worker environment extras
    # hf_env = f'HF_TOKEN="{hf_token}"'

    # Monitor worker - always 1 instance
    supervisor_conf += program_block("monitor_worker", "monitor", 1, startsecs=5, stopwaitsecs=10)
    
    if download_workers > 0:
        supervisor_conf += program_block("download_worker", "download", download_workers)
    
    if extract_workers > 0:
        supervisor_conf += program_block("extract_worker", "extract", extract_workers)
    
    if filter_workers > 0:
        supervisor_conf += program_block("filter_worker", "filter", filter_workers, startsecs=10, stopwaitsecs=60)
    
    if dedup_workers > 0:
        supervisor_conf += program_block("dedup_worker", "dedup", dedup_workers, startsecs=10, stopwaitsecs=60)
    
    if detect_workers > 0:
        supervisor_conf += program_block("detect_worker", "detect", detect_workers, startsecs=10, stopwaitsecs=60)
    

    # Write to temp and copy with sudo
    tmp_conf = Path("/tmp/data_miner.conf")
    tmp_conf.write_text(supervisor_conf)
    
    click.echo("Creating log directory...")
    subprocess.run(["sudo", "mkdir", "-p", log_dir], check=True)
    click.echo(f"Log directory created at {log_dir}")
    
    click.echo("Installing supervisor configuration...")
    subprocess.run(["sudo", "cp", str(tmp_conf), "/etc/supervisor/conf.d/data_miner.conf"], check=True)
    click.echo(f"Supervisor configuration installed at /etc/supervisor/conf.d/data_miner.conf")
    
    click.echo("Reloading supervisor...")
    subprocess.run(["sudo", "supervisorctl", "reread"], check=True)
    subprocess.run(["sudo", "supervisorctl", "update"], check=True)
    
    click.echo("\n✓ Setup complete! Use 'data-miner workers start' to start workers.")


@workers.command("start")
def workers_start():
    """Start all workers."""
    subprocess.run(["sudo", "supervisorctl", "start", "data_miner:*"], check=True)
    click.echo("Workers started.")


@workers.command("stop")
def workers_stop():
    """Stop all workers."""
    subprocess.run(["sudo", "supervisorctl", "stop", "data_miner:*"], check=True)
    click.echo("Workers stopped.")


@workers.command("restart")
def workers_restart():
    """Restart all workers."""
    subprocess.run(["sudo", "supervisorctl", "restart", "data_miner:*"], check=True)
    click.echo("Workers restarted.")


@workers.command("status")
def workers_status():
    """Show worker status."""
    subprocess.run(["sudo", "supervisorctl", "status", "data_miner:*"])


if __name__ == "__main__":
    main()

