#!/usr/bin/env python3
"""
Script to continuously sync frames_raw folders to remote destination.

This script monitors the video_miner_v3/output directory for frames_raw folders,
and syncs folders with 11-character names that haven't been modified in the last hour
to a remote destination via rsync.
"""

import os
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

# Import configuration
try:
    from sync_config import (
        SOURCE_BASE, REMOTE_DEST, SLEEP_INTERVAL, 
        MODIFICATION_THRESHOLD_HOURS, LOG_LEVEL, DELETE_AFTER_SYNC,
        VERIFICATION_TIMEOUT
    )
except ImportError:
    raise ImportError("Could not import sync_config.py. Please ensure it exists and is in the PYTHONPATH.")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_frames.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_ssh_connection(remote_dest: str) -> bool:
    """Test SSH connection to remote destination without password."""
    remote_host = remote_dest.split(':')[0]
    try:
        logger.info(f"Testing SSH connection to {remote_host}...")
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 
             remote_host, 'echo "SSH connection successful"'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            logger.info("✓ SSH connection test passed (passwordless)")
            return True
        else:
            logger.error(f"✗ SSH connection failed: {result.stderr.strip()}")
            logger.error("Hint: Set up SSH key authentication with setup_ssh_keys.sh")
            return False
    except subprocess.TimeoutExpired:
        logger.error("✗ SSH connection timeout")
        return False
    except Exception as e:
        logger.error(f"✗ SSH connection error: {e}")
        return False


def find_frames_raw_folders(base_path: str|Path) -> List[Path]:
    """Find all frames_raw folders in the directory tree."""
    frames_raw_folders = []
    base_path = Path(base_path)
    
    for root, dirs, files in os.walk(base_path):
        if 'frames_raw' in dirs:
            frames_raw_path = Path(root) / 'frames_raw'
            frames_raw_folders.append(frames_raw_path)
    frames_raw_folders = list(set(frames_raw_folders))  # Remove duplicates
    return frames_raw_folders


def find_11_char_folders(frames_raw_path: Path) -> List[Path]:
    """Find folders with exactly 11 characters in frames_raw directory."""
    eleven_char_folders = []
    
    try:
        for item in frames_raw_path.iterdir():
            if item.is_dir() and len(item.name) == 11:
                eleven_char_folders.append(item)
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not access {frames_raw_path}: {e}")
    
    return eleven_char_folders


def is_older_than_threshold(folder_path: Path, threshold_hours: int = 1) -> bool:
    """Check if folder was last modified before the threshold."""
    try:
        mtime = folder_path.stat().st_mtime
        modified_time = datetime.fromtimestamp(mtime)
        threshold_time = datetime.now() - timedelta(hours=threshold_hours)
        return modified_time < threshold_time
    except (OSError, FileNotFoundError) as e:
        logger.warning(f"Could not get modification time for {folder_path}: {e}")
        return False


def get_folder_stats(folder_path: Path) -> Tuple[int, int]:
    """Get file count and total size of folder."""
    file_count = 0
    total_size = 0
    
    try:
        for root, dirs, files in os.walk(folder_path):
            file_count += len(files)
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    logger.warning(f"Could not get size for {file_path}")
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not access {folder_path}: {e}")
    
    return file_count, total_size


def calculate_relative_path(source_folder: Path, base_source: str) -> str:
    """Calculate the relative path from base source to maintain directory structure."""
    base_source_path = Path(base_source)
    relative_path = source_folder.relative_to(base_source_path)
    return str(relative_path)


def create_remote_directory(remote_dest: str, relative_path: str) -> bool:
    """Create remote directory structure if it doesn't exist."""
    remote_host = remote_dest.split(':')[0]
    remote_base_dir = remote_dest.split(':')[1]
    remote_dir = f"{remote_base_dir.rstrip('/')}/{relative_path}"
    
    try:
        # Use SSH to create directory structure with mkdir -p
        mkdir_cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
                    remote_host, f'mkdir -p "{remote_dir}"']
        
        result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.debug(f"Remote directory created/verified: {remote_dir}")
            return True
        else:
            logger.error(f"Failed to create remote directory {remote_dir}: {result.stderr}")
            return False
            
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Error creating remote directory {remote_dir}: {e}")
        return False


def sync_folder_to_remote(source_folder: Path, remote_dest: str, relative_path: str) -> bool:
    """Sync folder to remote destination using rsync."""
    # Create the full remote destination path
    remote_path = f"{remote_dest.rstrip('/')}/{relative_path}"
    
    # # First, ensure the remote directory structure exists
    # if not create_remote_directory(remote_dest, relative_path):
    #     logger.error(f"Cannot create remote directory structure for {relative_path}")
    #     return False
    
    # Use rsync to copy the folder (keep --mkpath for double safety)
    rsync_cmd = [
        'rsync',
        '-avz',
        '--checksum',  # Use checksums to verify file content integrity
        '--progress',
        '--mkpath',  # Create destination directories as needed (if supported)
        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=30',  # Non-interactive SSH
        f"{source_folder}/",  # Source with trailing slash to copy contents
        f"{remote_path}/"     # Destination with trailing slash
    ]
    
    try:
        logger.info(f"Syncing {source_folder} to {remote_path}")
        result = subprocess.run(
            rsync_cmd, 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully synced {source_folder}")
            return True
        else:
            logger.error(f"rsync failed for {source_folder}: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"rsync timeout for {source_folder}")
        return False
    except subprocess.SubprocessError as e:
        logger.error(f"rsync error for {source_folder}: {e}")
        return False


def verify_remote_folder(source_folder: Path, remote_dest: str, relative_path: str) -> bool:
    """Verify that remote folder matches source using rsync checksum verification."""
    remote_path = f"{remote_dest.rstrip('/')}/{relative_path}"
    
    # Get source stats for logging
    source_file_count, source_size = get_folder_stats(source_folder)
    
    try:
        # Use rsync with --checksum and --dry-run to verify sync
        # If no differences are found, rsync will have no output and return code 0
        verify_cmd = [
            'rsync',
            '-avz',
            '--checksum',     # Compare checksums, not just size/time
            '--dry-run',      # Don't actually transfer anything
            '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=30',
            f"{source_folder}/",  # Source with trailing slash
            f"{remote_path}/"     # Destination with trailing slash
        ]
        
        logger.info(f"Verifying {source_file_count} files using rsync checksum comparison")
        result = subprocess.run(
            verify_cmd, 
            capture_output=True, 
            text=True, 
            timeout=VERIFICATION_TIMEOUT
        )
        
        if result.returncode == 0:
            # Check if rsync found any differences
            output_lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            # Filter out rsync header/footer messages
            file_differences = [line for line in output_lines if not line.startswith('sending incremental') 
                              and not line.startswith('sent ') 
                              and not line.startswith('total size')
                              and line != './' 
                              and line]
            
            if not file_differences:
                logger.info(f"✓ Verification passed: {source_file_count} files verified with rsync checksum")
                return True
            else:
                logger.warning(f"Found {len(file_differences)} differences:")
                for diff in file_differences[:10]:  # Show first 10 differences
                    logger.warning(f"  - {diff}")
                if len(file_differences) > 10:
                    logger.warning(f"  ... and {len(file_differences) - 10} more differences")
                return False
        else:
            logger.error(f"rsync verification failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"rsync verification timeout after {VERIFICATION_TIMEOUT} seconds")
        return False
    except subprocess.SubprocessError as e:
        logger.error(f"rsync verification error: {e}")
        return False


def delete_source_folder(folder_path: Path) -> bool:
    """Safely delete source folder after successful sync and verification.
    
    Safety check: Only deletes folders with exactly 11 characters to prevent
    accidental deletion of parent directories or other folders.
    """
    try:
        import shutil
        
        # SAFETY CHECK: Only delete folders with exactly 11 characters
        folder_name = folder_path.name
        if len(folder_name) != 11:
            logger.error(f"SAFETY CHECK FAILED: Refusing to delete folder '{folder_name}' - not exactly 11 characters")
            logger.error(f"This prevents accidental deletion of parent directories")
            return False
        
        # Additional safety check: ensure it's actually a directory
        if not folder_path.is_dir():
            logger.error(f"SAFETY CHECK FAILED: Path '{folder_path}' is not a directory")
            return False
        
        logger.info(f"Deleting source folder: {folder_path} (11-char folder: '{folder_name}')")
        shutil.rmtree(folder_path)
        logger.info(f"✓ Successfully deleted source folder: {folder_name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to delete source folder {folder_path}: {e}")
        return False


def process_sync_candidates() -> None:
    """Main processing function to find and sync eligible folders."""
    logger.info("Starting sync scan...")
    
    # Find all frames_raw folders
    frames_raw_folders = find_frames_raw_folders(SOURCE_BASE)
    logger.info(f"Found {len(frames_raw_folders)} frames_raw folders")
    
    total_synced = 0
    total_verified = 0
    total_deleted = 0
    
    for frames_raw_folder in frames_raw_folders:
        logger.info(f"Processing {frames_raw_folder}")
        
        # Find 11-character folders
        eleven_char_folders = find_11_char_folders(frames_raw_folder)
        logger.info(f"Found {len(eleven_char_folders)} folders with 11 characters")
        
        for folder in eleven_char_folders:
            # Check if folder is older than threshold
            if is_older_than_threshold(folder, MODIFICATION_THRESHOLD_HOURS):
                logger.info(f"Folder {folder.name} is older than {MODIFICATION_THRESHOLD_HOURS} hour(s)")
                
                # Calculate relative path for destination
                relative_path = calculate_relative_path(folder, SOURCE_BASE)
                
                # Sync to remote
                if sync_folder_to_remote(folder, REMOTE_DEST, relative_path):
                    total_synced += 1
                    
                    # Verify sync
                    if verify_remote_folder(folder, REMOTE_DEST, relative_path):
                        total_verified += 1
                        logger.info(f"✓ Successfully synced and verified {folder.name}")
                        
                        # Delete source folder only after successful verification (if enabled)
                        if DELETE_AFTER_SYNC:
                            if delete_source_folder(folder):
                                total_deleted += 1
                            else:
                                logger.warning(f"⚠ Synced and verified but failed to delete {folder.name}")
                        else:
                            logger.info(f"Source deletion disabled - keeping {folder.name}")
                    else:
                        logger.warning(f"⚠ Synced but verification failed for {folder.name} - NOT deleting source")
                else:
                    logger.error(f"✗ Failed to sync {folder.name} - NOT deleting source")
            else:
                logger.debug(f"Folder {folder.name} is too recent, skipping")
    
    deletion_status = f", {total_deleted} deleted from source" if DELETE_AFTER_SYNC else " (deletion disabled)"
    logger.info(f"Sync completed: {total_synced} synced, {total_verified} verified{deletion_status}")


def main():
    """Main loop that runs the sync process forever."""
    logger.info("Starting continuous sync process...")
    logger.info(f"Source: {SOURCE_BASE}")
    logger.info(f"Destination: {REMOTE_DEST}")
    logger.info(f"Sleep interval: {SLEEP_INTERVAL} seconds")
    logger.info(f"Modification threshold: {MODIFICATION_THRESHOLD_HOURS} hour(s)")
    logger.info(f"Delete after sync: {DELETE_AFTER_SYNC}")
    
    # Test SSH connection before starting
    if not test_ssh_connection(REMOTE_DEST):
        logger.error("Cannot proceed without working SSH connection")
        logger.error("Please run: ./setup_ssh_keys.sh and set up SSH keys")
        return
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            logger.info(f"--- Sync Cycle #{cycle_count} ---")
            
            process_sync_candidates()
            
            logger.info(f"Sleeping for {SLEEP_INTERVAL} seconds...")
            time.sleep(SLEEP_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            logger.info(f"Continuing after {SLEEP_INTERVAL} seconds...")
            time.sleep(SLEEP_INTERVAL)
    
    logger.info("Sync process stopped")


if __name__ == "__main__":
    main()