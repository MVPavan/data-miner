# Configuration for sync_frames.py
# Update these values according to your environment

# Remote destination (update with actual username, IP, and path)
REMOTE_DEST = "pavan@10.160.105.54:/media/data_2/datasets/yt_frames"

# Source directory to monitor
SOURCE_BASE = "/mnt/data_2/pavan/project_helpers/video_miner_v3/output"

# Time interval between sync cycles (seconds)
SLEEP_INTERVAL = 300  # 5 minutes

# Only sync folders older than this many hours
MODIFICATION_THRESHOLD_HOURS = 1

# Delete source folders after successful sync and verification
DELETE_AFTER_SYNC = True

# Verification timeout for rsync checksum verification (seconds)
VERIFICATION_TIMEOUT = 1800  # 30 minutes

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "INFO"