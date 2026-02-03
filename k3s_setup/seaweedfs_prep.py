# seaweedfs_prep.py
#
# Prepare nodes for SeaweedFS
#
# Usage:
#   pyinfra inventory.py seaweedfs_prep.py -y

from pyinfra import host
from pyinfra.operations import apt, files, server

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = "/data/seaweed"
MOUNT_PARENT_DIR = "/swdfs_mnt"
MOUNT_DIR = "/swdfs_mnt/swshared"

# ═══════════════════════════════════════════════════════════════════
# PACKAGES
# ═══════════════════════════════════════════════════════════════════

apt.packages(name="Install fuse", packages=["fuse3"], _sudo=True)

# ═══════════════════════════════════════════════════════════════════
# DIRECTORIES
# ═══════════════════════════════════════════════════════════════════

files.directory(name=f"Create {DATA_DIR}", path=DATA_DIR, mode="755", _sudo=True)
files.directory(name=f"Create {MOUNT_PARENT_DIR}", path=MOUNT_PARENT_DIR, mode="755", _sudo=True)
files.directory(name=f"Create {MOUNT_DIR}", path=MOUNT_DIR, mode="777", _sudo=True)

# ═══════════════════════════════════════════════════════════════════
# FUSE CONFIG
# ═══════════════════════════════════════════════════════════════════

files.line(
    name="Enable FUSE user_allow_other",
    path="/etc/fuse.conf",
    line="user_allow_other",
    replace="^#?user_allow_other.*",
    present=True,
    _sudo=True,
)

# ═══════════════════════════════════════════════════════════════════
# SYSCTL TUNING
# ═══════════════════════════════════════════════════════════════════

server.shell(name="Configure sysctl", commands=["""
cat > /etc/sysctl.d/99-seaweedfs.conf << 'EOF'
# SeaweedFS optimizations
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
net.core.somaxconn = 65535
vm.swappiness = 10
EOF
sysctl --system > /dev/null
"""], _sudo=True)