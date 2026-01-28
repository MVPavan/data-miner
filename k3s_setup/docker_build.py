import subprocess
import os
import sys
import fcntl
from pyinfra import host, local
from pyinfra.operations import server, files
from pyinfra.facts.files import File

# Config
IMAGE_NAME = "data-miner"
TAG = "latest"
FULL_IMAGE = f"{IMAGE_NAME}:{TAG}"
TAR_PATH = "/tmp/data-miner.tar"
LOCAL_TAR_PATH = "/tmp/data-miner.tar"
FORCE = os.environ.get("FORCE", "false").lower() == "true"

# 1. Build and Save Image (Run once on control machine)
# -----------------------------------------------------
# Use a file lock to ensure only one thread builds the image, but all wait for it.
LOCK_FILE = "/tmp/data_miner_build.lock"

# Check if local build exists
BUILT = os.path.exists(LOCAL_TAR_PATH)

if FORCE:
    BUILT = False

with open(LOCK_FILE, "w") as lock_f:
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        if not BUILT:
            print(f"[{host.name}] Building Docker Image: {FULL_IMAGE}...")
            subprocess.check_call(
                f"docker build -f k3s_setup/Dockerfile -t {FULL_IMAGE} .",
                shell=True
            )
            
            print(f"[{host.name}] Saving Image to {LOCAL_TAR_PATH}...")
            subprocess.check_call(
                f"docker save {FULL_IMAGE} -o {LOCAL_TAR_PATH}",
                shell=True
            )
            BUILT = True
    except Exception as e:
        print(f"[{host.name}] Error during build: {e}")
        # Release lock is handled by finally
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        sys.exit(1)
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        if not BUILT:
            print(f"[{host.name}] Build failed or was skipped incorrectly.")
            sys.exit(1)


# 2. Copy Image to Node
# ---------------------
# Check if remote file exists
remote_file = host.get_fact(File, path=TAR_PATH)

if FORCE or not remote_file:
    files.put(
        name="Copy image tar to node",
        src=LOCAL_TAR_PATH,
        dest=TAR_PATH,
    )
else:
    print(f"[{host.name}] Skipping copy (file exists on remote and FORCE=false)")

# 3. Import Image to K3s
# ----------------------
server.shell(
    name="Import image to K3s",
    commands=[
        f"sudo k3s ctr images import {TAR_PATH} --namespace k8s.io",
    ],
    _sudo=True,
)

# # 4. Cleanup
# # ----------
# server.shell(
#     name="Remove tar file",
#     commands=[
#         f"rm {TAR_PATH}",
#     ],
#     _sudo=True,
# )
