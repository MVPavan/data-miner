# inventory.py
# pyinfra inventory.py debug-inventory
import subprocess

MASTER_IP = "10.96.122.9"

# Auto-read token if running from master
try:
    result = subprocess.run(["sudo", "cat", "/var/lib/rancher/k3s/server/node-token"], 
                          capture_output=True, text=True)
    K3S_TOKEN = result.stdout.strip() if result.returncode == 0 else ""
    if K3S_TOKEN:
        print("Token found: {}".format(K3S_TOKEN))
    else:
        print("No token found")
except Exception as e:
    K3S_TOKEN = ""
    print("No token found: {}".format(e))

hosts = [
    (MASTER_IP, {
        "role": "master",
        "hostname": "k3s-master-pavanjci",
        "ssh_user": "pavanmv",
        "ssh_key": "~/.ssh/id_rsa_dm_k3s",
    }),
    ("10.96.122.132", {
        "role": "worker",
        "hostname": "k3s-worker-1-manthana",
        "ssh_user": "pavanmv",
        "ssh_key": "~/.ssh/id_rsa_dm_k3s",
    }),
    ("10.96.122.14", {
        "role": "worker",
        "hostname": "k3s-worker-2-arsenal",
        "ssh_user": "pavan",
        "ssh_key": "~/.ssh/id_rsa_dm_k3s",
    }),
]

for _, data in hosts:
    data["master_ip"] = MASTER_IP
    data["k3s_token"] = K3S_TOKEN

'''
for password less master

sudo visudo

Then add this line at the end of the file:

pavanmv ALL=(ALL) NOPASSWD: ALL
'''