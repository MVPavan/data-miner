#!/bin/bash

# use env_template.txt and create a .env with all those variables
# Load environment variables from .env
source .env

# Generate compose file
cat <<EOF > $COMPOSE_FILE_NAME
services:
  $SERVICE_NAME:
    build:
      context: ./
      dockerfile: $DOCKERFILE_NAME
    image: $IMAGE_NAME
    container_name: $CONTAINER_NAME
    network_mode: host
    environment:
      - HF_HOME=$HF_HOME
      - UV_CACHE_DIR
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - type: bind
        source: $CODE
        target: $CODE
      - type: bind
        source: $DATASET
        target: $DATASET
      - type: bind
        source: $PIXI_HOME
        target: $PIXI_HOME
      - type: bind
        source: $HF_HOME
        target: $HF_HOME
      - type: bind
        source: $UV_CACHE_DIR
        target: $UV_CACHE_DIR
      
    ulimits:
      memlock:
        soft: -1
        hard: -1
      stack:
        soft: 67108864
        hard: 67108864
    security_opt:
      - apparmor:unconfined
    ipc: host
    
    stdin_open: true 
    tty: true

    # Uncomment if below doesnt work
    # runtime: nvidia
    # default choice
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF

echo "$COMPOSE_FILE_NAME generated."


# Generate dev_entrypoint.sh
cat <<EOF > dev_entrypoint.sh
#!/bin/bash
# Set Hugging Face cache directory
echo 'export HF_HOME=$HF_HOME' >> ~/.bashrc

# Set uv package manager cache directory
echo 'export UV_CACHE_DIR=$UV_CACHE_DIR' >> ~/.bashrc

# Set pixi home for caches/config
echo 'export PIXI_HOME=$PIXI_HOME' >> ~/.bashrc

# Apply changes immediately
source ~/.bashrc
EOF

echo "dev_entrypoint generated."

# Generate devcontainer.json
cat <<EOF > devcontainer.json
{
    "name": "$DEVCONTAINER_NAME",
    "dockerComposeFile": "$COMPOSE_FILE_NAME",
    "service": "$SERVICE_NAME",
    "workspaceFolder": "$CODE",
    "customizations": {
      "vscode": {
          "extensions": [
              "ms-python.python",
              "charliermarsh.ruff",
              "astral-sh.ty",
              "eamodio.gitlens",
              "usernamehw.errorlens",
              "streetsidesoftware.code-spell-checker",
              "gruntfuggly.todo-tree",
              "oderwat.indent-rainbow",
              "aaron-bond.better-comments",
              "mikestead.dotenv",
              "mechatroner.rainbow-csv",
              "redhat.vscode-yaml",
              "tamasfe.even-better-toml",
              "tomoki1207.pdf",
              "jock.svg",
              "moshfeu.compare-folders",
              "christian-kohler.path-intellisense",
              "yzhang.markdown-all-in-one",
              "bierner.markdown-mermaid",
              "ms-vscode-remote.vscode-remote-extensionpack"
          ],
          "settings": {
              "terminal.integrated.shell.linux": "/bin/bash",
              "[python]": {
              "editor.formatOnSave": true,
              "editor.defaultFormatter": "charliermarsh.ruff",
              "editor.codeActionsOnSave": {
                  "source.fixAll.ruff": "explicit",
                  "source.organizeImports.ruff": "explicit"
              }
              },
              "ty.enable": true,
              "ty.check.enabled": true,
              "markdown.preview.breaks": true
          }
      }
    },
    "remoteUser": "root",
    "shutdownAction": "none",
    "postCreateCommand": "/bin/bash .devcontainer/dev_entrypoint.sh"
}
EOF

echo "devcontainer.json generated."

echo creating bind mount folders if they do not exist...
mkdir -p $DATASET
mkdir -p $PIXI_HOME
mkdir -p $HF_HOME
mkdir -p $UV_CACHE_DIR