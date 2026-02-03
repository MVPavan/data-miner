#!/bin/bash
# Set Hugging Face cache directory
echo 'export HF_HOME=/data/all_cache/hf_home' >> ~/.bashrc

# Set uv package manager cache directory
echo 'export UV_CACHE_DIR=/data/all_cache/uv_cache' >> ~/.bashrc

# Set pixi home for caches/config
echo 'export PIXI_HOME=/data/all_cache/pixi_home' >> ~/.bashrc

# Apply changes immediately
source ~/.bashrc
