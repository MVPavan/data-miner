"""FastAPI-based viewer for auto_annotation_v2 pipeline results.

Much faster than the NiceGUI viewer — all rendering happens client-side.

Usage:
    python -m data_miner.auto_annotation_v2.viewer_fast --output-dir output/auto_annotation_v2
    python -m data_miner.auto_annotation_v2.viewer_fast --image-dir /path/to/images --port 8090
"""
