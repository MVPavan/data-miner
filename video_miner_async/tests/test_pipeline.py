"""
Tests for the pipeline module.
"""

import pytest
from pathlib import Path

from video_miner_async.config import (
    PipelineConfig,
    DownloadConfig,
    ExtractionConfig,
    FilterConfig,
    DeduplicationConfig,
    DetectionConfig,
    DetectorType,
    SamplingStrategy,
)


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_default_config(self):
        config = PipelineConfig()
        assert config.device == "auto"
        assert len(config.stages) == 5
    
    def test_custom_stages(self):
        config = PipelineConfig(stages=["download", "extract"])
        assert len(config.stages) == 2
        assert "download" in config.stages
    
    def test_url_file_loading(self, tmp_path):
        # Create a temp URL file
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://youtube.com/watch?v=abc123\n"
            "https://youtube.com/watch?v=def456\n"
            "# This is a comment\n"
            "\n"  # Empty line
        )
        
        config = PipelineConfig(url_file=url_file)
        urls = config.get_urls()
        
        assert len(urls) == 2
        assert "abc123" in urls[0]
        assert "def456" in urls[1]
    
    def test_combined_urls(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://youtube.com/watch?v=file123\n")
        
        config = PipelineConfig(
            urls=["https://youtube.com/watch?v=direct123"],
            url_file=url_file,
        )
        urls = config.get_urls()
        
        assert len(urls) == 2


class TestDetectorType:
    """Test detector type enum."""
    
    def test_detector_values(self):
        assert DetectorType.DINO_X.value == "dino-x"
        assert DetectorType.MOONDREAM3.value == "moondream3"
        assert DetectorType.FLORENCE2.value == "florence2"
        assert DetectorType.GROUNDING_DINO.value == "grounding-dino"


class TestSamplingStrategy:
    """Test sampling strategy enum."""
    
    def test_sampling_values(self):
        assert SamplingStrategy.INTERVAL.value == "interval"
        assert SamplingStrategy.TIME_BASED.value == "time"
        assert SamplingStrategy.KEYFRAME.value == "keyframe"
