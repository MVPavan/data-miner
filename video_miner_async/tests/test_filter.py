"""
Tests for the filter module.
"""

import pytest
from pathlib import Path

from video_miner_async.config import FilterConfig
from video_miner_async.utils.validators import validate_classes


class TestClassValidation:
    """Test class/prompt validation."""
    
    def test_valid_classes(self):
        classes = ["glass door", "window", "sliding door"]
        is_valid, error = validate_classes(classes)
        assert is_valid is True
        assert error is None
    
    def test_empty_classes(self):
        classes = []
        is_valid, error = validate_classes(classes)
        assert is_valid is False
        assert "No classes" in error
    
    def test_empty_string_class(self):
        classes = ["glass door", "", "window"]
        is_valid, error = validate_classes(classes)
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_whitespace_class(self):
        classes = ["glass door", "   ", "window"]
        is_valid, error = validate_classes(classes)
        assert is_valid is False


class TestFilterConfig:
    """Test filter configuration."""
    
    def test_default_config(self):
        config = FilterConfig()
        assert config.threshold == 0.25
        assert config.batch_size == 16
        assert "siglip" in config.model_id.lower()
    
    def test_custom_threshold(self):
        config = FilterConfig(threshold=0.5)
        assert config.threshold == 0.5
    
    def test_path_conversion(self):
        config = FilterConfig(output_dir="./test_output")
        assert isinstance(config.output_dir, Path)
