"""
Tests for the downloader module.
"""

import pytest
from pathlib import Path

from video_miner_async.utils.io import get_video_id
from video_miner_async.utils.validators import validate_youtube_url


class TestVideoIdExtraction:
    """Test YouTube video ID extraction."""
    
    def test_standard_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert get_video_id(url) == "dQw4w9WgXcQ"
    
    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert get_video_id(url) == "dQw4w9WgXcQ"
    
    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert get_video_id(url) == "dQw4w9WgXcQ"
    
    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert get_video_id(url) == "dQw4w9WgXcQ"
    
    def test_with_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
        assert get_video_id(url) == "dQw4w9WgXcQ"
    
    def test_invalid_url(self):
        url = "https://example.com/video"
        assert get_video_id(url) is None


class TestUrlValidation:
    """Test YouTube URL validation."""
    
    def test_valid_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        is_valid, error = validate_youtube_url(url)
        assert is_valid is True
        assert error is None
    
    def test_invalid_domain(self):
        url = "https://vimeo.com/video/123456"
        is_valid, error = validate_youtube_url(url)
        assert is_valid is False
        assert "Not a YouTube domain" in error
    
    def test_empty_url(self):
        url = ""
        is_valid, error = validate_youtube_url(url)
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_invalid_scheme(self):
        url = "ftp://youtube.com/watch?v=dQw4w9WgXcQ"
        is_valid, error = validate_youtube_url(url)
        assert is_valid is False
