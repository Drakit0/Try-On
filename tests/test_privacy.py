"""
Tests for Privacy and Security Module
--------------------------------------
Validates capture management, privacy safeguards, and data handling.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.privacy import (
    CaptureManager,
    PrivacyAuditor,
    format_file_size,
    get_privacy_config,
)


class TestCaptureManager:
    """Test capture management functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def capture_manager(self, temp_dir):
        """Create capture manager with temporary directory."""
        return CaptureManager(
            capture_dir=temp_dir / "captures",
            max_captures=10,
            auto_cleanup_on_exit=False
        )
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample video frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red frame
        return frame
    
    def test_init_creates_directory(self, temp_dir):
        """Test capture manager creates capture directory."""
        capture_dir = temp_dir / "test_captures"
        assert not capture_dir.exists()
        
        CaptureManager(capture_dir=capture_dir)
        assert capture_dir.exists()
    
    def test_init_with_default_dir(self):
        """Test capture manager with default temp directory."""
        mgr = CaptureManager()
        assert mgr.capture_dir.exists()
        assert "virtual_tryon_captures" in str(mgr.capture_dir)
    
    def test_capture_frame_success(self, capture_manager, sample_frame):
        """Test successful frame capture."""
        success, filepath = capture_manager.capture_frame(sample_frame)
        
        assert success is True
        assert Path(filepath).exists()
        assert filepath.endswith('.png')
    
    def test_capture_frame_with_metadata(self, capture_manager, sample_frame):
        """Test frame capture with metadata."""
        metadata = {
            'garment': 'test_shirt',
            'settings': {'size': 1.0, 'offset': 0}
        }
        
        success, filepath = capture_manager.capture_frame(sample_frame, metadata)
        
        assert success is True
        
        # Check metadata file exists
        meta_path = Path(filepath).with_suffix('.json')
        assert meta_path.exists()
        
        # Verify metadata content
        with open(meta_path, 'r') as f:
            saved_meta = json.load(f)
            assert saved_meta['garment'] == 'test_shirt'
            assert saved_meta['settings']['size'] == 1.0
    
    def test_capture_limit(self, capture_manager, sample_frame):
        """Test max captures limit is enforced."""
        # Capture up to limit
        for i in range(10):
            success, _ = capture_manager.capture_frame(sample_frame)
            assert success is True
        
        # Next capture should fail
        success, msg = capture_manager.capture_frame(sample_frame)
        assert success is False
        assert "Maximum captures reached" in msg
    
    def test_list_captures(self, capture_manager, sample_frame):
        """Test listing captures."""
        assert len(capture_manager.list_captures()) == 0
        
        # Add captures
        capture_manager.capture_frame(sample_frame)
        capture_manager.capture_frame(sample_frame)
        
        captures = capture_manager.list_captures()
        assert len(captures) == 2
        
        # Should be sorted by time (newest first)
        assert captures[0].stat().st_mtime >= captures[1].stat().st_mtime
    
    def test_get_capture_count(self, capture_manager, sample_frame):
        """Test capture count."""
        assert capture_manager.get_capture_count() == 0
        
        capture_manager.capture_frame(sample_frame)
        assert capture_manager.get_capture_count() == 1
        
        capture_manager.capture_frame(sample_frame)
        assert capture_manager.get_capture_count() == 2
    
    def test_delete_capture(self, capture_manager, sample_frame):
        """Test deleting single capture."""
        success, filepath = capture_manager.capture_frame(sample_frame)
        assert Path(filepath).exists()
        
        deleted = capture_manager.delete_capture(filepath)
        assert deleted is True
        assert not Path(filepath).exists()
    
    def test_delete_capture_with_metadata(self, capture_manager, sample_frame):
        """Test deleting capture also deletes metadata."""
        metadata = {'test': 'data'}
        success, filepath = capture_manager.capture_frame(sample_frame, metadata)
        
        meta_path = Path(filepath).with_suffix('.json')
        assert meta_path.exists()
        
        capture_manager.delete_capture(filepath)
        
        assert not Path(filepath).exists()
        assert not meta_path.exists()
    
    def test_delete_all_captures(self, capture_manager, sample_frame):
        """Test deleting all captures."""
        # Create multiple captures
        for i in range(5):
            capture_manager.capture_frame(sample_frame, {'index': i})
        
        assert capture_manager.get_capture_count() == 5
        
        deleted, errors = capture_manager.delete_all_captures()
        
        assert deleted >= 5  # At least 5 PNGs (metadata files also deleted)
        assert errors == 0
        assert capture_manager.get_capture_count() == 0
    
    def test_get_storage_size(self, capture_manager, sample_frame):
        """Test storage size calculation."""
        assert capture_manager.get_storage_size() == 0
        
        capture_manager.capture_frame(sample_frame)
        size = capture_manager.get_storage_size()
        
        assert size > 0
        assert size < 1024 * 1024  # Should be less than 1MB for small frame
    
    def test_get_storage_size_mb(self, capture_manager, sample_frame):
        """Test storage size in MB."""
        capture_manager.capture_frame(sample_frame)
        size_mb = capture_manager.get_storage_size_mb()
        
        assert size_mb > 0
        assert size_mb < 1.0  # Should be less than 1MB


class TestPrivacyAuditor:
    """Test privacy auditing functionality."""
    
    def test_verify_no_network_calls(self):
        """Test network call verification."""
        is_compliant, issues = PrivacyAuditor.verify_no_network_calls()
        
        # Should be compliant (all processing is local)
        assert is_compliant is True
        assert len(issues) == 0
    
    def test_get_privacy_statement(self):
        """Test privacy statement retrieval."""
        statement = PrivacyAuditor.get_privacy_statement()
        
        assert "Local Processing" in statement
        assert "No Data Upload" in statement
        assert "No Persistent Storage" in statement
        assert "HTTPS" in statement
    
    def test_get_limitations_statement(self):
        """Test limitations statement retrieval."""
        statement = PrivacyAuditor.get_limitations_statement()
        
        assert "Limitations" in statement
        assert "2D Warping" in statement
        assert "Privacy" in statement
        assert "Browser" in statement


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_format_file_size_bytes(self):
        """Test file size formatting - bytes."""
        assert format_file_size(100) == "100.0 B"
        assert format_file_size(1023) == "1023.0 B"
    
    def test_format_file_size_kb(self):
        """Test file size formatting - kilobytes."""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"
        assert format_file_size(1536) == "1.5 KB"
    
    def test_format_file_size_mb(self):
        """Test file size formatting - megabytes."""
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 2.5) == "2.5 MB"
    
    def test_format_file_size_gb(self):
        """Test file size formatting - gigabytes."""
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_get_privacy_config(self):
        """Test privacy configuration retrieval."""
        config = get_privacy_config()
        
        assert 'capture_dir' in config
        assert 'max_captures' in config
        assert 'auto_cleanup_on_exit' in config
        assert 'show_privacy_notice' in config
        assert 'require_https_mobile' in config
        
        # Verify defaults
        assert config['capture_dir'] is None  # Use temp by default
        assert config['max_captures'] == 100
        assert config['auto_cleanup_on_exit'] is False
        assert config['show_privacy_notice'] is True
        assert config['require_https_mobile'] is True


class TestAutoCleanup:
    """Test auto-cleanup functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_auto_cleanup_disabled(self, temp_dir):
        """Test captures persist when auto-cleanup disabled."""
        capture_dir = temp_dir / "captures"
        
        # Create manager and capture
        mgr = CaptureManager(
            capture_dir=capture_dir,
            auto_cleanup_on_exit=False
        )
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        success, filepath = mgr.capture_frame(frame)
        assert success is True
        
        # Delete manager
        del mgr
        
        # File should still exist
        assert Path(filepath).exists()
    
    def test_auto_cleanup_enabled(self, temp_dir):
        """Test captures deleted when auto-cleanup enabled."""
        capture_dir = temp_dir / "captures"
        
        # Create manager with auto-cleanup
        mgr = CaptureManager(
            capture_dir=capture_dir,
            auto_cleanup_on_exit=True
        )
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        success, filepath = mgr.capture_frame(frame)
        assert success is True
        assert Path(filepath).exists()
        
        # Delete manager (triggers cleanup)
        del mgr
        
        # File should be deleted
        assert not Path(filepath).exists()


class TestPromptRequirements:
    """Test specific requirements from privacy/security prompt."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def capture_manager(self, temp_dir):
        return CaptureManager(capture_dir=temp_dir / "captures")
    
    def test_no_upload_frames(self):
        """Test frames not uploaded (no network calls)."""
        is_compliant, issues = PrivacyAuditor.verify_no_network_calls()
        assert is_compliant is True
    
    def test_ui_notice_text(self):
        """Test UI notice contains required text."""
        statement = PrivacyAuditor.get_privacy_statement()
        
        assert "Local" in statement or "local" in statement
        assert "not stored" in statement.lower() or "no persistent storage" in statement.lower()
    
    def test_delete_captures_button(self, capture_manager):
        """Test delete all captures functionality."""
        # Create test captures
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(3):
            capture_manager.capture_frame(frame)
        
        assert capture_manager.get_capture_count() == 3
        
        # Delete all
        deleted, errors = capture_manager.delete_all_captures()
        assert deleted > 0
        assert capture_manager.get_capture_count() == 0
    
    def test_captures_in_temp_or_configured_path(self, temp_dir):
        """Test captures saved to configured path or temp."""
        # With configured path
        mgr_configured = CaptureManager(capture_dir=temp_dir / "custom")
        assert str(temp_dir / "custom") in str(mgr_configured.capture_dir)
        
        # With default (temp)
        mgr_default = CaptureManager()
        assert "virtual_tryon_captures" in str(mgr_default.capture_dir)
    
    def test_https_reminder_in_privacy_statement(self):
        """Test HTTPS reminder present in privacy statement."""
        statement = PrivacyAuditor.get_privacy_statement()
        assert "HTTPS" in statement
        assert "mobile" in statement.lower() or "camera" in statement.lower()
