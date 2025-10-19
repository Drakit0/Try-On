"""
Tests for logging and telemetry utilities.

Test coverage for:
- Logger setup with rotating file handler
- PerformanceTracker with rolling averages
- Timing and formatting utilities
"""

import logging
import tempfile
from pathlib import Path

import pytest

from src.utils import (
    PerformanceTracker,
    format_fps,
    format_time_ms,
    setup_logger,
)


class TestLoggerSetup:
    """Tests for setup_logger function."""
    
    def test_logger_creation(self):
        """Test basic logger creation."""
        logger = setup_logger('test_logger', level=logging.INFO)
        
        assert logger is not None
        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO
    
    def test_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            logger = setup_logger(
                'test_file_logger',
                level=logging.DEBUG,
                log_file=str(log_file)
            )
            
            # Write log message
            logger.info("Test message")
            logger.debug("Debug message")
            
            # Close handlers before cleanup (important on Windows)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Check file exists
            assert log_file.exists()
            
            # Check content
            content = log_file.read_text()
            assert "Test message" in content
            assert "Debug message" in content
    
    def test_logger_levels(self):
        """Test different log levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            # INFO level logger
            logger = setup_logger(
                'test_levels',
                level=logging.INFO,
                log_file=str(log_file)
            )
            
            logger.debug("Debug message - should not appear")
            logger.info("Info message - should appear")
            
            # Close handlers before cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            content = log_file.read_text()
            assert "Debug message" not in content
            assert "Info message" in content
    
    def test_logger_singleton(self):
        """Test that calling setup_logger twice returns same logger."""
        logger1 = setup_logger('singleton_test')
        logger2 = setup_logger('singleton_test')
        
        assert logger1 is logger2


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceTracker(window_size=30)
        
        assert tracker.window_size == 30
        assert len(tracker.fps_samples) == 0
        assert 'pose_detection' in tracker.timing_samples
        assert 'tps_warping' in tracker.timing_samples
    
    def test_record_frame_time(self):
        """Test recording frame processing time."""
        tracker = PerformanceTracker(window_size=10)
        
        # Record some frame times (33ms = ~30 FPS)
        for _ in range(5):
            tracker.record_frame_time(0.033)
        
        assert len(tracker.fps_samples) == 5
        
        fps = tracker.get_fps()
        assert 29.0 < fps < 31.0  # Should be ~30 FPS
    
    def test_record_operation(self):
        """Test recording operation timing."""
        tracker = PerformanceTracker(window_size=10)
        
        # Record pose detection times
        tracker.record_operation('pose_detection', 0.015)  # 15ms
        tracker.record_operation('pose_detection', 0.018)  # 18ms
        tracker.record_operation('pose_detection', 0.012)  # 12ms
        
        stats = tracker.get_stats()
        
        # Average should be (15 + 18 + 12) / 3 = 15ms
        assert 14.0 < stats['avg_pose_detection'] < 16.0
        assert stats['min_pose_detection'] == 12.0
        assert stats['max_pose_detection'] == 18.0
    
    def test_rolling_window(self):
        """Test that tracker maintains rolling window."""
        tracker = PerformanceTracker(window_size=5)
        
        # Record more samples than window size
        for i in range(10):
            tracker.record_frame_time(0.033)
        
        # Should only keep last 5
        assert len(tracker.fps_samples) == 5
    
    def test_get_stats(self):
        """Test comprehensive statistics retrieval."""
        tracker = PerformanceTracker(window_size=10)
        
        # Record various operations
        tracker.record_frame_time(0.033)  # 30 FPS
        tracker.record_operation('pose_detection', 0.015)
        tracker.record_operation('tps_warping', 0.008)
        tracker.record_operation('segmentation', 0.012)
        
        stats = tracker.get_stats()
        
        # Check all required stats are present
        assert 'avg_fps' in stats
        assert 'min_fps' in stats
        assert 'max_fps' in stats
        assert 'avg_pose_detection' in stats
        assert 'avg_tps_warping' in stats
        assert 'avg_segmentation' in stats
        assert 'avg_total' in stats
        
        # Check values are reasonable
        assert stats['avg_fps'] > 0
        assert stats['avg_pose_detection'] == 15.0  # ms
        assert stats['avg_tps_warping'] == 8.0  # ms
        assert stats['avg_segmentation'] == 12.0  # ms
    
    def test_reset(self):
        """Test tracker reset."""
        tracker = PerformanceTracker(window_size=10)
        
        # Record some data
        tracker.record_frame_time(0.033)
        tracker.record_operation('pose_detection', 0.015)
        
        # Reset
        tracker.reset()
        
        assert len(tracker.fps_samples) == 0
        assert all(len(samples) == 0 for samples in tracker.timing_samples.values())
        assert tracker.get_fps() == 0.0
    
    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        tracker = PerformanceTracker(window_size=10)
        
        # Simulate frame processing
        tracker.record_operation('pose_detection', 0.015)
        tracker.record_operation('tps_warping', 0.008)
        tracker.record_operation('segmentation', 0.012)
        tracker.record_operation('overlay', 0.003)
        tracker.record_frame_time(0.038)  # Total time
        
        stats = tracker.get_stats()
        
        # Check all operations recorded
        assert stats['avg_pose_detection'] == 15.0
        assert stats['avg_tps_warping'] == 8.0
        assert stats['avg_segmentation'] == 12.0
        assert stats['avg_overlay'] == 3.0
        assert stats['avg_total'] == 38.0


class TestFormatters:
    """Tests for formatting utilities."""
    
    def test_format_time_ms(self):
        """Test millisecond formatting."""
        assert format_time_ms(0.001) == "1.0ms"
        assert format_time_ms(0.0152) == "15.2ms"
        assert format_time_ms(0.0999) == "99.9ms"
        assert format_time_ms(1.234) == "1234.0ms"
    
    def test_format_fps(self):
        """Test FPS formatting."""
        assert format_fps(30.0) == "30.0 FPS"
        assert format_fps(25.432) == "25.4 FPS"
        assert format_fps(60.0) == "60.0 FPS"
        assert format_fps(15.9) == "15.9 FPS"


class TestIntegration:
    """Integration tests for logging and telemetry."""
    
    def test_logger_and_tracker_together(self):
        """Test logger and tracker working together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'integration.log'
            logger = setup_logger('integration', log_file=str(log_file))
            tracker = PerformanceTracker(window_size=30)
            
            # Simulate frame processing
            logger.info("Starting frame processing")
            tracker.record_operation('pose_detection', 0.015)
            logger.debug(f"Pose detection: 15.0ms")
            
            tracker.record_operation('tps_warping', 0.008)
            logger.debug(f"TPS warping: 8.0ms")
            
            tracker.record_frame_time(0.025)
            fps = tracker.get_fps()
            logger.info(f"Frame processed: {fps:.1f} FPS")
            
            # Close handlers before cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Check log file
            content = log_file.read_text()
            assert "Starting frame processing" in content
            assert "Frame processed" in content
    
    def test_performance_warning_threshold(self):
        """Test logging performance warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'warnings.log'
            logger = setup_logger('warnings', log_file=str(log_file))
            tracker = PerformanceTracker(window_size=10)
            
            # Simulate slow frame (50ms = 20 FPS)
            slow_time = 0.050
            tracker.record_frame_time(slow_time)
            
            # Log warning if above threshold (40ms)
            if slow_time > 0.040:
                logger.warning(
                    f"Frame processing slow: {slow_time*1000:.1f}ms "
                    f"(target: 40.0ms)"
                )
            
            # Close handlers before cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            content = log_file.read_text()
            assert "Frame processing slow: 50.0ms" in content
