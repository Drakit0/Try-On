"""
Utility Functions for Virtual Try-On
=====================================

Provides logging, performance tracking, and helper utilities.

Author: Virtual Try-On Team
Date: 2025-01-19
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 3,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with rotating file handler and console output.
    
    Parameters
    ----------
    name : str
        Logger name (typically module name like 'app' or 'src.pose')
    level : int
        Logging level (logging.DEBUG, logging.INFO, etc.)
    log_file : str, optional
        Path to log file. If None, defaults to 'logs/{name}.log'
    max_bytes : int
        Maximum size of log file before rotation (default: 10 MB)
    backup_count : int
        Number of backup log files to keep (default: 3)
    console_output : bool
        Whether to also output logs to console (default: True)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Examples
    --------
    >>> logger = setup_logger('app', level=logging.INFO)
    >>> logger.info("Application started")
    >>> logger.debug("This only appears if level=DEBUG")
    
    >>> # For more verbose logging during development
    >>> logger = setup_logger('app', level=logging.DEBUG)
    >>> logger.debug("Detailed debugging info")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (if log_file specified or default)
    if log_file or True:  # Always create file handler by default
        # Default log file path
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{name.replace('.', '_')}.log"
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class PerformanceTracker:
    """
    Track performance metrics for video processing pipeline.
    
    Maintains rolling averages for FPS and timing breakdowns.
    
    Attributes
    ----------
    window_size : int
        Number of samples to keep for rolling average
    fps_samples : list
        Recent FPS measurements
    timing_samples : dict
        Recent timing measurements for each operation
    
    Examples
    --------
    >>> tracker = PerformanceTracker(window_size=30)
    >>> 
    >>> # Record frame processing time
    >>> tracker.record_frame_time(0.033)  # 33ms
    >>> 
    >>> # Record operation timing
    >>> tracker.record_operation('pose_detection', 0.015)  # 15ms
    >>> tracker.record_operation('tps_warping', 0.008)     # 8ms
    >>> 
    >>> # Get statistics
    >>> stats = tracker.get_stats()
    >>> print(f"Average FPS: {stats['avg_fps']:.1f}")
    >>> print(f"Pose detection: {stats['avg_pose_detection']:.1f}ms")
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance tracker.
        
        Parameters
        ----------
        window_size : int
            Number of samples for rolling average (default: 30)
        """
        self.window_size = window_size
        self.fps_samples = []
        self.timing_samples = {}
        
        # Operation timings we track
        self.tracked_operations = [
            'pose_detection',
            'tps_warping',
            'segmentation',
            'overlay',
            'total'
        ]
        
        # Initialize timing sample lists
        for op in self.tracked_operations:
            self.timing_samples[op] = []
    
    def record_frame_time(self, time_seconds: float):
        """
        Record total frame processing time.
        
        Parameters
        ----------
        time_seconds : float
            Processing time in seconds
        """
        # Calculate FPS
        if time_seconds > 0:
            fps = 1.0 / time_seconds
            self.fps_samples.append(fps)
            
            # Keep only last N samples
            if len(self.fps_samples) > self.window_size:
                self.fps_samples.pop(0)
        
        # Also record as 'total' timing
        self.record_operation('total', time_seconds)
    
    def record_operation(self, operation: str, time_seconds: float):
        """
        Record timing for a specific operation.
        
        Parameters
        ----------
        operation : str
            Operation name (e.g., 'pose_detection', 'tps_warping')
        time_seconds : float
            Operation time in seconds
        """
        # Create list if operation not tracked yet
        if operation not in self.timing_samples:
            self.timing_samples[operation] = []
        
        # Convert to milliseconds for easier reading
        time_ms = time_seconds * 1000.0
        self.timing_samples[operation].append(time_ms)
        
        # Keep only last N samples
        if len(self.timing_samples[operation]) > self.window_size:
            self.timing_samples[operation].pop(0)
    
    def get_stats(self) -> dict:
        """
        Get current performance statistics.
        
        Returns
        -------
        dict
            Statistics including average FPS and timing breakdowns
            
        Example return value:
        {
            'avg_fps': 25.4,
            'min_fps': 18.2,
            'max_fps': 30.0,
            'avg_pose_detection': 15.2,  # ms
            'avg_tps_warping': 8.1,       # ms
            'avg_segmentation': 12.5,     # ms
            'avg_total': 35.8             # ms
        }
        """
        stats = {}
        
        # FPS statistics
        if self.fps_samples:
            stats['avg_fps'] = sum(self.fps_samples) / len(self.fps_samples)
            stats['min_fps'] = min(self.fps_samples)
            stats['max_fps'] = max(self.fps_samples)
        else:
            stats['avg_fps'] = 0.0
            stats['min_fps'] = 0.0
            stats['max_fps'] = 0.0
        
        # Timing statistics (in milliseconds)
        for op, samples in self.timing_samples.items():
            if samples:
                stats[f'avg_{op}'] = sum(samples) / len(samples)
                stats[f'min_{op}'] = min(samples)
                stats[f'max_{op}'] = max(samples)
            else:
                stats[f'avg_{op}'] = 0.0
                stats[f'min_{op}'] = 0.0
                stats[f'max_{op}'] = 0.0
        
        return stats
    
    def get_fps(self) -> float:
        """
        Get current average FPS.
        
        Returns
        -------
        float
            Average FPS over last N frames
        """
        if self.fps_samples:
            return sum(self.fps_samples) / len(self.fps_samples)
        return 0.0
    
    def reset(self):
        """Reset all performance counters."""
        self.fps_samples.clear()
        for op in self.timing_samples:
            self.timing_samples[op].clear()


def format_time_ms(time_seconds: float) -> str:
    """
    Format time in seconds as milliseconds string.
    
    Parameters
    ----------
    time_seconds : float
        Time in seconds
    
    Returns
    -------
    str
        Formatted string like "15.2ms"
    
    Examples
    --------
    >>> format_time_ms(0.0152)
    '15.2ms'
    >>> format_time_ms(0.001)
    '1.0ms'
    """
    return f"{time_seconds * 1000:.1f}ms"


def format_fps(fps: float) -> str:
    """
    Format FPS value as string.
    
    Parameters
    ----------
    fps : float
        Frames per second
    
    Returns
    -------
    str
        Formatted string like "25.4 FPS"
    
    Examples
    --------
    >>> format_fps(25.432)
    '25.4 FPS'
    >>> format_fps(60.0)
    '60.0 FPS'
    """
    return f"{fps:.1f} FPS"
