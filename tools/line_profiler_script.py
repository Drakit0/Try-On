"""
Line-by-Line Profiler Setup
============================

Uses line_profiler to identify bottlenecks at line level in critical functions.

This script profiles:
- app.process_frame
- tps.warp_rgba
- overlay.render_overlay
- pose.detect

Usage:
    # Install line_profiler first
    pip install line_profiler
    
    # Run profiling
    kernprof -l -v tools/line_profiler_script.py
    
    # Or use the wrapper
    python tools/run_line_profiler.py

Authors: Pablo Tuñón Laguna, Lydia Ruiz Martínez
Date: 2025-01-19
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.pose import PoseEstimator
from src.tps import ThinPlateSpline
from src.overlay import render_overlay


def profile_tps_warp():
    """Profile TPS warping."""
    # Create test garment
    garment = np.random.randint(0, 255, (384, 256, 4), dtype=np.uint8)
    
    # TPS control points
    src_pts = np.array([
        [50, 75],
        [206, 75],
        [70, 250],
        [186, 250],
        [75, 350],
        [181, 350]
    ], dtype=np.float32)
    
    dst_pts = np.array([
        [200, 200],
        [400, 200],
        [220, 350],
        [380, 350],
        [225, 450],
        [375, 450]
    ], dtype=np.float32)
    
    tps = ThinPlateSpline(src_pts)
    tps.fit(dst_pts)
    
    # Profile 50 warps
    for i in range(50):
        warped = tps.warp_rgba(garment)
    
    return warped


def profile_pose_detection():
    """Profile pose detection."""
    pose = PoseEstimator()
    
    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Profile 50 detections
    for i in range(50):
        landmarks = pose.detect(frame)
    
    return landmarks


def profile_overlay():
    """Profile overlay rendering."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    garment = np.random.randint(0, 255, (384, 256, 4), dtype=np.uint8)
    
    transform = {
        'center': (320, 240),
        'angle': 0.0,
        'scale': 1.0
    }
    
    # Profile 50 overlays
    for i in range(50):
        output = render_overlay(frame, garment, transform)
    
    return output


def main():
    """Run all profiling."""
    print("Profiling TPS warping...")
    profile_tps_warp()
    
    print("Profiling pose detection...")
    profile_pose_detection()
    
    print("Profiling overlay rendering...")
    profile_overlay()
    
    print("\nProfiling complete!")


if __name__ == '__main__':
    # Add @profile decorators to functions you want to profile
    # Then run: kernprof -l -v tools/line_profiler_script.py
    main()
