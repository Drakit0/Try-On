"""
Performance Profiling Tool
===========================

Profile the virtual try-on pipeline to identify bottlenecks.

Uses cProfile for function-level profiling and provides detailed
timing breakdowns for each processing stage.

Usage:
    python tools/profile_performance.py
    python tools/profile_performance.py --frames 100
    python tools/profile_performance.py --output profile.txt

Authors: Pablo Tuñón Laguna, Lydia Ruiz Martínez
Date: 2025-01-19
"""

import argparse
import cProfile
import pstats
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.pose import PoseEstimator
from src.segmenter import PersonSegmenter
from src.smoothing import OneEuroFilter
from src.tps import ThinPlateSpline
from src.overlay import compute_transform, render_overlay


def profile_pipeline(num_frames: int = 100):
    """
    Profile the complete pipeline processing.
    
    Parameters
    ----------
    num_frames : int
        Number of frames to process
    """
    print(f"Profiling pipeline over {num_frames} frames...")
    
    # Initialize modules
    pose = PoseEstimator()
    segmenter = PersonSegmenter()
    
    # Create dummy test frame (640x480 BGR)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create dummy garment (256x384 RGBA)
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
    
    tps = ThinPlateSpline(src_pts)
    
    # Timing storage
    timings = {
        'pose': [],
        'transform': [],
        'tps_fit': [],
        'tps_warp': [],
        'overlay': [],
        'segmentation': [],
        'total': []
    }
    
    for i in range(num_frames):
        frame_start = time.perf_counter()
        
        # Pose detection
        t0 = time.perf_counter()
        landmarks = pose.detect(frame)
        t1 = time.perf_counter()
        timings['pose'].append((t1 - t0) * 1000)
        
        if landmarks:
            # Transform computation
            t0 = time.perf_counter()
            meta = {'base_shoulder_px': 200, 'base_torso_px': 300}
            transform = compute_transform(landmarks, meta)
            t1 = time.perf_counter()
            timings['transform'].append((t1 - t0) * 1000)
            
            # TPS warping
            t0 = time.perf_counter()
            dst_pts = np.array([
                [200, 200],
                [400, 200],
                [220, 350],
                [380, 350],
                [225, 450],
                [375, 450]
            ], dtype=np.float32)
            tps.fit(dst_pts)
            t1 = time.perf_counter()
            timings['tps_fit'].append((t1 - t0) * 1000)
            
            t0 = time.perf_counter()
            warped = tps.warp_rgba(garment)
            t1 = time.perf_counter()
            timings['tps_warp'].append((t1 - t0) * 1000)
            
            # Overlay
            t0 = time.perf_counter()
            output = render_overlay(frame, warped, transform)
            t1 = time.perf_counter()
            timings['overlay'].append((t1 - t0) * 1000)
            
            # Segmentation (occlusion)
            t0 = time.perf_counter()
            mask = segmenter.mask(frame, classes=['face', 'skin'])
            t1 = time.perf_counter()
            timings['segmentation'].append((t1 - t0) * 1000)
        
        frame_end = time.perf_counter()
        timings['total'].append((frame_end - frame_start) * 1000)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    # Print results
    print("\n" + "="*70)
    print("PERFORMANCE PROFILING RESULTS")
    print("="*70)
    
    for stage, times in timings.items():
        if times:
            avg = np.mean(times)
            std = np.std(times)
            min_t = np.min(times)
            max_t = np.max(times)
            p50 = np.percentile(times, 50)
            p95 = np.percentile(times, 95)
            
            print(f"\n{stage.upper()}")
            print(f"  Average:  {avg:7.2f} ms")
            print(f"  Std Dev:  {std:7.2f} ms")
            print(f"  Min:      {min_t:7.2f} ms")
            print(f"  Max:      {max_t:7.2f} ms")
            print(f"  P50:      {p50:7.2f} ms")
            print(f"  P95:      {p95:7.2f} ms")
    
    # FPS calculation
    avg_total = np.mean(timings['total'])
    fps = 1000 / avg_total if avg_total > 0 else 0
    
    print("\n" + "="*70)
    print(f"AVERAGE FRAME TIME: {avg_total:.2f} ms")
    print(f"EFFECTIVE FPS:      {fps:.1f}")
    print("="*70)
    
    # Check acceptance criteria
    print("\nACCEPTANCE CRITERIA:")
    if avg_total < 150:
        print(f"  [ok] Latency < 150ms: {avg_total:.2f} ms")
    else:
        print(f"  [fail] Latency < 150ms: {avg_total:.2f} ms (FAILED)")
    
    if fps >= 20:
        print(f"  [ok] FPS >= 20: {fps:.1f}")
    else:
        print(f"  [fail] FPS >= 20: {fps:.1f} (FAILED)")
    
    return timings


def profile_with_cprofile(output_file: str = None):
    """
    Profile using cProfile.
    
    Parameters
    ----------
    output_file : str, optional
        Output file for profile stats
    """
    print("Running cProfile profiling...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run profiling
    profile_pipeline(num_frames=50)
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    
    if output_file:
        print(f"\nWriting profile to {output_file}...")
        with open(output_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats(50)  # Top 50 functions
        print(f"Profile saved to {output_file}")
    else:
        print("\nTop 30 functions by cumulative time:")
        stats.sort_stats('cumulative')
        stats.print_stats(30)


def main():
    parser = argparse.ArgumentParser(description='Profile virtual try-on performance')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to process (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for cProfile results')
    parser.add_argument('--cprofile', action='store_true',
                        help='Use cProfile for detailed function profiling')
    
    args = parser.parse_args()
    
    if args.cprofile:
        profile_with_cprofile(args.output)
    else:
        profile_pipeline(args.frames)


if __name__ == '__main__':
    main()
