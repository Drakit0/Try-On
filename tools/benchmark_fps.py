"""
FPS Benchmark Tool
==================

Benchmark real-world FPS performance with actual webcam or video file.

Tests various configurations (TPS on/off, occlusion on/off) and measures
sustained FPS over time.

Usage:
    python tools/benchmark_fps.py
    python tools/benchmark_fps.py --duration 30
    python tools/benchmark_fps.py --video test_video.mp4

Author: Virtual Try-On Team
Date: 2025-01-19
"""

import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.pose import PoseEstimator
from src.segmenter import PersonSegmenter
from src.tps import ThinPlateSpline
from src.overlay import compute_transform, render_overlay, compose_with_occlusion


class FPSBenchmark:
    """FPS benchmark runner."""
    
    def __init__(self, video_source=0, duration=10):
        """
        Initialize benchmark.
        
        Parameters
        ----------
        video_source : int or str
            Video source (0 for webcam, or path to video file)
        duration : int
            Benchmark duration in seconds
        """
        self.video_source = video_source
        self.duration = duration
        
        # Initialize modules
        self.pose = PoseEstimator()
        self.segmenter = PersonSegmenter()
        
        # Load test garment
        garment_path = Path("assets/garments/sample_shirt.png")
        if garment_path.exists():
            self.garment = cv2.imread(str(garment_path), cv2.IMREAD_UNCHANGED)
        else:
            # Create dummy garment
            self.garment = np.random.randint(0, 255, (384, 256, 4), dtype=np.uint8)
        
        # TPS setup
        src_pts = np.array([
            [50, 75],
            [206, 75],
            [70, 250],
            [186, 250],
            [75, 350],
            [181, 350]
        ], dtype=np.float32)
        self.tps = ThinPlateSpline(src_pts)
        
        # Metadata
        self.meta = {'base_shoulder_px': 200, 'base_torso_px': 300}
    
    def benchmark_config(self, use_tps=False, use_occlusion=False):
        """
        Benchmark a specific configuration.
        
        Parameters
        ----------
        use_tps : bool
            Enable TPS warping
        use_occlusion : bool
            Enable occlusion
        
        Returns
        -------
        dict
            Benchmark results
        """
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video source: {self.video_source}")
            return None
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\nBenchmarking: TPS={use_tps}, Occlusion={use_occlusion}")
        print(f"Duration: {self.duration}s")
        
        start_time = time.time()
        frame_count = 0
        process_times = []
        detection_failures = 0
        
        while time.time() - start_time < self.duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.perf_counter()
            
            # Process frame
            landmarks = self.pose.detect(frame)
            
            if landmarks:
                transform = compute_transform(landmarks, self.meta)
                
                if use_tps:
                    # Compute TPS destination points
                    l_shoulder = landmarks.get('left_shoulder', (200, 200))
                    r_shoulder = landmarks.get('right_shoulder', (400, 200))
                    l_hip = landmarks.get('left_hip', (225, 450))
                    r_hip = landmarks.get('right_hip', (375, 450))
                    
                    # Estimate waist
                    l_waist_x = l_shoulder[0] * 0.4 + l_hip[0] * 0.6
                    l_waist_y = l_shoulder[1] * 0.4 + l_hip[1] * 0.6
                    r_waist_x = r_shoulder[0] * 0.4 + r_hip[0] * 0.6
                    r_waist_y = r_shoulder[1] * 0.4 + r_hip[1] * 0.6
                    
                    dst_pts = np.array([
                        l_shoulder,
                        r_shoulder,
                        [l_waist_x, l_waist_y],
                        [r_waist_x, r_waist_y],
                        l_hip,
                        r_hip
                    ], dtype=np.float32)
                    
                    self.tps.fit(dst_pts)
                    warped = self.tps.warp_rgba(self.garment)
                    output = render_overlay(frame, warped, transform)
                else:
                    output = render_overlay(frame, self.garment, transform)
                
                if use_occlusion:
                    mask = self.segmenter.mask(frame, classes=['face', 'skin'])
                    if mask is not None:
                        output = compose_with_occlusion(
                            frame, self.garment, mask, transform
                        )
            else:
                detection_failures += 1
                output = frame
            
            frame_end = time.perf_counter()
            process_times.append((frame_end - frame_start) * 1000)
            frame_count += 1
            
            # Display
            fps = frame_count / (time.time() - start_time)
            cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Benchmark', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        avg_time = np.mean(process_times) if process_times else 0
        
        results = {
            'config': f"TPS={use_tps}, Occlusion={use_occlusion}",
            'frames': frame_count,
            'duration': elapsed,
            'avg_fps': avg_fps,
            'avg_time_ms': avg_time,
            'min_time_ms': np.min(process_times) if process_times else 0,
            'max_time_ms': np.max(process_times) if process_times else 0,
            'p95_time_ms': np.percentile(process_times, 95) if process_times else 0,
            'detection_failures': detection_failures
        }
        
        return results
    
    def run_all_configs(self):
        """Run benchmarks for all configurations."""
        configs = [
            (False, False),  # Baseline
            (True, False),   # TPS only
            (False, True),   # Occlusion only
            (True, True),    # Both
        ]
        
        all_results = []
        
        print("="*70)
        print("FPS BENCHMARK - ALL CONFIGURATIONS")
        print("="*70)
        
        for use_tps, use_occlusion in configs:
            results = self.benchmark_config(use_tps, use_occlusion)
            if results:
                all_results.append(results)
                time.sleep(1)  # Brief pause between configs
        
        # Print summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        for results in all_results:
            print(f"\n{results['config']}")
            print(f"  Frames:      {results['frames']}")
            print(f"  Duration:    {results['duration']:.1f} s")
            print(f"  Avg FPS:     {results['avg_fps']:.1f}")
            print(f"  Avg Time:    {results['avg_time_ms']:.2f} ms")
            print(f"  Min Time:    {results['min_time_ms']:.2f} ms")
            print(f"  Max Time:    {results['max_time_ms']:.2f} ms")
            print(f"  P95 Time:    {results['p95_time_ms']:.2f} ms")
            print(f"  Failures:    {results['detection_failures']}")
            
            # Check acceptance
            if results['avg_fps'] >= 20:
                print(f"  Status:      ✅ PASS (>= 20 FPS)")
            else:
                print(f"  Status:      ❌ FAIL (< 20 FPS)")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark FPS performance')
    parser.add_argument('--video', type=str, default=0,
                        help='Video source (0 for webcam, or path to file)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Benchmark duration in seconds (default: 10)')
    parser.add_argument('--config', type=str, default='all',
                        choices=['all', 'baseline', 'tps', 'occlusion', 'full'],
                        help='Configuration to test (default: all)')
    
    args = parser.parse_args()
    
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    benchmark = FPSBenchmark(video_source, args.duration)
    
    if args.config == 'all':
        benchmark.run_all_configs()
    else:
        config_map = {
            'baseline': (False, False),
            'tps': (True, False),
            'occlusion': (False, True),
            'full': (True, True)
        }
        use_tps, use_occlusion = config_map[args.config]
        results = benchmark.benchmark_config(use_tps, use_occlusion)
        
        if results:
            print("\n" + "="*70)
            print("BENCHMARK RESULTS")
            print("="*70)
            print(f"\nConfiguration: {results['config']}")
            print(f"  Frames:      {results['frames']}")
            print(f"  Duration:    {results['duration']:.1f} s")
            print(f"  Avg FPS:     {results['avg_fps']:.1f}")
            print(f"  Avg Time:    {results['avg_time_ms']:.2f} ms")
            print("="*70)


if __name__ == '__main__':
    main()
