"""
Tests for TPS Module - Thin-Plate Spline 2D Warping
====================================================

Test suite for TPS warping functionality.

Test Coverage:
- ThinPlateSpline initialization and validation
- Control point fitting and accuracy
- Grid generation and mapping
- RGBA channel preservation
- Edge cases and error handling
- Performance benchmarks

Author: Virtual Try-On Team
Date: 2025-01-19
"""

import time

import cv2
import numpy as np
import pytest

from src.tps import (
    ThinPlateSpline,
    create_control_points_garment,
    create_control_points_square,
)


class TestThinPlateSpline:
    """Test suite for ThinPlateSpline class."""
    
    def test_initialization_valid(self):
        """Test initialization with valid control points."""
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        tps = ThinPlateSpline(src_pts)
        
        assert tps.n_pts == 4
        assert tps.src_pts.shape == (4, 2)
        assert tps.L_inv is not None
        assert tps.params_x is None
        assert tps.params_y is None
    
    def test_initialization_minimum_points(self):
        """Test initialization with minimum 3 points."""
        src_pts = np.array([[0, 0], [100, 0], [50, 100]])
        tps = ThinPlateSpline(src_pts)
        
        assert tps.n_pts == 3
    
    def test_initialization_many_points(self):
        """Test initialization with many control points."""
        src_pts = np.random.rand(10, 2) * 256
        tps = ThinPlateSpline(src_pts)
        
        assert tps.n_pts == 10
        assert tps.L_inv.shape == (13, 13)  # 10 + 3
    
    def test_initialization_invalid_shape(self):
        """Test initialization with invalid shape."""
        with pytest.raises(ValueError, match="must have shape"):
            ThinPlateSpline(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError, match="must have shape"):
            ThinPlateSpline(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_initialization_too_few_points(self):
        """Test initialization with too few points."""
        with pytest.raises(ValueError, match="at least 3"):
            ThinPlateSpline(np.array([[0, 0], [1, 1]]))
    
    def test_fit_valid(self):
        """Test fitting with valid destination points."""
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        dst_pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        assert tps.params_x is not None
        assert tps.params_y is not None
        assert tps.params_x.shape == (7,)  # 4 + 3
        assert tps.params_y.shape == (7,)
    
    def test_fit_invalid_shape(self):
        """Test fitting with wrong number of points."""
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        dst_pts = np.array([[10, 10], [90, 10], [90, 90]])  # Only 3 points
        
        tps = ThinPlateSpline(src_pts)
        
        with pytest.raises(ValueError, match="must have shape"):
            tps.fit(dst_pts)
    
    def test_control_point_accuracy(self):
        """Test that control points map exactly to destinations."""
        size = 256
        src_pts = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float64)
        
        dst_pts = np.array([
            [50, 20],
            [200, 25],
            [210, 230],
            [40, 235]
        ], dtype=np.float64)
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        # Generate grid
        map_x, map_y = tps.sample_grid(size, size)
        
        # Check each control point
        errors = []
        for i, (sx, sy) in enumerate(src_pts):
            tx = map_x[int(sy), int(sx)]
            ty = map_y[int(sy), int(sx)]
            
            dx, dy = dst_pts[i]
            error = np.sqrt((tx - dx)**2 + (ty - dy)**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # Control points should map very accurately
        assert mean_error < 1.5, f"Mean error {mean_error:.3f}px >= 1.5px"
        assert max_error < 3.0, f"Max error {max_error:.3f}px >= 3.0px"
    
    def test_square_to_trapezoid(self):
        """Test square to trapezoid transformation."""
        size = 256
        
        # Square corners
        src_pts = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float64)
        
        # Trapezoid (narrower at top)
        dst_pts = np.array([
            [50, 20],
            [size - 50, 20],
            [size - 10, size - 10],
            [10, size - 10]
        ], dtype=np.float64)
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        map_x, map_y = tps.sample_grid(size, size)
        
        # Verify corners
        errors = []
        for i, (sx, sy) in enumerate(src_pts):
            tx = map_x[int(sy), int(sx)]
            ty = map_y[int(sy), int(sx)]
            
            dx, dy = dst_pts[i]
            error = np.sqrt((tx - dx)**2 + (ty - dy)**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        assert mean_error < 1.5
    
    def test_sample_grid_before_fit(self):
        """Test that sample_grid raises error before fit."""
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        tps = ThinPlateSpline(src_pts)
        
        with pytest.raises(RuntimeError, match="Must call fit"):
            tps.sample_grid(256, 256)
    
    def test_sample_grid_output_shape(self):
        """Test that sample_grid returns correct shapes."""
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        dst_pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        # Test various output sizes
        for w, h in [(128, 128), (256, 256), (320, 240), (640, 480)]:
            map_x, map_y = tps.sample_grid(w, h)
            
            assert map_x.shape == (h, w)
            assert map_y.shape == (h, w)
            assert map_x.dtype == np.float32
            assert map_y.dtype == np.float32
    
    def test_warp_rgba_shape_preservation(self):
        """Test that warp_rgba preserves shape and channels."""
        size = 256
        
        # Create RGBA image
        img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        img_rgba[:, :, 0] = 255  # Red
        img_rgba[:, :, 1] = 128  # Green
        img_rgba[:, :, 2] = 64   # Blue
        img_rgba[:, :, 3] = 200  # Alpha
        
        src_pts = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        dst_pts = np.array([[10, 10], [size-10, 10], [size-10, size-10], [10, size-10]])
        
        tps = ThinPlateSpline(src_pts)
        warped = tps.warp_rgba(img_rgba, dst_pts)
        
        # Check shape and dtype
        assert warped.shape == img_rgba.shape
        assert warped.dtype == img_rgba.dtype
    
    def test_warp_rgba_channel_preservation(self):
        """Test that all RGBA channels are warped."""
        size = 128
        
        # Create RGBA image with distinct channels
        img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        img_rgba[30:70, 30:70, 0] = 255  # Red square
        img_rgba[30:70, 30:70, 1] = 128  # Green square
        img_rgba[30:70, 30:70, 2] = 64   # Blue square
        img_rgba[30:70, 30:70, 3] = 200  # Alpha square
        
        src_pts = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        dst_pts = src_pts + np.array([10, 5])  # Slight shift
        
        tps = ThinPlateSpline(src_pts)
        warped = tps.warp_rgba(img_rgba, dst_pts)
        
        # Check that all channels have non-zero values
        assert np.any(warped[:, :, 0] > 0), "Red channel lost"
        assert np.any(warped[:, :, 1] > 0), "Green channel lost"
        assert np.any(warped[:, :, 2] > 0), "Blue channel lost"
        assert np.any(warped[:, :, 3] > 0), "Alpha channel lost"
    
    def test_warp_rgba_invalid_input(self):
        """Test warp_rgba with invalid input."""
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        tps = ThinPlateSpline(src_pts)
        
        # RGB image (missing alpha)
        img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        dst_pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        
        with pytest.raises(ValueError, match="must have shape"):
            tps.warp_rgba(img_rgb, dst_pts)
    
    def test_interpolation_methods(self):
        """Test different interpolation methods."""
        size = 128
        img_rgba = np.random.randint(0, 256, (size, size, 4), dtype=np.uint8)
        
        src_pts = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        dst_pts = np.array([[10, 10], [size-15, 5], [size-5, size-10], [5, size-5]])
        
        tps = ThinPlateSpline(src_pts)
        
        # Test different interpolation modes
        for interp in [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]:
            warped = tps.warp_rgba(img_rgba, dst_pts, interpolation=interp)
            assert warped.shape == img_rgba.shape
    
    def test_identity_transformation(self):
        """Test that identity transformation (src == dst) works."""
        size = 128
        
        # Create image with pattern
        img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        cv2.circle(img_rgba, (size//2, size//2), 30, (255, 255, 255, 255), -1)
        
        # Identity: src == dst
        pts = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        
        tps = ThinPlateSpline(pts)
        warped = tps.warp_rgba(img_rgba, pts)
        
        # Result should be very similar to input
        diff = np.abs(warped.astype(float) - img_rgba.astype(float))
        mean_diff = np.mean(diff)
        
        # Allow small numerical errors
        assert mean_diff < 5.0, f"Identity transform changed image: {mean_diff:.2f}"
    
    def test_six_point_configuration(self):
        """Test with 6 control points (typical for garments)."""
        size = 256
        
        # 6 points: 4 corners + 2 side midpoints
        src_pts = np.array([
            [0, 0],                    # Top-left
            [size-1, 0],               # Top-right
            [size-1, size-1],          # Bottom-right
            [0, size-1],               # Bottom-left
            [0, size//2],              # Left middle
            [size-1, size//2]          # Right middle
        ], dtype=np.float64)
        
        # Deformed shape
        dst_pts = np.array([
            [30, 20],
            [size-30, 15],
            [size-20, size-10],
            [20, size-15],
            [10, size//2],
            [size-10, size//2 + 10]
        ], dtype=np.float64)
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        # Verify accuracy
        map_x, map_y = tps.sample_grid(size, size)
        
        errors = []
        for i, (sx, sy) in enumerate(src_pts):
            tx = map_x[int(sy), int(sx)]
            ty = map_y[int(sy), int(sx)]
            
            dx, dy = dst_pts[i]
            error = np.sqrt((tx - dx)**2 + (ty - dy)**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        assert mean_error < 1.5
    
    def test_eight_point_configuration(self):
        """Test with 8 control points (maximum typical)."""
        size = 256
        
        # 8 points: 4 corners + 4 edge midpoints
        src_pts = np.array([
            [0, 0],                    # Top-left
            [size-1, 0],               # Top-right
            [size-1, size-1],          # Bottom-right
            [0, size-1],               # Bottom-left
            [size//2, 0],              # Top middle
            [size-1, size//2],         # Right middle
            [size//2, size-1],         # Bottom middle
            [0, size//2]               # Left middle
        ], dtype=np.float64)
        
        # Add random deformation
        dst_pts = src_pts + np.random.randn(8, 2) * 15
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        # Should handle 8 points without issues
        map_x, map_y = tps.sample_grid(size, size)
        
        assert map_x.shape == (size, size)
        assert map_y.shape == (size, size)


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_control_points_square(self):
        """Test square control point generation."""
        pts = create_control_points_square(256)
        
        assert pts.shape == (4, 2)
        assert np.allclose(pts[0], [0, 0])
        assert np.allclose(pts[1], [255, 0])
        assert np.allclose(pts[2], [255, 255])
        assert np.allclose(pts[3], [0, 255])
    
    def test_create_control_points_square_different_sizes(self):
        """Test square points with different sizes."""
        for size in [128, 256, 512]:
            pts = create_control_points_square(size)
            assert pts.shape == (4, 2)
            assert pts[1, 0] == size - 1  # Top-right x
            assert pts[2, 1] == size - 1  # Bottom-right y
    
    def test_create_control_points_garment(self):
        """Test garment control point generation."""
        pts = create_control_points_garment(256, 384)
        
        # Should have shoulder + waist + side points
        # Default: 2 shoulder + 2 waist + 2 side = 6 points
        assert pts.shape[0] == 6
        assert pts.shape[1] == 2
        
        # Check bounds
        assert np.all(pts[:, 0] >= 0)
        assert np.all(pts[:, 0] < 256)
        assert np.all(pts[:, 1] >= 0)
        assert np.all(pts[:, 1] < 384)
    
    def test_create_control_points_garment_custom(self):
        """Test garment points with custom configuration."""
        pts = create_control_points_garment(256, 384, n_shoulder_pts=3, n_waist_pts=3)
        
        # 3 shoulder + 3 waist + 2 side = 8 points
        assert pts.shape[0] == 8


class TestPerformance:
    """Performance benchmarks for TPS operations."""
    
    def test_initialization_performance(self):
        """Test initialization time with various point counts."""
        times = {}
        
        for n_pts in [4, 6, 8, 10]:
            src_pts = np.random.rand(n_pts, 2) * 256
            
            start = time.perf_counter()
            tps = ThinPlateSpline(src_pts)
            elapsed = (time.perf_counter() - start) * 1000
            
            times[n_pts] = elapsed
            print(f"\nInitialization with {n_pts} points: {elapsed:.2f}ms")
        
        # Should be fast (<5ms for typical counts)
        assert times[6] < 5.0
        assert times[8] < 10.0
    
    def test_fitting_performance(self):
        """Test fitting time."""
        src_pts = np.random.rand(8, 2) * 256
        dst_pts = src_pts + np.random.randn(8, 2) * 10
        
        tps = ThinPlateSpline(src_pts)
        
        start = time.perf_counter()
        for _ in range(100):
            tps.fit(dst_pts)
        elapsed = (time.perf_counter() - start) * 1000 / 100
        
        print(f"\nFitting time: {elapsed:.3f}ms")
        
        # Should be very fast (< 1ms)
        assert elapsed < 1.0
    
    def test_grid_generation_performance(self):
        """Test grid generation time for various sizes."""
        src_pts = np.random.rand(8, 2) * 256
        dst_pts = src_pts + np.random.randn(8, 2) * 10
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        for size in [128, 256, 512]:
            start = time.perf_counter()
            for _ in range(10):
                map_x, map_y = tps.sample_grid(size, size)
            elapsed = (time.perf_counter() - start) * 1000 / 10
            
            print(f"\nGrid generation {size}×{size}: {elapsed:.2f}ms")
            
            # Should be reasonable
            if size == 256:
                assert elapsed < 50.0  # Target for real-time
    
    def test_warp_performance(self):
        """Test full warp time (fit + grid + remap)."""
        size = 256
        img_rgba = np.random.randint(0, 256, (size, size, 4), dtype=np.uint8)
        
        src_pts = np.random.rand(8, 2) * size
        dst_pts = src_pts + np.random.randn(8, 2) * 10
        
        tps = ThinPlateSpline(src_pts)
        
        start = time.perf_counter()
        n_warps = 30
        for _ in range(n_warps):
            warped = tps.warp_rgba(img_rgba, dst_pts)
        elapsed = (time.perf_counter() - start) * 1000 / n_warps
        
        fps = 1000 / elapsed if elapsed > 0 else 0
        
        print(f"\nFull warp {size}×{size}: {elapsed:.2f}ms ({fps:.1f} FPS)")
        
        # Target: >30 FPS (< 33ms per frame)
        assert elapsed < 50.0, f"Warp too slow: {elapsed:.1f}ms"
    
    def test_real_time_simulation(self):
        """Simulate real-time video processing."""
        size = 256
        img_rgba = np.random.randint(0, 256, (size, size, 4), dtype=np.uint8)
        
        # Simulate pose tracking: changing destination points
        src_pts = np.array([
            [0, 0], [size-1, 0], [size-1, size-1], [0, size-1],
            [size//2, 0], [size-1, size//2]
        ], dtype=np.float64)
        
        tps = ThinPlateSpline(src_pts)
        
        # Process 30 frames (1 second at 30 FPS)
        n_frames = 30
        start = time.perf_counter()
        
        for i in range(n_frames):
            # Simulate changing pose
            offset = np.sin(i / 5) * 10
            dst_pts = src_pts + np.array([offset, offset * 0.5])
            
            # Warp
            warped = tps.warp_rgba(img_rgba, dst_pts)
        
        total_time = time.perf_counter() - start
        avg_fps = n_frames / total_time
        avg_ms = total_time * 1000 / n_frames
        
        print(f"\nReal-time simulation ({n_frames} frames):")
        print(f"  Total time: {total_time*1000:.1f}ms")
        print(f"  Avg per frame: {avg_ms:.2f}ms")
        print(f"  Avg FPS: {avg_fps:.1f}")
        
        # Should maintain >25 FPS for real-time
        assert avg_fps > 25, f"Too slow for real-time: {avg_fps:.1f} FPS"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_colinear_points(self):
        """Test with colinear control points."""
        # All points on a line
        src_pts = np.array([[0, 0], [100, 0], [200, 0]], dtype=np.float64)
        
        # Should still initialize (though not ideal)
        tps = ThinPlateSpline(src_pts)
        assert tps.n_pts == 3
    
    def test_very_close_points(self):
        """Test with very close control points."""
        src_pts = np.array([
            [0, 0],
            [0.01, 0],
            [0, 0.01],
            [100, 100]
        ], dtype=np.float64)
        
        # Should handle with eps regularization
        tps = ThinPlateSpline(src_pts)
        assert tps.L_inv is not None
    
    def test_large_deformation(self):
        """Test with very large deformation."""
        size = 256
        
        src_pts = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        
        # Extreme deformation
        dst_pts = np.array([
            [size*0.3, size*0.3],
            [size*0.7, size*0.2],
            [size*0.8, size*0.8],
            [size*0.2, size*0.9]
        ])
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        # Should still work
        map_x, map_y = tps.sample_grid(size, size)
        assert map_x.shape == (size, size)
    
    def test_small_image(self):
        """Test with very small image."""
        size = 32
        img_rgba = np.random.randint(0, 256, (size, size, 4), dtype=np.uint8)
        
        src_pts = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        dst_pts = src_pts + 2
        
        tps = ThinPlateSpline(src_pts)
        warped = tps.warp_rgba(img_rgba, dst_pts)
        
        assert warped.shape == img_rgba.shape


class TestPromptRequirements:
    """Tests from 10_TESTS_PROMPT.txt"""
    
    def test_square_to_trapezoid_6_points(self):
        """Warp square to trapezoid with 6 points → mean error < 1.5 px @ 256×256."""
        size = 256
        
        # 6 control points on square
        src_pts = np.array([
            [0, 0],                # Top-left
            [size-1, 0],           # Top-right
            [0, size//2],          # Left middle
            [size-1, size//2],     # Right middle
            [0, size-1],           # Bottom-left
            [size-1, size-1]       # Bottom-right
        ], dtype=np.float64)
        
        # Trapezoid (narrower at top)
        dst_pts = np.array([
            [50, 0],
            [size-50, 0],
            [20, size//2],
            [size-20, size//2],
            [0, size-1],
            [size-1, size-1]
        ], dtype=np.float64)
        
        tps = ThinPlateSpline(src_pts)
        tps.fit(dst_pts)
        
        # Verify control points accuracy
        map_x, map_y = tps.sample_grid(size, size)
        
        errors = []
        for i, (sx, sy) in enumerate(src_pts):
            tx = map_x[int(sy), int(sx)]
            ty = map_y[int(sy), int(sx)]
            dx, dy = dst_pts[i]
            error = np.sqrt((tx - dx)**2 + (ty - dy)**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        assert mean_error < 1.5, f"Mean error {mean_error:.3f}px >= 1.5px"
    
    def test_alpha_preserved_after_warp(self):
        """Alpha channel preserved after warping."""
        size = 256
        
        # Create RGBA image with alpha gradient
        img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        img_rgba[:, :, 0:3] = 255  # White RGB
        img_rgba[:, :, 3] = np.linspace(0, 255, size).reshape(1, -1).astype(np.uint8)  # Alpha gradient
        
        src_pts = np.array([
            [0, 0], [size-1, 0], [0, size-1], [size-1, size-1]
        ], dtype=np.float64)
        
        dst_pts = np.array([
            [10, 10], [size-20, 5], [5, size-15], [size-10, size-10]
        ], dtype=np.float64)
        
        tps = ThinPlateSpline(src_pts)
        warped = tps.warp_rgba(img_rgba, dst_pts)
        
        # Verify alpha channel exists and has content
        assert warped.shape[2] == 4, "Alpha channel missing"
        assert np.any(warped[:, :, 3] > 0), "Alpha channel is all zeros"
        assert np.any(warped[:, :, 3] < 255), "Alpha channel has no variation"


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v', '--tb=short', '-s'])
