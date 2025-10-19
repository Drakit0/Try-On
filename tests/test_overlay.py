"""
Unit Tests for Overlay Module
==============================

Tests for garment composition, transform computation, and alpha blending.

Author: Virtual Try-On Team
Date: 2025-01-19
"""

import math

import cv2
import numpy as np
import pytest

from src.overlay import (
    bgr_to_rgba,
    compose_with_alpha,
    compose_with_occlusion,
    compute_transform,
    draw_transform_visualization,
    render_overlay,
    rgba_to_bgr,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_landmarks():
    """Sample pose landmarks for testing."""
    return {
        'left_shoulder': (280, 300),
        'right_shoulder': (360, 300),
        'left_hip': (290, 450),
        'right_hip': (350, 450)
    }


@pytest.fixture
def sample_garment_meta():
    """Sample garment metadata."""
    return {
        'base_shoulder_px': 200,
        'base_torso_px': 300,
        'y_offset_to_waist_px': 50
    }


@pytest.fixture
def sample_frame():
    """Sample BGR frame."""
    frame = np.full((480, 640, 3), [100, 150, 200], dtype=np.uint8)
    return frame


@pytest.fixture
def sample_garment():
    """Sample RGBA garment with simple pattern."""
    garment = np.zeros((400, 300, 4), dtype=np.uint8)
    
    # Draw a red rectangle with alpha
    garment[100:300, 50:250] = [0, 0, 255, 255]  # Red in RGB
    
    # Add a semi-transparent circle
    cv2.circle(garment, (150, 200), 50, (0, 255, 0, 200), -1)  # Green
    
    return garment


# ============================================================================
# Test compute_transform
# ============================================================================

def test_compute_transform_center(sample_landmarks, sample_garment_meta):
    """Test center computation (shoulder midpoint)."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Center should be midpoint of shoulders
    expected_x = (280 + 360) / 2
    expected_y = (300 + 300) / 2
    
    assert abs(transform['center'][0] - expected_x) < 0.1
    assert abs(transform['center'][1] - expected_y) < 0.1


def test_compute_transform_angle_horizontal(sample_landmarks, sample_garment_meta):
    """Test angle computation with horizontal shoulders."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Shoulders at same y = 0 degrees
    assert abs(transform['angle_deg']) < 0.1


def test_compute_transform_angle_tilted(sample_garment_meta):
    """Test angle computation with tilted shoulders."""
    landmarks = {
        'left_shoulder': (280, 300),
        'right_shoulder': (360, 320),  # 20px down
        'left_hip': (290, 450),
        'right_hip': (350, 470)
    }
    
    transform = compute_transform(landmarks, sample_garment_meta)
    
    # Angle should be positive (right shoulder lower)
    # tan(angle) = dy/dx = 20/80 ≈ 0.25 → angle ≈ 14°
    expected_angle = math.degrees(math.atan2(20, 80))
    
    assert abs(transform['angle_deg'] - expected_angle) < 1.0


def test_compute_transform_scale_x(sample_landmarks, sample_garment_meta):
    """Test scale_x computation."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Shoulder distance = 80px
    # Base shoulder = 200px
    # Scale = 80/200 = 0.4
    expected_scale_x = 80.0 / 200.0
    
    assert abs(transform['scale_x'] - expected_scale_x) < 0.01


def test_compute_transform_scale_y(sample_landmarks, sample_garment_meta):
    """Test scale_y computation."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Torso height = 150px (shoulder_y=300 to hip_y=450)
    # Base torso = 300px
    # Scale = 150/300 = 0.5
    expected_scale_y = 150.0 / 300.0
    
    assert abs(transform['scale_y'] - expected_scale_y) < 0.01


def test_compute_transform_scale_average(sample_landmarks, sample_garment_meta):
    """Test average scale computation."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    expected_scale = (transform['scale_x'] + transform['scale_y']) / 2.0
    
    assert abs(transform['scale'] - expected_scale) < 0.01


def test_compute_transform_waist_offset(sample_landmarks, sample_garment_meta):
    """Test waist offset extraction."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    assert transform['waist_y_offset'] == 50


def test_compute_transform_missing_landmarks(sample_garment_meta):
    """Test with missing landmarks (defaults to origin)."""
    landmarks = {
        'left_shoulder': (100, 200)
        # Missing right_shoulder, hips
    }
    
    transform = compute_transform(landmarks, sample_garment_meta)
    
    # Should not crash, compute with defaults
    assert 'center' in transform
    assert 'angle_deg' in transform
    assert 'scale_x' in transform


def test_compute_transform_zero_base(sample_landmarks):
    """Test with zero base dimensions (edge case)."""
    garment_meta = {
        'base_shoulder_px': 0,
        'base_torso_px': 0
    }
    
    transform = compute_transform(sample_landmarks, garment_meta)
    
    # Should default to 1.0 scale
    assert transform['scale_x'] == 1.0
    assert transform['scale_y'] == 1.0


# ============================================================================
# Test render_overlay
# ============================================================================

def test_render_overlay_basic(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test basic overlay rendering."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    result = render_overlay(sample_frame, sample_garment, transform)
    
    # Check output shape
    assert result.shape == sample_frame.shape
    assert result.dtype == np.uint8
    
    # Result should differ from frame (garment added)
    assert not np.array_equal(result, sample_frame)


def test_render_overlay_y_offset(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test y_offset parameter."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    result1 = render_overlay(sample_frame, sample_garment, transform, y_offset_px=0)
    result2 = render_overlay(sample_frame, sample_garment, transform, y_offset_px=50)
    
    # Results should differ due to offset
    assert not np.array_equal(result1, result2)


def test_render_overlay_scale_bias(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test scale_bias parameter."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    result1 = render_overlay(sample_frame, sample_garment, transform, scale_bias=1.0)
    result2 = render_overlay(sample_frame, sample_garment, transform, scale_bias=1.5)
    
    # Results should differ due to scale
    assert not np.array_equal(result1, result2)


def test_render_overlay_uniform_scale(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test uniform vs non-uniform scaling."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    result1 = render_overlay(sample_frame, sample_garment, transform, use_uniform_scale=True)
    result2 = render_overlay(sample_frame, sample_garment, transform, use_uniform_scale=False)
    
    # Results may differ if scale_x != scale_y
    if abs(transform['scale_x'] - transform['scale_y']) > 0.01:
        assert not np.array_equal(result1, result2)


def test_render_overlay_rotation(sample_frame, sample_garment, sample_garment_meta):
    """Test rotation handling."""
    # Create landmarks with 45-degree tilt
    landmarks = {
        'left_shoulder': (300, 240),
        'right_shoulder': (340, 280),
        'left_hip': (310, 400),
        'right_hip': (350, 440)
    }
    
    transform = compute_transform(landmarks, sample_garment_meta)
    
    # Angle should be ~45 degrees
    assert abs(transform['angle_deg'] - 45.0) < 5.0
    
    result = render_overlay(sample_frame, sample_garment, transform)
    
    # Should render without error
    assert result.shape == sample_frame.shape


def test_render_overlay_invalid_frame():
    """Test with invalid frame shape."""
    frame = np.zeros((480, 640, 4), dtype=np.uint8)  # Wrong channels
    garment = np.zeros((100, 100, 4), dtype=np.uint8)
    transform = {'center': (320, 240), 'angle_deg': 0, 'scale': 1.0, 'scale_x': 1.0, 'scale_y': 1.0}
    
    with pytest.raises(ValueError):
        render_overlay(frame, garment, transform)


def test_render_overlay_invalid_garment():
    """Test with invalid garment shape."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    garment = np.zeros((100, 100, 3), dtype=np.uint8)  # Missing alpha
    transform = {'center': (320, 240), 'angle_deg': 0, 'scale': 1.0, 'scale_x': 1.0, 'scale_y': 1.0}
    
    with pytest.raises(ValueError):
        render_overlay(frame, garment, transform)


def test_render_overlay_preserves_background():
    """Test that areas without garment preserve background."""
    frame = np.full((480, 640, 3), [100, 150, 200], dtype=np.uint8)
    
    # Small garment with full alpha
    garment = np.zeros((50, 50, 4), dtype=np.uint8)
    garment[:, :, :3] = [255, 0, 0]  # Red
    garment[:, :, 3] = 255  # Opaque
    
    transform = {'center': (100, 100), 'angle_deg': 0, 'scale': 1.0, 'scale_x': 1.0, 'scale_y': 1.0}
    
    result = render_overlay(frame, garment, transform)
    
    # Check far corner preserves background
    assert np.array_equal(result[400, 500], [100, 150, 200])


def test_render_overlay_alpha_transparency():
    """Test alpha transparency handling."""
    frame = np.full((480, 640, 3), [100, 150, 200], dtype=np.uint8)
    
    # Garment with 50% alpha
    garment = np.zeros((200, 200, 4), dtype=np.uint8)
    garment[:, :, :3] = [255, 0, 0]  # Red
    garment[:, :, 3] = 128  # 50% transparent
    
    transform = {'center': (320, 240), 'angle_deg': 0, 'scale': 1.0, 'scale_x': 1.0, 'scale_y': 1.0}
    
    result = render_overlay(frame, garment, transform)
    
    # Check center pixel is blend of red and background
    center_color = result[240, 320]
    
    # Should be between red [0, 0, 255] and background [100, 150, 200]
    assert 0 < center_color[0] < 255
    assert 0 < center_color[1] < 255
    assert 0 < center_color[2] < 255


# ============================================================================
# Test compose_with_alpha
# ============================================================================

def test_compose_with_alpha_builtin():
    """Test composition with built-in alpha channel."""
    bg = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
    fg = np.zeros((100, 100, 4), dtype=np.uint8)
    fg[:, :, :3] = [255, 0, 0]  # BGR: Blue=255, Green=0, Red=0
    fg[:, :, 3] = 255  # Opaque
    
    result = compose_with_alpha(bg, fg)
    
    # Result should be blue (channel 0 in BGR)
    assert np.allclose(result[:, :, 0], 255)  # Blue channel in BGR


def test_compose_with_alpha_separate_mask():
    """Test composition with separate alpha mask."""
    bg = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
    fg = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # BGR: Blue=255
    alpha = np.full((100, 100), 255, dtype=np.uint8)
    
    result = compose_with_alpha(bg, fg, alpha)
    
    # Result should be fg (blue channel = 255)
    assert np.allclose(result[:, :, 0], 255)


def test_compose_with_alpha_partial_transparency():
    """Test partial alpha blending."""
    bg = np.full((100, 100, 3), [0, 0, 0], dtype=np.uint8)
    fg = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)  # BGR: Red=255
    alpha = np.full((100, 100), 128, dtype=np.uint8)  # 50%
    
    result = compose_with_alpha(bg, fg, alpha)
    
    # Result should be approximately half red (channel 2 in BGR)
    assert 120 < result[0, 0, 2] < 135


def test_compose_with_alpha_zero_alpha():
    """Test with zero alpha (fully transparent)."""
    bg = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
    fg = np.full((100, 100, 4), [255, 0, 0, 0], dtype=np.uint8)  # Zero alpha
    
    result = compose_with_alpha(bg, fg)
    
    # Result should be background
    assert np.array_equal(result, bg)


def test_compose_with_alpha_no_alpha():
    """Test with no alpha channel (fully opaque)."""
    bg = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
    fg = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
    
    result = compose_with_alpha(bg, fg)
    
    # Result should be foreground
    assert np.array_equal(result, fg)


def test_compose_with_alpha_size_mismatch():
    """Test with mismatched sizes."""
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    fg = np.zeros((200, 200, 4), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        compose_with_alpha(bg, fg)


# ============================================================================
# Test compose_with_occlusion
# ============================================================================

def test_compose_with_occlusion_basic(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test basic occlusion composition."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Create occlusion mask (face region)
    occlusion = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(occlusion, (320, 200), 50, 255, -1)
    
    result = compose_with_occlusion(
        sample_frame, sample_garment, occlusion, transform
    )
    
    assert result.shape == sample_frame.shape
    assert result.dtype == np.uint8


def test_compose_with_occlusion_mask_effect(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test that occlusion mask brings original pixels forward."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Without occlusion
    result1 = render_overlay(sample_frame, sample_garment, transform)
    
    # With full occlusion
    occlusion = np.full((480, 640), 255, dtype=np.uint8)
    result2 = compose_with_occlusion(
        sample_frame, sample_garment, occlusion, transform
    )
    
    # With full occlusion, result should match original frame
    assert np.array_equal(result2, sample_frame)


def test_compose_with_occlusion_partial_mask(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test partial occlusion mask."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Half-screen occlusion
    occlusion = np.zeros((480, 640), dtype=np.uint8)
    occlusion[:, :320] = 255
    
    result = compose_with_occlusion(
        sample_frame, sample_garment, occlusion, transform
    )
    
    # Left half should match frame, right half should have garment
    left_matches = np.array_equal(result[:, 100], sample_frame[:, 100])
    assert left_matches


# ============================================================================
# Test Color Conversion
# ============================================================================

def test_bgr_to_rgba_shape():
    """Test BGR to RGBA conversion shape."""
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    rgba = bgr_to_rgba(bgr)
    
    assert rgba.shape == (100, 100, 4)
    assert rgba.dtype == np.uint8


def test_bgr_to_rgba_alpha_value():
    """Test BGR to RGBA alpha value."""
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    rgba = bgr_to_rgba(bgr, alpha=200)
    
    assert np.all(rgba[:, :, 3] == 200)


def test_bgr_to_rgba_color_preservation():
    """Test color preservation in BGR to RGBA."""
    bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Blue in BGR
    rgba = bgr_to_rgba(bgr)
    
    # Should be red in RGBA (RGB order)
    assert rgba[0, 0, 0] == 0  # R
    assert rgba[0, 0, 1] == 0  # G
    assert rgba[0, 0, 2] == 255  # B (was B in BGR)


def test_rgba_to_bgr_shape():
    """Test RGBA to BGR conversion shape."""
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    bgr = rgba_to_bgr(rgba)
    
    assert bgr.shape == (100, 100, 3)
    assert bgr.dtype == np.uint8


def test_rgba_to_bgr_discards_alpha():
    """Test that alpha is discarded."""
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    
    bgr = rgba_to_bgr(rgba)
    
    # Should only have 3 channels
    assert bgr.shape[2] == 3


def test_bgr_rgba_roundtrip():
    """Test round-trip conversion."""
    original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    rgba = bgr_to_rgba(original)
    back = rgba_to_bgr(rgba)
    
    assert np.array_equal(original, back)


def test_bgr_to_rgba_invalid_channels():
    """Test with invalid channel count."""
    invalid = np.zeros((100, 100, 4), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        bgr_to_rgba(invalid)


def test_rgba_to_bgr_invalid_channels():
    """Test with invalid channel count."""
    invalid = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        rgba_to_bgr(invalid)


# ============================================================================
# Test Visualization
# ============================================================================

def test_draw_transform_visualization_shape(sample_frame, sample_landmarks, sample_garment_meta):
    """Test visualization output shape."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    vis = draw_transform_visualization(sample_frame, transform)
    
    assert vis.shape == sample_frame.shape
    assert vis.dtype == np.uint8


def test_draw_transform_visualization_with_landmarks(sample_frame, sample_landmarks, sample_garment_meta):
    """Test visualization with landmarks."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    vis = draw_transform_visualization(sample_frame, transform, sample_landmarks)
    
    # Should have drawings (differs from frame)
    assert not np.array_equal(vis, sample_frame)


def test_draw_transform_visualization_without_landmarks(sample_frame, sample_landmarks, sample_garment_meta):
    """Test visualization without landmarks."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    vis = draw_transform_visualization(sample_frame, transform)
    
    # Should still render
    assert not np.array_equal(vis, sample_frame)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test complete overlay pipeline."""
    # 1. Compute transform
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # 2. Render overlay
    result = render_overlay(sample_frame, sample_garment, transform)
    
    # 3. Add visualization
    vis = draw_transform_visualization(result, transform, sample_landmarks)
    
    assert vis.shape == sample_frame.shape
    assert vis.dtype == np.uint8


def test_multiple_scale_biases(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test rendering with multiple scale biases."""
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    results = []
    for scale_bias in [0.8, 1.0, 1.2]:
        result = render_overlay(sample_frame, sample_garment, transform, scale_bias=scale_bias)
        results.append(result)
    
    # All should have same shape but different content
    assert all(r.shape == sample_frame.shape for r in results)
    assert not np.array_equal(results[0], results[1])
    assert not np.array_equal(results[1], results[2])


def test_performance_render_overlay(sample_frame, sample_garment, sample_landmarks, sample_garment_meta):
    """Test render_overlay performance."""
    import time
    
    transform = compute_transform(sample_landmarks, sample_garment_meta)
    
    # Warmup
    for _ in range(5):
        render_overlay(sample_frame, sample_garment, transform)
    
    # Measure
    n_iterations = 30
    start = time.perf_counter()
    for _ in range(n_iterations):
        render_overlay(sample_frame, sample_garment, transform)
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / n_iterations
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\nPerformance: {avg_time*1000:.2f}ms per frame, {fps:.1f} FPS")
    
    # Should be fast enough for real-time (< 50ms for 20 FPS target)
    # Allow variance for different systems and optimizations
    assert avg_time < 0.050, f"Too slow: {avg_time*1000:.2f}ms > 50ms"


class TestPromptRequirements:
    """Tests from 10_TESTS_PROMPT.txt"""
    
    def test_compute_transform_30_degree_angle(self):
        """synthetic landmarks with shoulders at 30° → angle within [29°, 31°]."""
        # Create landmarks with 30° shoulder angle
        # tan(30°) ≈ 0.577, so for dx=100, dy = 100 * tan(30°) ≈ 57.7
        import math
        
        angle_deg = 30.0
        shoulder_distance = 100.0
        dy = shoulder_distance * math.tan(math.radians(angle_deg))
        
        landmarks = {
            'left_shoulder': (250.0, 300.0),
            'right_shoulder': (250.0 + shoulder_distance, 300.0 + dy),
            'left_hip': (260.0, 450.0),
            'right_hip': (340.0, 450.0)
        }
        
        garment_meta = {
            'base_shoulder_px': 200,
            'base_torso_px': 300
        }
        
        transform = compute_transform(landmarks, garment_meta)
        
        # Verify angle is within [29°, 31°]
        assert 29.0 <= transform['angle_deg'] <= 31.0, \
            f"Angle {transform['angle_deg']:.2f}° not in [29°, 31°]"
    
    def test_compute_transform_shoulder_torso_coherence(self):
        """Shoulder/torso distances coherent with base scales."""
        landmarks = {
            'left_shoulder': (280, 300),
            'right_shoulder': (360, 300),  # 80px apart
            'left_hip': (290, 450),
            'right_hip': (350, 450)  # 60px apart, 150px down
        }
        
        garment_meta = {
            'base_shoulder_px': 200,
            'base_torso_px': 300
        }
        
        transform = compute_transform(landmarks, garment_meta)
        
        # Shoulder distance = 80px, base = 200px → scale_x = 0.4
        expected_scale_x = 80.0 / 200.0
        assert abs(transform['scale_x'] - expected_scale_x) < 0.01
        
        # Torso distance = 150px, base = 300px → scale_y = 0.5
        expected_scale_y = 150.0 / 300.0
        assert abs(transform['scale_y'] - expected_scale_y) < 0.01
        
        # Verify coherence: scales should be proportional to actual body measurements
        assert 0.3 < transform['scale_x'] < 0.6
        assert 0.3 < transform['scale_y'] < 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
