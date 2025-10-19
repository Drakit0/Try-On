"""
Tests for TPS Mapper Module
----------------------------
Validates destination point computation from pose landmarks.
"""

import pytest
import numpy as np
from src.tps_mapper import TPSMapper, create_mapper


class TestTPSMapperBasics:
    """Test basic initialization and configuration."""
    
    def test_init_default_params(self):
        """Test mapper initializes with default parameters."""
        mapper = TPSMapper()
        assert mapper.waist_alpha == 0.6
        assert mapper.waist_lateral_factor == 0.08
        assert mapper.hip_widening_factor == 0.05
        assert mapper.enable_pre_transform is False
    
    def test_init_custom_params(self):
        """Test mapper initializes with custom parameters."""
        mapper = TPSMapper(
            waist_alpha=0.7,
            waist_lateral_factor=0.1,
            hip_widening_factor=0.08,
            enable_pre_transform=True
        )
        assert mapper.waist_alpha == 0.7
        assert mapper.waist_lateral_factor == 0.1
        assert mapper.hip_widening_factor == 0.08
        assert mapper.enable_pre_transform is True
    
    def test_create_mapper_factory(self):
        """Test factory function creates mapper correctly."""
        mapper = create_mapper()
        assert isinstance(mapper, TPSMapper)
        assert mapper.waist_alpha == 0.6
    
    def test_create_mapper_with_config(self):
        """Test factory function with custom config."""
        config = {'waist_alpha': 0.5, 'enable_pre_transform': True}
        mapper = create_mapper(config)
        assert mapper.waist_alpha == 0.5
        assert mapper.enable_pre_transform is True


class TestComputeDestinationPoints:
    """Test destination point computation."""
    
    @pytest.fixture
    def sample_landmarks(self):
        """Sample pose landmarks in T-pose configuration."""
        return {
            'left_shoulder': (100.0, 150.0),
            'right_shoulder': (200.0, 150.0),
            'left_hip': (110.0, 300.0),
            'right_hip': (190.0, 300.0)
        }
    
    @pytest.fixture
    def garment_metadata(self):
        """Sample garment metadata."""
        return {
            'base_shoulder_px': 100,
            'base_torso_px': 150,
            'y_offset_to_waist_px': 0
        }
    
    def test_compute_dst_points_basic(self, sample_landmarks, garment_metadata):
        """Test basic destination point computation."""
        mapper = TPSMapper()
        dst_pts = mapper.compute_dst_points(sample_landmarks, garment_metadata)
        
        # Should return 6 points
        assert dst_pts.shape == (6, 2)
        assert dst_pts.dtype == np.float32
        
        # Shoulders should match landmarks exactly
        np.testing.assert_array_almost_equal(
            dst_pts[0], np.array([100.0, 150.0]), decimal=2
        )
        np.testing.assert_array_almost_equal(
            dst_pts[1], np.array([200.0, 150.0]), decimal=2
        )
    
    def test_waist_interpolation(self, sample_landmarks, garment_metadata):
        """Test waist position interpolates correctly between shoulders and hips."""
        mapper = TPSMapper(waist_alpha=0.6)
        dst_pts = mapper.compute_dst_points(sample_landmarks, garment_metadata)
        
        # Calculate expected waist Y position
        mid_shoulder_y = 150.0
        mid_hip_y = 300.0
        expected_waist_y = mid_shoulder_y + 0.6 * (mid_hip_y - mid_shoulder_y)
        
        # Check left and right waist Y coordinates
        assert abs(dst_pts[2][1] - expected_waist_y) < 1.0  # L_waist
        assert abs(dst_pts[3][1] - expected_waist_y) < 1.0  # R_waist
    
    def test_waist_lateral_adjustment(self, sample_landmarks, garment_metadata):
        """Test waist points are laterally adjusted."""
        mapper = TPSMapper(waist_lateral_factor=0.08)
        dst_pts = mapper.compute_dst_points(sample_landmarks, garment_metadata)
        
        # Shoulder distance is 100px
        shoulder_distance = 100.0
        expected_lateral_shift = 0.08 * shoulder_distance  # 8px
        
        # Mid-point should be at x=150
        mid_x = 150.0
        
        # Check lateral shifts
        assert abs(dst_pts[2][0] - (mid_x - expected_lateral_shift)) < 1.0  # L_waist
        assert abs(dst_pts[3][0] - (mid_x + expected_lateral_shift)) < 1.0  # R_waist
    
    def test_y_offset_applied(self, sample_landmarks):
        """Test y_offset_to_waist_px shifts waist vertically."""
        metadata = {'y_offset_to_waist_px': 20}
        mapper = TPSMapper(waist_alpha=0.6)
        dst_pts = mapper.compute_dst_points(sample_landmarks, metadata)
        
        # Calculate expected waist Y with offset
        mid_shoulder_y = 150.0
        mid_hip_y = 300.0
        expected_waist_y = mid_shoulder_y + 0.6 * (mid_hip_y - mid_shoulder_y) + 20
        
        assert abs(dst_pts[2][1] - expected_waist_y) < 1.0
    
    def test_hip_widening_with_segmentation(self, sample_landmarks, garment_metadata):
        """Test hip widening when segmentation indicates wider torso."""
        mapper = TPSMapper(hip_widening_factor=0.05)
        
        # Without segmentation - no widening
        dst_pts_no_seg = mapper.compute_dst_points(
            sample_landmarks, garment_metadata
        )
        
        # With wide segmentation - should widen
        shoulder_distance = 100.0
        wide_segmentation = shoulder_distance * 1.5  # Much wider than typical
        dst_pts_with_seg = mapper.compute_dst_points(
            sample_landmarks, garment_metadata, segmentation_width=wide_segmentation
        )
        
        # Hip points should be wider in second case
        hip_distance_no_seg = dst_pts_no_seg[5][0] - dst_pts_no_seg[4][0]
        hip_distance_with_seg = dst_pts_with_seg[5][0] - dst_pts_with_seg[4][0]
        
        assert hip_distance_with_seg > hip_distance_no_seg
    
    def test_missing_landmarks_raises_error(self, garment_metadata):
        """Test error raised when required landmarks missing."""
        incomplete_landmarks = {
            'left_shoulder': (100.0, 150.0),
            'right_shoulder': (200.0, 150.0)
            # Missing hips
        }
        
        mapper = TPSMapper()
        with pytest.raises(ValueError, match="Missing required landmarks"):
            mapper.compute_dst_points(incomplete_landmarks, garment_metadata)


class TestDestinationPointsWithHem:
    """Test hem/tail point computation."""
    
    @pytest.fixture
    def sample_landmarks(self):
        return {
            'left_shoulder': (100.0, 150.0),
            'right_shoulder': (200.0, 150.0),
            'left_hip': (110.0, 300.0),
            'right_hip': (190.0, 300.0)
        }
    
    @pytest.fixture
    def garment_metadata(self):
        return {'y_offset_to_waist_px': 0}
    
    def test_compute_with_hem(self, sample_landmarks, garment_metadata):
        """Test hem points are added below hips."""
        mapper = TPSMapper()
        hem_extension = 100.0
        
        dst_pts = mapper.compute_dst_points_with_hem(
            sample_landmarks, garment_metadata, hem_extension=hem_extension
        )
        
        # Should have 8 points (6 base + 2 hem)
        assert dst_pts.shape == (8, 2)
        
        # Hem Y should be hip Y + extension
        hip_y = 300.0
        expected_hem_y = hip_y + hem_extension
        
        assert abs(dst_pts[6][1] - expected_hem_y) < 1.0  # L_hem
        assert abs(dst_pts[7][1] - expected_hem_y) < 1.0  # R_hem
    
    def test_hem_x_matches_hips(self, sample_landmarks, garment_metadata):
        """Test hem X coordinates match hip X coordinates."""
        mapper = TPSMapper()
        dst_pts = mapper.compute_dst_points_with_hem(
            sample_landmarks, garment_metadata, hem_extension=100.0
        )
        
        # Hem X should match hip X
        assert abs(dst_pts[6][0] - dst_pts[4][0]) < 1.0  # L_hem ~ L_hip
        assert abs(dst_pts[7][0] - dst_pts[5][0]) < 1.0  # R_hem ~ R_hip


class TestParameterUpdates:
    """Test real-time parameter updates."""
    
    def test_update_waist_alpha(self):
        """Test waist_alpha can be updated."""
        mapper = TPSMapper(waist_alpha=0.6)
        assert mapper.waist_alpha == 0.6
        
        mapper.update_params(waist_alpha=0.7)
        assert mapper.waist_alpha == 0.7
    
    def test_update_lateral_factor(self):
        """Test waist_lateral_factor can be updated."""
        mapper = TPSMapper(waist_lateral_factor=0.08)
        mapper.update_params(waist_lateral_factor=0.1)
        assert mapper.waist_lateral_factor == 0.1
    
    def test_update_widening_factor(self):
        """Test hip_widening_factor can be updated."""
        mapper = TPSMapper(hip_widening_factor=0.05)
        mapper.update_params(hip_widening_factor=0.08)
        assert mapper.hip_widening_factor == 0.08
    
    def test_update_multiple_params(self):
        """Test multiple parameters can be updated simultaneously."""
        mapper = TPSMapper()
        mapper.update_params(
            waist_alpha=0.5,
            waist_lateral_factor=0.12,
            hip_widening_factor=0.06
        )
        assert mapper.waist_alpha == 0.5
        assert mapper.waist_lateral_factor == 0.12
        assert mapper.hip_widening_factor == 0.06


class TestVisualization:
    """Test visualization utilities."""
    
    def test_visualize_mapping(self):
        """Test visualization draws points on frame."""
        mapper = TPSMapper()
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create sample destination points
        dst_pts = np.array([
            [100, 150], [200, 150],  # Shoulders
            [120, 225], [180, 225],  # Waist
            [110, 300], [190, 300]   # Hips
        ], dtype=np.float32)
        
        overlay = mapper.visualize_mapping(frame, dst_pts, draw_labels=True)
        
        # Should return same size frame
        assert overlay.shape == frame.shape
        
        # Frame should be modified (not all zeros)
        assert not np.array_equal(overlay, frame)
    
    def test_get_control_point_names(self):
        """Test control point name retrieval."""
        mapper = TPSMapper()
        
        names_6 = mapper.get_control_point_names(6)
        assert len(names_6) == 6
        assert names_6[0] == 'L_shoulder_seam'
        assert names_6[5] == 'R_hip'
        
        names_8 = mapper.get_control_point_names(8)
        assert len(names_8) == 8
        assert names_8[6] == 'L_hem'
        assert names_8[7] == 'R_hem'


class TestLerpUtility:
    """Test linear interpolation utility."""
    
    def test_lerp_at_zero(self):
        """Test lerp at t=0 returns first point."""
        a = np.array([0.0, 0.0])
        b = np.array([100.0, 100.0])
        result = TPSMapper._lerp(a, b, 0.0)
        np.testing.assert_array_almost_equal(result, a)
    
    def test_lerp_at_one(self):
        """Test lerp at t=1 returns second point."""
        a = np.array([0.0, 0.0])
        b = np.array([100.0, 100.0])
        result = TPSMapper._lerp(a, b, 1.0)
        np.testing.assert_array_almost_equal(result, b)
    
    def test_lerp_at_half(self):
        """Test lerp at t=0.5 returns midpoint."""
        a = np.array([0.0, 0.0])
        b = np.array([100.0, 100.0])
        result = TPSMapper._lerp(a, b, 0.5)
        np.testing.assert_array_almost_equal(result, np.array([50.0, 50.0]))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_shoulder_distance(self):
        """Test handling when shoulders are at same position."""
        landmarks = {
            'left_shoulder': (150.0, 150.0),
            'right_shoulder': (150.0, 150.0),  # Same as left
            'left_hip': (140.0, 300.0),
            'right_hip': (160.0, 300.0)
        }
        metadata = {'y_offset_to_waist_px': 0}
        
        mapper = TPSMapper()
        # Should not raise error
        dst_pts = mapper.compute_dst_points(landmarks, metadata)
        assert dst_pts.shape == (6, 2)
    
    def test_inverted_shoulder_order(self):
        """Test handling when left/right shoulders are swapped."""
        landmarks = {
            'left_shoulder': (200.0, 150.0),  # Right position
            'right_shoulder': (100.0, 150.0),  # Left position
            'left_hip': (190.0, 300.0),
            'right_hip': (110.0, 300.0)
        }
        metadata = {'y_offset_to_waist_px': 0}
        
        mapper = TPSMapper()
        # Should not raise error - just compute with given positions
        dst_pts = mapper.compute_dst_points(landmarks, metadata)
        assert dst_pts.shape == (6, 2)
    
    def test_extreme_waist_alpha(self):
        """Test extreme waist_alpha values."""
        landmarks = {
            'left_shoulder': (100.0, 150.0),
            'right_shoulder': (200.0, 150.0),
            'left_hip': (110.0, 300.0),
            'right_hip': (190.0, 300.0)
        }
        metadata = {'y_offset_to_waist_px': 0}
        
        # Test alpha = 0 (waist at shoulders)
        mapper_zero = TPSMapper(waist_alpha=0.0)
        dst_pts_zero = mapper_zero.compute_dst_points(landmarks, metadata)
        assert abs(dst_pts_zero[2][1] - 150.0) < 1.0  # Waist Y ~ shoulder Y
        
        # Test alpha = 1 (waist at hips)
        mapper_one = TPSMapper(waist_alpha=1.0)
        dst_pts_one = mapper_one.compute_dst_points(landmarks, metadata)
        assert abs(dst_pts_one[2][1] - 300.0) < 1.0  # Waist Y ~ hip Y


class TestPromptRequirements:
    """Test specific requirements from TPS mapping recipes prompt."""
    
    def test_shoulder_direct_mapping(self):
        """Test shoulders map directly to pose landmarks."""
        landmarks = {
            'left_shoulder': (95.0, 142.0),
            'right_shoulder': (213.0, 158.0),
            'left_hip': (100.0, 300.0),
            'right_hip': (210.0, 300.0)
        }
        metadata = {'y_offset_to_waist_px': 0}
        
        mapper = TPSMapper()
        dst_pts = mapper.compute_dst_points(landmarks, metadata)
        
        # Shoulders should exactly match landmarks
        np.testing.assert_array_almost_equal(dst_pts[0], [95.0, 142.0], decimal=2)
        np.testing.assert_array_almost_equal(dst_pts[1], [213.0, 158.0], decimal=2)
    
    def test_waist_interpolation_with_offset(self):
        """Test waist = lerp(shoulders, hips, 0.6) + y_offset."""
        landmarks = {
            'left_shoulder': (100.0, 100.0),
            'right_shoulder': (200.0, 100.0),
            'left_hip': (110.0, 400.0),
            'right_hip': (190.0, 400.0)
        }
        metadata = {'y_offset_to_waist_px': 15}
        
        mapper = TPSMapper(waist_alpha=0.6)
        dst_pts = mapper.compute_dst_points(landmarks, metadata)
        
        # Expected waist Y: 100 + 0.6 * (400 - 100) + 15 = 100 + 180 + 15 = 295
        expected_y = 295.0
        assert abs(dst_pts[2][1] - expected_y) < 1.0
        assert abs(dst_pts[3][1] - expected_y) < 1.0
    
    def test_waist_lateral_shift_0_08(self):
        """Test lateral shift = ±(0.08 * shoulder_distance)."""
        landmarks = {
            'left_shoulder': (50.0, 100.0),
            'right_shoulder': (150.0, 100.0),  # Distance = 100px
            'left_hip': (60.0, 300.0),
            'right_hip': (140.0, 300.0)
        }
        metadata = {'y_offset_to_waist_px': 0}
        
        mapper = TPSMapper(waist_lateral_factor=0.08)
        dst_pts = mapper.compute_dst_points(landmarks, metadata)
        
        # Shoulder distance = 100px, shift = 0.08 * 100 = 8px
        # Mid X = 100, so L_waist = 92, R_waist = 108
        expected_shift = 8.0
        mid_x = 100.0
        
        assert abs(dst_pts[2][0] - (mid_x - expected_shift)) < 1.0  # 92
        assert abs(dst_pts[3][0] - (mid_x + expected_shift)) < 1.0  # 108
    
    def test_hip_widening_0_05(self):
        """Test hip widening = ±(0.05 * shoulder_distance) when segmentation wide."""
        landmarks = {
            'left_shoulder': (100.0, 100.0),
            'right_shoulder': (200.0, 100.0),  # Distance = 100px
            'left_hip': (90.0, 300.0),
            'right_hip': (210.0, 300.0)
        }
        metadata = {'y_offset_to_waist_px': 0}
        
        mapper = TPSMapper(hip_widening_factor=0.05)
        
        # Wide segmentation triggers widening
        wide_seg = 100.0 * 1.5  # Much wider than typical (120)
        dst_pts = mapper.compute_dst_points(landmarks, metadata, segmentation_width=wide_seg)
        
        # Expected widening = 0.05 * 100 = 5px per side
        # L_hip should be at 90 - 5 = 85, R_hip at 210 + 5 = 215
        expected_widening = 5.0
        
        assert abs(dst_pts[4][0] - (90.0 - expected_widening)) < 1.0  # 85
        assert abs(dst_pts[5][0] - (210.0 + expected_widening)) < 1.0  # 215
