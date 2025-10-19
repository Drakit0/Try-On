"""
Tests for pose estimation module.
Tests with synthetic landmarks and mock data.
"""

import math
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.pose import PoseEstimator, visualize_pose


class MockLandmark:
    """Mock landmark for testing."""
    def __init__(self, x, y, z=0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class TestPoseEstimator:
    """Test suite for PoseEstimator class."""
    
    def test_init_solutions_api(self):
        """Test initialization with Solutions API fallback."""
        # Force Solutions API by using invalid model path
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            assert estimator.api_type == "solutions"
            assert estimator.min_detection_conf == 0.5
            assert estimator.min_tracking_conf == 0.5
            estimator.close()
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator(
                min_detection_conf=0.7,
                min_tracking_conf=0.8,
                use_last_valid=False
            )
            assert estimator.min_detection_conf == 0.7
            assert estimator.min_tracking_conf == 0.8
            assert estimator.use_last_valid is False
            estimator.close()
    
    def test_detect_empty_frame(self):
        """Test detection with empty frame."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator(use_last_valid=False)
            result = estimator.detect(None)
            assert result is None
            estimator.close()
    
    def test_extract_landmarks(self):
        """Test landmark extraction from MediaPipe results."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            # Create synthetic landmarks (normalized coordinates)
            landmarks = [MockLandmark(0, 0) for _ in range(33)]  # MediaPipe has 33 landmarks
            
            # Set specific landmarks we care about
            landmarks[11] = MockLandmark(0.3, 0.4, visibility=0.9)  # Left shoulder
            landmarks[12] = MockLandmark(0.7, 0.4, visibility=0.95)  # Right shoulder
            landmarks[23] = MockLandmark(0.35, 0.8, visibility=0.85)  # Left hip
            landmarks[24] = MockLandmark(0.65, 0.8, visibility=0.9)  # Right hip
            
            # Extract landmarks
            result = estimator._extract_landmarks(landmarks, w=640, h=480)
            
            # Verify pixel coordinates
            assert result["left_shoulder"] == (192, 192)  # 0.3*640, 0.4*480
            assert result["right_shoulder"] == (448, 192)  # 0.7*640, 0.4*480
            assert result["left_hip"] == (224, 384)  # 0.35*640, 0.8*480
            assert result["right_hip"] == (416, 384)  # 0.65*640, 0.8*480
            assert result["image_size"] == (640, 480)
            assert result["confidence"] == 0.85  # Minimum visibility
            
            estimator.close()
    
    def test_shoulder_midpoint(self):
        """Test shoulder midpoint calculation."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (300, 200),
                "left_hip": (120, 400),
                "right_hip": (280, 400),
                "image_size": (640, 480)
            }
            
            midpoint = estimator.shoulder_midpoint(landmarks)
            assert midpoint == (200, 200)  # (100+300)/2, (200+200)/2
            
            estimator.close()
    
    def test_hip_midpoint(self):
        """Test hip midpoint calculation."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (300, 200),
                "left_hip": (120, 400),
                "right_hip": (280, 400),
                "image_size": (640, 480)
            }
            
            midpoint = estimator.hip_midpoint(landmarks)
            assert midpoint == (200, 400)  # (120+280)/2, (400+400)/2
            
            estimator.close()
    
    def test_shoulder_vector_angle_deg(self):
        """Test shoulder angle calculation."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            # Horizontal shoulders (0 degrees)
            landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (300, 200),
                "left_hip": (120, 400),
                "right_hip": (280, 400),
                "image_size": (640, 480)
            }
            angle = estimator.shoulder_vector_angle_deg(landmarks)
            assert abs(angle) < 0.01  # Should be close to 0
            
            # Right shoulder 100px lower (tilted)
            landmarks["right_shoulder"] = (300, 300)
            angle = estimator.shoulder_vector_angle_deg(landmarks)
            expected_angle = math.degrees(math.atan2(100, 200))  # dy=100, dx=200
            assert abs(angle - expected_angle) < 0.01
            
            estimator.close()
    
    def test_shoulder_distance_px(self):
        """Test shoulder distance calculation."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (400, 200),
                "left_hip": (120, 400),
                "right_hip": (380, 400),
                "image_size": (640, 480)
            }
            
            distance = estimator.shoulder_distance_px(landmarks)
            assert distance == 300  # 400 - 100
            
            # Tilted shoulders
            landmarks["right_shoulder"] = (400, 300)
            distance = estimator.shoulder_distance_px(landmarks)
            expected = math.sqrt(300**2 + 100**2)  # Pythagorean theorem
            assert abs(distance - expected) < 0.01
            
            estimator.close()
    
    def test_torso_height_px(self):
        """Test torso height calculation."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (300, 200),
                "left_hip": (100, 500),
                "right_hip": (300, 500),
                "image_size": (640, 480)
            }
            
            height = estimator.torso_height_px(landmarks)
            assert height == 300  # 500 - 200
            
            estimator.close()
    
    def test_get_stats(self):
        """Test getting all statistics at once."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            
            landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (300, 200),
                "left_hip": (120, 400),
                "right_hip": (280, 400),
                "image_size": (640, 480),
                "confidence": 0.95
            }
            
            stats = estimator.get_stats(landmarks)
            
            assert stats["shoulder_midpoint"] == (200, 200)
            assert stats["hip_midpoint"] == (200, 400)
            assert abs(stats["shoulder_angle_deg"]) < 0.01
            assert stats["shoulder_distance_px"] == 200
            assert stats["torso_height_px"] == 200
            assert stats["confidence"] == 0.95
            assert stats["is_cached"] is False
            
            estimator.close()
    
    def test_last_valid_result_caching(self):
        """Test caching of last valid result."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator(use_last_valid=True)
            
            # Set a valid result
            valid_landmarks = {
                "left_shoulder": (100, 200),
                "right_shoulder": (300, 200),
                "left_hip": (120, 400),
                "right_hip": (280, 400),
                "image_size": (640, 480),
                "confidence": 0.9,
                "is_cached": False
            }
            estimator.last_valid_result = valid_landmarks
            
            # Request fallback
            result = estimator._get_fallback_result()
            
            assert result is not None
            assert result["is_cached"] is True
            assert result["left_shoulder"] == (100, 200)
            
            estimator.close()
    
    def test_no_caching_when_disabled(self):
        """Test that caching doesn't happen when disabled."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator(use_last_valid=False)
            
            # Set a valid result
            estimator.last_valid_result = {
                "left_shoulder": (100, 200),
                "confidence": 0.9
            }
            
            # Request fallback - should return None
            result = estimator._get_fallback_result()
            assert result is None
            
            estimator.close()
    
    def test_context_manager(self):
        """Test context manager usage."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            with PoseEstimator() as estimator:
                assert estimator is not None
                assert hasattr(estimator, 'pose')
            # Should close automatically
    
    def test_visualize_pose_none_landmarks(self):
        """Test visualization with None landmarks."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = visualize_pose(frame, None)
        assert result.shape == frame.shape
        assert np.array_equal(result, frame)
    
    def test_visualize_pose_with_landmarks(self):
        """Test visualization with valid landmarks."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = {
            "left_shoulder": (200, 150),
            "right_shoulder": (400, 150),
            "left_hip": (220, 350),
            "right_hip": (380, 350),
            "image_size": (640, 480)
        }
        
        result = visualize_pose(frame, landmarks)
        assert result.shape == frame.shape
        # Check that frame was modified (some pixels should be non-zero)
        assert not np.array_equal(result, frame)
    
    def test_utility_methods_with_none(self):
        """Test utility methods return None when no landmarks available."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator()
            estimator.last_valid_result = None
            
            assert estimator.shoulder_midpoint() is None
            assert estimator.hip_midpoint() is None
            assert estimator.shoulder_vector_angle_deg() is None
            assert estimator.shoulder_distance_px() is None
            assert estimator.torso_height_px() is None
            assert estimator.get_stats() == {}
            
            estimator.close()
    
    def test_complexity_mapping(self):
        """Test model complexity mapping."""
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            estimator = PoseEstimator(model_complexity='lite')
            assert estimator._map_complexity() == 0
            
            estimator.model_complexity = 'full'
            assert estimator._map_complexity() == 1
            
            estimator.model_complexity = 'heavy'
            assert estimator._map_complexity() == 2
            
            estimator.model_complexity = 'invalid'
            assert estimator._map_complexity() == 1  # Default
            
            estimator.close()


class TestPoseEstimatorIntegration:
    """Integration tests with actual MediaPipe (if available)."""
    
    def test_detect_with_actual_frame(self):
        """Test detection with a real frame (will fail on blank, but tests pipeline)."""
        # Create a test frame with a simple figure
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple stick figure (won't be detected, but tests pipeline)
        
        try:
            with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
                with PoseEstimator(use_last_valid=False) as estimator:
                    result = estimator.detect(frame)
                    # Blank frame won't have a person, so result should be None
                    assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.skip(f"MediaPipe not available or error: {e}")
    
    def test_frame_count_increment(self):
        """Test that frame count increments."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('src.pose.MEDIAPIPE_TASKS_AVAILABLE', False):
            with PoseEstimator() as estimator:
                initial_count = estimator.frame_count
                estimator.detect(frame)
                assert estimator.frame_count == initial_count + 1
                estimator.detect(frame)
                assert estimator.frame_count == initial_count + 2


def test_mock_landmark_creation():
    """Test MockLandmark helper class."""
    landmark = MockLandmark(0.5, 0.6, 0.1, 0.95)
    assert landmark.x == 0.5
    assert landmark.y == 0.6
    assert landmark.z == 0.1
    assert landmark.visibility == 0.95


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
