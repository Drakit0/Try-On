"""
Pose estimation module using MediaPipe Tasks API.
Detects body landmarks (shoulders, hips) for virtual try-on anchoring.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Always import mediapipe base
import mediapipe as mp

try:
    from mediapipe import tasks
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError:
    # Fallback to classic MediaPipe Solutions API
    MEDIAPIPE_TASKS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Pose estimator using MediaPipe for landmark detection.
    
    Detects shoulder and hip landmarks for garment anchoring and scaling.
    Supports both MediaPipe Tasks API (recommended) and Solutions API (fallback).
    
    Args:
        model_path: Path to pose landmarker task model (Tasks API only)
        min_detection_conf: Minimum detection confidence threshold (0.0-1.0)
        min_tracking_conf: Minimum tracking confidence threshold (0.0-1.0)
        model_complexity: Model complexity ('lite', 'full', 'heavy')
        use_last_valid: If True, reuse last valid detection on failure
    """
    
    # MediaPipe pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    def __init__(
        self,
        model_path: str = 'pose_landmarker_full.task',
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        model_complexity: str = 'full',
        use_last_valid: bool = True
    ):
        """Initialize pose estimator with MediaPipe."""
        self.model_path = model_path
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.model_complexity = model_complexity
        self.use_last_valid = use_last_valid
        
        # State management
        self.last_valid_result: Optional[Dict] = None
        self.frame_count = 0
        
        # Initialize the appropriate pose detector
        if MEDIAPIPE_TASKS_AVAILABLE:
            self._init_tasks_api()
        else:
            self._init_solutions_api()  # Fallback
    
    def _init_tasks_api(self):
        """Initialize MediaPipe Tasks API pose landmarker."""
        logger.info("Initializing MediaPipe Tasks API for pose detection")
        
        try:
            # Configure base options
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            
            # Configure pose landmarker options
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf,
                num_poses=1  # Only detect one person
            )
            
            # Create pose landmarker
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.api_type = "tasks"
            logger.info("MediaPipe Tasks API pose landmarker initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Tasks API: {e}. Falling back to Solutions API.")
            self._init_solutions_api()  # Fallback
    
    def _init_solutions_api(self):
        """Initialize MediaPipe Solutions API pose detector (Fallback)."""
        logger.info("Initializing MediaPipe Solutions API for pose detection (Fallback)")
        
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self._map_complexity(),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        )
        self.api_type = "solutions"
        logger.info("MediaPipe Solutions API pose detector initialized successfully")
    
    def _map_complexity(self) -> int:
        """Map complexity string to integer for Solutions API."""
        complexity_map = {
            'lite': 0,
            'full': 1,
            'heavy': 2
        }
        return complexity_map.get(self.model_complexity.lower(), 1)
    
    def detect(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        Detect pose landmarks in a BGR frame.
        
        Args:
            frame_bgr: Input frame in BGR format (np.ndarray, uint8)
        
        Returns:
            Dictionary with landmarks or None if detection fails:
            {
                "left_shoulder": (x_px, y_px),
                "right_shoulder": (x_px, y_px),
                "left_hip": (x_px, y_px),
                "right_hip": (x_px, y_px),
                "image_size": (width, height),
                "confidence": float,
                "is_cached": bool  # True if using last valid result
            }
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Empty frame provided to pose detector")
            return self._get_fallback_result()
        
        h, w = frame_bgr.shape[:2]
        self.frame_count += 1
        
        try:
            if self.api_type == "tasks":
                result = self._detect_tasks_api(frame_bgr, w, h)
            else:
                result = self._detect_solutions_api(frame_bgr, w, h)
            
            if result is not None:
                result["is_cached"] = False
                self.last_valid_result = result
                return result
            else:
                return self._get_fallback_result()
                
        except Exception as e:
            logger.error(f"Error during pose detection: {e}")
            return self._get_fallback_result()
    
    def _detect_tasks_api(
        self, 
        frame_bgr: np.ndarray, 
        w: int, 
        h: int
    ) -> Optional[Dict]:
        """Detect using MediaPipe Tasks API."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose landmarks
        # Use frame count as timestamp (in milliseconds)
        detection_result = self.landmarker.detect_for_video(mp_image, self.frame_count)
        
        # Check if pose was detected
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            logger.debug(f"No pose detected in frame {self.frame_count}")
            return None
        
        # Extract first person's landmarks (we only track one person)
        landmarks = detection_result.pose_landmarks[0]
        
        # Extract required landmarks and convert to pixel coordinates
        result = self._extract_landmarks(landmarks, w, h)
        return result
    
    def _detect_solutions_api(
        self, 
        frame_bgr: np.ndarray, 
        w: int, 
        h: int
    ) -> Optional[Dict]:
        """Detect using MediaPipe Solutions API (Fallback)."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        # Check if pose was detected
        if not results.pose_landmarks:
            logger.debug(f"No pose detected in frame {self.frame_count}")
            return None
        
        # Extract required landmarks
        result = self._extract_landmarks(results.pose_landmarks.landmark, w, h)
        return result
    
    def _extract_landmarks(self, landmarks, w: int, h: int) -> Dict:
        """
        Extract shoulder and hip landmarks from MediaPipe results.
        
        Args:
            landmarks: MediaPipe landmarks (Tasks or Solutions format)
            w: Frame width
            h: Frame height
        
        Returns:
            Dictionary with extracted landmarks
        """
        # Get landmark coordinates
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        
        # Convert normalized coordinates to pixels
        result = {
            "left_shoulder": (int(left_shoulder.x * w), int(left_shoulder.y * h)),
            "right_shoulder": (int(right_shoulder.x * w), int(right_shoulder.y * h)),
            "left_hip": (int(left_hip.x * w), int(left_hip.y * h)),
            "right_hip": (int(right_hip.x * w), int(right_hip.y * h)),
            "image_size": (w, h),
            "confidence": min(
                getattr(left_shoulder, 'visibility', 1.0),
                getattr(right_shoulder, 'visibility', 1.0),
                getattr(left_hip, 'visibility', 1.0),
                getattr(right_hip, 'visibility', 1.0)
            )
        }
        
        return result
    
    def _get_fallback_result(self) -> Optional[Dict]:
        """Get fallback result (last valid or None)."""
        if self.use_last_valid and self.last_valid_result is not None:
            logger.debug("Using last valid pose detection result")
            cached_result = self.last_valid_result.copy()
            cached_result["is_cached"] = True
            return cached_result
        return None
    
    # Utility methods
    
    def shoulder_midpoint(self, landmarks: Optional[Dict] = None) -> Optional[Tuple[float, float]]:
        """
        Calculate midpoint between shoulders.
        
        Args:
            landmarks: Landmark dict (uses last_valid_result if None)
        
        Returns:
            (x, y) tuple or None
        """
        if landmarks is None:
            landmarks = self.last_valid_result
        
        if landmarks is None:
            return None
        
        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]
        return ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    
    def hip_midpoint(self, landmarks: Optional[Dict] = None) -> Optional[Tuple[float, float]]:
        """
        Calculate midpoint between hips.
        
        Args:
            landmarks: Landmark dict (uses last_valid_result if None)
        
        Returns:
            (x, y) tuple or None
        """
        if landmarks is None:
            landmarks = self.last_valid_result
        
        if landmarks is None:
            return None
        
        lh = landmarks["left_hip"]
        rh = landmarks["right_hip"]
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    
    def shoulder_vector_angle_deg(self, landmarks: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate shoulder line angle in degrees (from horizontal).
        
        Positive angle = right shoulder higher
        Negative angle = left shoulder higher
        
        Args:
            landmarks: Landmark dict (uses last_valid_result if None)
        
        Returns:
            Angle in degrees or None
        """
        if landmarks is None:
            landmarks = self.last_valid_result
        
        if landmarks is None:
            return None
        
        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]
        
        # Calculate angle from horizontal
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def shoulder_distance_px(self, landmarks: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate distance between shoulders in pixels.
        
        Args:
            landmarks: Landmark dict (uses last_valid_result if None)
        
        Returns:
            Distance in pixels or None
        """
        if landmarks is None:
            landmarks = self.last_valid_result
        
        if landmarks is None:
            return None
        
        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]
        
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        
        return math.sqrt(dx**2 + dy**2)
    
    def torso_height_px(self, landmarks: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate torso height (shoulder midpoint to hip midpoint) in pixels.
        
        Args:
            landmarks: Landmark dict (uses last_valid_result if None)
        
        Returns:
            Height in pixels or None
        """
        if landmarks is None:
            landmarks = self.last_valid_result
        
        if landmarks is None:
            return None
        
        shoulder_mid = self.shoulder_midpoint(landmarks)
        hip_mid = self.hip_midpoint(landmarks)
        
        if shoulder_mid is None or hip_mid is None:
            return None
        
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        
        return math.sqrt(dx**2 + dy**2)
    
    def get_stats(self, landmarks: Optional[Dict] = None) -> Dict:
        """
        Get all pose statistics in one call.
        
        Args:
            landmarks: Landmark dict (uses last_valid_result if None)
        
        Returns:
            Dictionary with all statistics
        """
        if landmarks is None:
            landmarks = self.last_valid_result
        
        if landmarks is None:
            return {}
        
        return {
            "shoulder_midpoint": self.shoulder_midpoint(landmarks),
            "hip_midpoint": self.hip_midpoint(landmarks),
            "shoulder_angle_deg": self.shoulder_vector_angle_deg(landmarks),
            "shoulder_distance_px": self.shoulder_distance_px(landmarks),
            "torso_height_px": self.torso_height_px(landmarks),
            "confidence": landmarks.get("confidence", 1.0),
            "is_cached": landmarks.get("is_cached", False)
        }
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()
        logger.info("Pose estimator closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility function for quick testing
def visualize_pose(frame_bgr: np.ndarray, landmarks: Dict) -> np.ndarray:
    """
    Draw pose landmarks on frame for visualization.
    
    Args:
        frame_bgr: Input frame
        landmarks: Landmark dictionary from PoseEstimator.detect()
    
    Returns:
        Frame with landmarks drawn
    """
    if landmarks is None:
        return frame_bgr
    
    frame = frame_bgr.copy()
    
    # Draw landmarks
    for key in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
        if key in landmarks:
            x, y = landmarks[key]
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, key.split('_')[0][0].upper(), (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw shoulder line
    if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
        cv2.line(frame, landmarks["left_shoulder"], landmarks["right_shoulder"], 
                (255, 0, 0), 2)
    
    # Draw hip line
    if "left_hip" in landmarks and "right_hip" in landmarks:
        cv2.line(frame, landmarks["left_hip"], landmarks["right_hip"], 
                (255, 0, 0), 2)
    
    # Draw torso line (shoulder midpoint to hip midpoint)
    estimator = PoseEstimator()
    shoulder_mid = estimator.shoulder_midpoint(landmarks)
    hip_mid = estimator.hip_midpoint(landmarks)
    if shoulder_mid and hip_mid:
        cv2.line(frame, 
                (int(shoulder_mid[0]), int(shoulder_mid[1])),
                (int(hip_mid[0]), int(hip_mid[1])),
                (0, 255, 255), 2)
    
    return frame


if __name__ == "__main__":
    # Test with a dummy frame
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PoseEstimator...")
    print(f"MediaPipe Tasks API available: {MEDIAPIPE_TASKS_AVAILABLE}")
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        # Initialize without model file (will use Solutions API as fallback)
        estimator = PoseEstimator(use_last_valid=True)
        print(f"Using API: {estimator.api_type}")
        
        # Try detection (will fail on blank frame, but tests initialization)
        result = estimator.detect(test_frame)
        print(f"Detection result: {result}")
        
        estimator.close()
        print("PoseEstimator test complete!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
