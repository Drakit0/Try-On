"""
MediaPipe compatibility utilities.
Checks MediaPipe availability and provides fallback options.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def check_mediapipe_availability() -> Tuple[bool, bool, str]:
    """
    Check MediaPipe and MediaPipe Tasks availability.
    
    Returns:
        Tuple of (has_mediapipe, has_tasks, version_info)
        - has_mediapipe: Whether mediapipe is installed
        - has_tasks: Whether mediapipe.tasks is available
        - version_info: Version string
    """
    try:
        import mediapipe as mp
        has_mediapipe = True
        version_info = mp.__version__
        
        try:
            from mediapipe import tasks
            has_tasks = True
            logger.info(f"MediaPipe {version_info} with Tasks API available")
        except ImportError:
            has_tasks = False
            logger.warning(
                f"MediaPipe {version_info} installed but Tasks API not available. "
                "Using classic solutions API."
            )
        
        return has_mediapipe, has_tasks, version_info
    
    except ImportError as e:
        logger.error(f"MediaPipe not installed: {e}")
        return False, False, "Not installed"


def get_pose_detector(use_tasks: bool = True):
    """
    Get pose detector based on availability.
    
    Args:
        use_tasks: Whether to prefer Tasks API (if available)
    
    Returns:
        Pose detector instance
    
    Raises:
        ImportError: If MediaPipe is not available
    """
    has_mediapipe, has_tasks, version = check_mediapipe_availability()
    
    if not has_mediapipe:
        raise ImportError("MediaPipe is not installed. Install with: pip install mediapipe>=0.10")
    
    if use_tasks and has_tasks:
        # Use new Tasks API
        from mediapipe import tasks
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        logger.info("Using MediaPipe Tasks API for pose detection")
        # Will be implemented in pose.py
        return "tasks_api"
    else:
        # Fallback to classic solutions API
        import mediapipe as mp
        
        logger.info("Using MediaPipe Solutions API for pose detection (Fallback)")
        # Will be implemented in pose.py
        return "solutions_api"


def get_segmenter(use_tasks: bool = True):
    """
    Get segmenter based on availability.
    
    Args:
        use_tasks: Whether to prefer Tasks API (if available)
    
    Returns:
        Segmenter instance type identifier
    
    Raises:
        ImportError: If MediaPipe is not available
    """
    has_mediapipe, has_tasks, version = check_mediapipe_availability()
    
    if not has_mediapipe:
        raise ImportError("MediaPipe is not installed. Install with: pip install mediapipe>=0.10")
    
    if use_tasks and has_tasks:
        # Use new Tasks API
        logger.info("Using MediaPipe Tasks API for segmentation")
        return "tasks_api"
    else:
        # Fallback to classic solutions API  # Fallback
        logger.info("Using MediaPipe Solutions API for segmentation (Fallback)")
        return "solutions_api"


if __name__ == "__main__":
    # Test availability
    logging.basicConfig(level=logging.INFO)
    has_mp, has_tasks, version = check_mediapipe_availability()
    
    print(f"MediaPipe installed: {has_mp}")
    print(f"MediaPipe version: {version}")
    print(f"Tasks API available: {has_tasks}")
    
    if has_mp:
        pose_type = get_pose_detector()
        seg_type = get_segmenter()
        print(f"Pose detector type: {pose_type}")
        print(f"Segmenter type: {seg_type}")
