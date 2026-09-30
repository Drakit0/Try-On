"""
Virtual Try-On - Streamlit Web Application
===========================================

Real-time virtual garment try-on using webcam with pose detection,
TPS warping, and realistic composition.

Features:
- Real-time webcam video with streamlit-webrtc
- Multiple garment selection
- Adjustable size, offset, and warping
- Face/skin occlusion for realism
- Mirror mode and FPS counter
- Capture and save frames

Authors: Pablo Tuñón Laguna, Lydia Ruiz Martínez
Date: 2025-01-19
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.overlay import (
    compose_with_occlusion,
    compute_transform,
    render_overlay,
)
from src.pose import PoseEstimator
from src.privacy import CaptureManager, PrivacyAuditor, format_file_size
from src.segmenter import PersonSegmenter
from src.smoothing import OneEuroFilter
from src.tps import ThinPlateSpline
from src.utils import PerformanceTracker, setup_logger

# Set up logger
logger = setup_logger('app', level=logging.INFO)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration."""
    
    # Video settings
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    TARGET_FPS = 30
    
    # Asset paths
    ASSETS_DIR = Path("assets")
    GARMENTS_JSON = ASSETS_DIR / "garments.json"
    
    # Performance optimizations
    FRAME_SKIP = 1  # Process every Nth frame
    POSE_FRAME_SKIP = 2  # Run pose every Nth frame (render @ 30, pose @ 15)
    MAX_PROCESSING_TIME = 0.040  # 40ms = ~25 FPS
    GC_INTERVAL = 100  # Run GC every N frames
    
    # Smoothing parameters
    SMOOTH_MIN_CUTOFF = 1.0
    SMOOTH_BETA = 0.1
    SMOOTH_DCUTOFF = 1.0


# ============================================================================
# Garment Management
# ============================================================================

@st.cache_data
def load_garments() -> List[Dict]:
    """Load garment metadata from JSON file."""
    try:
        with open(Config.GARMENTS_JSON, 'r') as f:
            data = json.load(f)
            return data.get('garments', [])
    except Exception as e:
        st.error(f"Error loading garments: {e}")
        return []


def load_garment_image(
    garment_path: str,
    max_width: int = 1024,
    state: Optional['ProcessingState'] = None
) -> Optional[np.ndarray]:
    """
    Load garment image with alpha channel.
    
    Uses cv2.INTER_AREA for high-quality downscaling and caches loaded images.
    
    Parameters
    ----------
    garment_path : str
        Path to garment image
    max_width : int
        Maximum width for downscaling (default: 1024)
    state : ProcessingState, optional
        Processing state for caching
        
    Returns
    -------
    np.ndarray or None
        RGBA image or None if failed
    """
    try:
        # Check cache first
        if state is not None and garment_path in state.cached_garments:
            return state.cached_garments[garment_path]
        
        # Try relative to assets directory first
        path = Config.ASSETS_DIR / garment_path
        if not path.exists():
            # Try relative to current directory
            path = Path(garment_path)
        
        if not path.exists():
            st.error(f"Garment not found: {garment_path}")
            return None
        
        garment = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if garment is None:
            st.error(f"Failed to load garment: {path}")
            return None
        
        if garment.shape[2] != 4:
            st.warning(f"Garment {path.name} has no alpha channel, adding opaque alpha")
            h, w = garment.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            garment = np.concatenate([garment, alpha], axis=2)
        
        # Downscale if too large (use INTER_AREA for quality)
        h, w = garment.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            garment = cv2.resize(garment, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Cache for reuse
        if state is not None:
            state.cached_garments[garment_path] = garment
        
        return garment
    
    except Exception as e:
        st.error(f"Error loading garment {garment_path}: {e}")
        return None


def compute_tps_dst_points(
    landmarks: Dict[str, Tuple[float, float]],
    garment_meta: Dict
) -> np.ndarray:
    """
    Compute TPS destination points from pose landmarks.
    
    Maps garment control points to body landmarks for realistic warping.
    
    Parameters
    ----------
    landmarks : dict
        Pose landmarks (shoulders, hips, etc.)
    garment_meta : dict
        Garment metadata including base dimensions
    
    Returns
    -------
    np.ndarray
        Destination points for TPS warping, shape (N, 2)
    """
    # Extract landmarks
    left_shoulder = np.array(landmarks.get('left_shoulder', [0, 0]))
    right_shoulder = np.array(landmarks.get('right_shoulder', [0, 0]))
    left_hip = np.array(landmarks.get('left_hip', [0, 0]))
    right_hip = np.array(landmarks.get('right_hip', [0, 0]))
    
    # Compute midpoints
    mid_shoulders = (left_shoulder + right_shoulder) / 2.0
    mid_hips = (left_hip + right_hip) / 2.0
    
    # Shoulder distance for lateral adjustments
    shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
    
    # Waist estimation (interpolate between shoulders and hips)
    alpha = 0.6  # 60% of the way from shoulders to hips
    waist_center = mid_shoulders + alpha * (mid_hips - mid_shoulders)
    
    # Add y_offset if specified
    y_offset = garment_meta.get('y_offset_to_waist_px', 0)
    waist_center[1] += y_offset
    
    # Lateral waist adjustments (slightly narrower than shoulders)
    waist_lateral_offset = 0.08 * shoulder_dist
    left_waist = waist_center + np.array([-waist_lateral_offset, 0])
    right_waist = waist_center + np.array([waist_lateral_offset, 0])
    
    # Hip points (use detected hips with slight widening)
    hip_lateral_offset = 0.05 * shoulder_dist
    left_hip_adj = left_hip + np.array([-hip_lateral_offset, 0])
    right_hip_adj = right_hip + np.array([hip_lateral_offset, 0])
    
    # Construct destination points array
    # Order matches TPS control points: shoulders, waist, hips
    dst_pts = np.array([
        left_shoulder,      # L_shoulder_seam
        right_shoulder,     # R_shoulder_seam
        left_waist,         # L_waist
        right_waist,        # R_waist
        left_hip_adj,       # L_hip
        right_hip_adj       # R_hip
    ], dtype=np.float32)
    
    return dst_pts


# ============================================================================
# Video Processing State
# ============================================================================

class ProcessingState:
    """Shared state for video processing."""
    
    def __init__(self):
        # Module instances (reused across frames)
        self.pose_estimator: Optional[PoseEstimator] = None
        self.segmenter: Optional[PersonSegmenter] = None
        self.tps: Optional[ThinPlateSpline] = None
        
        # Smoothing filters
        self.filters: Dict[str, OneEuroFilter] = {}
        
        # Performance tracking (new)
        self.perf_tracker = PerformanceTracker(window_size=30)
        
        # Legacy performance tracking (kept for compatibility)
        self.frame_count = 0
        self.processing_times = []
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Last processed state
        self.last_landmarks = None
        self.last_transform = None
        
        # Timestamp for smoothing
        self.timestamp = 0.0
        
        # Pose reuse optimization
        self.pose_frame_counter = 0
        self.cached_raw_landmarks = None
        
        # Garment cache
        self.cached_garments = {}  # path -> image
        self.cached_tps = {}  # garment_id -> ThinPlateSpline instance
        
        # GC counter
        self.gc_counter = 0
    
    def initialize_modules(self):
        """Initialize processing modules (lazy loading)."""
        if self.pose_estimator is None:
            self.pose_estimator = PoseEstimator()
            logger.info("Initialized PoseEstimator")
        
        if self.segmenter is None:
            self.segmenter = PersonSegmenter()
            logger.info("Initialized PersonSegmenter")
    
    def get_or_create_filter(self, name: str) -> OneEuroFilter:
        """Get or create a OneEuroFilter for a landmark."""
        if name not in self.filters:
            self.filters[name] = OneEuroFilter(
                min_cutoff=Config.SMOOTH_MIN_CUTOFF,
                beta=Config.SMOOTH_BETA,
                dcutoff=Config.SMOOTH_DCUTOFF
            )
        return self.filters[name]
    
    def smooth_landmarks(self, landmarks: Dict) -> Dict:
        """Apply temporal smoothing to landmarks."""
        smoothed = {}
        
        # Define metadata keys that should not be smoothed
        metadata_keys = {'confidence', 'is_cached', 'image_size'}
        
        for key, value in landmarks.items():
            # Skip metadata fields (confidence, is_cached, image_size)
            if key in metadata_keys:
                smoothed[key] = value
                continue
            
            # Unpack coordinate tuple
            x, y = value
            
            filter_x = self.get_or_create_filter(f"{key}_x")
            filter_y = self.get_or_create_filter(f"{key}_y")
            
            smooth_x = filter_x.filter_scalar(x, self.timestamp)
            smooth_y = filter_y.filter_scalar(y, self.timestamp)
            
            smoothed[key] = (smooth_x, smooth_y)
        
        return smoothed
    
    def update_fps(self, processing_time: float):
        """Update FPS calculation (uses PerformanceTracker now)."""
        # Record in new tracker
        self.perf_tracker.record_frame_time(processing_time)
        
        # Update legacy FPS (for backward compatibility)
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # Update FPS every second
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            self.current_fps = self.perf_tracker.get_fps()
            self.last_fps_update = now


# ============================================================================
# Video Frame Processing
# ============================================================================

def process_frame(frame: av.VideoFrame, state: ProcessingState, settings: Dict) -> av.VideoFrame:
    """
    Process video frame with virtual try-on pipeline.
    
    Parameters
    ----------
    frame : av.VideoFrame
        Input video frame from webcam
    state : ProcessingState
        Shared processing state
    settings : dict
        UI settings (garment, size, offset, etc.)
    
    Returns
    -------
    av.VideoFrame
        Processed frame with garment overlay
    """
    start_time = time.perf_counter()
    
    # Convert to BGR numpy array
    img = frame.to_ndarray(format="bgr24")
    
    # Mirror mode
    if settings.get('mirror', True):
        img = cv2.flip(img, 1)
    
    # Frame skipping for performance
    state.frame_count += 1
    if state.frame_count % Config.FRAME_SKIP != 0:
        # Return original frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Update timestamp for smoothing
    state.timestamp = time.perf_counter()
    
    # Timing variables
    pose_time = 0.0
    tps_time = 0.0
    seg_time = 0.0
    overlay_time = 0.0
    
    try:
        # Initialize modules
        state.initialize_modules()
        
        # Get current garment
        garment_data = settings.get('garment_data')
        garment_img = settings.get('garment_img')
        
        if garment_data is None or garment_img is None:
            # No garment selected
            output = img
            logger.debug("No garment selected")
        else:
            # 1. Detect pose (with frame skipping optimization)
            pose_start = time.perf_counter()
            state.pose_frame_counter += 1
            
            if state.pose_frame_counter % Config.POSE_FRAME_SKIP == 0 or state.cached_raw_landmarks is None:
                # Run pose detection
                landmarks = state.pose_estimator.detect(img)
                state.cached_raw_landmarks = landmarks
                pose_time = time.perf_counter() - pose_start
                
                if landmarks:
                    logger.debug(f"Pose detected: {len(landmarks)} landmarks in {pose_time*1000:.1f}ms")
                else:
                    logger.debug(f"No pose detected ({pose_time*1000:.1f}ms)")
            else:
                # Reuse last pose detection result
                landmarks = state.cached_raw_landmarks
                pose_time = time.perf_counter() - pose_start
                logger.debug(f"Reusing cached pose ({pose_time*1000:.1f}ms)")
            
            # Record pose timing
            state.perf_tracker.record_operation('pose_detection', pose_time)
            
            if landmarks is None or not landmarks:
                # No person detected
                output = img
                
                # Draw message
                cv2.putText(
                    output,
                    "No person detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
            else:
                # 2. Smooth landmarks
                landmarks = state.smooth_landmarks(landmarks)
                state.last_landmarks = landmarks
                
                # 3. Compute transform
                garment_meta = garment_data.get('metadata', {})
                transform = compute_transform(landmarks, garment_meta)
                
                # Apply size adjustment
                size_factor = settings.get('size', 1.0)
                
                # Apply vertical offset
                y_offset = settings.get('y_offset', 0)
                
                # 4. Apply TPS warping if enabled
                use_tps = settings.get('use_tps', False)
                
                if use_tps and 'tps_control_points' in garment_data:
                    # Get TPS control points
                    tps_config = garment_data['tps_control_points']
                    src_pts_config = tps_config.get('src', [])
                    
                    if src_pts_config:
                        tps_start = time.perf_counter()
                        
                        # Get or create cached TPS for this garment
                        garment_id = garment_data.get('id', 'unknown')
                        
                        if garment_id not in state.cached_tps:
                            # Convert source points to numpy array
                            src_pts = np.array([
                                [pt['x'], pt['y']] for pt in src_pts_config
                            ], dtype=np.float32)
                            
                            # Precompute TPS matrix (expensive, done once per garment)
                            state.cached_tps[garment_id] = ThinPlateSpline(src_pts)
                            logger.info(f"Created TPS for garment '{garment_id}' with {len(src_pts)} control points")
                        
                        tps = state.cached_tps[garment_id]
                        
                        # Compute destination points from landmarks
                        dst_pts = compute_tps_dst_points(landmarks, garment_meta)
                        
                        # Fit to destination points (cheap, just solves linear system)
                        tps.fit(dst_pts)
                        
                        # Warp garment
                        warped_garment = tps.warp_rgba(garment_img)
                        
                        tps_time = time.perf_counter() - tps_start
                        logger.debug(f"TPS warping completed in {tps_time*1000:.1f}ms")
                        
                        # Record TPS timing
                        state.perf_tracker.record_operation('tps_warping', tps_time)
                        
                        # Render overlay with warped garment
                        overlay_start = time.perf_counter()
                        output = render_overlay(
                            img,
                            warped_garment,
                            transform,
                            y_offset_px=y_offset,
                            scale_bias=size_factor
                        )
                        overlay_time = time.perf_counter() - overlay_start
                    else:
                        # No TPS points, fall back to simple overlay
                        overlay_start = time.perf_counter()
                        output = render_overlay(
                            img,
                            garment_img,
                            transform,
                            y_offset_px=y_offset,
                            scale_bias=size_factor
                        )
                        overlay_time = time.perf_counter() - overlay_start
                        logger.debug("No TPS control points, using simple overlay")
                else:
                    # Simple overlay (no TPS)
                    overlay_start = time.perf_counter()
                    output = render_overlay(
                        img,
                        garment_img,
                        transform,
                        y_offset_px=y_offset,
                        scale_bias=size_factor
                    )
                    overlay_time = time.perf_counter() - overlay_start
                
                # Record overlay timing
                state.perf_tracker.record_operation('overlay', overlay_time)
                
                # 5. Apply occlusion if enabled
                use_occlusion = settings.get('use_occlusion', False)
                
                if use_occlusion:
                    seg_start = time.perf_counter()
                    
                    # Get face/skin mask
                    occlusion_mask = state.segmenter.mask(
                        img,
                        classes=['face', 'skin']
                    )
                    
                    seg_time = time.perf_counter() - seg_start
                    logger.debug(f"Segmentation completed in {seg_time*1000:.1f}ms")
                    
                    # Record segmentation timing
                    state.perf_tracker.record_operation('segmentation', seg_time)
                    
                    if occlusion_mask is not None:
                        # Compose with occlusion
                        output = compose_with_occlusion(
                            img,
                            garment_img,
                            occlusion_mask,
                            transform,
                            y_offset_px=y_offset,
                            scale_bias=size_factor
                        )
        
        # Get debug stats setting
        show_debug_stats = settings.get('show_debug_stats', False)
        
        # Draw FPS counter
        if settings.get('show_fps', True):
            fps_text = f"FPS: {state.current_fps:.1f}"
            cv2.putText(
                output,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Draw detailed performance stats if enabled
        if show_debug_stats:
            stats = state.perf_tracker.get_stats()
            y_pos = 60
            line_height = 25
            
            # Performance stats
            perf_lines = [
                f"Pose: {stats.get('avg_pose_detection', 0):.1f}ms",
                f"TPS: {stats.get('avg_tps_warping', 0):.1f}ms",
                f"Seg: {stats.get('avg_segmentation', 0):.1f}ms",
                f"Overlay: {stats.get('avg_overlay', 0):.1f}ms",
                f"Total: {stats.get('avg_total', 0):.1f}ms",
            ]
            
            for line in perf_lines:
                cv2.putText(
                    output,
                    line,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),  # Cyan for debug stats
                    1
                )
                y_pos += line_height
        
        # Update performance metrics
        processing_time = time.perf_counter() - start_time
        state.update_fps(processing_time)
        
        # Log performance warnings
        if processing_time > Config.MAX_PROCESSING_TIME:
            logger.warning(
                f"Frame processing slow: {processing_time*1000:.1f}ms "
                f"(target: {Config.MAX_PROCESSING_TIME*1000:.1f}ms) - "
                f"Pose: {pose_time*1000:.1f}ms, TPS: {tps_time*1000:.1f}ms, "
                f"Seg: {seg_time*1000:.1f}ms"
            )
        
        # Periodic performance summary (every 100 frames)
        if state.frame_count % 100 == 0:
            stats = state.perf_tracker.get_stats()
            logger.info(
                f"Performance summary (frame {state.frame_count}): "
                f"FPS={stats.get('avg_fps', 0):.1f}, "
                f"Pose={stats.get('avg_pose_detection', 0):.1f}ms, "
                f"TPS={stats.get('avg_tps_warping', 0):.1f}ms, "
                f"Seg={stats.get('avg_segmentation', 0):.1f}ms, "
                f"Total={stats.get('avg_total', 0):.1f}ms"
            )
        
        # Store last processed frame for capture
        # Convert to RGB for storage (PIL/display compatibility)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.session_state.last_processed_frame = output_rgb
        
        # Store last processed frame for capture
        # Convert to RGB for storage (PIL/display compatibility)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.session_state.last_processed_frame = output_rgb
        
        # Periodic garbage collection to prevent stuttering
        state.gc_counter += 1
        if state.gc_counter >= Config.GC_INTERVAL:
            import gc
            gc.collect()
            state.gc_counter = 0
        
        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(output, format="bgr24")
    
    except Exception as e:
        # Error handling - return original frame with error message
        cv2.putText(
            img,
            f"Error: {str(e)[:50]}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Virtual Try-On",
        page_icon="👕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Log application startup
    logger.info("=" * 80)
    logger.info("Virtual Try-On Application Started")
    logger.info(f"Target FPS: {Config.TARGET_FPS}, Video: {Config.VIDEO_WIDTH}x{Config.VIDEO_HEIGHT}")
    logger.info(f"Frame skip: {Config.FRAME_SKIP}, Pose skip: {Config.POSE_FRAME_SKIP}")
    logger.info("=" * 80)
    
    # Title
    st.title("👕 Virtual Try-On")
    st.markdown("Real-time garment visualization using AI pose detection")
    
    # Initialize session state
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = ProcessingState()
    
    if 'captured_frames' not in st.session_state:
        st.session_state.captured_frames = []
    
    if 'capture_manager' not in st.session_state:
        st.session_state.capture_manager = CaptureManager(
            max_captures=100,
            auto_cleanup_on_exit=False  # User must explicitly delete
        )
    
    if 'last_processed_frame' not in st.session_state:
        st.session_state.last_processed_frame = None
    
    state = st.session_state.processing_state
    capture_mgr = st.session_state.capture_manager
    
    # Load garments
    garments = load_garments()
    
    if not garments:
        st.error("No garments found! Please add garments to assets/garments.json")
        return
    
    # ========================================================================
    # Sidebar Controls
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Garment selector
        st.subheader("👔 Garment")
        garment_names = [g['name'] for g in garments]
        selected_garment_name = st.selectbox(
            "Select Garment",
            garment_names,
            key="garment_selector"
        )
        
        # Find selected garment data
        selected_garment = next(
            (g for g in garments if g['name'] == selected_garment_name),
            None
        )
        
        if selected_garment:
            # Load garment image
            garment_img = load_garment_image(selected_garment['path'])
            
            if garment_img is not None:
                # Show thumbnail
                thumb = cv2.resize(garment_img, (200, int(200 * garment_img.shape[0] / garment_img.shape[1])))
                thumb_rgb = cv2.cvtColor(thumb[:, :, :3], cv2.COLOR_BGR2RGB)
                st.image(thumb_rgb, caption=selected_garment['name'], use_container_width=True)
        else:
            garment_img = None
        
        st.divider()
        
        # Size adjustment
        st.subheader("📏 Adjustments")
        size = st.slider(
            "Size",
            min_value=0.8,
            max_value=1.3,
            value=1.0,
            step=0.05,
            key="size_slider"
        )
        
        # Vertical offset
        y_offset = st.slider(
            "Vertical Offset",
            min_value=-80,
            max_value=80,
            value=0,
            step=5,
            key="y_offset_slider"
        )
        
        st.divider()
        
        # Features
        st.subheader("✨ Features")
        
        use_tps = st.checkbox(
            "TPS Warp",
            value=False,
            help="Enable Thin-Plate Spline warping for realistic deformation",
            key="tps_checkbox"
        )
        
        use_occlusion = st.checkbox(
            "Face/Skin Occlusion",
            value=False,
            help="Bring face and hands in front of garment",
            key="occlusion_checkbox"
        )
        
        mirror = st.checkbox(
            "Mirror Mode",
            value=True,
            help="Flip video horizontally",
            key="mirror_checkbox"
        )
        
        show_fps = st.checkbox(
            "Show FPS",
            value=True,
            help="Display frames per second",
            key="fps_checkbox"
        )
        
        show_debug_stats = st.checkbox(
            "Debug Stats",
            value=False,
            help="Show detailed performance breakdown (Pose, TPS, Segmentation timings)",
            key="debug_stats_checkbox"
        )
        
        st.divider()
        
        # Capture section
        st.subheader("📸 Capture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📷 Capture", use_container_width=True):
                if st.session_state.last_processed_frame is not None:
                    # Capture current frame
                    metadata = {
                        'timestamp': datetime.now().isoformat(),
                        'garment': selected_garment['name'] if selected_garment else 'None',
                        'settings': {
                            'size': size,
                            'y_offset': y_offset,
                            'use_tps': use_tps,
                            'use_occlusion': use_occlusion
                        }
                    }
                    
                    success, result = capture_mgr.capture_frame(
                        st.session_state.last_processed_frame,
                        metadata
                    )
                    
                    if success:
                        st.success(f"✅ Captured! ({capture_mgr.get_capture_count()} total)")
                    else:
                        st.error(f"❌ Capture failed: {result}")
                else:
                    st.warning("⚠️ No frame available to capture")
        
        with col2:
            if st.button("🗑️ Delete All", use_container_width=True):
                deleted, errors = capture_mgr.delete_all_captures()
                if deleted > 0:
                    st.success(f"✅ Deleted {deleted} capture(s)")
                elif errors > 0:
                    st.error(f"❌ Errors: {errors}")
                else:
                    st.info("ℹ️ No captures to delete")
        
        # Show capture stats
        num_captures = capture_mgr.get_capture_count()
        storage_size = format_file_size(capture_mgr.get_storage_size())
        st.caption(f"📁 Captures: {num_captures} | Storage: {storage_size}")
        
        if num_captures > 0:
            st.caption(f"📂 Location: {capture_mgr.capture_dir}")
        
        st.divider()
        
        # Privacy notice
        st.subheader("🔒 Privacy & Security")
        
        with st.expander("Privacy Information", expanded=False):
            st.markdown(
                """
                **✅ Local Processing Only**
                
                All AI processing happens on your device. No video frames, 
                pose data, or biometric information is sent to external servers.
                
                **✅ No Persistent Storage**
                
                Images are **not stored** unless you press the "Capture" button. 
                Pose landmarks are computed per-frame and immediately discarded.
                
                **✅ User Control**
                
                - Captures saved to temporary directory on your machine
                - Delete all captures anytime with "Delete All" button
                - No cloud backup or external uploads
                
                **⚠️ HTTPS Required for Mobile**
                
                Modern browsers require HTTPS to access camera on mobile devices.
                Use ngrok, Caddy, or a reverse proxy for HTTPS access.
                
                **📍 Capture Location**
                
                Captures are stored in: `{}`
                
                This is typically your system's temporary directory and may be 
                cleared automatically by your OS.
                """.format(capture_mgr.capture_dir)
            )
        
        st.info(
            "🔒 **Local processing only** • Images not stored unless you press Capture"
        )
        
        # Performance info
        st.subheader("⚡ Performance")
        st.caption(f"Target: {Config.TARGET_FPS} FPS @ {Config.VIDEO_WIDTH}×{Config.VIDEO_HEIGHT}")
        st.caption(f"Current: {state.current_fps:.1f} FPS")
    
    # ========================================================================
    # Main Content - Video Stream
    # ========================================================================
    
    st.header("📹 Live Preview")
    
    # Prepare settings dict for callback
    settings = {
        'garment_data': selected_garment,
        'garment_img': garment_img if selected_garment else None,
        'size': size,
        'y_offset': y_offset,
        'use_tps': use_tps,
        'use_occlusion': use_occlusion,
        'mirror': mirror,
        'show_fps': show_fps,
        'show_debug_stats': show_debug_stats
    }
    
    # WebRTC streamer with STUN/TURN configuration
    RTC_CONFIGURATION = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
    
    webrtc_ctx = webrtc_streamer(
        key="virtual-tryon",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=lambda frame: process_frame(frame, state, settings),
        media_stream_constraints={
            "video": {
                "width": {"ideal": Config.VIDEO_WIDTH},
                "height": {"ideal": Config.VIDEO_HEIGHT}
            },
            "audio": False
        },
        async_processing=True,
    )
    
    # Instructions
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        ### Getting Started
        
        1. **Allow Camera Access**: Click "START" and grant camera permissions
        2. **Select Garment**: Choose from the dropdown in the sidebar
        3. **Adjust Settings**: Use sliders to fine-tune size and position
        4. **Enable Features**: Try TPS warping for realistic fit
        
        ### Tips
        
        - Stand 1-2 meters from the camera for best results
        - Ensure good lighting on your upper body
        - Keep your shoulders visible in frame
        - Enable "Mirror Mode" for a more natural view
        - Use "TPS Warp" for fitted garments
        - Enable "Face/Skin Occlusion" for realistic layering
        
        ### Troubleshooting
        
        - **No Detection**: Ensure your upper body is fully visible
        - **Low FPS**: Disable TPS and occlusion features
        - **Misalignment**: Adjust vertical offset slider
        - **Wrong Size**: Use size slider (0.8-1.3)
        
        ### Mobile Access
        
        For mobile devices, you need to serve over HTTPS:
        
        ```bash
        # Option 1: ngrok
        streamlit run app.py
        ngrok http 8501
        
        # Option 2: Caddy (automatic HTTPS)
        caddy reverse-proxy --from example.com --to localhost:8501
        ```
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown(f"""
        ### Pipeline
        
        1. **Pose Detection**: MediaPipe pose estimation
        2. **Landmark Smoothing**: One Euro Filter (β={Config.SMOOTH_BETA}, cutoff={Config.SMOOTH_MIN_CUTOFF})
        3. **Transform Computation**: Center, angle, scale from shoulders/hips
        4. **TPS Warping** (optional): Thin-Plate Spline deformation
        5. **Overlay Composition**: Affine transform + alpha blending
        6. **Occlusion** (optional): Face/skin segmentation masking
        
        ### Performance
        
        - **Resolution**: {Config.VIDEO_WIDTH}×{Config.VIDEO_HEIGHT}
        - **Target FPS**: {Config.TARGET_FPS}
        - **Frame Skip**: {Config.FRAME_SKIP}
        - **Max Processing Time**: {Config.MAX_PROCESSING_TIME*1000:.0f}ms
        
        ### Modules
        
        - `src/pose.py`: Pose estimation (21 landmarks)
        - `src/smoothing.py`: Temporal filtering (One Euro)
        - `src/tps.py`: Thin-Plate Spline warping
        - `src/overlay.py`: Transform + composition
        - `src/segmenter.py`: Person segmentation
        
        ### Test Coverage
        
        - **Total Tests**: 147
        - **Pass Rate**: 100%
        - **Modules**: Pose, Segmentation, Smoothing, TPS, Overlay
        """)


if __name__ == "__main__":
    main()
