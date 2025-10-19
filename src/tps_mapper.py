"""
TPS Mapping Module
------------------
Computes per-frame destination control points from pose landmarks for TPS warping.

Implements heuristics for mapping garment control points to body landmarks:
- Shoulders: Direct mapping to pose shoulder landmarks
- Waist: Interpolation between shoulders and hips with lateral adjustment
- Hips: Hip landmarks with optional widening based on segmentation
- Hem/tail: Extrapolation from hip points

Supports optional pre-rotation/pre-scale using rigid transform before TPS.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TPSMapper:
    """
    Maps garment control points to body pose landmarks for TPS warping.
    
    Computes destination points frame-by-frame based on pose landmarks,
    enabling real-time garment deformation to match body pose and proportions.
    """
    
    def __init__(
        self,
        waist_alpha: float = 0.6,
        waist_lateral_factor: float = 0.08,
        hip_widening_factor: float = 0.05,
        enable_pre_transform: bool = False
    ):
        """
        Initialize TPS mapper with configurable heuristics.
        
        Args:
            waist_alpha: Interpolation factor for waist position [0,1]
                        0 = at shoulders, 1 = at hips, 0.6 = 60% toward hips
            waist_lateral_factor: Lateral adjustment for waist points as 
                                 fraction of shoulder distance (±0.08 typical)
            hip_widening_factor: Hip widening factor based on shoulder distance
                                (0.05 = 5% expansion for loose garments)
            enable_pre_transform: If True, apply rigid transform before TPS
        """
        self.waist_alpha = waist_alpha
        self.waist_lateral_factor = waist_lateral_factor
        self.hip_widening_factor = hip_widening_factor
        self.enable_pre_transform = enable_pre_transform
        
        logger.info(
            f"TPSMapper initialized: waist_alpha={waist_alpha}, "
            f"lateral={waist_lateral_factor}, widening={hip_widening_factor}"
        )
    
    def compute_dst_points(
        self,
        landmarks: Dict[str, Tuple[float, float]],
        garment_metadata: Dict,
        segmentation_width: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute destination control points from pose landmarks.
        
        Args:
            landmarks: Dictionary of pose landmarks, e.g.:
                      {'left_shoulder': (x, y), 'right_shoulder': (x, y),
                       'left_hip': (x, y), 'right_hip': (x, y), ...}
            garment_metadata: Garment metadata dict with 'y_offset_to_waist_px'
            segmentation_width: Optional torso width from segmentation mask
                               (used for hip widening adjustment)
        
        Returns:
            dst_pts: Nx2 array of destination control points matching the
                    order of source points in garment's tps_control_points
        
        Raises:
            ValueError: If required landmarks are missing
        """
        # Validate required landmarks
        required = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        missing = [lm for lm in required if lm not in landmarks]
        if missing:
            raise ValueError(f"Missing required landmarks: {missing}")
        
        # Extract landmark positions
        l_shoulder = np.array(landmarks['left_shoulder'])
        r_shoulder = np.array(landmarks['right_shoulder'])
        l_hip = np.array(landmarks['left_hip'])
        r_hip = np.array(landmarks['right_hip'])
        
        # Compute reference points
        mid_shoulders = (l_shoulder + r_shoulder) / 2.0
        mid_hips = (l_hip + r_hip) / 2.0
        shoulder_distance = np.linalg.norm(r_shoulder - l_shoulder)
        
        # 1. Shoulder points: Direct mapping
        dst_l_shoulder = l_shoulder.copy()
        dst_r_shoulder = r_shoulder.copy()
        
        # 2. Waist points: Interpolation with lateral adjustment
        y_offset = garment_metadata.get('y_offset_to_waist_px', 0)
        mid_waist = self._lerp(mid_shoulders, mid_hips, self.waist_alpha)
        mid_waist[1] += y_offset  # Apply vertical offset
        
        lateral_shift = self.waist_lateral_factor * shoulder_distance
        dst_l_waist = mid_waist + np.array([-lateral_shift, 0])
        dst_r_waist = mid_waist + np.array([lateral_shift, 0])
        
        # 3. Hip points: With optional widening
        hip_widening = 0.0
        if segmentation_width is not None:
            # If segmentation indicates wider torso, expand hips
            estimated_width = shoulder_distance * 1.2  # Typical hip/shoulder ratio
            if segmentation_width > estimated_width:
                hip_widening = self.hip_widening_factor * shoulder_distance
        
        dst_l_hip = l_hip + np.array([-hip_widening, 0])
        dst_r_hip = r_hip + np.array([hip_widening, 0])
        
        # Assemble destination points in standard order:
        # [L_shoulder_seam, R_shoulder_seam, L_waist, R_waist, L_hip, R_hip]
        dst_pts = np.array([
            dst_l_shoulder,
            dst_r_shoulder,
            dst_l_waist,
            dst_r_waist,
            dst_l_hip,
            dst_r_hip
        ], dtype=np.float32)
        
        logger.debug(
            f"Computed {len(dst_pts)} destination points: "
            f"shoulder_dist={shoulder_distance:.1f}, widening={hip_widening:.1f}"
        )
        
        return dst_pts
    
    def compute_dst_points_with_hem(
        self,
        landmarks: Dict[str, Tuple[float, float]],
        garment_metadata: Dict,
        hem_extension: float = 100.0,
        segmentation_width: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute destination points including hem/tail extrapolation.
        
        For longer garments (dresses, skirts), adds hem points below hips.
        
        Args:
            landmarks: Pose landmarks dictionary
            garment_metadata: Garment metadata dict
            hem_extension: Vertical distance below hips for hem (pixels)
            segmentation_width: Optional torso width from segmentation
        
        Returns:
            dst_pts: Nx2 array with additional hem points
        """
        # Get base 6 points
        base_pts = self.compute_dst_points(
            landmarks, garment_metadata, segmentation_width
        )
        
        # Extract hip points (indices 4, 5)
        l_hip = base_pts[4]
        r_hip = base_pts[5]
        
        # Extrapolate hem points vertically
        dst_l_hem = l_hip + np.array([0, hem_extension])
        dst_r_hem = r_hip + np.array([0, hem_extension])
        
        # Append hem points: [6 base points + 2 hem points]
        dst_pts_with_hem = np.vstack([base_pts, [dst_l_hem, dst_r_hem]])
        
        logger.debug(f"Added hem points at {hem_extension}px below hips")
        
        return dst_pts_with_hem
    
    def apply_pre_transform(
        self,
        garment_img: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        overlay_module
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rigid pre-transform (rotation + scale) before TPS warping.
        
        This reduces stress on TPS by handling global rotation/scale separately,
        leaving only residual deformation for TPS.
        
        Args:
            garment_img: RGBA garment image
            src_pts: Source control points from garment
            dst_pts: Destination control points from pose
            overlay_module: Instance of OverlayModule with compute_transform
        
        Returns:
            (transformed_img, adjusted_src_pts): Pre-transformed garment and
                                                 adjusted source points
        """
        if not self.enable_pre_transform:
            return garment_img, src_pts
        
        # Compute rigid transform using overlay module
        # Use first 2 points (shoulders) for alignment
        transform_matrix = overlay_module.compute_transform(
            src_pts[:2], dst_pts[:2], garment_img.shape[:2]
        )
        
        # Apply transform to garment image
        import cv2
        h, w = garment_img.shape[:2]
        transformed_img = cv2.warpAffine(
            garment_img, transform_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        # Transform source points to match
        src_pts_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        adjusted_src_pts = (transform_matrix @ src_pts_homo.T).T
        
        logger.debug("Applied pre-transform (rigid rotation + scale)")
        
        return transformed_img, adjusted_src_pts.astype(np.float32)
    
    def visualize_mapping(
        self,
        frame: np.ndarray,
        dst_pts: np.ndarray,
        draw_labels: bool = True
    ) -> np.ndarray:
        """
        Draw destination points on frame for real-time validation.
        
        Args:
            frame: Video frame (RGB or BGR)
            dst_pts: Destination control points
            draw_labels: If True, draw point labels
        
        Returns:
            frame_with_overlay: Frame with points and labels drawn
        """
        import cv2
        
        overlay = frame.copy()
        labels = ['L_Shoulder', 'R_Shoulder', 'L_Waist', 'R_Waist', 'L_Hip', 'R_Hip']
        
        # Extend labels if more points exist (e.g., hem)
        if len(dst_pts) > 6:
            for i in range(6, len(dst_pts)):
                labels.append(f'Point_{i}')
        
        # Draw points
        for i, pt in enumerate(dst_pts):
            x, y = int(pt[0]), int(pt[1])
            
            # Draw circle
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 6, (0, 0, 0), 1)
            
            # Draw label
            if draw_labels and i < len(labels):
                cv2.putText(
                    overlay, labels[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2
                )
                cv2.putText(
                    overlay, labels[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
        
        # Draw connecting lines for structure
        pairs = [(0, 1), (2, 3), (4, 5)]  # Shoulders, waist, hips
        for i, j in pairs:
            if i < len(dst_pts) and j < len(dst_pts):
                pt1 = tuple(dst_pts[i].astype(int))
                pt2 = tuple(dst_pts[j].astype(int))
                cv2.line(overlay, pt1, pt2, (255, 255, 0), 1)
        
        return overlay
    
    def update_params(
        self,
        waist_alpha: Optional[float] = None,
        waist_lateral_factor: Optional[float] = None,
        hip_widening_factor: Optional[float] = None
    ):
        """
        Update mapping parameters for real-time tuning.
        
        Args:
            waist_alpha: New waist interpolation factor
            waist_lateral_factor: New lateral adjustment factor
            hip_widening_factor: New hip widening factor
        """
        if waist_alpha is not None:
            self.waist_alpha = waist_alpha
            logger.info(f"Updated waist_alpha to {waist_alpha}")
        
        if waist_lateral_factor is not None:
            self.waist_lateral_factor = waist_lateral_factor
            logger.info(f"Updated waist_lateral_factor to {waist_lateral_factor}")
        
        if hip_widening_factor is not None:
            self.hip_widening_factor = hip_widening_factor
            logger.info(f"Updated hip_widening_factor to {hip_widening_factor}")
    
    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation: a + t * (b - a)"""
        return a + t * (b - a)
    
    def get_control_point_names(self, num_points: int = 6) -> List[str]:
        """
        Get standard control point names for given count.
        
        Args:
            num_points: Number of control points (6 or 8)
        
        Returns:
            List of control point names
        """
        base_names = [
            'L_shoulder_seam', 'R_shoulder_seam',
            'L_waist', 'R_waist',
            'L_hip', 'R_hip'
        ]
        
        if num_points == 6:
            return base_names
        elif num_points == 8:
            return base_names + ['L_hem', 'R_hem']
        else:
            return base_names + [f'Point_{i}' for i in range(6, num_points)]


def create_mapper(config: Optional[Dict] = None) -> TPSMapper:
    """
    Factory function to create TPSMapper with optional config.
    
    Args:
        config: Optional configuration dict with keys:
               - waist_alpha: float
               - waist_lateral_factor: float
               - hip_widening_factor: float
               - enable_pre_transform: bool
    
    Returns:
        Configured TPSMapper instance
    """
    if config is None:
        config = {}
    
    return TPSMapper(
        waist_alpha=config.get('waist_alpha', 0.6),
        waist_lateral_factor=config.get('waist_lateral_factor', 0.08),
        hip_widening_factor=config.get('hip_widening_factor', 0.05),
        enable_pre_transform=config.get('enable_pre_transform', False)
    )
