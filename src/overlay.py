"""
Overlay Module - Garment Composition and Transform
===================================================

This module handles the final composition of garments onto video frames,
including rotation, scaling, translation, and alpha blending.

It provides both simple affine transforms (rotate/scale/translate) and
integration with TPS warping for more complex deformations.

Functions
---------
compute_transform(landmarks, garment_meta)
    Compute transformation parameters from pose landmarks.

render_overlay(frame_bgr, garment_rgba, transform, ...)
    Compose garment onto frame with affine transformation.

compose_with_alpha(background, foreground, alpha_mask)
    Alpha blend foreground onto background.

Workflow
--------
1. Detect pose landmarks
2. Compute transform from landmarks and garment metadata
3. Apply transform (simple) or TPS warp (complex) to garment
4. Compose warped garment onto frame with alpha blending
5. Optionally apply segmentation mask for occlusion

Authors: Pablo Tuñón Laguna, Lydia Ruiz Martínez
Date: 2025-01-19
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def compute_transform(
    landmarks: Dict[str, Tuple[float, float]],
    garment_meta: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute transformation parameters from pose landmarks.
    
    Calculates the center, rotation angle, and scale factors needed
    to position a garment on a person based on their pose.
    
    Parameters
    ----------
    landmarks : dict
        Pose landmarks with keys like 'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip'. Each value is (x, y) tuple in pixels.
    garment_meta : dict
        Garment metadata with keys:
        - 'base_shoulder_px': Reference shoulder width in garment image
        - 'base_torso_px': Reference torso height in garment image
        - 'y_offset_to_waist_px': Vertical offset to waist (optional)
    
    Returns
    -------
    dict
        Transform parameters:
        - 'center': (x, y) tuple - Shoulder midpoint in frame
        - 'angle_deg': float - Rotation angle in degrees
        - 'scale_x': float - Horizontal scale factor
        - 'scale_y': float - Vertical scale factor
        - 'scale': float - Average scale (for uniform scaling)
        - 'waist_y_offset': float - Y offset to waist line
    
    Examples
    --------
    >>> landmarks = {
    ...     'left_shoulder': (280, 300),
    ...     'right_shoulder': (360, 300),
    ...     'left_hip': (290, 450),
    ...     'right_hip': (350, 450)
    ... }
    >>> garment_meta = {
    ...     'base_shoulder_px': 200,
    ...     'base_torso_px': 300,
    ...     'y_offset_to_waist_px': 50
    ... }
    >>> transform = compute_transform(landmarks, garment_meta)
    >>> print(transform['center'])
    (320.0, 300.0)
    
    Notes
    -----
    - Rotation is computed from shoulder vector angle
    - Scale factors preserve aspect ratio by default
    - Center is the shoulder midpoint (garment anchor point)
    """
    # Extract landmarks
    left_shoulder = np.array(landmarks.get('left_shoulder', [0, 0]))
    right_shoulder = np.array(landmarks.get('right_shoulder', [0, 0]))
    left_hip = np.array(landmarks.get('left_hip', [0, 0]))
    right_hip = np.array(landmarks.get('right_hip', [0, 0]))
    
    # Compute shoulder midpoint (garment center)
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    
    # Compute shoulder vector and angle
    shoulder_vec = right_shoulder - left_shoulder
    angle_rad = math.atan2(shoulder_vec[1], shoulder_vec[0])
    angle_deg = math.degrees(angle_rad)
    
    # Compute shoulder distance (width)
    shoulder_dist = np.linalg.norm(shoulder_vec)
    
    # Compute torso height (shoulder to hip)
    hip_mid = (left_hip + right_hip) / 2.0
    torso_vec = hip_mid - shoulder_mid
    torso_height = np.linalg.norm(torso_vec)
    
    # Compute scale factors
    base_shoulder = garment_meta.get('base_shoulder_px', 200)
    base_torso = garment_meta.get('base_torso_px', 300)
    
    scale_x = shoulder_dist / base_shoulder if base_shoulder > 0 else 1.0
    scale_y = torso_height / base_torso if base_torso > 0 else 1.0
    
    # Average scale (for uniform scaling)
    scale = (scale_x + scale_y) / 2.0
    
    # Waist offset
    waist_y_offset = garment_meta.get('y_offset_to_waist_px', 0)
    
    return {
        'center': tuple(shoulder_mid),
        'angle_deg': angle_deg,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'scale': scale,
        'waist_y_offset': waist_y_offset
    }


def render_overlay(
    frame_bgr: np.ndarray,
    garment_rgba: np.ndarray,
    transform: Dict[str, float],
    y_offset_px: float = 0,
    scale_bias: float = 1.0,
    use_uniform_scale: bool = True
) -> np.ndarray:
    """
    Render garment overlay onto frame with affine transformation.
    
    Applies rotation, scaling, and translation to position a garment
    on a person, then alpha blends it onto the frame.
    
    Parameters
    ----------
    frame_bgr : np.ndarray
        Background frame in BGR format, shape (H, W, 3).
    garment_rgba : np.ndarray
        Garment image with alpha channel, shape (H, W, 4).
    transform : dict
        Transform parameters from compute_transform().
    y_offset_px : float, default=0
        Additional vertical offset in pixels (positive = down).
    scale_bias : float, default=1.0
        Scale multiplier (e.g., 1.1 for 10% larger).
    use_uniform_scale : bool, default=True
        If True, uses average scale for both X and Y.
        If False, uses separate scale_x and scale_y.
    
    Returns
    -------
    np.ndarray
        Composed image in BGR format, same shape as frame_bgr.
    
    Examples
    --------
    >>> frame = cv2.imread('frame.jpg')
    >>> garment = cv2.imread('shirt.png', cv2.IMREAD_UNCHANGED)
    >>> result = render_overlay(frame, garment, transform)
    
    Notes
    -----
    - Rotation is performed around the garment center
    - Alpha channel is used for transparent composition
    - Preserves aspect ratio by default (use_uniform_scale=True)
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"frame_bgr must be (H, W, 3), got {frame_bgr.shape}")
    
    if garment_rgba.ndim != 3 or garment_rgba.shape[2] != 4:
        raise ValueError(f"garment_rgba must be (H, W, 4), got {garment_rgba.shape}")
    
    # Create output (copy of frame)
    output = frame_bgr.copy()
    
    # Extract transform parameters
    center_x, center_y = transform['center']
    center_y += y_offset_px  # Apply additional offset
    
    angle_deg = transform['angle_deg']
    
    # Determine scale
    if use_uniform_scale:
        scale_x = scale_y = transform['scale'] * scale_bias
    else:
        scale_x = transform['scale_x'] * scale_bias
        scale_y = transform['scale_y'] * scale_bias
    
    # Get garment dimensions
    garment_h, garment_w = garment_rgba.shape[:2]
    garment_center = np.array([garment_w / 2.0, garment_h / 2.0])
    
    # Build transformation matrix (center -> rotate -> scale -> translate)
    # Step 1: Translate to origin
    M1 = np.array([
        [1, 0, -garment_center[0]],
        [0, 1, -garment_center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Step 2: Rotate
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    M2 = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Step 3: Scale
    M3 = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Step 4: Translate to target position
    M4 = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Combine transformations
    M = M4 @ M3 @ M2 @ M1
    
    # Extract 2x3 matrix for cv2.warpAffine
    M_affine = M[:2, :]
    
    # Warp garment
    frame_h, frame_w = frame_bgr.shape[:2]
    warped_rgba = cv2.warpAffine(
        garment_rgba,
        M_affine,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    # Split RGB and alpha
    warped_rgb = warped_rgba[:, :, :3]
    warped_alpha = warped_rgba[:, :, 3:4].astype(np.float32) / 255.0
    
    # Alpha blend
    output = output.astype(np.float32)
    warped_rgb = warped_rgb.astype(np.float32)
    
    output = output * (1.0 - warped_alpha) + warped_rgb * warped_alpha
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output


def compose_with_alpha(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Alpha blend foreground onto background.
    
    Supports both foreground with built-in alpha channel (RGBA/BGRA)
    or separate alpha mask. Preserves color channel order.
    
    Parameters
    ----------
    background : np.ndarray
        Background image, shape (H, W, 3) in BGR or RGB.
    foreground : np.ndarray
        Foreground image, shape (H, W, 3) or (H, W, 4).
        If 4 channels, uses built-in alpha (channel 3).
        Color order must match background.
    alpha_mask : np.ndarray, optional
        Separate alpha mask, shape (H, W) or (H, W, 1).
        Values 0-255 where 255 = fully opaque.
        If None and foreground has 4 channels, uses built-in alpha.
    
    Returns
    -------
    np.ndarray
        Composed image, same format as background.
    
    Examples
    --------
    >>> # With built-in alpha (BGRA)
    >>> result = compose_with_alpha(frame_bgr, garment_bgra)
    
    >>> # With separate mask
    >>> result = compose_with_alpha(frame_bgr, garment_bgr, alpha_mask)
    
    Notes
    -----
    Uses pre-multiplied alpha blending:
        result = fg * alpha + bg * (1 - alpha)
    
    Important: This function does NOT convert between BGR and RGB.
    It preserves whatever color order you provide.
    """
    if background.shape[:2] != foreground.shape[:2]:
        raise ValueError(
            f"Background and foreground size mismatch: "
            f"{background.shape[:2]} vs {foreground.shape[:2]}"
        )
    
    # Extract foreground RGB channels (before any conversion)
    if foreground.shape[2] == 4:
        fg_rgb = foreground[:, :, :3].copy()
    else:
        fg_rgb = foreground.copy()
    
    # Extract alpha channel
    if alpha_mask is not None:
        # Use provided mask
        if alpha_mask.ndim == 2:
            alpha = alpha_mask[:, :, np.newaxis]
        else:
            alpha = alpha_mask
    elif foreground.shape[2] == 4:
        # Use built-in alpha
        alpha = foreground[:, :, 3:4]
    else:
        # No alpha - fully opaque
        return fg_rgb
    
    # Normalize alpha to [0, 1]
    alpha = alpha.astype(np.float32) / 255.0
    
    # Alpha blend
    background = background.astype(np.float32)
    fg_rgb = fg_rgb.astype(np.float32)
    
    result = fg_rgb * alpha + background * (1.0 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def compose_with_occlusion(
    frame_bgr: np.ndarray,
    garment_rgba: np.ndarray,
    occlusion_mask: np.ndarray,
    transform: Dict[str, float],
    y_offset_px: float = 0,
    scale_bias: float = 1.0
) -> np.ndarray:
    """
    Compose garment with occlusion mask (e.g., face/hands over garment).
    
    This is the complete composition pipeline:
    1. Transform garment to match pose
    2. Blend garment onto frame
    3. Apply occlusion mask to bring body parts over garment
    
    Parameters
    ----------
    frame_bgr : np.ndarray
        Background frame in BGR format.
    garment_rgba : np.ndarray
        Garment with alpha channel.
    occlusion_mask : np.ndarray
        Binary mask (0/255) where 255 = occlude garment, shape (H, W).
    transform : dict
        Transform parameters.
    y_offset_px : float, default=0
        Vertical offset.
    scale_bias : float, default=1.0
        Scale multiplier.
    
    Returns
    -------
    np.ndarray
        Final composed image in BGR format.
    
    Examples
    --------
    >>> # Get segmentation mask for face/hands
    >>> from src.segmenter import PersonSegmenter
    >>> segmenter = PersonSegmenter()
    >>> occlusion_mask = segmenter.mask(frame, classes=['face', 'skin'])
    >>> 
    >>> # Compose with occlusion
    >>> result = compose_with_occlusion(
    ...     frame, garment, occlusion_mask, transform
    ... )
    
    Notes
    -----
    The occlusion mask brings body parts (face, hands) in front of
    the garment for realistic composition.
    """
    # Step 1: Render garment onto frame
    with_garment = render_overlay(
        frame_bgr, garment_rgba, transform,
        y_offset_px=y_offset_px,
        scale_bias=scale_bias
    )
    
    # Step 2: Apply occlusion mask
    # Where mask = 255, use original frame (body part visible)
    # Where mask = 0, use garment composite
    
    # Normalize mask to [0, 1]
    mask_norm = occlusion_mask.astype(np.float32) / 255.0
    if mask_norm.ndim == 2:
        mask_norm = mask_norm[:, :, np.newaxis]
    
    # Blend: original where masked, garment composite where not
    frame_bgr = frame_bgr.astype(np.float32)
    with_garment = with_garment.astype(np.float32)
    
    result = frame_bgr * mask_norm + with_garment * (1.0 - mask_norm)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def bgr_to_rgba(image_bgr: np.ndarray, alpha: int = 255) -> np.ndarray:
    """
    Convert BGR image to RGBA.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Image in BGR format, shape (H, W, 3).
    alpha : int, default=255
        Alpha value for all pixels (0-255).
    
    Returns
    -------
    np.ndarray
        Image in RGBA format, shape (H, W, 4).
    """
    if image_bgr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {image_bgr.shape[2]}")
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Add alpha channel
    h, w = rgb.shape[:2]
    alpha_channel = np.full((h, w, 1), alpha, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha_channel], axis=2)
    
    return rgba


def rgba_to_bgr(image_rgba: np.ndarray) -> np.ndarray:
    """
    Convert RGBA image to BGR (discards alpha).
    
    Parameters
    ----------
    image_rgba : np.ndarray
        Image in RGBA format, shape (H, W, 4).
    
    Returns
    -------
    np.ndarray
        Image in BGR format, shape (H, W, 3).
    """
    if image_rgba.shape[2] != 4:
        raise ValueError(f"Expected 4 channels, got {image_rgba.shape[2]}")
    
    # Extract RGB
    rgb = image_rgba[:, :, :3]
    
    # Convert RGB to BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    return bgr


def draw_transform_visualization(
    frame: np.ndarray,
    transform: Dict[str, float],
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Draw visualization of transform parameters on frame.
    
    Useful for debugging and demonstration.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame to draw on (will be copied).
    transform : dict
        Transform parameters.
    landmarks : dict, optional
        Original landmarks to visualize.
    
    Returns
    -------
    np.ndarray
        Frame with visualization overlay.
    """
    vis = frame.copy()
    
    # Draw center point
    center = transform['center']
    center_int = (int(center[0]), int(center[1]))
    cv2.circle(vis, center_int, 10, (0, 255, 0), -1)
    cv2.circle(vis, center_int, 11, (0, 0, 0), 2)
    
    # Draw rotation angle
    angle_deg = transform['angle_deg']
    angle_rad = math.radians(angle_deg)
    length = 100
    end_x = int(center[0] + length * math.cos(angle_rad))
    end_y = int(center[1] + length * math.sin(angle_rad))
    cv2.arrowedLine(vis, center_int, (end_x, end_y), (0, 255, 255), 3)
    
    # Draw scale box
    scale = transform['scale']
    box_size = int(50 * scale)
    pt1 = (center_int[0] - box_size, center_int[1] - box_size)
    pt2 = (center_int[0] + box_size, center_int[1] + box_size)
    cv2.rectangle(vis, pt1, pt2, (255, 0, 255), 2)
    
    # Draw landmarks if provided
    if landmarks:
        for name, (x, y) in landmarks.items():
            if 'shoulder' in name or 'hip' in name:
                color = (255, 255, 0) if 'shoulder' in name else (0, 255, 255)
                cv2.circle(vis, (int(x), int(y)), 5, color, -1)
                cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 0), 1)
    
    # Draw info text
    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"Angle: {angle_deg:.1f}°", (10, y),
               font, 0.6, (255, 255, 255), 2)
    y += 30
    cv2.putText(vis, f"Scale: {scale:.2f}x", (10, y),
               font, 0.6, (255, 255, 255), 2)
    
    return vis


if __name__ == '__main__':
    """
    Demonstration of overlay functionality.
    """
    print("=" * 70)
    print("Overlay Module - Garment Composition")
    print("=" * 70)
    
    # Test 1: Compute transform
    print("\n1. Transform Computation")
    print("-" * 70)
    
    landmarks = {
        'left_shoulder': (280, 300),
        'right_shoulder': (360, 300),
        'left_hip': (290, 450),
        'right_hip': (350, 450)
    }
    
    garment_meta = {
        'base_shoulder_px': 200,
        'base_torso_px': 300,
        'y_offset_to_waist_px': 50
    }
    
    transform = compute_transform(landmarks, garment_meta)
    
    print(f"Input landmarks:")
    for name, pos in landmarks.items():
        print(f"  {name}: {pos}")
    
    print(f"\nGarment metadata:")
    for key, val in garment_meta.items():
        print(f"  {key}: {val}")
    
    print(f"\nComputed transform:")
    for key, val in transform.items():
        if isinstance(val, tuple):
            print(f"  {key}: ({val[0]:.1f}, {val[1]:.1f})")
        else:
            print(f"  {key}: {val:.3f}")
    
    # Verify calculations
    shoulder_dist = np.linalg.norm(
        np.array(landmarks['right_shoulder']) - 
        np.array(landmarks['left_shoulder'])
    )
    expected_scale_x = shoulder_dist / garment_meta['base_shoulder_px']
    
    print(f"\nVerification:")
    print(f"  Shoulder distance: {shoulder_dist:.1f}px")
    print(f"  Expected scale_x: {expected_scale_x:.3f}")
    print(f"  Computed scale_x: {transform['scale_x']:.3f}")
    
    if abs(transform['scale_x'] - expected_scale_x) < 0.001:
        print("  [ok] Scale calculation correct")
    else:
        print("  [warn]  Scale mismatch")
    
    # Test 2: Alpha composition
    print("\n2. Alpha Composition")
    print("-" * 70)
    
    # Create test images
    bg = np.full((480, 640, 3), [100, 150, 200], dtype=np.uint8)
    fg_rgba = np.zeros((480, 640, 4), dtype=np.uint8)
    
    # Draw a circle on foreground
    cv2.circle(fg_rgba, (320, 240), 100, (255, 0, 0, 255), -1)
    cv2.circle(fg_rgba, (320, 240), 50, (0, 255, 0, 200), -1)
    
    print(f"Background shape: {bg.shape}")
    print(f"Foreground shape: {fg_rgba.shape}")
    
    # Compose
    result = compose_with_alpha(bg, fg_rgba)
    
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    
    # Check composition
    # Center should be green (fg), edges should be blue (bg)
    center_color = result[240, 320]
    edge_color = result[50, 50]
    
    print(f"\nColor at center (should be green): {center_color}")
    print(f"Color at edge (should be bg): {edge_color}")
    
    if center_color[1] > center_color[0]:  # More green than blue
        print("[ok] Alpha composition working")
    else:
        print("[warn]  Alpha composition issue")
    
    # Test 3: BGR/RGBA conversion
    print("\n3. Color Space Conversion")
    print("-" * 70)
    
    test_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Blue in BGR
    test_rgba = bgr_to_rgba(test_bgr, alpha=200)
    
    print(f"BGR: {test_bgr[0, 0]}")
    print(f"RGBA: {test_rgba[0, 0]}")
    
    # Should be red in RGB (with alpha)
    if test_rgba[0, 0, 0] == 255 and test_rgba[0, 0, 3] == 200:
        print("[ok] BGR to RGBA conversion correct")
    else:
        print("[warn]  Conversion issue")
    
    # Convert back
    back_bgr = rgba_to_bgr(test_rgba)
    print(f"Back to BGR: {back_bgr[0, 0]}")
    
    if np.array_equal(test_bgr, back_bgr):
        print("[ok] Round-trip conversion successful")
    else:
        print("[warn]  Round-trip mismatch")
    
    print("\n" + "=" * 70)
    print("[ok] Overlay module demonstration complete")
    print("=" * 70)
