"""
TPS Module - Thin-Plate Spline 2D Warping
==========================================

This module implements 2D Thin-Plate Spline (TPS) warping for real-time
garment deformation in virtual try-on applications.

TPS is a non-rigid transformation that smoothly interpolates between
control points while minimizing bending energy. It's ideal for warping
garments to fit body poses.

Classes
-------
ThinPlateSpline
    Efficient TPS implementation with precomputed factorization.

Theory
------
Thin-Plate Splines minimize the "bending energy" of a surface while
interpolating control points. The transformation is defined by:

    f(x, y) = a₁ + aₓx + aᵧy + Σᵢ wᵢU(|(x,y) - (xᵢ,yᵢ)|)

where U(r) = r² log(r) is the radial basis function.

The parameters are found by solving a linear system:
    L · [w; a] = [v; 0]

where L is the TPS matrix and v are the target coordinates.

Performance
-----------
- Precompute L matrix factorization once
- Reuse factorization for multiple warps
- Vectorized grid generation (no Python loops)
- Typical: 6-8 control points, <10ms per warp

References
----------
Bookstein, F. L. (1989). "Principal Warps: Thin-Plate Splines and the
Decomposition of Deformations." IEEE PAMI, 11(6), 567-585.

Authors: Pablo Tuñón Laguna, Lydia Ruiz Martínez
Date: 2025-01-19
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class ThinPlateSpline:
    """
    Thin-Plate Spline (TPS) for 2D image warping.
    
    This class efficiently computes TPS warps by precomputing and
    factorizing the TPS matrix based on source control points.
    
    The transformation can then be applied to any destination points
    without recomputing the factorization, making it suitable for
    real-time video processing.
    
    Parameters
    ----------
    src_pts : np.ndarray
        Source control points, shape (N, 2) where N >= 3.
        These are coordinates in the source image (e.g., garment PNG).
    
    Attributes
    ----------
    n_pts : int
        Number of control points.
    src_pts : np.ndarray
        Source control points (N, 2).
    L_inv : np.ndarray
        Precomputed inverse of TPS matrix L.
    params_x : np.ndarray or None
        Parameters for X transformation (set by fit()).
    params_y : np.ndarray or None
        Parameters for Y transformation (set by fit()).
    
    Examples
    --------
    Basic warping:
    
        >>> src = np.array([[0, 0], [256, 0], [256, 256], [0, 256]])
        >>> dst = np.array([[10, 20], [246, 15], [250, 240], [5, 245]])
        >>> tps = ThinPlateSpline(src)
        >>> warped = tps.warp_rgba(image, dst)
    
    Reuse for multiple warps:
    
        >>> tps = ThinPlateSpline(garment_control_points)
        >>> for pose in poses:
        ...     body_points = extract_body_points(pose)
        ...     warped = tps.warp_rgba(garment_image, body_points)
    
    Notes
    -----
    - Requires N >= 3 control points
    - Source points should span the image region
    - Destination points define the target shape
    - More control points = more flexible but slower
    - Typical use: 6-8 points for garment warping
    """
    
    def __init__(self, src_pts: np.ndarray, eps: float = 1e-6):
        """
        Initialize TPS with source control points.
        
        Precomputes and factorizes the TPS matrix L, which depends
        only on the relative positions of source control points.
        
        Parameters
        ----------
        src_pts : np.ndarray
            Source control points, shape (N, 2) where N >= 3.
            Coordinates should be in pixels (e.g., over garment PNG).
        eps : float, default=1e-6
            Small value for numerical stability in U(r) = r² log(r + eps).
        
        Raises
        ------
        ValueError
            If src_pts has invalid shape or too few points.
        
        Examples
        --------
        >>> src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> tps = ThinPlateSpline(src)
        """
        src_pts = np.asarray(src_pts, dtype=np.float64)
        
        if src_pts.ndim != 2 or src_pts.shape[1] != 2:
            raise ValueError(
                f"src_pts must have shape (N, 2), got {src_pts.shape}"
            )
        
        if src_pts.shape[0] < 3:
            raise ValueError(
                f"Need at least 3 control points, got {src_pts.shape[0]}"
            )
        
        self.n_pts = src_pts.shape[0]
        self.src_pts = src_pts
        self.eps = eps
        
        # Precompute TPS matrix and factorize
        self.L_inv = self._compute_L_inverse()
        
        # Parameters (computed by fit())
        self.params_x = None
        self.params_y = None
    
    def _compute_L_inverse(self) -> np.ndarray:
        """
        Compute and factorize the TPS matrix L.
        
        L has structure:
            L = [[K,   P  ],
                 [P^T, 0  ]]
        
        where:
        - K is (N, N) with K_ij = U(|src_i - src_j|)
        - P is (N, 3) with rows [1, x_i, y_i]
        - 0 is (3, 3) zero block
        
        Returns
        -------
        np.ndarray
            Inverse of L, shape (N+3, N+3).
        
        Notes
        -----
        This is the most expensive operation but only done once.
        For N=6, this is a 9×9 system.
        """
        N = self.n_pts
        
        # Build K matrix (N, N)
        # K_ij = U(|src_i - src_j|) where U(r) = r² log(r + eps)
        K = np.zeros((N, N), dtype=np.float64)
        
        for i in range(N):
            for j in range(i + 1, N):
                # Distance between control points
                r = np.linalg.norm(self.src_pts[i] - self.src_pts[j])
                
                # Radial basis U(r) = r² log(r + eps)
                if r > self.eps:
                    U_val = r * r * np.log(r + self.eps)
                else:
                    U_val = 0.0
                
                K[i, j] = U_val
                K[j, i] = U_val  # Symmetric
        
        # Build P matrix (N, 3)
        # Rows are [1, x_i, y_i]
        P = np.ones((N, 3), dtype=np.float64)
        P[:, 1] = self.src_pts[:, 0]  # x coordinates
        P[:, 2] = self.src_pts[:, 1]  # y coordinates
        
        # Build full L matrix (N+3, N+3)
        L = np.zeros((N + 3, N + 3), dtype=np.float64)
        L[:N, :N] = K
        L[:N, N:] = P
        L[N:, :N] = P.T
        # L[N:, N:] is already zero
        
        # Compute inverse (or use LU factorization for solve later)
        # For real-time, we precompute the inverse
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Singular matrix - add small regularization
            L += np.eye(N + 3) * 1e-8
            L_inv = np.linalg.inv(L)
        
        return L_inv
    
    def fit(self, dst_pts: np.ndarray):
        """
        Fit TPS parameters to map src_pts to dst_pts.
        
        Solves for transformation parameters that map each source
        control point to its corresponding destination point.
        
        Parameters
        ----------
        dst_pts : np.ndarray
            Destination control points, shape (N, 2).
            Must have same number of points as src_pts.
        
        Raises
        ------
        ValueError
            If dst_pts has wrong shape or point count.
        
        Examples
        --------
        >>> tps = ThinPlateSpline(src_pts)
        >>> tps.fit(dst_pts)
        >>> # Now ready to generate warped grid
        """
        dst_pts = np.asarray(dst_pts, dtype=np.float64)
        
        if dst_pts.shape != (self.n_pts, 2):
            raise ValueError(
                f"dst_pts must have shape ({self.n_pts}, 2), "
                f"got {dst_pts.shape}"
            )
        
        # Build right-hand side: [v_x; 0] and [v_y; 0]
        # where v_x and v_y are the destination coordinates
        rhs_x = np.zeros(self.n_pts + 3, dtype=np.float64)
        rhs_x[:self.n_pts] = dst_pts[:, 0]
        
        rhs_y = np.zeros(self.n_pts + 3, dtype=np.float64)
        rhs_y[:self.n_pts] = dst_pts[:, 1]
        
        # Solve using precomputed inverse
        # L · params = rhs  =>  params = L_inv · rhs
        self.params_x = self.L_inv @ rhs_x
        self.params_y = self.L_inv @ rhs_y
    
    def _U(self, r: np.ndarray) -> np.ndarray:
        """
        Radial basis function U(r) = r² log(r + eps).
        
        Parameters
        ----------
        r : np.ndarray
            Distances (can be any shape).
        
        Returns
        -------
        np.ndarray
            U(r) values, same shape as input.
        """
        # Avoid log(0) by adding eps
        r_safe = r + self.eps
        
        # U(r) = r² log(r + eps)
        U = r * r * np.log(r_safe)
        
        # Handle very small r
        U = np.where(r > self.eps, U, 0.0)
        
        return U
    
    def sample_grid(
        self,
        out_w: int,
        out_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dense sampling grid for cv2.remap.
        
        Computes the TPS transformation at every pixel location
        in the output image, creating inverse mapping arrays.
        
        Parameters
        ----------
        out_w : int
            Output width in pixels.
        out_h : int
            Output height in pixels.
        
        Returns
        -------
        map_x : np.ndarray
            X coordinates in source image, shape (out_h, out_w), dtype float32.
        map_y : np.ndarray
            Y coordinates in source image, shape (out_h, out_w), dtype float32.
        
        Notes
        -----
        The returned maps are "inverse" mappings (output→source) as required
        by cv2.remap. For each output pixel (x, y), the maps give the
        corresponding source coordinate to sample from.
        
        This uses vectorized NumPy operations for efficiency (no Python loops).
        
        Examples
        --------
        >>> tps = ThinPlateSpline(src_pts)
        >>> tps.fit(dst_pts)
        >>> map_x, map_y = tps.sample_grid(256, 256)
        >>> warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        """
        if self.params_x is None or self.params_y is None:
            raise RuntimeError("Must call fit() before sample_grid()")
        
        # Create output coordinate grid (vectorized)
        # y_grid has shape (out_h, out_w), x_grid has shape (out_h, out_w)
        y_coords = np.arange(out_h, dtype=np.float64)
        x_coords = np.arange(out_w, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Flatten grids for vectorized computation
        # shape: (out_h * out_w,)
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        n_pixels = len(x_flat)
        
        # Stack into (n_pixels, 2) array
        query_pts = np.stack([x_flat, y_flat], axis=1)
        
        # Compute affine part: a1 + ax*x + ay*y
        # params shape: (n_pts + 3,)
        # Last 3 params are [a1, ax, ay]
        affine_x = (
            self.params_x[-3] +
            self.params_x[-2] * x_flat +
            self.params_x[-1] * y_flat
        )
        affine_y = (
            self.params_y[-3] +
            self.params_y[-2] * x_flat +
            self.params_y[-1] * y_flat
        )
        
        # Compute non-rigid part: Σᵢ wᵢ U(|p - pᵢ|)
        # Vectorized computation for all control points
        nonrigid_x = np.zeros(n_pixels, dtype=np.float64)
        nonrigid_y = np.zeros(n_pixels, dtype=np.float64)
        
        # For each control point, compute contribution
        for i in range(self.n_pts):
            # Distance from all query points to control point i
            dx = x_flat - self.src_pts[i, 0]
            dy = y_flat - self.src_pts[i, 1]
            r = np.sqrt(dx * dx + dy * dy)
            
            # Apply radial basis
            U_vals = self._U(r)
            
            # Add weighted contribution
            nonrigid_x += self.params_x[i] * U_vals
            nonrigid_y += self.params_y[i] * U_vals
        
        # Combine affine + non-rigid
        result_x = affine_x + nonrigid_x
        result_y = affine_y + nonrigid_y
        
        # Reshape to (out_h, out_w)
        map_x = result_x.reshape(out_h, out_w).astype(np.float32)
        map_y = result_y.reshape(out_h, out_w).astype(np.float32)
        
        return map_x, map_y
    
    def warp_rgba(
        self,
        image_rgba: np.ndarray,
        dst_pts: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Warp RGBA image using TPS transformation.
        
        This is a convenience method that:
        1. Fits TPS to dst_pts
        2. Generates sampling grid
        3. Remaps each channel (including alpha)
        
        Parameters
        ----------
        image_rgba : np.ndarray
            Input image with alpha channel, shape (H, W, 4), dtype uint8.
        dst_pts : np.ndarray
            Destination control points, shape (N, 2).
        interpolation : int, default=cv2.INTER_LINEAR
            OpenCV interpolation method.
            - cv2.INTER_LINEAR: Bilinear (fast, smooth)
            - cv2.INTER_CUBIC: Bicubic (slower, smoother)
            - cv2.INTER_LANCZOS4: Lanczos (best quality, slowest)
        
        Returns
        -------
        np.ndarray
            Warped RGBA image, same shape as input, dtype uint8.
        
        Examples
        --------
        >>> src = np.array([[0, 0], [256, 0], [256, 256], [0, 256]])
        >>> dst = np.array([[10, 20], [246, 15], [250, 240], [5, 245]])
        >>> tps = ThinPlateSpline(src)
        >>> warped = tps.warp_rgba(garment_rgba, dst)
        
        Notes
        -----
        - Each channel is warped independently
        - Alpha channel is preserved and warped
        - Output size matches input size
        - Use INTER_CUBIC for better quality at ~2x cost
        """
        if image_rgba.ndim != 3 or image_rgba.shape[2] != 4:
            raise ValueError(
                f"image_rgba must have shape (H, W, 4), got {image_rgba.shape}"
            )
        
        # Fit transformation
        self.fit(dst_pts)
        
        # Get output size from input
        h, w = image_rgba.shape[:2]
        
        # Generate sampling grid
        map_x, map_y = self.sample_grid(w, h)
        
        # Warp each channel independently
        warped_channels = []
        for c in range(4):
            channel = image_rgba[:, :, c]
            warped_channel = cv2.remap(
                channel,
                map_x,
                map_y,
                interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            warped_channels.append(warped_channel)
        
        # Stack channels back together
        warped_rgba = np.stack(warped_channels, axis=2)
        
        return warped_rgba


def create_control_points_square(size: int = 256) -> np.ndarray:
    """
    Create control points for a square image.
    
    Generates 4 corner points + optional edge midpoints.
    
    Parameters
    ----------
    size : int, default=256
        Image size (assumes square).
    
    Returns
    -------
    np.ndarray
        Control points, shape (4, 2) for corners only.
    
    Examples
    --------
    >>> pts = create_control_points_square(256)
    >>> print(pts)
    [[  0   0]
     [255   0]
     [255 255]
     [  0 255]]
    """
    # 4 corners
    corners = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float64)
    
    return corners


def create_control_points_garment(
    width: int,
    height: int,
    n_shoulder_pts: int = 2,
    n_waist_pts: int = 2
) -> np.ndarray:
    """
    Create control points for a garment image.
    
    Places control points at key garment locations:
    - Shoulders (top)
    - Waist/hips (bottom)
    - Edges
    
    Parameters
    ----------
    width : int
        Garment image width.
    height : int
        Garment image height.
    n_shoulder_pts : int, default=2
        Number of points along shoulder line.
    n_waist_pts : int, default=2
        Number of points along waist line.
    
    Returns
    -------
    np.ndarray
        Control points, shape (N, 2).
    
    Examples
    --------
    >>> pts = create_control_points_garment(256, 384)
    >>> tps = ThinPlateSpline(pts)
    """
    pts = []
    
    # Top (shoulder) points
    for i in range(n_shoulder_pts):
        x = (i + 1) * width / (n_shoulder_pts + 1)
        pts.append([x, 0])
    
    # Bottom (waist) points
    for i in range(n_waist_pts):
        x = (i + 1) * width / (n_waist_pts + 1)
        pts.append([x, height - 1])
    
    # Side points (middle)
    pts.append([0, height / 2])
    pts.append([width - 1, height / 2])
    
    return np.array(pts, dtype=np.float64)


if __name__ == '__main__':
    """
    Demonstration of TPS warping functionality.
    """
    print("=" * 70)
    print("TPS Module - Thin-Plate Spline 2D Warping")
    print("=" * 70)
    
    # Test 1: Square to trapezoid
    print("\n1. Square → Trapezoid Transformation")
    print("-" * 70)
    
    size = 256
    
    # Create source control points (square corners)
    src_pts = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float64)
    
    # Create destination control points (trapezoid)
    # Narrower at top, wider at bottom
    dst_pts = np.array([
        [50, 20],           # Top-left (moved in and down)
        [size - 50, 20],    # Top-right (moved in and down)
        [size - 10, size - 10],  # Bottom-right (near corner)
        [10, size - 10]     # Bottom-left (near corner)
    ], dtype=np.float64)
    
    print(f"Source points (square):")
    print(src_pts)
    print(f"\nDestination points (trapezoid):")
    print(dst_pts)
    
    # Create TPS
    tps = ThinPlateSpline(src_pts)
    print(f"\n[ok] TPS initialized with {tps.n_pts} control points")
    
    # Fit to destination
    tps.fit(dst_pts)
    print("[ok] Fitted TPS transformation")
    
    # Test control point accuracy
    # Transform source points and compare to destination
    map_x, map_y = tps.sample_grid(size, size)
    
    errors = []
    for i, (sx, sy) in enumerate(src_pts):
        # Get transformed coordinate
        tx = map_x[int(sy), int(sx)]
        ty = map_y[int(sy), int(sx)]
        
        # Compare to destination
        dx, dy = dst_pts[i]
        error = np.sqrt((tx - dx)**2 + (ty - dy)**2)
        errors.append(error)
        
        print(f"  Point {i}: ({sx:.0f}, {sy:.0f}) → ({tx:.1f}, {ty:.1f}) "
              f"[target: ({dx:.0f}, {dy:.0f}), error: {error:.3f}px]")
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"\nControl point errors:")
    print(f"  Mean: {mean_error:.3f}px")
    print(f"  Max:  {max_error:.3f}px")
    
    if mean_error < 1.5:
        print("  [ok] Mean error < 1.5px (acceptable)")
    else:
        print("  [warn]  Mean error >= 1.5px")
    
    # Test 2: RGBA warping
    print("\n2. RGBA Channel Preservation")
    print("-" * 70)
    
    # Create synthetic RGBA image
    test_img = np.zeros((size, size, 4), dtype=np.uint8)
    test_img[:, :, 0] = 255  # Red channel
    test_img[:, :, 1] = 128  # Green channel
    test_img[:, :, 2] = 64   # Blue channel
    test_img[:, :, 3] = 200  # Alpha channel
    
    # Add a pattern to see warping
    cv2.circle(test_img, (size // 2, size // 2), 50, (255, 255, 0, 255), -1)
    
    print(f"Input image shape: {test_img.shape}")
    print(f"Input channels: R={test_img[0, 0, 0]}, G={test_img[0, 0, 1]}, "
          f"B={test_img[0, 0, 2]}, A={test_img[0, 0, 3]}")
    
    # Warp
    warped = tps.warp_rgba(test_img, dst_pts)
    
    print(f"\nWarped image shape: {warped.shape}")
    print(f"Warped dtype: {warped.dtype}")
    
    # Check channel preservation
    has_r = np.any(warped[:, :, 0] > 0)
    has_g = np.any(warped[:, :, 1] > 0)
    has_b = np.any(warped[:, :, 2] > 0)
    has_a = np.any(warped[:, :, 3] > 0)
    
    print(f"Channels preserved: R={has_r}, G={has_g}, B={has_b}, A={has_a}")
    
    if has_r and has_g and has_b and has_a:
        print("[ok] All RGBA channels preserved")
    else:
        print("[warn]  Some channels missing")
    
    # Test 3: Performance
    print("\n3. Performance Test")
    print("-" * 70)
    
    import time
    
    # Test with typical garment control points
    n_pts = 8
    src_8 = np.random.rand(n_pts, 2) * size
    dst_8 = src_8 + np.random.randn(n_pts, 2) * 10
    
    # Initialization (one-time)
    start = time.perf_counter()
    tps_perf = ThinPlateSpline(src_8)
    init_time = (time.perf_counter() - start) * 1000
    
    print(f"Initialization ({n_pts} points): {init_time:.2f}ms")
    
    # Warping (per-frame)
    n_warps = 100
    start = time.perf_counter()
    for _ in range(n_warps):
        warped = tps_perf.warp_rgba(test_img, dst_8)
    warp_time = (time.perf_counter() - start) * 1000 / n_warps
    
    print(f"Warping ({size}×{size}): {warp_time:.2f}ms per frame")
    print(f"Estimated FPS: {1000/warp_time:.1f}")
    
    if warp_time < 33:  # 30 FPS
        print("[ok] Real-time performance (>30 FPS)")
    else:
        print("[warn]  Below real-time (<30 FPS)")
    
    print("\n" + "=" * 70)
    print("[ok] TPS module demonstration complete")
    print("=" * 70)
