"""
Smoothing Module - Anti-Jitter Filters for Temporal Landmark Stabilization
===========================================================================

This module provides temporal filtering for pose landmarks to reduce jitter
and noise while maintaining responsiveness and minimal latency (<50ms).

Implements:
- One Euro Filter: Adaptive low-pass filter balancing noise reduction and lag
- EMA (Exponential Moving Average): Simple temporal smoothing

Classes
-------
OneEuroFilter
    Adaptive low-pass filter for single point or landmark dict smoothing.

Functions
---------
ema_point(pt, alpha, prev=None)
    Simple exponential moving average for a single point.

Usage
-----
Basic One Euro Filter:
    >>> filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, fps=30)
    >>> smoothed = filter.filter_point((100, 200), t=0.0)
    >>> smoothed = filter.filter_point((102, 198), t=0.033)

Filter multiple landmarks:
    >>> landmarks = {
    ...     'nose': (320, 240),
    ...     'left_shoulder': (280, 300),
    ...     'right_shoulder': (360, 300)
    ... }
    >>> smoothed = filter.filter_landmarks(landmarks, t=0.0)

EMA smoothing:
    >>> from src.smoothing import ema_point
    >>> smoothed = ema_point((100, 200), alpha=0.3, prev=(95, 205))

Theory
------
One Euro Filter:
    Combines two exponential filters:
    1. Low-pass filter for position (reduces jitter)
    2. Low-pass filter for velocity (adapts cutoff frequency)
    
    When movement is slow: Strong smoothing (low jitter)
    When movement is fast: Less smoothing (low lag)
    
    Parameters:
    - min_cutoff: Minimum cutoff frequency (Hz) - controls smoothing strength
    - beta: Speed coefficient - how much cutoff increases with velocity
    - dcutoff: Cutoff frequency for derivative - controls velocity smoothing

References
----------
Casiez, G., Roussel, N., & Vogel, D. (2012).
"1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
CHI '12: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems

Authors: Pablo Tuñón Laguna, Lydia Ruiz Martínez
Date: 2025-01-19
"""

import math
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np


# Type aliases
Point = Union[Tuple[float, float], Tuple[float, float, float]]
Landmarks = Dict[str, Point]


class LowPassFilter:
    """
    Simple low-pass filter using exponential smoothing.
    
    This is an internal helper for OneEuroFilter. It applies
    exponential smoothing with a cutoff frequency.
    
    Parameters
    ----------
    alpha : float
        Smoothing factor between 0 and 1.
        Higher alpha = less smoothing (more responsive).
    
    Attributes
    ----------
    y : float or None
        Previous filtered value.
    """
    
    def __init__(self, alpha: float):
        """Initialize low-pass filter with smoothing factor."""
        self.alpha = alpha
        self.y = None
    
    def __call__(self, x: float, alpha: Optional[float] = None) -> float:
        """
        Apply filter to new value.
        
        Parameters
        ----------
        x : float
            New input value.
        alpha : float, optional
            Override smoothing factor for this call.
        
        Returns
        -------
        float
            Filtered value.
        """
        if alpha is not None:
            self.alpha = alpha
        
        if self.y is None:
            # First value - no history
            self.y = x
        else:
            # Exponential smoothing
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        
        return self.y


class OneEuroFilter:
    """
    One Euro Filter - Adaptive low-pass filter for jitter reduction.
    
    This filter automatically adjusts its cutoff frequency based on the
    speed of movement. Fast movements get less smoothing (lower lag),
    while slow movements get more smoothing (lower jitter).
    
    The "One Euro" name comes from the original paper's claim that the
    filter is simple enough to implement in one Euro's worth of time.
    
    Parameters
    ----------
    min_cutoff : float, default=1.0
        Minimum cutoff frequency (Hz). Lower values = more smoothing.
        Typical range: 0.1 to 10.0
    beta : float, default=0.0
        Speed coefficient. Higher values = more adaptation to velocity.
        Typical range: 0.0 to 1.0
    dcutoff : float, default=1.0
        Cutoff frequency for derivative filter (Hz).
        Controls how much velocity estimate is smoothed.
    fps : float, default=30
        Expected frame rate for timestamp conversion.
        Only used if timestamps are None.
    
    Attributes
    ----------
    x_filter : LowPassFilter
        Filter for position values.
    dx_filter : LowPassFilter
        Filter for velocity (derivative) values.
    
    Examples
    --------
    Filter a noisy point sequence:
    
        >>> filter = OneEuroFilter(min_cutoff=1.0, beta=0.7, fps=30)
        >>> points = [(100, 200), (102, 198), (99, 201), (101, 199)]
        >>> t = 0.0
        >>> for pt in points:
        ...     smoothed = filter.filter_point(pt, t)
        ...     t += 1.0/30  # 30 FPS
        ...     print(smoothed)
    
    Filter pose landmarks:
    
        >>> landmarks = {'nose': (320, 240), 'left_eye': (300, 220)}
        >>> smoothed = filter.filter_landmarks(landmarks, t=0.0)
    
    Notes
    -----
    Latency is typically <10ms on modern hardware for <100 landmarks.
    The filter maintains state between calls, so it should be reused
    across frames rather than recreated each time.
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
        fps: float = 30
    ):
        """Initialize One Euro Filter with parameters."""
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.fps = fps
        
        # Create filters for position and velocity
        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(dcutoff))
        
        # State
        self.prev_x = None
        self.prev_t = None
    
    def _alpha(self, cutoff: float) -> float:
        """
        Calculate smoothing factor from cutoff frequency.
        
        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz.
        
        Returns
        -------
        float
            Smoothing factor alpha between 0 and 1.
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.fps
        return 1.0 / (1.0 + tau / te)
    
    def filter_scalar(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a single scalar value.
        
        Parameters
        ----------
        x : float
            Input value to filter.
        t : float, optional
            Timestamp in seconds. If None, uses 1/fps intervals.
        
        Returns
        -------
        float
            Filtered value.
        """
        # Calculate time delta
        if t is None:
            dt = 1.0 / self.fps
        elif self.prev_t is None:
            dt = 1.0 / self.fps
        else:
            dt = t - self.prev_t
        
        # Prevent division by zero
        if dt <= 0:
            dt = 1.0 / self.fps
        
        # Calculate velocity
        if self.prev_x is None:
            dx = 0.0
        else:
            dx = (x - self.prev_x) / dt
        
        # Smooth velocity
        edx = self.dx_filter(dx, self._alpha(self.dcutoff))
        
        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Filter position with adaptive cutoff
        filtered_x = self.x_filter(x, self._alpha(cutoff))
        
        # Update state
        self.prev_x = filtered_x
        self.prev_t = t
        
        return filtered_x
    
    def filter_point(
        self,
        pt: Point,
        t: Optional[float] = None
    ) -> Point:
        """
        Filter a 2D or 3D point.
        
        Each coordinate is filtered independently using the same
        velocity-adaptive cutoff frequency.
        
        Parameters
        ----------
        pt : tuple of float
            Point as (x, y) or (x, y, z).
        t : float, optional
            Timestamp in seconds. If None, uses 1/fps intervals.
        
        Returns
        -------
        tuple of float
            Filtered point with same dimensionality as input.
        
        Examples
        --------
        >>> filter = OneEuroFilter()
        >>> smoothed = filter.filter_point((100.5, 200.3), t=0.0)
        >>> smoothed = filter.filter_point((102.1, 198.7), t=0.033)
        """
        # Create separate filter for each dimension
        if not hasattr(self, '_point_filters'):
            self._point_filters = {}
        
        # Get or create filter for this point dimension
        key = 'point'
        if key not in self._point_filters:
            self._point_filters[key] = [
                OneEuroFilter(
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    dcutoff=self.dcutoff,
                    fps=self.fps
                )
                for _ in range(len(pt))
            ]
        
        # Filter each coordinate
        filtered = tuple(
            self._point_filters[key][i].filter_scalar(pt[i], t)
            for i in range(len(pt))
        )
        
        return filtered
    
    def filter_landmarks(
        self,
        landmarks: Landmarks,
        t: Optional[float] = None
    ) -> Landmarks:
        """
        Filter a dictionary of landmarks.
        
        Each landmark is filtered independently. The filter maintains
        separate state for each landmark key.
        
        Parameters
        ----------
        landmarks : dict
            Dictionary mapping landmark names to (x, y) or (x, y, z) tuples.
        t : float, optional
            Timestamp in seconds. If None, uses 1/fps intervals.
        
        Returns
        -------
        dict
            Dictionary with same keys, containing filtered landmark positions.
        
        Examples
        --------
        >>> filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        >>> landmarks = {
        ...     'nose': (320, 240),
        ...     'left_shoulder': (280, 300),
        ...     'right_shoulder': (360, 300)
        ... }
        >>> smoothed = filter.filter_landmarks(landmarks, t=0.0)
        >>> print(smoothed['nose'])
        (320.0, 240.0)
        
        Notes
        -----
        The filter creates and maintains separate state for each landmark
        key. If a landmark disappears and reappears, its filter state is
        preserved.
        """
        # Create separate filter for each landmark
        if not hasattr(self, '_landmark_filters'):
            self._landmark_filters = {}
        
        smoothed = {}
        
        for name, pt in landmarks.items():
            # Get or create filter for this landmark
            if name not in self._landmark_filters:
                self._landmark_filters[name] = [
                    OneEuroFilter(
                        min_cutoff=self.min_cutoff,
                        beta=self.beta,
                        dcutoff=self.dcutoff,
                        fps=self.fps
                    )
                    for _ in range(len(pt))
                ]
            
            # Filter each coordinate
            filtered = tuple(
                self._landmark_filters[name][i].filter_scalar(pt[i], t)
                for i in range(len(pt))
            )
            
            smoothed[name] = filtered
        
        return smoothed
    
    def reset(self):
        """
        Reset filter state.
        
        Call this when starting a new tracking sequence or when
        there's been a discontinuity in the input data.
        """
        self.x_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(self.dcutoff))
        self.prev_x = None
        self.prev_t = None
        
        # Reset point and landmark filters
        if hasattr(self, '_point_filters'):
            self._point_filters = {}
        if hasattr(self, '_landmark_filters'):
            self._landmark_filters = {}


def ema_point(
    pt: Point,
    alpha: float,
    prev: Optional[Point] = None
) -> Point:
    """
    Apply exponential moving average to a point.
    
    Simple temporal smoothing using exponential moving average (EMA).
    This is a lightweight alternative to OneEuroFilter when you don't
    need adaptive behavior.
    
    Formula: result = alpha * pt + (1 - alpha) * prev
    
    Parameters
    ----------
    pt : tuple of float
        Current point as (x, y) or (x, y, z).
    alpha : float
        Smoothing factor between 0 and 1.
        - 0.0 = maximum smoothing (only use previous value)
        - 1.0 = no smoothing (only use current value)
        - 0.3 = typical value for moderate smoothing
    prev : tuple of float, optional
        Previous smoothed point. If None, returns pt unchanged.
    
    Returns
    -------
    tuple of float
        Smoothed point with same dimensionality as input.
    
    Examples
    --------
    >>> pt = (100.0, 200.0)
    >>> prev = (95.0, 205.0)
    >>> smoothed = ema_point(pt, alpha=0.3, prev=prev)
    >>> print(smoothed)
    (96.5, 203.5)
    
    Without previous value (first frame):
    >>> smoothed = ema_point((100.0, 200.0), alpha=0.3, prev=None)
    >>> print(smoothed)
    (100.0, 200.0)
    
    Notes
    -----
    EMA is simpler and faster than OneEuroFilter but doesn't adapt to
    velocity. Use EMA when:
    - You need very low latency (<1ms)
    - Movement speed is relatively constant
    - You want predictable smoothing behavior
    
    Use OneEuroFilter when:
    - Movement speed varies significantly
    - You need to balance jitter reduction and responsiveness
    - Latency <50ms is acceptable
    """
    if prev is None:
        return pt
    
    # Apply EMA to each coordinate
    smoothed = tuple(
        alpha * pt[i] + (1.0 - alpha) * prev[i]
        for i in range(len(pt))
    )
    
    return smoothed


def calculate_jitter(points: list, window: int = 5) -> float:
    """
    Calculate jitter metric for a sequence of points.
    
    Jitter is measured as the average acceleration magnitude,
    which indicates how much the velocity is changing.
    
    Parameters
    ----------
    points : list of tuple
        Sequence of (x, y) or (x, y, z) points.
    window : int, default=5
        Window size for calculating local jitter.
    
    Returns
    -------
    float
        Jitter metric (average acceleration magnitude).
        Lower values = smoother motion.
    
    Examples
    --------
    >>> noisy = [(100, 200), (102, 198), (99, 201), (101, 199)]
    >>> jitter = calculate_jitter(noisy)
    >>> print(f"Jitter: {jitter:.2f}")
    
    Notes
    -----
    This is useful for quantifying the effectiveness of smoothing.
    Compare jitter before and after filtering to measure improvement.
    """
    if len(points) < 3:
        return 0.0
    
    # Convert to numpy for easier calculation
    pts = np.array(points)
    
    # Calculate velocity (first derivative)
    velocity = np.diff(pts, axis=0)
    
    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    
    # Jitter is average acceleration magnitude
    jitter = np.mean(np.linalg.norm(acceleration, axis=1))
    
    return float(jitter)


def calculate_lag(
    original: list,
    filtered: list,
    fps: float = 30
) -> float:
    """
    Calculate lag (latency) introduced by filtering.
    
    Lag is estimated by finding the time shift that maximizes
    cross-correlation between original and filtered signals.
    
    Parameters
    ----------
    original : list of tuple
        Original unfiltered points.
    filtered : list of tuple
        Filtered points.
    fps : float, default=30
        Frame rate for converting lag to milliseconds.
    
    Returns
    -------
    float
        Estimated lag in milliseconds.
    
    Examples
    --------
    >>> original = [(i, 100) for i in range(100)]
    >>> filtered = [(i-1, 100) for i in range(100)]
    >>> lag = calculate_lag(original, filtered, fps=30)
    >>> print(f"Lag: {lag:.1f}ms")
    
    Notes
    -----
    This is useful for verifying that filtering doesn't introduce
    perceptible latency (>50ms). The target is <50ms for interactive
    applications.
    """
    if len(original) != len(filtered) or len(original) < 2:
        return 0.0
    
    # Extract x-coordinates for correlation
    orig_x = np.array([pt[0] for pt in original])
    filt_x = np.array([pt[0] for pt in filtered])
    
    # Calculate cross-correlation
    correlation = np.correlate(orig_x, filt_x, mode='full')
    
    # Find lag (in frames)
    center = len(correlation) // 2
    lag_frames = np.argmax(correlation) - center
    
    # Convert to milliseconds
    lag_ms = abs(lag_frames) * 1000.0 / fps
    
    return float(lag_ms)


if __name__ == '__main__':
    """
    Demonstration of smoothing functionality.
    """
    print("=" * 70)
    print("Smoothing Module - Anti-Jitter Filters")
    print("=" * 70)
    
    # Generate noisy signal
    np.random.seed(42)
    t_values = np.linspace(0, 2, 60)  # 2 seconds at 30 FPS
    clean_signal = np.sin(2 * np.pi * t_values)
    noise = np.random.normal(0, 0.1, len(t_values))
    noisy_signal = clean_signal + noise
    
    print("\n1. One Euro Filter Demo")
    print("-" * 70)
    
    # Apply One Euro Filter
    filter_1euro = OneEuroFilter(min_cutoff=1.0, beta=0.7, fps=30)
    filtered_1euro = []
    
    for i, val in enumerate(noisy_signal):
        pt = (val, 0)
        smoothed = filter_1euro.filter_point(pt, t=t_values[i])
        filtered_1euro.append(smoothed[0])
    
    # Calculate jitter reduction
    noisy_pts = [(noisy_signal[i], 0) for i in range(len(noisy_signal))]
    filtered_pts = [(filtered_1euro[i], 0) for i in range(len(filtered_1euro))]
    
    jitter_before = calculate_jitter(noisy_pts)
    jitter_after = calculate_jitter(filtered_pts)
    jitter_reduction = (1 - jitter_after / jitter_before) * 100
    
    print(f"Jitter before: {jitter_before:.4f}")
    print(f"Jitter after:  {jitter_after:.4f}")
    print(f"Reduction:     {jitter_reduction:.1f}%")
    
    # Calculate lag
    clean_pts = [(clean_signal[i], 0) for i in range(len(clean_signal))]
    lag = calculate_lag(clean_pts, filtered_pts, fps=30)
    print(f"Estimated lag: {lag:.1f}ms")
    
    if lag < 50:
        print("✅ Latency < 50ms (acceptable)")
    else:
        print("⚠️  Latency > 50ms (perceptible)")
    
    print("\n2. EMA Filter Demo")
    print("-" * 70)
    
    # Apply EMA
    filtered_ema = []
    prev = None
    
    for val in noisy_signal:
        pt = (val, 0)
        smoothed = ema_point(pt, alpha=0.3, prev=prev)
        filtered_ema.append(smoothed[0])
        prev = smoothed
    
    # Calculate jitter reduction
    ema_pts = [(filtered_ema[i], 0) for i in range(len(filtered_ema))]
    jitter_ema = calculate_jitter(ema_pts)
    jitter_reduction_ema = (1 - jitter_ema / jitter_before) * 100
    
    print(f"Jitter before: {jitter_before:.4f}")
    print(f"Jitter after:  {jitter_ema:.4f}")
    print(f"Reduction:     {jitter_reduction_ema:.1f}%")
    
    # Calculate lag
    lag_ema = calculate_lag(clean_pts, ema_pts, fps=30)
    print(f"Estimated lag: {lag_ema:.1f}ms")
    
    if lag_ema < 50:
        print("✅ Latency < 50ms (acceptable)")
    else:
        print("⚠️  Latency > 50ms (perceptible)")
    
    print("\n3. Landmark Filtering Demo")
    print("-" * 70)
    
    # Create synthetic landmarks
    landmarks_sequence = []
    for i in range(10):
        landmarks = {
            'nose': (320 + np.random.normal(0, 2), 240 + np.random.normal(0, 2)),
            'left_shoulder': (280 + np.random.normal(0, 3), 300 + np.random.normal(0, 3)),
            'right_shoulder': (360 + np.random.normal(0, 3), 300 + np.random.normal(0, 3))
        }
        landmarks_sequence.append(landmarks)
    
    # Filter landmarks
    filter_landmarks = OneEuroFilter(min_cutoff=1.0, beta=0.5, fps=30)
    smoothed_sequence = []
    
    for i, landmarks in enumerate(landmarks_sequence):
        smoothed = filter_landmarks.filter_landmarks(landmarks, t=i/30.0)
        smoothed_sequence.append(smoothed)
    
    print("Original nose positions (first 5 frames):")
    for i in range(5):
        x, y = landmarks_sequence[i]['nose']
        print(f"  Frame {i}: ({x:.1f}, {y:.1f})")
    
    print("\nSmoothed nose positions (first 5 frames):")
    for i in range(5):
        x, y = smoothed_sequence[i]['nose']
        print(f"  Frame {i}: ({x:.1f}, {y:.1f})")
    
    # Calculate variance reduction
    nose_orig = np.array([lm['nose'] for lm in landmarks_sequence])
    nose_smooth = np.array([lm['nose'] for lm in smoothed_sequence])
    
    var_orig = np.var(nose_orig)
    var_smooth = np.var(nose_smooth)
    var_reduction = (1 - var_smooth / var_orig) * 100
    
    print(f"\nNose variance reduction: {var_reduction:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ Smoothing module demonstration complete")
    print("=" * 70)
