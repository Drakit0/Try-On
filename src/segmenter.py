"""
Person segmentation module for occlusion masks.
Creates masks for face, skin, and person regions using MediaPipe.
"""

import logging
from typing import Optional, Tuple, Union, List

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
    MEDIAPIPE_TASKS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PersonSegmenter:
    """
    Person segmentation for creating occlusion masks.
    
    Supports multiclass segmentation (face, skin, hair, body) and binary
    segmentation (person vs background) with automatic fallback.
    
    Args:
        model: Path to segmentation model (Tasks API) or model type string
        output_type: Output type - 'category_mask' or 'confidence_mask'
        use_gpu: Whether to use GPU acceleration if available
    """
    
    # MediaPipe selfie segmentation class indices
    # For multiclass segmentation
    CLASS_BACKGROUND = 0
    CLASS_HAIR = 1
    CLASS_BODY_SKIN = 2
    CLASS_FACE_SKIN = 3
    CLASS_CLOTHES = 4
    CLASS_OTHERS = 5
    
    # Class name mapping
    CLASS_NAMES = {
        'background': CLASS_BACKGROUND,
        'hair': CLASS_HAIR,
        'body': CLASS_BODY_SKIN,
        'skin': CLASS_BODY_SKIN,
        'face': CLASS_FACE_SKIN,
        'clothes': CLASS_CLOTHES,
        'person': [CLASS_HAIR, CLASS_BODY_SKIN, CLASS_FACE_SKIN, CLASS_CLOTHES]
    }
    
    def __init__(
        self,
        model: str = 'selfie_multiclass_256x256.tflite',
        output_type: str = 'category_mask',
        use_gpu: bool = False
    ):
        """Initialize person segmenter."""
        self.model = model
        self.output_type = output_type
        self.use_gpu = use_gpu
        
        # State management
        self.frame_count = 0
        self.is_multiclass = False
        
        # Initialize segmenter
        if MEDIAPIPE_TASKS_AVAILABLE:
            self._init_tasks_api()
        else:
            self._init_solutions_api()  # Fallback
    
    def _init_tasks_api(self):
        """Initialize MediaPipe Tasks API image segmenter."""
        logger.info("Initializing MediaPipe Tasks API for segmentation")
        
        try:
            # Configure base options
            base_options = python.BaseOptions(model_asset_path=self.model)
            
            # Configure segmenter options
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_category_mask=(self.output_type == 'category_mask'),
                output_confidence_masks=(self.output_type == 'confidence_mask')
            )
            
            # Create segmenter
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
            self.api_type = "tasks"
            self.is_multiclass = True  # Tasks API supports multiclass
            logger.info("MediaPipe Tasks API segmenter initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Tasks API: {e}. Falling back to Solutions API.")
            self._init_solutions_api()  # Fallback
    
    def _init_solutions_api(self):
        """Initialize MediaPipe Solutions API segmenter (Fallback)."""
        logger.info("Initializing MediaPipe Solutions API for segmentation (Fallback)")
        
        # Use SelfieSegmentation from solutions
        # Model 0: general model, Model 1: landscape model (faster)
        model_selection = 0 if 'general' in self.model.lower() else 1
        
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        self.api_type = "solutions"
        self.is_multiclass = False  # Solutions API only supports binary
        logger.info("MediaPipe Solutions API segmenter initialized successfully (binary only)")
    
    def mask(
        self,
        frame_bgr: np.ndarray,
        classes: Optional[Union[Tuple[str, ...], List[str]]] = ('face', 'skin')
    ) -> np.ndarray:
        """
        Generate segmentation mask for specified classes.
        
        Args:
            frame_bgr: Input frame in BGR format (np.ndarray, uint8)
            classes: Tuple/list of class names to include in mask.
                    Options: 'face', 'skin', 'hair', 'clothes', 'person', 'body'
                    If None, returns full person mask (all non-background)
        
        Returns:
            Binary mask (0/255, uint8) with same dimensions as input frame
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Empty frame provided to segmenter")
            return np.zeros((480, 640), dtype=np.uint8)
        
        h, w = frame_bgr.shape[:2]
        self.frame_count += 1
        
        try:
            if self.api_type == "tasks":
                mask = self._segment_tasks_api(frame_bgr, classes, w, h)
            else:
                mask = self._segment_solutions_api(frame_bgr, classes, w, h)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return np.zeros((h, w), dtype=np.uint8)
    
    def _segment_tasks_api(
        self,
        frame_bgr: np.ndarray,
        classes: Optional[Union[Tuple[str, ...], List[str]]],
        w: int,
        h: int
    ) -> np.ndarray:
        """Segment using MediaPipe Tasks API."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame_rgb)
        
        # Segment image
        segmentation_result = self.segmenter.segment_for_video(mp_image, self.frame_count)
        
        # Extract mask
        if self.output_type == 'category_mask' and segmentation_result.category_mask:
            category_mask = segmentation_result.category_mask.numpy_view()
            mask = self._create_class_mask(category_mask, classes, w, h)
        elif segmentation_result.confidence_masks:
            # Use confidence masks (multiple masks, one per class)
            mask = self._create_confidence_mask(
                segmentation_result.confidence_masks, classes, w, h
            )
        else:
            logger.warning("No segmentation mask returned")
            mask = np.zeros((h, w), dtype=np.uint8)
        
        return mask
    
    def _segment_solutions_api(
        self,
        frame_bgr: np.ndarray,
        classes: Optional[Union[Tuple[str, ...], List[str]]],
        w: int,
        h: int
    ) -> np.ndarray:
        """Segment using MediaPipe Solutions API (Fallback - binary only)."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.segmenter.process(frame_rgb)
        
        if results.segmentation_mask is None:
            logger.warning("No segmentation mask returned")
            return np.zeros((h, w), dtype=np.uint8)
        
        # Solutions API only provides binary person mask
        # Threshold to create binary mask
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Resize if needed
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return mask
    
    def _create_class_mask(
        self,
        category_mask: np.ndarray,
        classes: Optional[Union[Tuple[str, ...], List[str]]],
        w: int,
        h: int
    ) -> np.ndarray:
        """Create binary mask from category mask based on specified classes."""
        # Resize category mask to match frame size
        if category_mask.shape[0] != h or category_mask.shape[1] != w:
            category_mask = cv2.resize(
                category_mask.astype(np.uint8), (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if classes is None:
            # Include all non-background classes (full person)
            mask[category_mask > 0] = 255
        else:
            # Include specified classes
            for class_name in classes:
                class_name = class_name.lower()
                if class_name in self.CLASS_NAMES:
                    class_indices = self.CLASS_NAMES[class_name]
                    if isinstance(class_indices, list):
                        # Multiple indices (e.g., 'person')
                        for idx in class_indices:
                            mask[category_mask == idx] = 255
                    else:
                        # Single index
                        mask[category_mask == class_indices] = 255
                else:
                    logger.warning(f"Unknown class name: {class_name}")
        
        return mask
    
    def _create_confidence_mask(
        self,
        confidence_masks,
        classes: Optional[Union[Tuple[str, ...], List[str]]],
        w: int,
        h: int
    ) -> np.ndarray:
        """Create binary mask from confidence masks."""
        # Combine confidence masks for specified classes
        combined_mask = np.zeros((h, w), dtype=np.float32)
        
        if classes is None:
            # Combine all non-background masks
            for i, conf_mask in enumerate(confidence_masks):
                if i > 0:  # Skip background (index 0)
                    mask_data = conf_mask.numpy_view()
                    if mask_data.shape[0] != h or mask_data.shape[1] != w:
                        mask_data = cv2.resize(mask_data, (w, h))
                    combined_mask = np.maximum(combined_mask, mask_data)
        else:
            # Combine specified class masks
            for class_name in classes:
                class_name = class_name.lower()
                if class_name in self.CLASS_NAMES:
                    class_indices = self.CLASS_NAMES[class_name]
                    if isinstance(class_indices, list):
                        for idx in class_indices:
                            if idx < len(confidence_masks):
                                mask_data = confidence_masks[idx].numpy_view()
                                if mask_data.shape[0] != h or mask_data.shape[1] != w:
                                    mask_data = cv2.resize(mask_data, (w, h))
                                combined_mask = np.maximum(combined_mask, mask_data)
                    else:
                        if class_indices < len(confidence_masks):
                            mask_data = confidence_masks[class_indices].numpy_view()
                            if mask_data.shape[0] != h or mask_data.shape[1] != w:
                                mask_data = cv2.resize(mask_data, (w, h))
                            combined_mask = np.maximum(combined_mask, mask_data)
        
        # Threshold to binary
        binary_mask = (combined_mask > 0.5).astype(np.uint8) * 255
        return binary_mask
    
    def refine_mask(self, mask: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Refine mask using morphological operations.
        
        Applies opening (erosion followed by dilation) to remove noise,
        then closing (dilation followed by erosion) to fill holes.
        
        Args:
            mask: Input binary mask (0/255, uint8)
            ksize: Kernel size for morphological operations (odd number)
        
        Returns:
            Refined binary mask (0/255, uint8)
        """
        if mask is None or mask.size == 0:
            return mask
        
        # Ensure ksize is odd
        if ksize % 2 == 0:
            ksize += 1
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        # Opening: remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing: fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return mask
    
    def apply_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        background: Optional[np.ndarray] = None,
        blur_edges: bool = True,
        blur_radius: int = 3
    ) -> np.ndarray:
        """
        Apply segmentation mask to frame for alpha composition.
        
        Args:
            frame: Input frame (BGR)
            mask: Binary mask (0/255, uint8)
            background: Optional background image. If None, uses black background
            blur_edges: Whether to blur mask edges for smoother composition
            blur_radius: Radius for edge blurring
        
        Returns:
            Composited frame
        """
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Blur edges for smoother composition
        if blur_edges and blur_radius > 0:
            alpha_mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
        else:
            alpha_mask = mask.copy()
        
        # Normalize to 0-1 range
        alpha_mask = alpha_mask.astype(np.float32) / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)  # Add channel dimension
        
        # Create background
        if background is None:
            background = np.zeros_like(frame)
        elif background.shape != frame.shape:
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        
        # Alpha blend
        result = (frame * alpha_mask + background * (1 - alpha_mask)).astype(np.uint8)
        
        return result
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'segmenter') and self.segmenter is not None:
            if self.api_type == "tasks":
                self.segmenter.close()
            elif self.api_type == "solutions":
                self.segmenter.close()
        logger.info("Segmenter closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility functions

def blend_with_mask(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Blend foreground and background using mask.
    
    Args:
        foreground: Foreground image (BGR)
        background: Background image (BGR)
        mask: Binary mask (0/255, uint8)
    
    Returns:
        Blended image
    """
    # Ensure mask is same size as images
    if mask.shape[:2] != foreground.shape[:2]:
        mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))
    
    # Normalize mask to 0-1
    alpha = mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    # Blend
    result = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
    return result


def create_soft_edge_mask(mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """
    Create soft-edged mask for seamless composition.
    
    Args:
        mask: Binary mask (0/255, uint8)
        blur_radius: Radius for Gaussian blur
    
    Returns:
        Soft-edged mask (0/255, uint8)
    """
    # Apply Gaussian blur to soften edges
    kernel_size = blur_radius * 2 + 1
    soft_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return soft_mask


if __name__ == "__main__":
    # Test with a dummy frame
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PersonSegmenter...")
    print(f"MediaPipe Tasks API available: {MEDIAPIPE_TASKS_AVAILABLE}")
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a simple colored rectangle to simulate a person
    test_frame[100:400, 200:440] = [128, 128, 128]
    
    try:
        # Initialize segmenter
        segmenter = PersonSegmenter()
        print(f"Using API: {segmenter.api_type}")
        print(f"Multiclass support: {segmenter.is_multiclass}")
        
        # Try segmentation
        mask = segmenter.mask(test_frame, classes=None)
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask values - min: {mask.min()}, max: {mask.max()}")
        
        # Try refinement
        refined_mask = segmenter.refine_mask(mask, ksize=5)
        print(f"Refined mask shape: {refined_mask.shape}")
        
        segmenter.close()
        print("PersonSegmenter test complete!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
