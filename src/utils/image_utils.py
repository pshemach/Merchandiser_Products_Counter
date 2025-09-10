import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Tuple, Optional, Union
import logging
from src.utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)

def validate_image(image: Union[str, Path, np.ndarray, Image.Image]) -> bool:
    """Validate if input is a valid image"""
    try:
        if isinstance(image, (str, Path)):
            path = Path(image)
            
            if not path.exists():
                return False
            if path.suffix not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                return False
            
            # Try to open
            img = cv2.imread(str(path))
            
            if img is not None:
                return True
        elif isinstance(image, (np.ndarray)):
            return len(image.shape) in {2,3} and image.size > 0 
        
        elif isinstance(image, (Image.Image)):
            return image.size[0]>0 and image.size[1]>0
        
        return False
    
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        raise
    
def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """load image from path"""
    try:
        image_path = Path(image_path)
        
        if not validate_image(image_path):
            raise ValueError(f"Invalid image: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise
    
def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 maintain_aspect: bool = True) -> np.ndarray:
    """Resize image to target size"""
    try:
        if not maintain_aspect:
            return cv2.resize(image, target_size)
        
        # Maintain aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize and pad if necessary
        resized = cv2.resize(image, (new_w, new_h))
        
        if new_w != target_w or new_h != target_h:
            # Create padded image
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return padded
        
        return resized
    
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        raise
    
def normalize_image(image:np.ndarray, method: str='standard') -> np.ndarray:
    """Normalize image values"""
    try:
        if method == 'standard':
            return image.astype(np.float32)/255.0
        elif method == 'mean_std':
            mean = np.mean(image)
            std = np.std(image)
            return (image.astype(np.float32) - mean) / (std + 1e-8)
        elif method == 'min_max':
            min_val = np.min(image)
            max_val = np.max(image)
            return (image.astype(np.float32) - min_val)/ (max_val -min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    except Exception as e:
        logger.error(f"Failed to normalize image: {e}")
        raise
    
def enhance_image(image:np.ndarray, brightness: float=1.0,
                  contrast: float=1.0) -> np.ndarray:
    """Enhance image brightness and contrast"""
    try:
        enhanced = cv2.convertScaleAbs(image, alpha=brightness, beta=contrast)
        return enhanced
    except Exception as e:
        logger.error(f"Failed to enhanced image: {e}")
        raise
    
def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using bounding box (x1, y1, x2, y2)"""
    try:
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            raise ValueError(f"Invalid crop coordinates: {bbox}")
        
        return cropped
    
    except Exception as e:
        logger.error(f"Failed to crop image: {e}")
        raise
    
def save_image(image: np.ndarray, filepath: Path, quality: int = 95) -> None:
    """Save image to file"""
    try:
        ensure_dir(Path(filepath).parent)
        
        if filepath.suffix.lower() == '.jpg':
            cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(str(filepath), image)
        
        logger.debug(f"Saved image to {filepath}")
    
    except Exception as e:
        logger.error(f"Failed to save image to {filepath}: {e}")
        raise