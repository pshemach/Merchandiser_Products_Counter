from pathlib import Path
import re
from typing import Any, List, Union, Dict

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration dictionary"""
    errors = []
    
    # Required fields
    required_fields = [
        'yolo_model_name', 'embedding_model_name', 'similarity_threshold',
        'data_dir', 'models_dir'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate threshold values
    if 'similarity_threshold' in config:
        threshold = config['similarity_threshold']
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            errors.append("similarity_threshold must be between 0.0 and 1.0")
    
    if 'yolo_confidence_threshold' in config:
        threshold = config['yolo_confidence_threshold']
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            errors.append("yolo_confidence_threshold must be between 0.0 and 1.0")
    
    # Validate paths
    path_fields = ['data_dir', 'models_dir', 'logs_dir']
    for field in path_fields:
        if field in config:
            path = Path(config[field])
            if not path.exists():
                errors.append(f"Path does not exist: {field} = {path}")
    
    return errors

def validate_image_path(image_path: Union[str, Path]) -> bool:
    """Validate image file path"""
    try:
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            return False
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if path.suffix.lower() not in valid_extensions:
            return False
        
        # Check if it's a file (not directory)
        if not path.is_file():
            return False
        
        return True
    
    except Exception:
        return False

def validate_product_data(product_data: Dict[str, Any]) -> List[str]:
    """Validate product data dictionary"""
    errors = []
    
    required_fields = ['product_id', 'name']
    for field in required_fields:
        if field not in product_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate product_id format (alphanumeric with underscores/hyphens)
    if 'product_id' in product_data:
        product_id = product_data['product_id']
        if not isinstance(product_id, str) or not re.match(r'^[a-zA-Z0-9_-]+$', product_id):
            errors.append("product_id must be alphanumeric with underscores/hyphens only")
    
    # Validate name
    if 'name' in product_data:
        name = product_data['name']
        if not isinstance(name, str) or len(name.strip()) == 0:
            errors.append("name must be a non-empty string")
    
    # Validate reference images
    if 'reference_images' in product_data:
        images = product_data['reference_images']
        if not isinstance(images, list):
            errors.append("reference_images must be a list")
        else:
            for i, img_path in enumerate(images):
                if not validate_image_path(img_path):
                    errors.append(f"Invalid reference image {i}: {img_path}")
    
    return errors

def validate_bounding_box(bbox: tuple, image_shape: tuple) -> bool:
    """Validate bounding box coordinates"""
    try:
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2]
        
        # Check coordinates are numeric
        # if not all(isinstance(coord, (int, float, )) for coord in bbox):
        #     print("Check coordinates are numeric is False")
        #     return False
        
        # Check coordinates are within image bounds
        if not (0 <= x1 < w and 0 <= x2 <= w and 0 <= y1 < h and 0 <= y2 <= h):
            print("Check coordinates are within image bounds is False")
            return False
        
        # Check box has positive area
        if x2 <= x1 or y2 <= y1:
            print("Check box has positive area is False")
            return False
        
        return True
    
    except Exception:
        return False

def validate_similarity_score(score: float) -> bool:
    """Validate similarity score"""
    return isinstance(score, (int, float)) and 0.0 <= score <= 1.0

def validate_product_counts(counts: Dict[str, int]) -> List[str]:
    """Validate product counts dictionary"""
    errors = []
    
    if not isinstance(counts, dict):
        errors.append("Product counts must be a dictionary")
        return errors
    
    for product_id, count in counts.items():
        if not isinstance(product_id, str):
            errors.append(f"Product ID must be string: {product_id}")
        
        if not isinstance(count, int) or count < 0:
            errors.append(f"Count must be non-negative integer for {product_id}: {count}")
    
    return errors