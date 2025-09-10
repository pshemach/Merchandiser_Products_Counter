from typing import Optional, Any
import traceback
from src.exceptions import ProductCountingException

class SystemInitializationError(ProductCountingException):
    """Raised when system initialization fails"""
    
    def __init__(self, message: str, component: str = None, original_error: Exception = None):
        self.component = component
        self.original_error = original_error
        super().__init__(message)

class EmbeddingExtractionError(ProductCountingException):
    """Raised when embedding extraction fails"""
    
    def __init__(self, message: str, image_path: str = None, model_name: str = None):
        self.image_path = image_path
        self.model_name = model_name
        super().__init__(message)

class SimilarityMatchingError(ProductCountingException):
    """Raised when similarity matching fails"""
    
    def __init__(self, message: str, embedding_shape: tuple = None, catalog_size: int = None):
        self.embedding_shape = embedding_shape
        self.catalog_size = catalog_size
        super().__init__(message)

class ObjectDetectionError(ProductCountingException):
    """Raised when object detection fails"""
    
    def __init__(self, message: str, image_shape: tuple = None, model_name: str = None):
        self.image_shape = image_shape
        self.model_name = model_name
        super().__init__(message)