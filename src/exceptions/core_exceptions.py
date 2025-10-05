from typing import Optional, Any, Tuple, List
from src.exceptions.base_exceptions import SystemError, ValidationError, ResourceError

class SystemInitializationError(SystemError):
    """Raised when system initialization fails"""
    
    def __init__(self, message: str, component: Optional[str] = None, 
                 initialization_step: Optional[str] = None, **kwargs):
        super().__init__(message, component=component, **kwargs)
        if initialization_step:
            self.add_context('initialization_step', initialization_step)
        
        # Add common suggestions
        self.add_suggestion("Check system requirements are met")
        self.add_suggestion("Verify all dependencies are installed")
        self.add_suggestion("Check GPU/CUDA availability if using GPU")

class ModelLoadingError(ResourceError):
    """Raised when model loading fails"""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 model_path: Optional[str] = None, **kwargs):
        super().__init__(message, resource_type='model', **kwargs)
        if model_name:
            self.add_context('model_name', model_name)
        if model_path:
            self.add_context('model_path', model_path)
        
        # Add specific suggestions
        self.add_suggestion("Check if model file exists and is accessible")
        self.add_suggestion("Verify model format compatibility")
        self.add_suggestion("Check available memory for model loading")

class EmbeddingExtractionError(SystemError):
    """Raised when embedding extraction fails"""
    
    def __init__(self, message: str, image_path: Optional[str] = None,
                 model_name: Optional[str] = None, image_shape: Optional[Tuple] = None,
                 **kwargs):
        super().__init__(message, component='embedding_extractor', **kwargs)
        if image_path:
            self.add_context('image_path', image_path)
        if model_name:
            self.add_context('model_name', model_name)
        if image_shape:
            self.add_context('image_shape', image_shape)
        
        # Add specific suggestions
        self.add_suggestion("Verify image format and quality")
        self.add_suggestion("Check if image is corrupted")
        self.add_suggestion("Ensure image preprocessing is correct")

class ObjectDetectionError(SystemError):
    """Raised when object detection fails"""
    
    def __init__(self, message: str, image_shape: Optional[Tuple] = None,
                 model_name: Optional[str] = None, confidence_threshold: Optional[float] = None,
                 **kwargs):
        super().__init__(message, component='object_detector', **kwargs)
        if image_shape:
            self.add_context('image_shape', image_shape)
        if model_name:
            self.add_context('model_name', model_name)
        if confidence_threshold:
            self.add_context('confidence_threshold', confidence_threshold)
        
        # Add specific suggestions
        self.add_suggestion("Check image resolution and quality")
        self.add_suggestion("Adjust confidence threshold if needed")
        self.add_suggestion("Verify YOLO model compatibility")

class SimilarityMatchingError(SystemError):
    """Raised when similarity matching fails"""
    
    def __init__(self, message: str, embedding_shape: Optional[Tuple] = None,
                 catalog_size: Optional[int] = None, index_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, component='similarity_matcher', **kwargs)
        if embedding_shape:
            self.add_context('embedding_shape', embedding_shape)
        if catalog_size is not None:
            self.add_context('catalog_size', catalog_size)
        if index_type:
            self.add_context('index_type', index_type)
        
        # Add specific suggestions
        self.add_suggestion("Check embedding dimensions match catalog")
        self.add_suggestion("Verify catalog is not empty")
        self.add_suggestion("Rebuild FAISS index if corrupted")

class CatalogError(SystemError):
    """Raised when catalog operations fail"""
    
    def __init__(self, message: str, product_id: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, component='catalog_manager', **kwargs)
        if product_id:
            self.add_context('product_id', product_id)
        if operation:
            self.add_context('operation', operation)
        
        # Add specific suggestions
        self.add_suggestion("Check product ID format and uniqueness")
        self.add_suggestion("Verify reference images are accessible")
        self.add_suggestion("Check catalog file permissions")

class NormalizationError(SystemError):
    """Raised when embedding normalization fails"""
    
    def __init__(self, message: str, strategy: Optional[str] = None,
                 embedding_count: Optional[int] = None, **kwargs):
        super().__init__(message, component='embedding_extractor', **kwargs)
        if strategy:
            self.add_context('normalization_strategy', strategy)
        if embedding_count is not None:
            self.add_context('embedding_count', embedding_count)
        
        # Add specific suggestions
        self.add_suggestion("Check if catalog has sufficient embeddings")
        self.add_suggestion("Verify normalization strategy is supported")
        self.add_suggestion("Check for NaN or infinite values in embeddings")
