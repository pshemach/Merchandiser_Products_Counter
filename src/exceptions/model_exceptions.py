from typing import Optional, Any, List
from src.exceptions.base_exceptions import ResourceError, SystemError

class ModelError(SystemError):
    """Base class for model-related errors"""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 model_type: Optional[str] = None, **kwargs):
        super().__init__(message, component='model', **kwargs)
        if model_name:
            self.add_context('model_name', model_name)
        if model_type:
            self.add_context('model_type', model_type)

class ModelNotFoundError(ModelError, ResourceError):
    """Model file or resource not found"""
    
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_path:
            self.add_context('model_path', model_path)
        
        self.add_suggestion("Check if model file exists")
        self.add_suggestion("Verify model path is correct")
        self.add_suggestion("Download model if it's not cached")

class ModelLoadError(ModelError):
    """Model loading failed"""
    
    def __init__(self, message: str, model_format: Optional[str] = None,
                 memory_required: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_format:
            self.add_context('model_format', model_format)
        if memory_required:
            self.add_context('memory_required', memory_required)
        
        self.add_suggestion("Check available system memory")
        self.add_suggestion("Verify model format compatibility")
        self.add_suggestion("Try loading model on CPU if GPU fails")

class InferenceError(ModelError):
    """Model inference failed"""
    
    def __init__(self, message: str, input_shape: Optional[tuple] = None,
                 expected_shape: Optional[tuple] = None, **kwargs):
        super().__init__(message, **kwargs)
        if input_shape:
            self.add_context('input_shape', input_shape)
        if expected_shape:
            self.add_context('expected_shape', expected_shape)
        
        self.add_suggestion("Check input data format and shape")
        self.add_suggestion("Verify preprocessing steps")
        self.add_suggestion("Check for NaN or infinite values in input")

class ModelCompatibilityError(ModelError):
    """Model compatibility issue"""
    
    def __init__(self, message: str, framework_version: Optional[str] = None,
                 required_version: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if framework_version:
            self.add_context('framework_version', framework_version)
        if required_version:
            self.add_context('required_version', required_version)
        
        self.add_suggestion("Check framework version compatibility")
        self.add_suggestion("Update to required framework version")
        self.add_suggestion("Use model conversion tools if available")

class QuantizationError(ModelError):
    """Model quantization failed"""
    
    def __init__(self, message: str, quantization_type: Optional[str] = None,
                 target_dtype: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if quantization_type:
            self.add_context('quantization_type', quantization_type)
        if target_dtype:
            self.add_context('target_dtype', target_dtype)
        
        self.add_suggestion("Check if model supports quantization")
        self.add_suggestion("Try different quantization method")
        self.add_suggestion("Use calibration dataset if required")