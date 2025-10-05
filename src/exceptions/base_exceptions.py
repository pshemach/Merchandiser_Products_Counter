import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

class ProductCountingException(Exception):
    """
    Base exception class for all product counting system errors.
    
    Provides common functionality for error handling, logging, and context preservation.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause  # Original exception if this is a wrapper
        self.suggestions = suggestions or []
        self.timestamp = datetime.now()
        self.traceback_str = traceback.format_exc() if cause else None
        
        # Log the error
        logger = logging.getLogger(self.__class__.__module__)
        logger.error(f"{self.error_code}: {self.message}", exc_info=True)
    
    def add_context(self, key: str, value: Any) -> 'ProductCountingException':
        """Add context information to the exception"""
        self.context[key] = value
        return self
    
    def add_suggestion(self, suggestion: str) -> 'ProductCountingException':
        """Add a suggestion for resolving the error"""
        self.suggestions.append(suggestion)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """Enhanced string representation with context"""
        base_msg = f"{self.error_code}: {self.message}"
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base_msg += f" (Context: {context_str})"
        
        if self.cause:
            base_msg += f" (Caused by: {self.cause})"
        
        return base_msg

class SystemError(ProductCountingException):
    """Base class for system-level errors"""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if component:
            self.add_context('component', component)

class ValidationError(ProductCountingException):
    """Base class for validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        if field:
            self.add_context('field', field)
        if value is not None:
            self.add_context('value', str(value))

class ConfigurationError(ProductCountingException):
    """Errors related to system configuration"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        if config_key:
            self.add_context('config_key', config_key)
        if config_value is not None:
            self.add_context('config_value', str(config_value))
        
        # Add common suggestions
        self.add_suggestion("Check your configuration file")
        self.add_suggestion("Verify environment variables are set correctly")

class ResourceError(ProductCountingException):
    """Errors related to resource availability or access"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 resource_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if resource_type:
            self.add_context('resource_type', resource_type)
        if resource_path:
            self.add_context('resource_path', resource_path)