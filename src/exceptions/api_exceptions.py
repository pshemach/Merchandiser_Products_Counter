from typing import Optional, Any, Dict, List
from src.exceptions.base_exceptions import ProductCountingException, ValidationError

class APIError(ProductCountingException):
    """Base class for API-related errors"""
    
    def __init__(self, message: str, status_code: int = 500, 
                 endpoint: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        if endpoint:
            self.add_context('endpoint', endpoint)

class AuthenticationError(APIError):
    """Authentication failed"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)
        self.add_suggestion("Check API key or authentication credentials")
        self.add_suggestion("Verify token has not expired")

class AuthorizationError(APIError):
    """Authorization failed - insufficient permissions"""
    
    def __init__(self, message: str = "Insufficient permissions", 
                 required_permission: Optional[str] = None, **kwargs):
        super().__init__(message, status_code=403, **kwargs)
        if required_permission:
            self.add_context('required_permission', required_permission)
        self.add_suggestion("Check user permissions")
        self.add_suggestion("Contact administrator for access")

class RequestValidationError(ValidationError):
    """Request validation failed"""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 validation_errors: Optional[List[Dict]] = None, **kwargs):
        super().__init__(message, field=field, error_code='REQUEST_VALIDATION_ERROR', **kwargs)
        self.status_code = 400
        if validation_errors:
            self.add_context('validation_errors', validation_errors)
        
        # Add specific suggestions
        self.add_suggestion("Check request format and required fields")
        self.add_suggestion("Verify data types match API specification")

class FileUploadError(APIError):
    """File upload operation failed"""
    
    def __init__(self, message: str, filename: Optional[str] = None,
                 file_size: Optional[int] = None, max_size: Optional[int] = None,
                 **kwargs):
        super().__init__(message, status_code=400, **kwargs)
        if filename:
            self.add_context('filename', filename)
        if file_size is not None:
            self.add_context('file_size', file_size)
        if max_size is not None:
            self.add_context('max_size', max_size)
        
        # Add specific suggestions
        self.add_suggestion("Check file format is supported")
        self.add_suggestion("Verify file size is within limits")
        self.add_suggestion("Ensure file is not corrupted")

class RateLimitError(APIError):
    """Rate limit exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", 
                 limit: Optional[int] = None, window: Optional[str] = None,
                 retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        if limit is not None:
            self.add_context('rate_limit', limit)
        if window:
            self.add_context('time_window', window)
        if retry_after is not None:
            self.add_context('retry_after', retry_after)
        
        self.add_suggestion(f"Wait {retry_after} seconds before retrying" if retry_after else "Wait before retrying")
        self.add_suggestion("Consider reducing request frequency")

class ServiceUnavailableError(APIError):
    """Service temporarily unavailable"""
    
    def __init__(self, message: str = "Service temporarily unavailable",
                 service: Optional[str] = None, **kwargs):
        super().__init__(message, status_code=503, **kwargs)
        if service:
            self.add_context('service', service)
        
        self.add_suggestion("Try again later")
        self.add_suggestion("Check system status page")