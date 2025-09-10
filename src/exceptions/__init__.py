class ProductCountingException(Exception):
    """Base exception for product counting system"""
    pass

class ModelLoadingError(ProductCountingException):
    """Raised when model loading fails"""
    pass

class ImageProcessingError(ProductCountingException):
    """Raised when image processing fails"""
    pass

class CatalogError(ProductCountingException):
    """Raised when catalog operations fail"""
    pass

class ConfigurationError(ProductCountingException):
    """Raised when configuration is invalid"""
    pass

class APIError(ProductCountingException):
    """Raised when API operations fail"""
    pass