import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logging(
    log_level:str = 'INFO',
    log_file: Optional[Path] = None,
    max_file_size: int = 10*1024*1024,
    backup_count: int = 5,
    format_string: Optional[str] = None
    ) -> logging.Logger:
    """Setup comprehensive logging configuration"""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete - Level: {log_level}")
    return logger


class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: logging.Logger, operation: str, log_level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.log_level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(self.log_level, f"Completed {self.operation} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
