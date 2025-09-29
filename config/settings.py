import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from enum import Enum

class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    
class Settings(BaseSettings):
    """Main application settings with environment support"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra env vars like HF_API
    )
    
    # =============================================================================
    # ENVIRONMENT CONFIGURATION
    # =============================================================================
    
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Current environment",
        alias="ENVIRONMENT"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode",
        alias="DEBUG"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        alias="LOG_LEVEL"
    )

    # =============================================================================
    # PROJECT PATHS
    # =============================================================================
    
    project_root: Path = Field(
        default=Path(__file__).parent.parent,
        description="Project root directory"
    )
    
    data_dir: Optional[Path] = Field(
        default=None,
        description="Data directory path"
    )
    
    models_dir: Optional[Path] = Field(
        default=None,
        description="Models directory path"
    )
    
    yolo_model_dir: Optional[Path] = Field(
        default=None,
        description="Yolo Model directory path"
    )
    
    logs_dir: Optional[Path] = Field(
        default=None,
        description="Logs directory path"
    )
    
    results_dir: Optional[Path] = Field(
        default=None,
        description="Results directory path"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set default paths relative to project root
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.models_dir is None:
            self.models_dir = self.data_dir / "models"
        if self.yolo_model_dir is None:
            self.yolo_model_dir = self.models_dir / "yolo_weights"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "monitoring" / "logs"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"
        
        # Automatically create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
            
            
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    # YOLO Configuration
    yolo_model_name: str = Field(
        default="yolov9.pt",
        description="YOLO model variant (n, s, m, l, x)",
        alias="YOLO_MODEL"
    )
    
    
    yolo_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="YOLO detection confidence threshold"
    )
    
    yolo_iou_threshold: float = Field(
        default=0.45,
        ge=0.0, le=1.0,
        description="YOLO IoU threshold for NMS"
    )
    
    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="facebook/dinov2-large",
        description="Embedding model from HuggingFace",
        alias="EMBEDDING_MODEL"
    )
    
    embedding_dimension: int = Field(
        default=768,
        description="Embedding vector dimension"
    )
    
    normalization_strategy: str = Field(
        default="individual",
        description="Embedding normalization strategy"
    )
    
    # Similarity Matching Configuration
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0, le=1.0,
        description="Product similarity threshold"
    )
    
    max_catalog_matches: int = Field(
        default=3,
        ge=1, le=10,
        description="Maximum catalog matches to return"
    )
    
    # =============================================================================
    # PROCESSING CONFIGURATION
    # =============================================================================
    
    max_image_size: int = Field(
        default=1024,
        description="Maximum image size for processing"
    )
    
    batch_size: int = Field(
        default=8,
        ge=1, le=64,
        description="Batch size for processing"
    )
    
    device: str = Field(
        default="auto",
        description="Processing device (auto, cpu, cuda)",
        alias="DEVICE"
    )
    
    num_workers: int = Field(
        default=4,
        ge=1, le=16,
        description="Number of worker threads"
    )
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    
    database_url: str = Field(
        default="sqlite:///./monitoring.db",
        description="Database connection URL",
        alias="DATABASE_URL"
    )
    
    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (debug mode)"
    )
    
    # =============================================================================
    # API CONFIGURATION
    # =============================================================================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address",
        alias="API_HOST"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1024, le=65535,
        description="API port",
        alias="API_PORT"
    )
    
    api_workers: int = Field(
        default=1,
        ge=1, le=8,
        description="Number of API worker processes"
    )
    
    max_upload_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum upload file size in bytes"
    )
    
    api_timeout: int = Field(
        default=60,
        description="API request timeout in seconds"
    )
    
    # =============================================================================
    # MONITORING CONFIGURATION
    # =============================================================================
    
    monitoring_enabled: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    
    monitoring_interval: int = Field(
        default=60,
        description="Monitoring collection interval in seconds"
    )
    
    health_check_interval: int = Field(
        default=300,
        description="Health check interval in seconds"
    )
    
    metrics_retention_days: int = Field(
        default=30,
        description="Metrics retention period in days"
    )
    
    # =============================================================================
    # ALERTING CONFIGURATION
    # =============================================================================
    
    alerting_enabled: bool = Field(
        default=True,
        description="Enable alerting system"
    )
    
    email_alerts_enabled: bool = Field(
        default=False,
        description="Enable email alerts"
    )
    
    webhook_alerts_enabled: bool = Field(
        default=False,
        description="Enable webhook alerts"
    )
    
    alert_cooldown_minutes: int = Field(
        default=15,
        description="Minimum time between duplicate alerts"
    )
    
    # Alert Thresholds
    accuracy_drop_threshold: float = Field(
        default=0.1,
        description="Accuracy drop threshold for alerts"
    )
    
    processing_time_threshold: float = Field(
        default=5.0,
        description="Processing time threshold in seconds"
    )
    
    memory_usage_threshold: float = Field(
        default=80.0,
        description="Memory usage threshold percentage"
    )
    
    error_rate_threshold: float = Field(
        default=0.05,
        description="Error rate threshold for alerts"
    )
    
    # =============================================================================
    # SECURITY CONFIGURATION
    # =============================================================================
    
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for API authentication",
        alias="SECRET_KEY"
    )
    
    api_key_enabled: bool = Field(
        default=False,
        description="Enable API key authentication"
    )
    
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # =============================================================================
    # PERFORMANCE CONFIGURATION
    # =============================================================================
    
    auto_optimization_enabled: bool = Field(
        default=True,
        description="Enable automatic performance optimization"
    )
    
    optimization_interval_hours: int = Field(
        default=24,
        description="Automatic optimization interval in hours"
    )
    
    model_quantization_enabled: bool = Field(
        default=True,
        description="Enable model quantization for speed"
    )
    
    embedding_cache_enabled: bool = Field(
        default=True,
        description="Enable embedding caching"
    )
    
    embedding_cache_size: int = Field(
        default=1000,
        description="Maximum cached embeddings"
    )

# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

class DevelopmentSettings(Settings):
    """Development environment settings"""
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug: bool = True
    log_level: str = "DEBUG"
    yolo_model_name: str = "yolov9.pt"  # Fastest for development
    monitoring_enabled: bool = False
    alerting_enabled: bool = False

class TestingSettings(Settings):
    """Testing environment settings"""
    environment: EnvironmentType = EnvironmentType.TESTING
    debug: bool = True
    log_level: str = "WARNING"
    database_url: str = "sqlite:///:memory:"  # In-memory DB for tests
    monitoring_enabled: bool = False
    alerting_enabled: bool = False
    yolo_model_name: str = "yolov8n.pt"  # Fast for testing

class StagingSettings(Settings):
    """Staging environment settings"""
    environment: EnvironmentType = EnvironmentType.STAGING
    debug: bool = False
    log_level: str = "INFO"
    yolo_model_name: str = "yolov9.pt"  # Balance speed/accuracy
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    auto_optimization_enabled: bool = True

class ProductionSettings(Settings):
    """Production environment settings"""
    environment: EnvironmentType = EnvironmentType.PRODUCTION
    debug: bool = False
    log_level: str = "WARNING"
    yolo_model_name: str = "yolov9.pt"  # Best accuracy
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    auto_optimization_enabled: bool = True
    model_quantization_enabled: bool = True
    embedding_cache_enabled: bool = True
    api_key_enabled: bool = True

# =============================================================================
# SETTINGS FACTORY
# =============================================================================

def get_settings(environment: Optional[str] = None) -> Settings:
    """Get settings based on environment"""
    
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()
    
    settings_map = {
        "development": DevelopmentSettings,
        "testing": TestingSettings, 
        "staging": StagingSettings,
        "production": ProductionSettings
    }
    
    settings_class = settings_map.get(environment, DevelopmentSettings)
    return settings_class()

# Global settings instance
settings = get_settings()

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_settings(settings: Settings) -> List[str]:
    """Validate settings and return list of issues"""
    issues = []
    
    # Check required directories exist
    required_dirs = [settings.data_dir, settings.models_dir, settings.logs_dir, settings.results_dir]
    for dir_path in required_dirs:
        if not dir_path.exists():
            issues.append(f"Required directory does not exist: {dir_path}")
    
    # Validate thresholds
    if not 0.1 <= settings.similarity_threshold <= 0.99:
        issues.append("Similarity threshold should be between 0.1 and 0.99")
    
    if not 0.1 <= settings.yolo_confidence_threshold <= 0.9:
        issues.append("YOLO confidence threshold should be between 0.1 and 0.9")
    
    # Production-specific validations
    if settings.environment == EnvironmentType.PRODUCTION:
        if settings.secret_key == "your-secret-key-change-in-production":
            issues.append("SECRET_KEY must be changed in production")
        
        if not settings.api_key_enabled:
            issues.append("API key authentication should be enabled in production")
    
    return issues

if __name__ == "__main__":
    # Test settings loading
    print("----Configuration Settings Test----")
    print("=" * 50)
    
    test_settings = get_settings()
    print(f"Environment: {test_settings.environment}")
    print(f"Debug Mode: {test_settings.debug}")
    print(f"YOLO Model: {test_settings.yolo_model_name}")
    print(f"Embedding Model: {test_settings.embedding_model_name}")
    print(f"Data Directory: {test_settings.data_dir}")
    print(test_settings.model_dump())  # Use model_dump() instead of dict() to avoid deprecation
    
    # Validate settings
    issues = validate_settings(test_settings)
    if issues:
        print("\n----Configuration Issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n----Configuration is valid!")

# import os
# from pathlib import Path
# from enum import Enum

# class EnvironmentType(str, Enum):
#     DEVELOPMENT = "development"
#     TESTING = "testing"
#     STAGING = "staging"
#     PRODUCTION = "production"

# class Settings:
#     """Main application settings without Pydantic"""

#     def __init__(self):
#         # ENVIRONMENT CONFIGURATION
#         self.environment = EnvironmentType(os.environ.get("ENVIRONMENT", "development").lower())
#         self.debug = os.environ.get("DEBUG", "True").lower() == "true"
#         self.log_level = os.environ.get("LOG_LEVEL", "INFO")

#         # PROJECT PATHS
#         self.project_root = Path(__file__).parent.parent
#         self.data_dir = Path(os.environ.get("DATA_DIR", self.project_root / "data"))
#         self.models_dir = Path(os.environ.get("MODELS_DIR", self.data_dir / "models"))
#         self.logs_dir = Path(os.environ.get("LOGS_DIR", self.project_root / "monitoring" / "logs"))
#         self.results_dir = Path(os.environ.get("RESULTS_DIR", self.project_root / "results"))

#         # MODEL CONFIGURATION
#         self.yolo_model_name = os.environ.get("YOLO_MODEL", "yolov9.pt")
#         self.yolo_confidence_threshold = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", 0.5))
#         self.yolo_iou_threshold = float(os.environ.get("YOLO_IOU_THRESHOLD", 0.45))

#         self.embedding_model_name = os.environ.get("EMBEDDING_MODEL", "facebook/dinov2-base")
#         self.embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", 768))
#         self.normalization_strategy = os.environ.get("NORMALIZATION_STRATEGY", "catalog_norm")

#         # SIMILARITY MATCHING
#         self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.8))
#         self.max_catalog_matches = int(os.environ.get("MAX_CATALOG_MATCHES", 3))

#         # PROCESSING CONFIGURATION
#         self.max_image_size = int(os.environ.get("MAX_IMAGE_SIZE", 1024))
#         self.batch_size = int(os.environ.get("BATCH_SIZE", 8))
#         self.device = os.environ.get("DEVICE", "auto")
#         self.num_workers = int(os.environ.get("NUM_WORKERS", 4))

#         # DATABASE CONFIGURATION
#         self.database_url = os.environ.get("DATABASE_URL", "sqlite:///./monitoring.db")
#         self.database_echo = os.environ.get("DATABASE_ECHO", "False").lower() == "true"

#         # API CONFIGURATION
#         self.api_host = os.environ.get("API_HOST", "0.0.0.0")
#         self.api_port = int(os.environ.get("API_PORT", 8000))
#         self.api_workers = int(os.environ.get("API_WORKERS", 1))
#         self.max_upload_size = int(os.environ.get("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))
#         self.api_timeout = int(os.environ.get("API_TIMEOUT", 60))

#         # MONITORING CONFIGURATION
#         self.monitoring_enabled = os.environ.get("MONITORING_ENABLED", "True").lower() == "true"
#         self.monitoring_interval = int(os.environ.get("MONITORING_INTERVAL", 60))
#         self.health_check_interval = int(os.environ.get("HEALTH_CHECK_INTERVAL", 300))
#         self.metrics_retention_days = int(os.environ.get("METRICS_RETENTION_DAYS", 30))

#         # ALERTING CONFIGURATION
#         self.alerting_enabled = os.environ.get("ALERTING_ENABLED", "True").lower() == "true"
#         self.email_alerts_enabled = os.environ.get("EMAIL_ALERTS_ENABLED", "False").lower() == "true"
#         self.webhook_alerts_enabled = os.environ.get("WEBHOOK_ALERTS_ENABLED", "False").lower() == "true"
#         self.alert_cooldown_minutes = int(os.environ.get("ALERT_COOLDOWN_MINUTES", 15))
#         self.accuracy_drop_threshold = float(os.environ.get("ACCURACY_DROP_THRESHOLD", 0.1))
#         self.processing_time_threshold = float(os.environ.get("PROCESSING_TIME_THRESHOLD", 5.0))
#         self.memory_usage_threshold = float(os.environ.get("MEMORY_USAGE_THRESHOLD", 80.0))
#         self.error_rate_threshold = float(os.environ.get("ERROR_RATE_THRESHOLD", 0.05))

#         # SECURITY CONFIGURATION
#         self.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
#         self.api_key_enabled = os.environ.get("API_KEY_ENABLED", "False").lower() == "true"
#         self.cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")

#         # PERFORMANCE CONFIGURATION
#         self.auto_optimization_enabled = os.environ.get("AUTO_OPTIMIZATION_ENABLED", "True").lower() == "true"
#         self.optimization_interval_hours = int(os.environ.get("OPTIMIZATION_INTERVAL_HOURS", 24))
#         self.model_quantization_enabled = os.environ.get("MODEL_QUANTIZATION_ENABLED", "True").lower() == "true"
#         self.embedding_cache_enabled = os.environ.get("EMBEDDING_CACHE_ENABLED", "True").lower() == "true"
#         self.embedding_cache_size = int(os.environ.get("EMBEDDING_CACHE_SIZE", 1000))

# def validate_settings(settings: Settings):
#     """Validate settings and return list of issues"""
#     issues = []
#     required_dirs = [settings.data_dir, settings.models_dir, settings.logs_dir]
#     for dir_path in required_dirs:
#         if not Path(dir_path).exists():
#             issues.append(f"Required directory does not exist: {dir_path}")

#     if not 0.1 <= settings.similarity_threshold <= 0.99:
#         issues.append("Similarity threshold should be between 0.1 and 0.99")
#     if not 0.1 <= settings.yolo_confidence_threshold <= 0.9:
#         issues.append("YOLO confidence threshold should be between 0.1 and 0.9")
#     if settings.environment == EnvironmentType.PRODUCTION:
#         if settings.secret_key == "your-secret-key-change-in-production":
#             issues.append("SECRET_KEY must be changed in production")
#         if not settings.api_key_enabled:
#             issues.append("API key authentication should be enabled in production")
#     return issues

# if __name__ == "__main__":
#     print("----Configuration Settings Test----")
#     print("=" * 50)
#     settings = Settings()
#     print(f"Environment: {settings.environment}")
#     print(f"Debug Mode: {settings.debug}")
#     print(f"Data Directory: {settings.data_dir}")
#     print(f"YOLO Model: {settings.yolo_model_name}")
#     print(f"Embedding Model: {settings.embedding_model_name}")
#     print(f"Models Directory: {settings.models_dir}")
#     print(vars(settings))

#     issues = validate_settings(settings)
#     if issues:
#         print("\n----Configuration Issues:")
#         for issue in issues:
#             print(f"  • {issue}")
#     else:
#         print("\n----Configuration is valid!")