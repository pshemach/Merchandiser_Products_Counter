import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    
class Settings(BaseSettings):
    """Main application settings with environment support"""
    
    # =============================================================================
    # ENVIRONMENT CONFIGURATION
    # =============================================================================
    
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Current environment"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # =============================================================================
    # PROJECT PATHS
    # =============================================================================
    
    project_root: Path = Field(
        default=Path(__file__).parent.parent,
        description="Project root directory"
    )
    
    data_dir: Path = Field(
        default=None,
        description="Data directory path"
    )
    
    models_dir: Path = Field(
        default=None,
        description="Models directory path"
    )
    
    logs_dir: Path = Field(
        default=None,
        description="Logs directory path"
    )
    
    results_dir: Path = Field(
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
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "monitoring" / "logs"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"