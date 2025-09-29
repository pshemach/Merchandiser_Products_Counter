from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from pathlib import Path
import tempfile
import shutil

from src.core.product_counting_system import ProductCountingSystem
from src.utils.logging_utils import setup_logging
from src.api.endpoints import router
from src.api.schemas import ErrorResponse

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


logger = logging.getLogger(__name__)

class ProductCountingAPI:
    """FastAPI application wrapper"""
    
    def __init__(self, system: ProductCountingSystem, config: Dict = None):
        self.system = system
        self.config = config or {}
        self.app = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="product_counting_"))
        self.results_dir = self.temp_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        
        app = FastAPI(
            title="AI-Powered Product Counting System",
            description="REST API for automated product counting using computer vision",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Add exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.error(f"Global exception handler: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal Server Error",
                    message=str(exc)
                ).dict()
            )
        
        # Mount static files for serving results
        app.mount("/results", StaticFiles(directory=str(self.results_dir)), name="results")
        
        # Add dependency injection
        def get_system():
            return self.system
        
        def get_results_dir():
            return self.results_dir
        
        # Include routers with dependencies
        app.include_router(
            router,
            dependencies=[Depends(get_system), Depends(get_results_dir)]
        )
        
        self.app = app
        return app
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

def create_app(system: ProductCountingSystem, config: Dict = None) -> FastAPI:
    """Factory function to create FastAPI app"""
    api = ProductCountingAPI(system, config)
    return api._create_app()

if __name__ == "__main__":
    # Initialize the ProductCountingSystem with settings
    from config.settings import get_settings
    settings = get_settings()
    system = ProductCountingSystem(config=settings.model_dump())
    app = ProductCountingAPI(system, settings.model_dump())
    app.run(host=settings.api_host, port=settings.api_port)