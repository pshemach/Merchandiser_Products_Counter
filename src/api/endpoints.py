from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Optional
import uuid
import time
from datetime import datetime
import logging

from src.core.product_counting_system import ProductCountingSystem
from src.api.schemas import *

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
async def get_system():
    # This will be overridden by the dependency injection in app.py
    pass

async def get_results_dir():
    # This will be overridden by the dependency injection in app.py
    pass

@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "AI-Powered Product Counting System",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }

@router.get("/health", response_model=Dict[str, Any])
async def health_check(system: ProductCountingSystem = Depends(get_system)):
    """Health check endpoint"""
    try:
        stats = system.get_system_statistics()
        return {
            "status": "healthy" if system.is_initialized else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": stats["system"],
            "catalog_size": stats["catalog"]["total_products"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Product Management Endpoints
@router.post("/products", response_model=ProductResponse)
async def create_product(
    request: ProductCreateRequest,
    images: List[UploadFile] = File(...),
    system: ProductCountingSystem = Depends(get_system),
    results_dir: Path = Depends(get_results_dir)
):
    """Create a new product with reference images"""
    
    if not images:
        raise HTTPException(status_code=400, detail="At least one reference image is required")
    
    try:
        # Save uploaded images
        image_paths = []
        product_dir = results_dir / "products" / request.product_id
        product_dir.mkdir(parents=True, exist_ok=True)
        
        for i, image_file in enumerate(images):
            # Validate file type
            if not image_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {image_file.content_type}")
            
            # Save file
            file_extension = Path(image_file.filename).suffix or '.jpg'
            image_path = product_dir / f"ref_{i}{file_extension}"
            
            async with aiofiles.open(image_path, 'wb') as f:
                content = await image_file.read()
                await f.write(content)
            
            image_paths.append(str(image_path))
        
        # Add product to system
        product_info = system.add_product_to_catalog(
            product_id=request.product_id,
            name=request.name,
            image_paths=image_paths,
            category=request.category,
            description=request.description,
            barcode=request.barcode,
            price=request.price,
            metadata=request.metadata
        )
        
        # Convert to response
        return ProductResponse(
            product_id=product_info.product_id,
            name=product_info.name,
            category=product_info.category,
            description=product_info.description,
            barcode=product_info.barcode,
            price=product_info.price,
            reference_images_count=len(product_info.reference_images),
            embedding_indices_count=len(product_info.embedding_indices),
            created_at=product_info.created_at,
            updated_at=product_info.updated_at,
            metadata=product_info.metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to create product {request.product_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/products", response_model=List[ProductResponse])
async def list_products(
    category: Optional[str] = None,
    system: ProductCountingSystem = Depends(get_system)
):
    """List all products or filter by category"""
    try:
        products = system.catalog_manager.list_products(category=category)
        
        return [
            ProductResponse(
                product_id=p.product_id,
                name=p.name,
                category=p.category,
                description=p.description,
                barcode=p.barcode,
                price=p.price,
                reference_images_count=len(p.reference_images),
                embedding_indices_count=len(p.embedding_indices),
                created_at=p.created_at,
                updated_at=p.updated_at,
                metadata=p.metadata
            )
            for p in products
        ]
        
    except Exception as e:
        logger.error(f"Failed to list products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: str,
    system: ProductCountingSystem = Depends(get_system)
):
    """Get specific product information"""
    try:
        product = system.catalog_manager.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")
        
        return ProductResponse(
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            description=product.description,
            barcode=product.barcode,
            price=product.price,
            reference_images_count=len(product.reference_images),
            embedding_indices_count=len(product.embedding_indices),
            created_at=product.created_at,
            updated_at=product.updated_at,
            metadata=product.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/products/{product_id}")
async def delete_product(
    product_id: str,
    system: ProductCountingSystem = Depends(get_system)
):
    """Delete a product"""
    try:
        system.catalog_manager.remove_product(product_id)
        return {"message": f"Product {product_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete product {product_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Counting Endpoints
@router.post("/count", response_model=CountingResponse)
async def count_products(
    image: UploadFile = File(...),
    request: CountingRequest = CountingRequest(),
    system: ProductCountingSystem = Depends(get_system),
    results_dir: Path = Depends(get_results_dir)
):
    """Count products in a single image"""
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")
    
    try:
        # Save uploaded image
        image_id = str(uuid.uuid4())
        file_extension = Path(image.filename).suffix or '.jpg'
        image_path = results_dir / f"input_{image_id}{file_extension}"
        
        async with aiofiles.open(image_path, 'wb') as f:
            content = await image.read()
            await f.write(content)
        
        # Process image
        result = system.count_products_in_image(
            image_path=image_path,
            confidence_threshold=request.confidence_threshold,
            similarity_threshold=request.similarity_threshold
        )
        
        # Create visualization if requested
        visualization_url = None
        if request.return_visualization:
            viz_path = results_dir / f"viz_{image_id}.jpg"
            system.visualize_results(image_path, result, viz_path)
            visualization_url = f"/results/viz_{image_id}.jpg"
        
        return CountingResponse(
            image_name=image.filename,
            processing_time=result.processing_time,
            total_detections=result.total_detections,
            matched_detections_count=len(result.matched_detections),
            unmatched_detections_count=len(result.unmatched_detections),
            product_counts=result.product_counts,
            errors=result.errors,
            visualization_url=visualization_url
        )
        
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/count/batch", response_model=BatchCountingResponse)
async def batch_count_products(
    images: List[UploadFile] = File(...),
    request: CountingRequest = CountingRequest(),
    background_tasks: BackgroundTasks = None,
    system: ProductCountingSystem = Depends(get_system),
    results_dir: Path = Depends(get_results_dir)
):
    """Count products in multiple images"""
    
    if len(images) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    try:
        # Save all uploaded images
        image_paths = []
        image_names = []
        
        for i, image_file in enumerate(images):
            if not image_file.content_type.startswith('image/'):
                continue
            
            image_id = f"batch_{int(time.time())}_{i}"
            file_extension = Path(image_file.filename).suffix or '.jpg'
            image_path = results_dir / f"input_{image_id}{file_extension}"
            
            async with aiofiles.open(image_path, 'wb') as f:
                content = await image_file.read()
                await f.write(content)
            
            image_paths.append(image_path)
            image_names.append(image_file.filename)
        
        # Process images
        results = system.batch_count_products(
            image_paths=image_paths,
            confidence_threshold=request.confidence_threshold,
            similarity_threshold=request.similarity_threshold
        )
        
        # Create response
        counting_responses = []
        successful_counts = 0
        failed_counts = 0
        total_processing_time = 0
        summary = {}
        
        for result, image_name in zip(results, image_names):
            total_processing_time += result.processing_time
            
            if result.errors:
                failed_counts += 1
            else:
                successful_counts += 1
                
                # Update summary
                for product_id, count in result.product_counts.items():
                    if product_id in summary:
                        summary[product_id] += count
                    else:
                        summary[product_id] = count
            
            counting_responses.append(CountingResponse(
                image_name=image_name,
                processing_time=result.processing_time,
                total_detections=result.total_detections,
                matched_detections_count=len(result.matched_detections),
                unmatched_detections_count=len(result.unmatched_detections),
                product_counts=result.product_counts,
                errors=result.errors
            ))
        
        return BatchCountingResponse(
            total_images=len(images),
            successful_counts=successful_counts,
            failed_counts=failed_counts,
            total_processing_time=total_processing_time,
            results=counting_responses,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# System Information Endpoints
@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(system: ProductCountingSystem = Depends(get_system)):
    """Get comprehensive system statistics"""
    try:
        stats = system.get_system_statistics()
        
        return SystemStatsResponse(
            system_info=stats["system"],
            catalog_stats=stats["catalog"],
            performance_stats={
                "embedding_extractor": stats["embedding_extractor"],
                "similarity_matcher": stats["similarity_matcher"],
                "object_detector": stats["object_detector"]
            },
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/catalog/export")
async def export_catalog(
    format: str = "json",
    system: ProductCountingSystem = Depends(get_system),
    results_dir: Path = Depends(get_results_dir)
):
    """Export product catalog"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        timestamp = int(time.time())
        filename = f"catalog_export_{timestamp}.{format}"
        export_path = results_dir / filename
        
        system.catalog_manager.export_catalog(export_path, format)
        
        return FileResponse(
            path=export_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Catalog export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))