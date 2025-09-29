from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import asyncio
import aiofiles
from pathlib import Path
from typing import List
import uuid
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.core.product_counting_system import ProductCountingSystem
from src.api.schemas import *

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
    confidence_threshold: float = 0.5,
    similarity_threshold: float = 0.8,
    return_visualization: bool = False,
    return_all_detections: bool = False,
    return_bounding_boxes: bool = False,
    system: ProductCountingSystem = Depends(get_system),
    results_dir: Path = Depends(get_results_dir)
):
    """
    Count products in a single image with detailed per-product information
    
    Args:
        image: Image file to process
        confidence_threshold: Detection confidence threshold (0.0-1.0)
        similarity_threshold: Product matching threshold (0.0-1.0)
        return_visualization: Include visualization image URL in response
        return_all_detections: Include all detection details in response
        return_bounding_boxes: Include bounding box coordinates in response
    
    Returns:
        Detailed counting results with per-product counts and confidence scores
    """
    
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
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold
        )
        
        # Build detailed product counts
        product_details = {}
        
        for match_info in result.matched_detections:
            detection = match_info['detection']
            match = match_info['match']
            product_id = match.product_id
            
            if product_id not in product_details:
                product_details[product_id] = {
                    'product_id': product_id,
                    'product_name': match.product_name,
                    'confidence_scores': [],
                    'bounding_boxes': []
                }
            
            product_details[product_id]['confidence_scores'].append(match.similarity)
            
            if return_bounding_boxes:
                product_details[product_id]['bounding_boxes'].append(list(detection.bbox))
        
        # Create ProductCountDetail objects
        product_count_list = []
        product_counts_simple = {}
        
        for product_id, details in product_details.items():
            count = len(details['confidence_scores'])
            avg_confidence = sum(details['confidence_scores']) / count if count > 0 else 0
            
            product_count_detail = ProductCountDetail(
                product_id=details['product_id'],
                product_name=details['product_name'],
                count=count,
                confidence_scores=details['confidence_scores'],
                avg_confidence=avg_confidence,
                bounding_boxes=details['bounding_boxes'] if return_bounding_boxes else None
            )
            
            product_count_list.append(product_count_detail)
            product_counts_simple[product_id] = count
        
        # Sort by count (descending)
        product_count_list.sort(key=lambda x: x.count, reverse=True)
        
        # Create summary
        total_products = sum(pc.count for pc in product_count_list)
        unique_products = len(product_count_list)
        total_detections = result.total_detections
        unmatched = len(result.unmatched_detections) if hasattr(result, 'unmatched_detections') else 0
        detection_rate = total_products / total_detections if total_detections > 0 else 0
        
        summary = CountingSummary(
            total_products_detected=total_products,
            unique_products_detected=unique_products,
            total_objects_detected=total_detections,
            unmatched_objects=unmatched,
            detection_rate=detection_rate
        )
        
        # Create visualization if requested
        visualization_url = None
        if return_visualization:
            viz_path = results_dir / f"viz_{image_id}.jpg"
            system.visualize_results(image_path, result, viz_path)
            visualization_url = f"/results/viz_{image_id}.jpg"
        
        # Prepare all detections if requested
        all_detections = None
        unmatched_detections_list = None
        
        if return_all_detections:
            all_detections = []
            
            for match_info in result.matched_detections:
                detection = match_info['detection']
                match = match_info['match']
                
                all_detections.append(DetectionInfo(
                    bbox=list(detection.bbox),
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    area=detection.area,
                    matched_product=match.product_id,
                    similarity_score=match.similarity
                ))
            
            # Add unmatched detections
            if hasattr(result, 'unmatched_detections'):
                unmatched_detections_list = []
                for detection in result.unmatched_detections:
                    unmatched_detections_list.append(DetectionInfo(
                        bbox=list(detection.bbox),
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        area=detection.area,
                        matched_product=None,
                        similarity_score=None
                    ))
        
        # Create response
        response = CountingResponse(
            image_name=image.filename,
            processing_time=result.processing_time,
            summary=summary,
            product_counts=product_count_list,
            product_counts_simple=product_counts_simple,
            all_detections=all_detections,
            unmatched_detections=unmatched_detections_list,
            visualization_url=visualization_url,
            errors=result.errors if hasattr(result, 'errors') else [],
            warnings=[]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/count/batch", response_model=BatchCountingResponse)
async def batch_count_products(
    images: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5,
    similarity_threshold: float = 0.8,
    return_visualization: bool = False,
    return_all_detections: bool = False,
    background_tasks: BackgroundTasks = None,
    system: ProductCountingSystem = Depends(get_system),
    results_dir: Path = Depends(get_results_dir)
):
    """
    Count products in multiple images with aggregated results
    
    Returns detailed per-product counts aggregated across all images
    """
    
    if len(images) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    try:
        # Process all images
        all_results = []
        
        for image_file in images:
            if not image_file.content_type.startswith('image/'):
                continue
            
            # Process individual image (reuse single image endpoint logic)
            image_id = f"batch_{int(time.time())}_{len(all_results)}"
            file_extension = Path(image_file.filename).suffix or '.jpg'
            image_path = results_dir / f"input_{image_id}{file_extension}"
            
            async with aiofiles.open(image_path, 'wb') as f:
                content = await image_file.read()
                await f.write(content)
            
            # Process
            result = system.count_products_in_image(
                image_path=image_path,
                confidence_threshold=confidence_threshold,
                similarity_threshold=similarity_threshold
            )
            
            # Build detailed product counts
            product_details = {}
            
            for match_info in result.matched_detections:
                detection = match_info['detection']
                match = match_info['match']
                product_id = match.product_id
                
                if product_id not in product_details:
                    product_details[product_id] = {
                        'product_id': product_id,
                        'product_name': match.product_name,
                        'confidence_scores': [],
                        'bounding_boxes': []
                    }
                
                product_details[product_id]['confidence_scores'].append(match.similarity)
            
            # Create product count list for this image
            product_count_list = []
            product_counts_simple = {}
            
            for product_id, details in product_details.items():
                count = len(details['confidence_scores'])
                avg_confidence = sum(details['confidence_scores']) / count if count > 0 else 0
                
                product_count_detail = ProductCountDetail(
                    product_id=details['product_id'],
                    product_name=details['product_name'],
                    count=count,
                    confidence_scores=details['confidence_scores'],
                    avg_confidence=avg_confidence
                )
                
                product_count_list.append(product_count_detail)
                product_counts_simple[product_id] = count
            
            # Create summary for this image
            total_products = sum(pc.count for pc in product_count_list)
            
            summary = CountingSummary(
                total_products_detected=total_products,
                unique_products_detected=len(product_count_list),
                total_objects_detected=result.total_detections,
                unmatched_objects=len(getattr(result, 'unmatched_detections', [])),
                detection_rate=total_products / result.total_detections if result.total_detections > 0 else 0
            )
            
            # Create counting response for this image
            counting_response = CountingResponse(
                image_name=image_file.filename,
                processing_time=result.processing_time,
                summary=summary,
                product_counts=product_count_list,
                product_counts_simple=product_counts_simple,
                errors=getattr(result, 'errors', [])
            )
            
            all_results.append(counting_response)
        
        # Aggregate results across all images
        aggregated_products = {}
        total_processing_time = 0
        successful_counts = 0
        failed_counts = 0
        
        # Aggregate statistics
        total_products_all = 0
        total_objects_all = 0
        total_unmatched_all = 0
        
        for counting_response in all_results:
            total_processing_time += counting_response.processing_time
            
            if counting_response.errors:
                failed_counts += 1
            else:
                successful_counts += 1
            
            # Aggregate summaries
            total_products_all += counting_response.summary.total_products_detected
            total_objects_all += counting_response.summary.total_objects_detected
            total_unmatched_all += counting_response.summary.unmatched_objects
            
            # Aggregate product counts
            for product_detail in counting_response.product_counts:
                product_id = product_detail.product_id
                
                if product_id not in aggregated_products:
                    aggregated_products[product_id] = {
                        'product_id': product_id,
                        'product_name': product_detail.product_name,
                        'total_count': 0,
                        'confidence_scores': []
                    }
                
                aggregated_products[product_id]['total_count'] += product_detail.count
                aggregated_products[product_id]['confidence_scores'].extend(product_detail.confidence_scores)
        
        # Create aggregated product count details
        aggregated_product_list = []
        total_product_counts_simple = {}
        unique_products_all = len(aggregated_products)
        
        for product_id, details in aggregated_products.items():
            total_count = details['total_count']
            avg_confidence = sum(details['confidence_scores']) / len(details['confidence_scores']) if details['confidence_scores'] else 0
            
            aggregated_detail = ProductCountDetail(
                product_id=details['product_id'],
                product_name=details['product_name'],
                count=total_count,
                confidence_scores=details['confidence_scores'],
                avg_confidence=avg_confidence
            )
            
            aggregated_product_list.append(aggregated_detail)
            total_product_counts_simple[product_id] = total_count
        
        # Sort by count (descending)
        aggregated_product_list.sort(key=lambda x: x.count, reverse=True)
        
        # Create aggregated summary
        aggregated_summary = CountingSummary(
            total_products_detected=total_products_all,
            unique_products_detected=unique_products_all,
            total_objects_detected=total_objects_all,
            unmatched_objects=total_unmatched_all,
            detection_rate=total_products_all / total_objects_all if total_objects_all > 0 else 0
        )
        
        # Create batch response
        batch_response = BatchCountingResponse(
            total_images=len(images),
            successful_counts=successful_counts,
            failed_counts=failed_counts,
            total_processing_time=total_processing_time,
            results=all_results,
            aggregated_summary=aggregated_summary,
            aggregated_product_counts=aggregated_product_list,
            total_product_counts=total_product_counts_simple
        )
        
        return batch_response
        
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

import logging
logger = logging.getLogger(__name__)