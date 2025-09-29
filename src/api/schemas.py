from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime

class ProductCreateRequest(BaseModel):
    """Request schema for creating a product"""
    product_id: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=200)
    category: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    barcode: Optional[str] = Field(None, max_length=50)
    price: Optional[float] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProductResponse(BaseModel):
    """Response schema for product data"""
    product_id: str
    name: str
    category: Optional[str]
    description: Optional[str]
    barcode: Optional[str]
    price: Optional[float]
    reference_images_count: int
    embedding_indices_count: int
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

class CountingRequest(BaseModel):
    """Request schema for product counting"""
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    similarity_threshold: Optional[float] = Field(0.8, ge=0.0, le=1.0)
    return_visualization: bool = Field(False)
    return_crops: bool = Field(False)

class DetectionInfo(BaseModel):
    """Detection information schema"""
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str
    area: int
    matched_product: Optional[str] = Field(None, description="Matched product ID if recognized")
    similarity_score: Optional[float] = Field(None, description="Similarity score if matched")

class MatchInfo(BaseModel):
    """Match information schema"""
    product_id: str
    product_name: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    embedding_index: int

class ProductCountDetail(BaseModel):
    """Detailed count information for a single product"""
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product display name")
    count: int = Field(..., ge=0, description="Number of times this product was detected")
    confidence_scores: List[float] = Field(default_factory=list, description="Similarity scores for each detection")
    avg_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    bounding_boxes: Optional[List[List[int]]] = Field(default=None, description="Bounding boxes for detections")
    
    class Config:
        schema_extra = {
            "example": {
                "product_id": "coca_cola_500ml",
                "product_name": "Coca Cola 500ml",
                "count": 5,
                "confidence_scores": [0.92, 0.89, 0.95, 0.88, 0.91],
                "avg_confidence": 0.91,
                "bounding_boxes": [[100, 150, 200, 350], [250, 160, 350, 360]]
            }
        }

class CountingSummary(BaseModel):
    """Summary statistics for counting results"""
    total_products_detected: int = Field(..., description="Total number of catalog products detected")
    unique_products_detected: int = Field(..., description="Number of unique product types detected")
    total_objects_detected: int = Field(..., description="Total objects detected (including non-catalog items)")
    unmatched_objects: int = Field(..., description="Objects that couldn't be matched to catalog")
    detection_rate: float = Field(..., description="Percentage of objects that were matched")
    
    class Config:
        schema_extra = {
            "example": {
                "total_products_detected": 15,
                "unique_products_detected": 3,
                "total_objects_detected": 18,
                "unmatched_objects": 3,
                "detection_rate": 0.833
            }
        }

class CountingResponse(BaseModel):
    """Enhanced response schema for counting results with detailed product counts"""
    image_name: str
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Summary statistics
    summary: CountingSummary
    
    # Detailed product counts
    product_counts: List[ProductCountDetail] = Field(
        ..., 
        description="Detailed count information for each detected product"
    )
    
    # Legacy field for backward compatibility
    product_counts_simple: Dict[str, int] = Field(
        ..., 
        description="Simple product_id: count mapping (for backward compatibility)"
    )
    
    # Detection details
    all_detections: Optional[List[DetectionInfo]] = Field(
        None, 
        description="All object detections (if requested)"
    )
    
    unmatched_detections: Optional[List[DetectionInfo]] = Field(
        None,
        description="Detections that couldn't be matched (if requested)"
    )
    
    # Visual outputs
    visualization_url: Optional[str] = Field(None, description="URL to visualization image")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        schema_extra = {
            "example": {
                "image_name": "shelf_001.jpg",
                "processing_time": 1.23,
                "summary": {
                    "total_products_detected": 15,
                    "unique_products_detected": 3,
                    "total_objects_detected": 18,
                    "unmatched_objects": 3,
                    "detection_rate": 0.833
                },
                "product_counts": [
                    {
                        "product_id": "coca_cola_500ml",
                        "product_name": "Coca Cola 500ml",
                        "count": 8,
                        "confidence_scores": [0.92, 0.89, 0.95, 0.88, 0.91, 0.93, 0.87, 0.90],
                        "avg_confidence": 0.906
                    },
                    {
                        "product_id": "pepsi_500ml",
                        "product_name": "Pepsi 500ml", 
                        "count": 5,
                        "confidence_scores": [0.85, 0.88, 0.82, 0.89, 0.86],
                        "avg_confidence": 0.860
                    },
                    {
                        "product_id": "sprite_500ml",
                        "product_name": "Sprite 500ml",
                        "count": 2,
                        "confidence_scores": [0.91, 0.89],
                        "avg_confidence": 0.900
                    }
                ],
                "product_counts_simple": {
                    "coca_cola_500ml": 8,
                    "pepsi_500ml": 5,
                    "sprite_500ml": 2
                },
                "errors": [],
                "warnings": []
            }
        }

class BatchCountingResponse(BaseModel):
    """Response schema for batch counting with aggregated product counts"""
    total_images: int
    successful_counts: int
    failed_counts: int
    total_processing_time: float
    
    # Per-image results
    results: List[CountingResponse]
    
    # Aggregated summary across all images
    aggregated_summary: CountingSummary = Field(
        ...,
        description="Aggregated statistics across all images"
    )
    
    # Aggregated product counts across all images
    aggregated_product_counts: List[ProductCountDetail] = Field(
        ...,
        description="Total counts for each product across all images"
    )
    
    # Simple aggregation for backward compatibility
    total_product_counts: Dict[str, int] = Field(
        ...,
        description="Total count per product across all images"
    )

class SystemStatsResponse(BaseModel):
    """Response schema for system statistics"""
    system_info: Dict[str, Any]
    catalog_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    last_updated: str

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())