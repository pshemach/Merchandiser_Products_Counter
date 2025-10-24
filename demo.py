# app/main.py
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
    status,
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------
from src.core.product_counting_system import ProductCountingSystem
from src.utils.file_utils import convert_paths_to_str, get_relative_image_path
from src.utils.logging_utils import setup_logging
from config.settings import get_settings
from src.api.schemas import *          # all your Pydantic models

# ----------------------------------------------------------------------
# Logging & settings
# ----------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# ----------------------------------------------------------------------
# Global directories (same as Flask)
# ----------------------------------------------------------------------
DATA_DIR = Path("data/reference_images")
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")

for d in (DATA_DIR, RESULTS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Global system (single instance – no client cache)
# ----------------------------------------------------------------------
system = ProductCountingSystem(config=settings.dict())
system.load_system_state(RESULTS_DIR)

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(
    title="Product Counting API (Single Client)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static files & templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------------------------------------------------
# Global exception handlers
# ----------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal Server Error", message=str(exc)).model_dump(),
    )

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    try:
        stats = system.get_system_statistics()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": stats.get("system", {}),
            "catalog_size": stats.get("catalog", {}).get("total_products", 0),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

# ------------------- PRODUCTS -------------------
@app.post("/products", response_model=ProductResponse)
async def create_product(
    product_id: str = Form(...),
    name: str = Form(...),
    category: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    barcode: Optional[str] = Form(None),
    price: Optional[float] = Form(None),
    metadata: Optional[dict] = Form(None),
    images: List[UploadFile] = File(...),
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one reference image is required")

    image_dir = DATA_DIR / product_id
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for idx, img in enumerate(images):
        if not img.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {img.content_type}")

        suffix = Path(img.filename).suffix or ".jpg"
        dest = image_dir / f"ref_{idx}{suffix}"
        content = await img.read()
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    prod = system.add_product_to_catalog(
        product_id=product_id,
        name=name,
        image_paths=saved_paths,
        category=category,
        description=description,
        barcode=barcode,
        price=price,
        metadata=metadata,
    )
    system.save_system_state(RESULTS_DIR)

    rel_imgs = [get_relative_image_path(p) for p in prod.reference_images] if prod.reference_images else []
    return ProductResponse(
        product_id=prod.product_id,
        name=prod.name,
        category=prod.category,
        description=prod.description,
        barcode=prod.barcode,
        price=prod.price,
        reference_images=rel_imgs,
        reference_images_count=len(rel_imgs),
        embedding_indices_count=len(prod.embedding_indices),
        created_at=prod.created_at,
        updated_at=prod.updated_at,
        metadata=prod.metadata,
    )

@app.get("/products", response_model=List[ProductResponse])
async def list_products(category: Optional[str] = None):
    products = system.catalog_manager.list_products(category=category)
    return [
        ProductResponse(
            product_id=p.product_id,
            name=p.name,
            category=p.category,
            description=p.description,
            barcode=p.barcode,
            price=p.price,
            reference_images=[get_relative_image_path(i) for i in p.reference_images] if p.reference_images else [],
            reference_images_count=len(p.reference_images),
            embedding_indices_count=len(p.embedding_indices),
            created_at=p.created_at,
            updated_at=p.updated_at,
            metadata=p.metadata,
        )
        for p in products
    ]

@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str):
    p = system.catalog_manager.get_product(product_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")

    rel_imgs = [get_relative_image_path(i) for i in p.reference_images] if p.reference_images else []
    return ProductResponse(
        product_id=p.product_id,
        name=p.name,
        category=p.category,
        description=p.description,
        barcode=p.barcode,
        price=p.price,
        reference_images=rel_imgs,
        reference_images_count=len(p.reference_images),
        embedding_indices_count=len(p.embedding_indices),
        created_at=p.created_at,
        updated_at=p.updated_at,
        metadata=p.metadata,
    )

@app.delete("/products/{product_id}")
async def delete_product(product_id: str):
    system.remove_product_from_catalog(product_id, save_state_dir=RESULTS_DIR)
    return {"message": f"Product {product_id} deleted successfully"}

# ------------------- IMAGE SERVING -------------------
@app.get("/data/reference_images/{path:path}")
async def serve_reference_image(path: str):
    file_path = DATA_DIR / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(file_path))

@app.get("/outputs/{path:path}")
async def serve_output(path: str):
    file_path = OUTPUTS_DIR / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))

@app.get("/results/{path:path}")
async def serve_result(path: str):
    file_path = RESULTS_DIR / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))

# ------------------- COUNT SINGLE -------------------
@app.post("/count", response_model=CountingResponse)
async def count_products(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    similarity_threshold: float = Form(0.8),
    return_visualization: bool = Form(False),
    return_all_detections: bool = Form(False),
    return_bounding_boxes: bool = Form(False),
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")

    image_id = str(uuid.uuid4())
    suffix = Path(image.filename).suffix or ".jpg"
    input_path = OUTPUTS_DIR / f"input_{image_id}{suffix}"
    input_path.write_bytes(await image.read())

    result = system.count_products_in_image(
        str(input_path),
        confidence_threshold=confidence_threshold,
        similarity_threshold=similarity_threshold,
    )

    # Build product details
    product_details: dict = {}
    for m in result.matched_detections:
        det = m["detection"]
        match = m["match"]
        pid = match.product_id
        if pid not in product_details:
            product_details[pid] = {
                "product_id": pid,
                "product_name": match.product_name,
                "confidence_scores": [],
                "bounding_boxes": [],
            }
        product_details[pid]["confidence_scores"].append(match.similarity)
        if return_bounding_boxes:
            product_details[pid]["bounding_boxes"].append(list(det.bbox))

    product_counts = []
    simple_counts = {}
    for pid, d in product_details.items():
        cnt = len(d["confidence_scores"])
        avg = sum(d["confidence_scores"]) / cnt if cnt else 0.0
        product_counts.append(
            ProductCountDetail(
                product_id=pid,
                product_name=d["product_name"],
                count=cnt,
                confidence_scores=d["confidence_scores"],
                avg_confidence=avg,
                bounding_boxes=[[int(round(c)) for c in bb] for bb in d["bounding_boxes"]]
                if return_bounding_boxes else None,
            )
        )
        simple_counts[pid] = cnt

    product_counts.sort(key=lambda x: x.count, reverse=True)

    summary = CountingSummary(
        total_products_detected=sum(pc.count for pc in product_counts),
        unique_products_detected=len(product_counts),
        total_objects_detected=result.total_detections,
        unmatched_objects=len(getattr(result, "unmatched_detections", [])),
        detection_rate=(
            sum(pc.count for pc in product_counts) / result.total_detections
            if result.total_detections else 0
        ),
    )

    viz_url = None
    if return_visualization:
        viz_path = OUTPUTS_DIR / f"viz_{image_id}.jpg"
        system.visualize_results(str(input_path), result, str(viz_path))
        viz_url = f"outputs/viz_{image_id}.jpg"

    all_dets = None
    unmatched_dets = None
    if return_all_detections:
        all_dets = [
            DetectionInfo(
                bbox=[int(round(c)) for c in m["detection"].bbox],
                confidence=m["detection"].confidence,
                class_id=m["detection"].class_id,
                class_name=m["detection"].class_name,
                area=int(m["detection"].area),
                matched_product=m["match"].product_id,
                similarity_score=m["match"].similarity,
            ).model_dump()
            for m in result.matched_detections
        ]
        if hasattr(result, "unmatched_detections"):
            unmatched_dets = [
                DetectionInfo(
                    bbox=[int(round(c)) for c in d.bbox],
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    area=int(d.area),
                    matched_product=None,
                    similarity_score=None,
                ).model_dump()
                for d in result.unmatched_detections
            ]

    return CountingResponse(
        image_name=image.filename,
        processing_time=result.processing_time,
        summary=summary,
        product_counts=product_counts,
        product_counts_simple=simple_counts,
        all_detections=all_dets,
        unmatched_detections=unmatched_dets,
        visualization_url=viz_url,
        errors=getattr(result, "errors", []),
        warnings=[],
    )

# ------------------- BATCH COUNT -------------------
@app.post("/count/batch", response_model=BatchCountingResponse)
async def batch_count_products(
    images: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5),
    similarity_threshold: float = Form(0.8),
    return_visualization: bool = Form(False),
):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    if len(images) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")

    per_image_results: List[CountingResponse] = []

    for idx, img in enumerate(images):
        if not img.content_type.startswith("image/"):
            continue

        img_id = f"batch_{int(time.time())}_{idx}"
        suffix = Path(img.filename).suffix or ".jpg"
        input_path = OUTPUTS_DIR / f"input_{img_id}{suffix}"
        input_path.write_bytes(await img.read())

        result = system.count_products_in_image(
            str(input_path),
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
        )

        # === Re-use single-count logic (simplified) ===
        product_details = {}
        for m in result.matched_detections:
            det = m["detection"]
            match = m["match"]
            pid = match.product_id
            if pid not in product_details:
                product_details[pid] = {
                    "product_id": pid,
                    "product_name": match.product_name,
                    "confidence_scores": [],
                }
            product_details[pid]["confidence_scores"].append(match.similarity)

        product_counts = []
        simple_counts = {}
        for pid, d in product_details.items():
            cnt = len(d["confidence_scores"])
            avg = sum(d["confidence_scores"]) / cnt if cnt else 0.0
            product_counts.append(
                ProductCountDetail(
                    product_id=pid,
                    product_name=d["product_name"],
                    count=cnt,
                    confidence_scores=d["confidence_scores"],
                    avg_confidence=avg,
                )
            )
            simple_counts[pid] = cnt

        summary = CountingSummary(
            total_products_detected=sum(pc.count for pc in product_counts),
            unique_products_detected=len(product_counts),
            total_objects_detected=result.total_detections,
            unmatched_objects=len(getattr(result, "unmatched_detections", [])),
            detection_rate=(
                sum(pc.count for pc in product_counts) / result.total_detections
                if result.total_detections else 0
            ),
        )

        per_image_results.append(
            CountingResponse(
                image_name=img.filename,
                processing_time=result.processing_time,
                summary=summary,
                product_counts=product_counts,
                product_counts_simple=simple_counts,
                errors=getattr(result, "errors", []),
            )
        )

    # === Aggregate ===
    agg_products = {}
    total_proc = sum(r.processing_time for r in per_image_results)
    successful = sum(1 for r in per_image_results if not r.errors)
    failed = len(per_image_results) - successful
    total_prod = sum(r.summary.total_products_detected for r in per_image_results)
    total_objs = sum(r.summary.total_objects_detected for r in per_image_results)
    total_unm = sum(r.summary.unmatched_objects for r in per_image_results)

    for resp in per_image_results:
        for pc in resp.product_counts:
            pid = pc.product_id
            if pid not in agg_products:
                agg_products[pid] = {
                    "product_id": pid,
                    "product_name": pc.product_name,
                    "total_count": 0,
                    "confidence_scores": [],
                }
            agg_products[pid]["total_count"] += pc.count
            agg_products[pid]["confidence_scores"].extend(pc.confidence_scores)

    agg_list = []
    simple_agg = {}
    for pid, d in agg_products.items():
        cnt = d["total_count"]
        avg = sum(d["confidence_scores"]) / len(d["confidence_scores"]) if d["confidence_scores"] else 0.0
        agg_list.append(
            ProductCountDetail(
                product_id=pid,
                product_name=d["product_name"],
                count=cnt,
                confidence_scores=d["confidence_scores"],
                avg_confidence=avg,
            )
        )
        simple_agg[pid] = cnt

    agg_list.sort(key=lambda x: x.count, reverse=True)

    agg_summary = CountingSummary(
        total_products_detected=total_prod,
        unique_products_detected=len(agg_products),
        total_objects_detected=total_objs,
        unmatched_objects=total_unm,
        detection_rate=total_prod / total_objs if total_objs else 0,
    )

    return BatchCountingResponse(
        total_images=len(images),
        successful_counts=successful,
        failed_counts=failed,
        total_processing_time=total_proc,
        results=per_image_results,
        aggregated_summary=agg_summary,
        aggregated_product_counts=agg_list,
        total_product_counts=simple_agg,
    )

# ------------------- STATS & EXPORT -------------------
@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    stats = system.get_system_statistics()
    resp = SystemStatsResponse(
        system_info=stats["system"],
        catalog_stats=stats["catalog"],
        performance_stats={
            "embedding_extractor": stats["embedding_extractor"],
            "similarity_matcher": stats["similarity_matcher"],
            "object_detector": stats["object_detector"],
        },
        last_updated=datetime.utcnow().isoformat(),
    )
    return JSONResponse(content=convert_paths_to_str(resp.model_dump()))

@app.get("/catalog/export")
async def export_catalog(format: str = "json"):
    if format not in ("json", "csv"):
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
    ts = int(time.time())
    fname = f"catalog_export_{ts}.{format}"
    path = RESULTS_DIR / fname
    system.catalog_manager.export_catalog(str(path), format)
    return FileResponse(str(path), media_type="application/octet-stream", filename=fname)

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )