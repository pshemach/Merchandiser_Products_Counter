from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from flask_compress import Compress
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil
import traceback
from functools import wraps
from datetime import datetime
from src.core.product_counting_system import ProductCountingSystem
from src.exceptions.api_exceptions import APIError, RequestValidationError, FileUploadError
from src.utils.logging_utils import setup_logging
from src.utils.file_utils import convert_paths_to_str, get_relative_image_path
from config.settings import get_settings
from src.api.schemas import *

sys.path.insert(0, str(Path(__file__).parent))

# results_dir and outputs_dir
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)

# Initialize ProductCountingSystem
settings = get_settings()
system = ProductCountingSystem(config=settings.dict())
system.load_system_state(results_dir)

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()

# Initialize Flask app
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)

# Error handler
@app.errorhandler(Exception)
def global_exception_handler(exc):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return jsonify(ErrorResponse(
        error="Internal Server Error",
        message=str(exc)
    ).model_dump()), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404
    
@app.route('/', methods=["GET"])
def root():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        return jsonify({"error": "Template not found. Please ensure templates/index.html exists"}), 500

@app.route('/health', methods=["GET"])
def health():
    try: 
        stats = system.get_system_statistics()
        response = {
            "status": "healthy" if system.is_initialized else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": stats["system"],
            "catalog_size": stats["catalog"]["total_products"]
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"error": f"Service unhealthy: {str(e)}"}), 503
    
@app.route('/products', methods=["POST"])
def create_product():
    try:
        data = request.form.to_dict()
        request_data = ProductCreateRequest(**data)
        images = request.files.getlist("images")
            
        if not images:
            return jsonify({"error": "At least one reference image is required"}), 400
        
        image_paths = []
        image_dir = Path("data/reference_images") / request_data.product_id
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Loop through files
        for i, image_file in enumerate(images):
            # Verify image
            if not image_file.content_type.startswith("image/"):
                return jsonify({"error": f"Invalid file type: {image_file.content_type}"}), 400

            filename = secure_filename(image_file.filename)
            file_extension = Path(filename).suffix or ".jpg"
            image_path = image_dir / f"ref_{i}{file_extension}"
            image_file.save(image_path)
            image_paths.append(str(image_path))
            
        product_info = system.add_product_to_catalog(
            product_id=request_data.product_id,
            name=request_data.name,
            image_paths=image_paths,
            category=request_data.category,
            description=request_data.description,
            barcode=request_data.barcode,
            price=request_data.price,
            metadata=request_data.metadata
        )
        
        system.save_system_state(results_dir)
        
        # Fix: Relativize image paths for response (using helper from before)
        rel_images = [get_relative_image_path(img) for img in product_info.reference_images] if product_info.reference_images else []
        
        response = ProductResponse(
            product_id=product_info.product_id,
            name=product_info.name,
            category=product_info.category,
            description=product_info.description,
            barcode=product_info.barcode,
            price=product_info.price,
            reference_images=rel_images,  # New: Pass relativized images
            reference_images_count=len(rel_images),  # Updated: Use rel_images len
            embedding_indices_count=len(product_info.embedding_indices),
            created_at=product_info.created_at,
            updated_at=product_info.updated_at,
            metadata=product_info.metadata
        )    
        
        return jsonify(response.model_dump())
    except Exception as e:
        logger.error(f"Failed to create product: {e}")
        return jsonify({"error": str(e)}), 400
    
@app.route('/products', methods=['GET'])
def list_products():
    category = request.args.get('category')
    try:
        products = system.catalog_manager.list_products(category=category)
        responses = [
            ProductResponse(
                product_id=p.product_id,
                name=p.name,
                category=p.category,
                description=p.description,
                barcode=p.barcode,
                price=p.price,
                reference_images=[get_relative_image_path(img) for img in p.reference_images] if p.reference_images else [],
                reference_images_count=len(p.reference_images),
                embedding_indices_count=len(p.embedding_indices),
                created_at=p.created_at,
                updated_at=p.updated_at,
                metadata=p.metadata
            ).model_dump()
            for p in products
        ]
        return jsonify(responses)
    except Exception as e:
        logger.error(f"Failed to list products: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/products/<product_id>', methods=['GET'])
def get_product(product_id):
    try:
        product = system.catalog_manager.get_product(product_id)
        if not product:
            return jsonify({"error": f"Product not found: {product_id}"}), 404
        response = ProductResponse(
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            description=product.description,
            barcode=product.barcode,
            price=product.price,
            reference_images=[get_relative_image_path(img) for img in product.reference_images] if product.reference_images else [],
            reference_images_count=len(product.reference_images),
            embedding_indices_count=len(product.embedding_indices),
            created_at=product.created_at,
            updated_at=product.updated_at,
            metadata=product.metadata
        )
        return jsonify(response.model_dump())
    except Exception as e:
        logger.error(f"Failed to get product {product_id}: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    try:
        # Use coordinated removal to ensure FAISS index and extractor state are rebuilt
        system.remove_product_from_catalog(product_id, save_state_dir=results_dir)
        return jsonify({"message": f"Product {product_id} deleted successfully"})
    except Exception as e:
        logger.error(f"Failed to delete product {product_id}: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/data/reference_images/<path:filename>', methods=['GET'])
def serve_reference_image(filename):
    try:
        data_dir = Path("data/reference_images")
        return send_from_directory(data_dir, filename)
    except Exception as e:
        logger.error(f"Failed to serve reference image {filename}: {e}")
        return jsonify({"error": "Image not found"}), 404
       
@app.route("/count", methods=["POST"])
def count_products():
    confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
    similarity_threshold = float(request.form.get('similarity_threshold', 0.8))
    return_visualization = request.form.get('return_visualization', 'false').lower() == 'true'
    return_all_detections = request.form.get('return_all_detections', 'false').lower() == 'true'
    return_bounding_boxes = request.form.get('return_bounding_boxes', 'false').lower() == 'true'
    
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image = request.files['image']
    if not image.content_type.startswith('image/'):
        return jsonify({"error": f"Invalid file type: {image.content_type}"}), 400
    
    try:
        image_id = str(uuid.uuid4())
        filename = secure_filename(image.filename)
        file_extension = Path(filename).suffix or '.jpg'
        
        # Use outputs_dir instead of image_dir
        image_path = outputs_dir / f"input_{image_id}{file_extension}"
        image.save(image_path)
        
        result = system.count_products_in_image(
            str(image_path),
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold
        )
        
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
                bounding_boxes=[
                    [int(round(coord)) for coord in bbox]
                    for bbox in details['bounding_boxes']
                ] if return_bounding_boxes else None
            )
            product_count_list.append(product_count_detail)
            product_counts_simple[product_id] = count
        
        product_count_list.sort(key=lambda x: x.count, reverse=True)
        
        total_products = sum(pc.count for pc in product_count_list)
        unique_products = len(product_count_list)
        total_detections = result.total_detections
        unmatched = len(getattr(result, 'unmatched_detections', []))
        detection_rate = total_products / total_detections if total_detections > 0 else 0
        
        summary = CountingSummary(
            total_products_detected=total_products,
            unique_products_detected=unique_products,
            total_objects_detected=total_detections,
            unmatched_objects=unmatched,
            detection_rate=detection_rate
        )
        
        visualization_url = None
        if return_visualization:
            viz_path = outputs_dir / f"viz_{image_id}.jpg"
            system.visualize_results(str(image_path), result, str(viz_path))
            # Return relative path for serving
            visualization_url = f"outputs/viz_{image_id}.jpg"
        
        all_detections = None
        unmatched_detections_list = None
        if return_all_detections:
            all_detections = [
                DetectionInfo(
                    bbox=list(map(int, detection.bbox)),
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    area=int(detection.area),
                    matched_product=match_info['match'].product_id,
                    similarity_score=match_info['match'].similarity
                ).model_dump()
                for match_info in result.matched_detections
            ]
            if hasattr(result, 'unmatched_detections'):
                unmatched_detections_list = [
                    DetectionInfo(
                        bbox=list(map(int, detection.bbox)),
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        area=int(detection.area),
                        matched_product=None,
                        similarity_score=None
                    ).model_dump()
                    for detection in result.unmatched_detections
                ]
        
        response = CountingResponse(
            image_name=filename,
            processing_time=result.processing_time,
            summary=summary,
            product_counts=product_count_list,
            product_counts_simple=product_counts_simple,
            all_detections=all_detections,
            unmatched_detections=unmatched_detections_list,
            visualization_url=visualization_url,
            errors=getattr(result, 'errors', []),
            warnings=[]
        )
        
        return jsonify(response.model_dump())
        
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/count/batch', methods=['POST'])
def batch_count_products():
    confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
    similarity_threshold = float(request.form.get('similarity_threshold', 0.8))
    return_visualization = request.form.get('return_visualization', 'false').lower() == 'true'
    
    images = request.files.getlist('images')
    if not images:
        return jsonify({"error": "No images provided"}), 400
        
    if len(images) > 50:
        return jsonify({"error": "Maximum 50 images per batch"}), 400
    
    try:
        all_results = []
        for i, image_file in enumerate(images):
            if not image_file.content_type.startswith('image/'):
                continue
            image_id = f"batch_{int(time.time())}_{i}"
            filename = secure_filename(image_file.filename)
            file_extension = Path(filename).suffix or '.jpg'
            image_path = outputs_dir / f"input_{image_id}{file_extension}"
            image_file.save(image_path)
            
            result = system.count_products_in_image(
                image_path=str(image_path),
                confidence_threshold=confidence_threshold,
                similarity_threshold=similarity_threshold
            )
            
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
            
            total_products = sum(pc.count for pc in product_count_list)
            summary = CountingSummary(
                total_products_detected=total_products,
                unique_products_detected=len(product_count_list),
                total_objects_detected=result.total_detections,
                unmatched_objects=len(getattr(result, 'unmatched_detections', [])),
                detection_rate=total_products / result.total_detections if result.total_detections > 0 else 0
            )
            
            counting_response = CountingResponse(
                image_name=filename,
                processing_time=result.processing_time,
                summary=summary,
                product_counts=product_count_list,
                product_counts_simple=product_counts_simple,
                errors=getattr(result, 'errors', [])
            )
            all_results.append(counting_response)
        
        aggregated_products = {}
        total_processing_time = 0
        successful_counts = 0
        failed_counts = 0
        total_products_all = 0
        total_objects_all = 0
        total_unmatched_all = 0
        
        for counting_response in all_results:
            total_processing_time += counting_response.processing_time
            if counting_response.errors:
                failed_counts += 1
            else:
                successful_counts += 1
            total_products_all += counting_response.summary.total_products_detected
            total_objects_all += counting_response.summary.total_objects_detected
            total_unmatched_all += counting_response.summary.unmatched_objects
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
        
        aggregated_product_list = []
        total_product_counts_simple = {}
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
        
        aggregated_product_list.sort(key=lambda x: x.count, reverse=True)
        
        aggregated_summary = CountingSummary(
            total_products_detected=total_products_all,
            unique_products_detected=len(aggregated_products),
            total_objects_detected=total_objects_all,
            unmatched_objects=total_unmatched_all,
            detection_rate=total_products_all / total_objects_all if total_objects_all > 0 else 0
        )
        
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
        
        return jsonify(batch_response.model_dump())
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
       
@app.route('/stats', methods=['GET'])
def get_system_stats():
    try:
        stats = system.get_system_statistics()
        response = SystemStatsResponse(
            system_info=stats["system"],
            catalog_stats=stats["catalog"],
            performance_stats={
                "embedding_extractor": stats["embedding_extractor"],
                "similarity_matcher": stats["similarity_matcher"],
                "object_detector": stats["object_detector"]
            },
            last_updated=datetime.now().isoformat()
        )
        # Convert all Path objects to str before jsonify
        response_dict = convert_paths_to_str(response.model_dump())
        return jsonify(response_dict)
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/catalog/export', methods=['GET'])
def export_catalog():
    format_type = request.args.get('format', 'json')
    try:
        if format_type not in ["json", "csv"]:
            return jsonify({"error": "Format must be 'json' or 'csv'"}), 400
        timestamp = int(time.time())
        filename = f"catalog_export_{timestamp}.{format_type}"
        export_path = results_dir / filename
        system.catalog_manager.export_catalog(str(export_path), format_type)
        return send_from_directory(results_dir, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Catalog export failed: {e}")
        return jsonify({"error": str(e)}), 500

# Serve files from outputs directory
@app.route('/outputs/<path:filename>', methods=['GET'])
def serve_output(filename):
    try:
        return send_from_directory(outputs_dir, filename)
    except Exception as e:
        logger.error(f"Failed to serve file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/results/<path:filename>', methods=['GET'])
def serve_result(filename):
    try:
        return send_from_directory(results_dir, filename)
    except Exception as e:
        logger.error(f"Failed to serve file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("data/reference_images", exist_ok=True)
    
    print(f"Starting server on {settings.api_host}:{settings.api_port}")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    
    app.run(host=settings.api_host, port=settings.api_port, debug=settings.debug)