import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from src.core.object_detector import Detection, YOLODetector
from src.core.embedding_extractor import ImprovedEmbeddingExtractor
from src.core.similarity_matcher import SimilarityMatch, SimilarityMatcher
from src.core.catalog_manager import ProductCatalogManager, ProductInfo

from src.exceptions.core_exceptions import (
    SystemInitializationError, ObjectDetectionError, 
    EmbeddingExtractionError, SimilarityMatchingError, CatalogError
)

from src.utils.image_utils import validate_image, load_image, crop_image, save_image
from src.utils.validation import validate_bounding_box, validate_similarity_score
from src.utils.file_utils import ensure_dir, save_json, load_json
from src.utils.logging_utils import PerformanceLogger
from datetime import datetime

logger = logging.getLogger(__name__)

class ProductCountingResult:
    """Result container for product counting operations"""
    def __init__(self, image_path: str, processing_time: float):
        self.image_path = image_path
        self.processing_time = processing_time
        self.total_detections = 0
        self.matched_detections = []
        self.product_counts = {}
        self.unmatched_detections = []
        self.errors = []
        self.metadata = {}
        
    def add_matched_detection(self, detection: Detection, match: SimilarityMatch) -> None:
        """Add a matched detection"""
        self.matched_detections.append({
            'detection': detection,
            'match': match,
            'crop_coords': detection.bbox
        })
        
        # Update product counts
        product_id = match.product_id
        if product_id in self.product_counts:
            self.product_counts[product_id] += 1
        else:
            self.product_counts[product_id] = 1
                   
    def add_unmatched_detection(self, detection: Detection) -> None:
        """Add an unmatched detection"""
        self.unmatched_detections.append(detection)
    
    def add_error(self, error_message: str) -> None:
        """Add an error message"""
        self.errors.append(error_message)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'image_path': self.image_path,
            'processing_time': self.processing_time,
            'total_detections': self.total_detections,
            'matched_detections_count': len(self.matched_detections),
            'product_counts': self.product_counts,
            'unmatched_detections_count': len(self.unmatched_detections),
            'errors': self.errors,
            'metadata': self.metadata
        }
        
        
class ProductCountingSystem:
    """Main product counting system integrating all components"""
    
    def __init__(self, config: Dict = None):
        """Initialize the product counting system"""
        
        # Default configuration
        default_config = {
            'yolo_model_dir':"data/models/yolo_weights",
            'yolo_model_name': 'yolov9.pt',
            'embedding_model': 'facebook/dinov2-base', 
            'normalization_strategy': 'catalog_norm',
            'similarity_threshold': 0.8,
            'confidence_threshold': 0.5,
            'device': 'auto',
            'index_type': 'IndexFlatIP'
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize components
        self.detector = None
        self.embedding_extractor = None
        self.similarity_matcher = None
        self.catalog_manager = None
        
        # System state
        self.is_initialized = False
        self.system_stats = {
            'total_predictions': 0,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'catalog_size': 0
        }
        
        logger.info("Initializing Product Counting System...")
        self._initialize_system()
        
    def _initialize_system(self) -> None:
        """Initialize all system components"""
        
        try:
            with PerformanceLogger(logger, "System initialization"):
                # Initialize object detector
                logger.info("Loading YOLO detector...")
                model_path = self.config['yolo_model_dir'] / self.config['yolo_model_name']
                self.detector = YOLODetector(
                    model_name=model_path,
                    device=self.config['device']
                )
                
                # Initialize embedding extractor
                logger.info("Loading embedding extractor...")
                self.embedding_extractor = ImprovedEmbeddingExtractor(
                    model_name=self.config['embedding_model'],
                    normalization_strategy=self.config['normalization_strategy'],
                    device=self.config['device']
                )
                
                # Initialize similarity matcher
                logger.info("Loading similarity matcher...")
                embedding_dim = self.embedding_extractor.embedding_dim
                self.similarity_matcher = SimilarityMatcher(
                    embedding_dim=embedding_dim,
                    index_type=self.config['index_type']
                )
                
                # Initialize catalog manager
                logger.info("Initializing catalog manager...")
                self.catalog_manager = ProductCatalogManager()
                
                self.is_initialized = True
                logger.info("System initialization completed successfully")
                           
        except Exception as e:
            logger.error(f"System initialization fail: {e}")
            raise SystemInitializationError(f"Initialization fail: {e}")
        
    def add_product_to_catalog(self, product_id: str, name: str, 
                               image_paths: List[str], **kwargs) -> ProductInfo:
        """Add product to the catalog with embedding"""
        if not self.is_initialized:
            raise SystemInitializationError("System not initialized")
        
        try:
            with PerformanceLogger(logger, f"Adding product {product_id} to catalog"):
                # Add product to catalog manager
                product_info = self.catalog_manager.add_product(product_id=product_id, name=name,
                                                        reference_images=image_paths, **kwargs)
                # Process reference images and create embeddings
                embeddings = []
                for image_path in image_paths:
                    try:
                        image = load_image(image_path)
                        
                        embedding = self.embedding_extractor.extract_embedding(image)
                        embeddings.append(embedding)
                        
                        # Add to similarity matcher
                        embedding_index = self.similarity_matcher.index.ntotal
                        self.similarity_matcher.add_embedding(
                            embedding=embedding,
                            product_id=product_id,
                            metadata={'name': name, 'image_path': image_path}
                        )
                        
                        # Update catalog with embedding index
                        self.catalog_manager.add_embedding_index(product_id, embedding_index)
                        
                    except Exception as e:
                        logger.error(f"Failed to add reference image {image_path}: {e}")
                        continue

                if not embeddings:
                    raise CatalogError(f"No valid embeddings created for product {product_id}")
                
                # Update embedding extractor with catalog embeddings
                for embedding in embeddings:
                    self.embedding_extractor.add_catalog_embedding(embedding=embedding)
                
                # Update system statistics
                self.system_stats['catalog_size'] = len(self.catalog_manager.products)
                
                logger.info(f"Successfully added product {product_id} with {len(embeddings)} embeddings")
                return product_info
                
        except Exception as e:
            logger.error(f"Failed to add product {product_id}: {e}")
            raise CatalogError(f"Failed to add product: {e}")
        
    def count_products_in_image(self, image_path: Union[str, Path],
                                confidence_threshold: Optional[float] = None,
                                similarity_threshold: Optional[float] = None) -> ProductCountingResult:
        """Count products in a single image"""
        if not self.is_initialized:
            raise SystemInitializationError("System not initialized")
        
        # Use config defaults if not specified
        if confidence_threshold is None:
            confidence_threshold = self.config['confidence_threshold']
        if similarity_threshold is None:
            similarity_threshold = self.config['similarity_threshold']
            
        start_time = time.time()
        result = ProductCountingResult(str(image_path), 0)
        
        try:
            # Load and validate image
            if not validate_image(image_path):
                raise ValueError(f"Invalid image: {image_path}")
            
            image = load_image(image_path)
            
            with PerformanceLogger(logger, f"Processing image {Path(image_path).name}"):
                detections = self.detector.detect(
                    image=image,
                    confidence_threshold=confidence_threshold
                )
                
                result.total_detections = len(detections)
                logger.info(f"Detected {len(detections)} detections")
                
                for i, detection in enumerate(detections):
                    try:
                        # Validate bounding box
                        if not validate_bounding_box(detection.bbox, image.shape):
                            logger.warning(f"Invalid bounding box for detection {i}")
                            continue
                        
                        # Cropped detected object
                        cropped_image = crop_image(image, detection.bbox)
                        # Extract embedding 
                        embedding = self.embedding_extractor.extract_embedding(cropped_image)
            
                        # Find similar products
                        matches = self.similarity_matcher.search(
                            query_embedding=embedding,
                            k=3,
                            similarity_threshold=similarity_threshold
                        ) 
                        
                        if matches:
                            # Use best match
                            best_match = matches[0]
                    
                            result.add_matched_detection(detection=detection,
                                                         match=best_match)   
                            logger.debug(f"Detection {i}: matched {best_match.product_id} "
                                       f"(similarity: {best_match.similarity:.3f})")
                        else:
                            # No match found
                            result.add_unmatched_detection(detection=detection)
                            logger.debug(f"Detection {i}: No found")
                        
                        
                    except Exception as e:
                        error_msg = f"Failed to process detection {i}: {str(e)}"
                        logger.error(error_msg)
                        result.add_error(error_msg)
                        continue
                # Calculate processing time
                result.processing_time = time.time() - start_time
                
                # Update system statistics
                self.system_stats['total_predictions'] += 1
                self.system_stats['total_processing_time'] += result.processing_time
                self.system_stats['average_processing_time'] = self.system_stats['total_processing_time'] / self.system_stats['total_predictions']
                
                # Add metadata
                result.metadata = {
                    'confidence_threshold': confidence_threshold,
                    'similarity_threshold': similarity_threshold,
                    'image_shape': image.shape,
                    'config': self.config
                }
                
                logger.info(f"Processed {Path(image_path).name}: "
                          f"{len(result.matched_detections)} matches in {result.processing_time:.2f}s")
                
                return result
  
        except Exception as e:
            result.processing_time = time.time() - start_time
            error_msg = f"Image processing failed: {str(e)}"
            logger.error(error_msg)
            result.add_error(error_msg)
            return result
    
    def batch_count_products(self, image_paths: List[Union[str, Path]], 
                           **kwargs) -> List[ProductCountingResult]:
        """Count products in batch of images"""
        
        results = []
        
        with PerformanceLogger(logger, f"Batch processing {len(image_paths)} images"):
            for i, image_path in enumerate(image_paths):
                try:
                    result = self.count_products_in_image(image_path, **kwargs)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process image {i} ({image_path}): {e}")
                    # Create error result
                    error_result = ProductCountingResult(str(image_path), 0)
                    error_result.add_error(f"Processing failed: {e}")
                    results.append(error_result)
        
        return results
    
    def visualize_results(self, image_path: Union[str, Path], 
                         result: ProductCountingResult,
                         save_path: Optional[Path] = None) -> np.ndarray:
        """Visualize counting results on image"""
        
        try:
            image = load_image(image_path)
            vis_image = image.copy()
            
            # Draw matched detections in green
            for match_info in result.matched_detections:
                detection = match_info['detection']
                match = match_info['match']
                
                x1, y1, x2, y2 = map(int, detection.bbox)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{match.product_name}\n{match.similarity:.3f}"
                
                # Calculate label size and position
                lines = label.split('\n')
                line_height = 20
                
                for i, line in enumerate(lines):
                    y_pos = y1 - 10 - (len(lines) - i - 1) * line_height
                    cv2.putText(vis_image, line, (x1, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw unmatched detections in red
            for detection in result.unmatched_detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_image, "Unknown", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add summary text
            summary_text = [
                f"Total: {result.total_detections}",
                f"Matched: {len(result.matched_detections)}",
                f"Time: {result.processing_time:.2f}s"
            ]
            
            for i, text in enumerate(summary_text):
                cv2.putText(vis_image, text, (10, 30 + i * 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save if requested
            if save_path:
                save_image(vis_image, save_path)
                logger.info(f"Saved visualization to {save_path}")
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise
        
    def get_system_statistics(self) -> Dict[str, any]:
        """Get comprehensive system statistics"""
        
        catalog_stats = self.catalog_manager.get_statistics() if self.catalog_manager else {}
        similarity_stats = self.similarity_matcher.get_statistics() if self.similarity_matcher else {}
        extractor_stats = self.embedding_extractor.get_stats_summary() if self.embedding_extractor else {}
        detector_info = self.detector.get_model_info() if self.detector else {}
        
        return {
            'system': {
                'is_initialized': self.is_initialized,
                'config': self.config,
                **self.system_stats
            },
            'catalog': catalog_stats,
            'similarity_matcher': similarity_stats,
            'embedding_extractor': extractor_stats,
            'object_detector': detector_info
        }
    def save_system_state(self, directory: Path) -> None:
        """Save complete system state"""
        
        try:
            directory = Path(directory)
            ensure_dir(directory=directory)
            
            with PerformanceLogger(logger, "Saving system state"):
                
                # Save catalog
                if self.catalog_manager:
                    catalog_file = directory / "catalog.json"
                    self.catalog_manager.save_catalog(catalog_file)
                    
                # Save similarity index
                if self.similarity_matcher:
                    index_file = directory / "similarity_index.fiass"
                    self.similarity_matcher.save_index(index_file)
                    
                # Save embedding extractor state
                if self.embedding_extractor:
                    extractor_file = directory / "extractor_state.pkl"
                    self.embedding_extractor.save_state(extractor_file)
                    
                # Save system config and stats
                system_info = {
                    'config': self.config,
                    'stats': self.system_stats,
                    'version': '1.0',
                    'saved_at': datetime.now().isoformat()
                }
                
                system_file = directory / "system_info.json"
                save_json(system_info, system_file)
                
            logger.info(f"System state saved to {directory}")
            
            
        except Exception as e:
            logger.error(f"Failed to save system state {e}")
            raise
        
    def load_system_state(self, directory: Path) -> None:
        """Load complete system state"""
        try:
            directory = Path(directory)
            
            if not directory.exists():
                raise FileNotFoundError(f"State directory not found: {directory}")
            
            with PerformanceLogger(logger, "Loading system state"):
                
                # Load system info
                system_file = directory / "system_info.json"
                if system_file.exists():
                    system_info = load_json(system_file)
                    self.config.update(system_info.get('config', {}))
                    self.system_stats.update(system_info.get('stats', {}))
                
                # Load catalog
                catalog_file = directory / "catalog.json"
                if catalog_file.exists() and self.catalog_manager:
                    self.catalog_manager.load_catalog(catalog_file)
                
                # Load similarity index
                index_file = directory / "similarity_index.fiass"
                if index_file.exists() and self.similarity_matcher:
                    self.similarity_matcher.load_index(index_file)
                
                # Load embedding extractor state
                extractor_file = directory / "extractor_state.pkl"
                if extractor_file.exists() and self.embedding_extractor:
                    self.embedding_extractor.load_state(extractor_file)
                
            logger.info(f"System state loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            raise