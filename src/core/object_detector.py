import torch
import logging
import cv2
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from src.utils.image_utils import validate_image
from src.utils.logging_utils import PerformanceLogger
from src.exceptions.core_exceptions import ObjectDetectionError

logger = logging.getLogger(__name__)

class Detection:
    """Data class for object detection results"""
    def __init__(self, bbox: Tuple[int,int,int,int], confidence: float,
                 class_id: int, class_name: str=None):
        self.bbox = bbox #(x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
        self.center = (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )
        
        self.width = self.bbox[2] - self.bbox[0]
        self.hight = self.bbox[3] - self.bbox[1]
        self.area = self.width * self.hight
        
    def to_dict(self) -> Dict:
        """Convert detection to dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area
        }
        
class YOLODetector:
    """YOLO object detector wrapper with enhanced functionality"""
    def __init__(self, model_name: str = 'yolov9.pt', device: str = 'auto'):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.class_names = {}
        
        logger.info(f"Initiating YOLO detector: {self.model_name} on {self.device}")
        self._load_model()
        
    def _resolve_device(self, device:str) -> str:
        """Resolve device string to actual device"""
        if device != 'auto':
            return 'cpu'
        try: 
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu' 
        
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            with PerformanceLogger(logger, f"Loading YOLO model {self.model_name}"):
                self.model = YOLO(self.model_name)

                if hasattr(self.model.model, 'to'):
                    self.model.model.to(self.device)
                    
                if hasattr(self.model, 'names'):
                    self.class_names = self.model.names
                    
            logger.info(f"Loaded YOLO model successfully: {self.class_names} classes")
        
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise ObjectDetectionError(
                f"Failed to load YOLO model: {e}",
                model_name=self.model_name
            )  
            
    def detect(self, image:np.ndarray, confidence_threshold: float = 0.5,
               iou_threshold: float = 0.45, max_detections: int = 100) -> List[Detection]:
        """Detect object in images"""
        
        if not validate_image(image):
            raise ObjectDetectionError("input image input", image_shape=image.shape)
        
        try: 
            with PerformanceLogger(logger, "YOLO object detection", logging.DEBUG):
                results = self.model(
                    image,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    max_det=max_detections,
                    verbose=False
                )
                
                detections = []
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            detection = Detection(
                                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                                confidence=conf,
                                class_id=cls,
                                class_name=self.class_names.get(cls, f"class_{cls}")
                            )
                            
                            detections.append(detection)
                logger.debug(f"Detected {len(detections)} objects")           
                return detections
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise ObjectDetectionError(
                f"Detection failed: {e}",
                image_shape=image.shape,
                model_name=self.model_name
            )
            
    def detect_from_file(self, image_path: Path, **kwargs) -> List[Detection]:
        """Detect objects from file"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ObjectDetectionError(f"Could not load image: {image_path}")
        
        return self.detect(image=image, **kwargs)
    
    def batch_detect(self, images: List[np.ndarray], **kwargs) -> List[List[Detection]]:
        """Detect objects in batch of images"""
        results = []
        
        with PerformanceLogger(logger, f"Batch detection on {len(images)} images"):
            for i, image in enumerate(images):
                try:
                    detections = self.detect(image, **kwargs)
                    results.append(detections)
                except Exception as e:
                    logger.error(f"Failed to detect objects in image {i}: {e}")
                    results.append([])  # Empty list for failed detection
        
        return results
    
    def filter_detections(self, detections: List[Detection], 
                         min_area: int = 100, max_area: Optional[int] = None,
                         allowed_classes: Optional[List[int]] = None) -> List[Detection]:
        """Filter detections based on criteria"""
        filtered = []
        
        for detection in detections:
            # Filter by area
            if detection.area < min_area:
                continue
            if max_area is not None and detection.area > max_area:
                continue
            
            # Filter by class
            if allowed_classes is not None and detection.class_id not in allowed_classes:
                continue
            
            filtered.append(detection)
        
        logger.debug(f"Filtered detections: {len(filtered)}/{len(detections)} kept")
        return filtered
    
    def visualize_detections(self, image: np.ndarray, detections: List[Detection],
                           show_confidence: bool = True, thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image"""
        vis_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)  # Ensure integer coordinates
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label
            label = detection.class_name
            if show_confidence:
                label += f" {detection.confidence:.2f}"
            
            # Calculate label size and position
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
        
        return vis_image
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        }