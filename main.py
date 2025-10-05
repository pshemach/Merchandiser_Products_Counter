import sys
from src.core.product_counting_system import ProductCountingSystem
from src.utils.logging_utils import PerformanceLogger
import logging
from pathlib import Path 
import cv2
import matplotlib.pyplot as plt
from config.settings import get_settings
from src.utils.file_utils import list_images
from src.api.schemas import *

logger=logging.getLogger(__name__)

def main():
    setting = get_settings()
    config = setting.dict()
    image_dir = Path('data/reference_images')
    output_dir = Path('results')
    test_image = Path('data/test_images/IMG_2329.jpeg')
    
    with PerformanceLogger(logger, "Product counting system"):
        pcs = ProductCountingSystem(config) 
        
        
        # products = {}
        # for item in image_dir.iterdir():
        #     product_id = item.name
        #     image_files = list_images(item)
        
        #     if image_files:
        #         products[product_id] = image_files
        
        # for product_id, image_paths in products.items():
        #     product_name = product_id
        #     product_info = pcs.add_product_to_catalog(
        #                     product_id=product_id,
        #                     name=product_name,
        #                     image_paths=[str(img) for img in image_paths],
        #                     category='retail_product',  # Default category
        #                     description=f"Product {product_name} with {len(image_paths)} reference images"
        #                 )
            
        # pcs.save_system_state(output_dir)
        
        pcs.load_system_state(output_dir)
        
        result = pcs.count_products_in_image(test_image)
        
        # print(result.to_dict())
        
        # img = pcs.visualize_results(test_image, result)
        
        # stat = pcs.get_system_statistics()
        # print(stat)
        
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        # print(pcs.catalog_manager.list_products())
        # print("\n\n")
        # pcs.catalog_manager.remove_product(product_id="1001")
        # print(pcs.catalog_manager.list_products())
        return_bounding_boxes = True
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
        print(product_details)
        
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
        
        print("\n\n")
        print(product_count_list)
        
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
        
        print("\n\n")
        print(summary)
        
        return_visualization = True
        visualization_url = None
        image_id = 1
        if return_visualization:
            viz_path = output_dir / f"viz_{image_id}.jpg"
            pcs.visualize_results(str(test_image), result, str(viz_path))
            visualization_url = f"/results/viz_{image_id}.jpg"
        
        return_all_detections = True
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
            image_name=str(test_image),
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
        
        print("\n\n")
        print(response)
if __name__ == "__main__":
    main()
