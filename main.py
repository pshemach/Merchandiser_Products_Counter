import sys
from src.core.product_counting_system import ProductCountingSystem
from src.utils.logging_utils import PerformanceLogger
import logging
from pathlib import Path 
import cv2
import matplotlib.pyplot as plt
from config.settings import get_settings
from src.utils.file_utils import list_images

logger=logging.getLogger(__name__)

def main():
    setting = get_settings()
    config = setting.dict()
    image_dir = Path('data/reference_images')
    output_dir = Path('results')
    test_image = Path('data/test_images/IMG_2277.jpeg')
    
    with PerformanceLogger(logger, "Product counting system"):
        pcs = ProductCountingSystem(config) 
        
        
        products = {}
        for item in image_dir.iterdir():
            product_id = item.name
            image_files = list_images(item)
        
            if image_files:
                products[product_id] = image_files
        
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
        
        print(result.to_dict())
        
        img = pcs.visualize_results(test_image, result)
        
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
if __name__ == "__main__":
    main()
