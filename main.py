# import sys
from src.core.object_detector import YOLODetector
from src.core.embedding_extractor import ImprovedEmbeddingExtractor
from src.utils.image_utils import crop_image
from pathlib import Path 
import cv2
import matplotlib.pyplot as plt


def main():
    # print("Hello from merchandiser-products-counter!")
    # print(f"Environment: {sys.prefix}")
    model_name = "data/models/yolov9.pt"
    model = YOLODetector(model_name=model_name)
        
    print(f"\nYOLO Model Info: {model.get_model_info()}\n")
        
    image = Path('data/test_images/IMG_2277.jpeg')
    detections = model.detect_from_file(image_path=image)
    print(f"number of detections: {len(detections)}\n")
    print(detections[0].to_dict())
    
    image_array = cv2.imread(str(image))
    # vis_image = model.visualize_detections(image_array, detections)

    # plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    
    extractor = ImprovedEmbeddingExtractor(model_name='facebook/dinov2-base', normalization_strategy='catalog_norm')
    
    print(extractor.get_stats_summary())
    
    cropped_img = crop_image(image_array, detections[0].bbox)
    embedding =extractor.extract_embedding(cropped_img)
    
    print(embedding)
    


if __name__ == "__main__":
    main()
