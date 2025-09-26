# import sys
from src.core.object_detector import YOLODetector
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
        
    vis_image = model.visualize_detections(cv2.imread(str(image)), detections)

    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
