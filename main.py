# import sys
from ultralytics import YOLO
from src.exceptions.core_exceptions import ObjectDetectionError
def main():
    # print("Hello from merchandiser-products-counter!")
    # print(f"Environment: {sys.prefix}")
    
    try:
        model_name = "data/models/yolov9.pt"
        model = YOLO(model_name)
        
        
    except Exception as e:
        raise ObjectDetectionError(
            message="An error occurred when loading model",
            image_shape=(640, 480),
            model_name=model_name,
        )
        


if __name__ == "__main__":
    main()
