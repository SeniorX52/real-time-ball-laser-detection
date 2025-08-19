# YOLOv8 Nano weights download script
from ultralytics import YOLO

# Download YOLOv8 Nano pretrained weights if not present
def download_yolov8n():
    model = YOLO('yolov8n.pt')
    print("YOLOv8-Nano weights ready.")

if __name__ == "__main__":
    download_yolov8n()
