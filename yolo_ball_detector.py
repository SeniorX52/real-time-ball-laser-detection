import cv2
from ultralytics import YOLO
import numpy as np
import torch

class YOLOBallDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.3):
        # Check GPU availability
        cuda_available = torch.cuda.is_available()
        device = 'cuda' if cuda_available else 'cpu'
        
        print("🔍 GPU Status Check:")
        print(f"  CUDA available: {cuda_available}")
        print(f"  Device selected: {device}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {gpu_memory:.1f} GB")
        
        # Load and configure YOLO model
        self.model = YOLO(model_path)
        if cuda_available:
            self.model.to(device)
            print("🚀 YOLO model moved to GPU!")
        else:
            print("⚠️ YOLO running on CPU only")
            
        self.conf = conf
        
        # Check if using custom model
        self.is_custom_model = 'custom' in str(model_path).lower() or 'ball' in str(model_path).lower()
        if self.is_custom_model:
            print(f"🎯 Using CUSTOM trained model: {model_path}")
            print("This model is specifically trained on your ball and laser data!")
        else:
            print(f"📦 Using generic model: {model_path}")
            # Class 0 is 'person', 32 is 'sports ball' in COCO, but yellow ball may not be labeled as such
            # We'll filter by color after detection

    def detect_yellow_ball(self, frame, hsv_low, hsv_high):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        found = None
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            ratio = np.sum(mask > 0) / mask.size
            if ratio > 0.3:  # At least 30% of the box is yellow (or user-defined)
                # Return center in image coordinates
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                found = (cx, cy, x1, y1, x2, y2)
                break
        return found

    def detect_any_ball(self, frame):
        """Pure YOLO detection without HSV post-filtering for better performance."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        
        if len(bboxes) > 0:
            # For custom models, class IDs are: 0=ball, 1=laser
            # For generic models, we take any detection
            if self.is_custom_model:
                # Look specifically for ball class (class 0)
                classes = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
                for i, (box, cls) in enumerate(zip(bboxes, classes)):
                    if int(cls) == 0:  # Ball class in custom model
                        x1, y1, x2, y2 = map(int, box[:4])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        return (cx, cy, x1, y1, x2, y2)
            else:
                # Return the first (highest confidence) detection for generic models
                x1, y1, x2, y2 = map(int, bboxes[0][:4])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                return (cx, cy, x1, y1, x2, y2)
        
        return None
    
    def detect_laser(self, frame):
        """Detect laser pointer using custom model (only works with trained model)"""
        if not self.is_custom_model:
            return None
            
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        
        if len(bboxes) > 0:
            classes = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
            # Look specifically for laser class (class 1)
            for i, (box, cls) in enumerate(zip(bboxes, classes)):
                if int(cls) == 1:  # Laser class in custom model
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    return (cx, cy, x1, y1, x2, y2)
        
        return None
