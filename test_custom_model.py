#!/usr/bin/env python3
"""
Test the trained custom model on new images or video stream.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

class CustomModelTester:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = ['ball', 'laser']
        self.colors = {
            'ball': (0, 255, 0),    # Green
            'laser': (0, 0, 255)    # Red
        }
    
    def test_on_image(self, image_path):
        """Test the model on a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Run inference
        results = self.model(image)[0]
        
        # Draw detections
        annotated_image = image.copy()
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.class_names[cls]
                    color = self.colors[class_name]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display result
        cv2.imshow('Custom Model Test', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def test_on_camera(self, camera_index=0):
        """Test the model on camera feed"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Could not open camera {camera_index}")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame)[0]
            
            # Draw detections
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    if conf > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.class_names[cls]
                        color = self.colors[class_name]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow('Custom Model Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test custom YOLO model')
    parser.add_argument('--model', type=str, default='custom_ball_laser_model.pt', help='Path to trained model weights (default: custom_ball_laser_model.pt)')
    parser.add_argument('--image', type=str, help='Test on single image')
    parser.add_argument('--camera', type=int, default=0, help='Test on camera (default: 0)')
    parser.add_argument('--mode', choices=['image', 'camera'], default='camera', help='Test mode (default: camera)')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try looking in current directory
        current_dir = Path.cwd()
        alt_model_path = current_dir / args.model
        if alt_model_path.exists():
            model_path = alt_model_path
        else:
            print(f"❌ Model file not found: {args.model}")
            print(f"Searched in:")
            print(f"   - {Path(args.model).absolute()}")
            print(f"   - {alt_model_path}")
            print("\nAvailable models:")
            # Look for .pt files in current directory
            pt_files = list(Path.cwd().glob('*.pt'))
            if pt_files:
                for pt_file in pt_files:
                    print(f"   - {pt_file.name}")
            else:
                print("   - No .pt files found in current directory")
            print("\n💡 Tip: Make sure you've run setup_custom_model.py first!")
            return
    
    print(f"🎯 Testing model: {model_path}")
    tester = CustomModelTester(model_path)
    
    if args.mode == 'image':
        if args.image:
            tester.test_on_image(args.image)
        else:
            print("Please provide --image path for image mode")
    else:
        tester.test_on_camera(args.camera)

if __name__ == "__main__":
    main()
