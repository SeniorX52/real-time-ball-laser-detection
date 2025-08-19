#!/usr/bin/env python3
"""
Simple annotation tool for ball and laser detection.
Alternative to labelme that works reliably on Windows.
"""

import cv2
import json
import os
from pathlib import Path
import time

class SimpleAnnotator:
    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)
        self.current_image = None
        self.current_image_path = None
        self.annotations = {}
        self.drawing = False
        self.bbox_start = None
        self.bbox_end = None
        self.current_label = "ball"  # Default label
        self.labels = ["ball", "laser"]
        
        # Colors for different labels
        self.colors = {
            "ball": (0, 255, 0),    # Green
            "laser": (0, 0, 255)    # Red
        }
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bbox_start = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Show preview of bounding box
            temp_image = self.current_image.copy()
            cv2.rectangle(temp_image, self.bbox_start, (x, y), 
                         self.colors[self.current_label], 2)
            try:
                cv2.imshow('Annotator', temp_image)
            except cv2.error:
                pass  # Ignore display errors during dragging
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox_end = (x, y)
            self.add_annotation()
            
    def add_annotation(self):
        """Add the drawn bounding box as an annotation"""
        if self.bbox_start and self.bbox_end:
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            
            # Ensure proper order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Don't save very small boxes (likely accidental clicks)
            if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                return
            
            image_name = self.current_image_path.name
            if image_name not in self.annotations:
                self.annotations[image_name] = []
            
            self.annotations[image_name].append({
                'label': self.current_label,
                'bbox': [x1, y1, x2, y2],
                'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # For labelme compatibility
            })
            
            print(f"Added {self.current_label} annotation: ({x1}, {y1}) to ({x2}, {y2})")
            
    def draw_annotations(self, image):
        """Draw existing annotations on the image"""
        image_name = self.current_image_path.name
        if image_name in self.annotations:
            for ann in self.annotations[image_name]:
                x1, y1, x2, y2 = ann['bbox']
                label = ann['label']
                color = self.colors.get(label, (255, 255, 255))
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
    
    def save_annotations(self):
        """Save annotations in labelme JSON format"""
        for image_name, anns in self.annotations.items():
            if not anns:  # Skip images with no annotations
                continue
                
            image_path = self.images_dir / image_name
            if not image_path.exists():
                continue
                
            # Read image to get dimensions
            img = cv2.imread(str(image_path))
            h, w, _ = img.shape
            
            # Create labelme-compatible JSON
            labelme_data = {
                "version": "5.2.1",
                "flags": {},
                "shapes": [],
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w
            }
            
            for ann in anns:
                shape = {
                    "label": ann['label'],
                    "points": ann['points'],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None
                }
                labelme_data['shapes'].append(shape)
            
            # Save JSON file
            json_path = image_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(labelme_data, f, indent=2)
            
            print(f"Saved annotations for {image_name}")
    
    def annotate(self):
        """Main annotation loop"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(ext))
        
        if not image_files:
            print(f"No image files found in {self.images_dir}")
            return
        
        print(f"Found {len(image_files)} images to annotate")
        print("\nControls:")
        print("  Left click + drag: Draw bounding box")
        print("  'b': Switch to ball label")
        print("  'l': Switch to laser label")
        print("  'n': Next image")
        print("  'p': Previous image")
        print("  'u': Undo last annotation")
        print("  's': Save annotations")
        print("  'q': Quit")
        print("="*50)
        
        current_idx = 0
        
        # Create window
        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self.mouse_callback)
        
        while current_idx < len(image_files):
            self.current_image_path = image_files[current_idx]
            self.current_image = cv2.imread(str(self.current_image_path))
            
            if self.current_image is None:
                print(f"Could not load image: {self.current_image_path}")
                current_idx += 1
                continue
            
            # Load existing annotations if they exist
            json_path = self.current_image_path.with_suffix('.json')
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    image_name = self.current_image_path.name
                    self.annotations[image_name] = []
                    
                    for shape in data.get('shapes', []):
                        if shape['shape_type'] == 'rectangle':
                            points = shape['points']
                            x1, y1 = map(int, points[0])
                            x2, y2 = map(int, points[2])
                            
                            self.annotations[image_name].append({
                                'label': shape['label'],
                                'bbox': [x1, y1, x2, y2],
                                'points': points
                            })
                except Exception as e:
                    print(f"Error loading annotations for {self.current_image_path.name}: {e}")
            
            while True:
                display_image = self.current_image.copy()
                display_image = self.draw_annotations(display_image)
                
                # Add status text
                status_text = f"Image {current_idx+1}/{len(image_files)} | Label: {self.current_label} | File: {self.current_image_path.name}"
                cv2.putText(display_image, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Annotator', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.save_annotations()
                    cv2.destroyAllWindows()
                    return
                    
                elif key == ord('n'):  # Next image
                    current_idx = min(current_idx + 1, len(image_files) - 1)
                    break
                    
                elif key == ord('p'):  # Previous image
                    current_idx = max(current_idx - 1, 0)
                    break
                    
                elif key == ord('b'):  # Ball label
                    self.current_label = "ball"
                    print(f"Switched to label: {self.current_label}")
                    
                elif key == ord('l'):  # Laser label
                    self.current_label = "laser"
                    print(f"Switched to label: {self.current_label}")
                    
                elif key == ord('u'):  # Undo
                    image_name = self.current_image_path.name
                    if image_name in self.annotations and self.annotations[image_name]:
                        removed = self.annotations[image_name].pop()
                        print(f"Removed annotation: {removed['label']}")
                        
                elif key == ord('s'):  # Save
                    self.save_annotations()
                    print("Annotations saved!")
        
        self.save_annotations()
        cv2.destroyAllWindows()
        print("Annotation complete!")

def main():
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        print("Please run collect_training_data.py first to collect some images.")
        return
    
    annotator = SimpleAnnotator(images_dir)
    annotator.annotate()

if __name__ == "__main__":
    main()
