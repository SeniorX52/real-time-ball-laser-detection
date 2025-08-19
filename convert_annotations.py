#!/usr/bin/env python3
"""
Convert labelme JSON annotations to YOLO format for training.
This script processes all JSON files in the images directory and converts them to YOLO format.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path

class LabelmeToYOLO:
    def __init__(self, images_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping
        self.class_names = ['ball', 'laser']
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
    def convert_polygon_to_bbox(self, points, img_width, img_height):
        """Convert polygon points to YOLO bounding box format"""
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        center_x = (x_min + x_max) / 2 / img_width
        center_y = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return center_x, center_y, width, height
    
    def process_json_file(self, json_path):
        """Process a single JSON annotation file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # Create YOLO annotation file
        txt_filename = json_path.stem + '.txt'
        txt_path = self.output_dir / txt_filename
        
        annotations = []
        for shape in data['shapes']:
            label = shape['label']
            if label not in self.class_to_id:
                print(f"Warning: Unknown label '{label}' in {json_path}")
                continue
            
            class_id = self.class_to_id[label]
            points = shape['points']
            
            # Convert polygon to bounding box
            center_x, center_y, width, height = self.convert_polygon_to_bbox(
                points, img_width, img_height
            )
            
            annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Save YOLO format file
        with open(txt_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        return len(annotations)
    
    def convert_all(self):
        """Convert all JSON files in the images directory"""
        json_files = list(self.images_dir.glob('*.json'))
        
        if not json_files:
            print(f"No JSON files found in {self.images_dir}")
            return
        
        print(f"Found {len(json_files)} JSON annotation files")
        total_annotations = 0
        
        for json_file in json_files:
            try:
                count = self.process_json_file(json_file)
                total_annotations += count
                print(f"Processed {json_file.name}: {count} annotations")
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
        
        # Create classes.txt file
        classes_file = self.output_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            f.write('\n'.join(self.class_names))
        
        print(f"\nConversion complete!")
        print(f"Total annotations: {total_annotations}")
        print(f"Output directory: {self.output_dir}")
        print(f"Classes file created: {classes_file}")

def main():
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    output_dir = project_root / "training_data" / "processed"
    
    converter = LabelmeToYOLO(images_dir, output_dir)
    converter.convert_all()

if __name__ == "__main__":
    main()
