#!/usr/bin/env python3
"""
Console-based annotation tool for ball and laser detection.
No GUI dependencies - works entirely through console input.
"""

import json
import cv2
from pathlib import Path
import time

class ConsoleAnnotator:
    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)
        self.annotations = {}
        self.labels = ["ball", "laser"]
        
    def show_image_info(self, image_path, image):
        """Display image information and save a preview"""
        h, w, _ = image.shape
        print(f"\nImage: {image_path.name}")
        print(f"Dimensions: {w}x{h}")
        print(f"Size: {image_path.stat().st_size} bytes")
        
        # Save a small preview for reference
        preview_path = self.images_dir / f"preview_{image_path.name}"
        small = cv2.resize(image, (min(400, w), min(300, h)))
        cv2.imwrite(str(preview_path), small)
        print(f"Preview saved as: {preview_path.name}")
        
    def get_bounding_box(self, image_shape, label):
        """Get bounding box coordinates from user input"""
        h, w, _ = image_shape
        print(f"\n📍 Annotating: {label}")
        print(f"Image size: {w} x {h} (width x height)")
        print("💡 Tip: Look at the preview image to estimate coordinates")
        print()
        
        while True:
            try:
                print("Enter bounding box (top-left corner):")
                x1 = int(input(f"  X coordinate (0 to {w-1}): "))
                y1 = int(input(f"  Y coordinate (0 to {h-1}): "))
                
                print("Enter bounding box (bottom-right corner):")
                x2 = int(input(f"  X coordinate ({x1+10} to {w-1}): "))
                y2 = int(input(f"  Y coordinate ({y1+10} to {h-1}): "))
                
                # Validate coordinates
                if x1 >= x2 or y1 >= y2:
                    print("❌ Bottom-right must be greater than top-left. Try again.")
                    continue
                    
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w-1))
                y2 = max(y1+1, min(y2, h-1))
                
                # Show confirmation
                box_w = x2 - x1
                box_h = y2 - y1
                print(f"✅ Box: ({x1}, {y1}) to ({x2}, {y2}) - Size: {box_w}x{box_h}")
                
                confirm = input("Is this correct? (y/n): ").lower().strip()
                if confirm in ['y', 'yes', '']:
                    return [x1, y1, x2, y2]
                elif confirm in ['n', 'no']:
                    print("Let's try again...")
                    continue
                else:
                    return [x1, y1, x2, y2]  # Default to yes
                    
            except ValueError:
                print("❌ Invalid input. Please enter numbers only.")
            except KeyboardInterrupt:
                print("\n⏭️  Skipping this annotation...")
                return None
    
    def save_annotations(self):
        """Save annotations in labelme JSON format"""
        saved_count = 0
        for image_name, anns in self.annotations.items():
            if not anns:
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
                x1, y1, x2, y2 = ann['bbox']
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                
                shape = {
                    "label": ann['label'],
                    "points": points,
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
            
            saved_count += 1
            print(f"✓ Saved annotations for {image_name}")
        
        print(f"\nSaved annotations for {saved_count} images")
    
    def load_existing_annotations(self, image_path):
        """Load existing annotations if they exist"""
        json_path = image_path.with_suffix('.json')
        if not json_path.exists():
            return []
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            annotations = []
            for shape in data.get('shapes', []):
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    x1, y1 = map(int, points[0])
                    x2, y2 = map(int, points[2])
                    
                    annotations.append({
                        'label': shape['label'],
                        'bbox': [x1, y1, x2, y2]
                    })
            
            return annotations
            
        except Exception as e:
            print(f"Error loading existing annotations: {e}")
            return []
    
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
        
        print(f"Console Annotation Tool")
        print(f"Found {len(image_files)} images to annotate")
        print("\nCommands:")
        print("  'ball' or 'b' - Add ball annotation")
        print("  'laser' or 'l' - Add laser annotation")
        print("  'skip' or 's' - Skip this image")
        print("  'done' or 'd' - Finish current image")
        print("  'quit' or 'q' - Save and quit")
        print("  'list' - List current annotations")
        print("="*60)
        
        for i, image_path in enumerate(image_files):
            print(f"\n{'='*60}")
            print(f"IMAGE {i+1}/{len(image_files)}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            
            self.show_image_info(image_path, image)
            
            # Load existing annotations
            existing_anns = self.load_existing_annotations(image_path)
            image_name = image_path.name
            self.annotations[image_name] = existing_anns.copy()
            
            if existing_anns:
                print(f"\nExisting annotations:")
                for j, ann in enumerate(existing_anns):
                    bbox = ann['bbox']
                    print(f"  {j+1}. {ann['label']}: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
            
            # Annotation loop for current image
            while True:
                print(f"\nCurrent image: {image_path.name}")
                print(f"Annotations: {len(self.annotations.get(image_name, []))}")
                
                command = input("Command: ").lower().strip()
                
                if command in ['quit', 'q']:
                    self.save_annotations()
                    return
                    
                elif command in ['done', 'd', 'skip', 's']:
                    break
                    
                elif command in ['ball', 'b']:
                    bbox = self.get_bounding_box(image.shape, 'ball')
                    if bbox:
                        if image_name not in self.annotations:
                            self.annotations[image_name] = []
                        self.annotations[image_name].append({
                            'label': 'ball',
                            'bbox': bbox
                        })
                        print(f"✓ Added ball annotation: {bbox}")
                        
                elif command in ['laser', 'l']:
                    bbox = self.get_bounding_box(image.shape, 'laser')
                    if bbox:
                        if image_name not in self.annotations:
                            self.annotations[image_name] = []
                        self.annotations[image_name].append({
                            'label': 'laser',
                            'bbox': bbox
                        })
                        print(f"✓ Added laser annotation: {bbox}")
                        
                elif command == 'list':
                    current_anns = self.annotations.get(image_name, [])
                    if current_anns:
                        print("\nCurrent annotations:")
                        for j, ann in enumerate(current_anns):
                            bbox = ann['bbox']
                            print(f"  {j+1}. {ann['label']}: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                    else:
                        print("No annotations for this image yet.")
                        
                else:
                    print("Unknown command. Try 'ball', 'laser', 'done', or 'quit'")
        
        self.save_annotations()
        print("Annotation complete!")

def main():
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        print("Please run collect_training_data.py first to collect some images.")
        return
    
    annotator = ConsoleAnnotator(images_dir)
    annotator.annotate()

if __name__ == "__main__":
    main()
