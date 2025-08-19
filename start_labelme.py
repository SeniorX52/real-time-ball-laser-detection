#!/usr/bin/env python3
"""
Script to start labelme for annotating ball and laser images.
This script sets up the proper configuration and launches labelme.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Set up paths
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    annotations_dir = project_root / "training_data" / "annotations"
    
    # Create directories if they don't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create labelme config file
    config_content = '''
{
  "labels": [
    "ball",
    "laser"
  ],
  "auto_save": true,
  "display_label_popup": true,
  "validate_label": "exact"
}
'''
    
    config_file = project_root / "labelme_config.json"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Starting labelme...")
    print(f"Images directory: {images_dir}")
    print(f"Annotations will be saved alongside images")
    print(f"Labels available: ball, laser")
    print("="*50)
    
    # Try to launch labelme
    try:
        cmd = [
            sys.executable, "-m", "labelme",
            str(images_dir),
            "--config", str(config_file),
            "--output", str(images_dir)  # Save annotations in same folder as images
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching labelme: {e}")
        print("\nLabelme failed to start due to dependency issues.")
        print("This is common on Windows with ONNX runtime conflicts.")
        print("\nChoose an alternative annotation method:")
        print("1. Simple GUI annotator (OpenCV-based)")
        print("2. Console annotator (no GUI, text-based)")
        print("3. Exit")
        
        try:
            choice = input("Enter choice (1, 2, or 3): ").strip()
            if choice == '1':
                print("\nStarting simple GUI annotator...")
                # Import and run simple annotator
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "simple_annotator", 
                    project_root / "simple_annotator.py"
                )
                simple_annotator = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(simple_annotator)
                simple_annotator.main()
            elif choice == '2':
                print("\nStarting console annotator...")
                # Import and run console annotator
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "console_annotator", 
                    project_root / "console_annotator.py"
                )
                console_annotator = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(console_annotator)
                console_annotator.main()
            else:
                print("Annotation cancelled.")
        except KeyboardInterrupt:
            print("\nCancelled by user")
    except KeyboardInterrupt:
        print("Labelme closed by user")

if __name__ == "__main__":
    main()
