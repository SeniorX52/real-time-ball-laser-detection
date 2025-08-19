#!/usr/bin/env python3
"""
Annotation Launcher - Choose your preferred annotation method
"""

import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    
    # Check if we have images to annotate
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        print("Please run collect_training_data.py first to collect some images.")
        return
    
    print("🎯 Ball & Laser Annotation Tool Launcher")
    print("="*50)
    print(f"Found {len(image_files)} images to annotate")
    print(f"Location: {images_dir}")
    print()
    print("Choose annotation method:")
    print()
    print("1. 📋 Console Annotator (Recommended)")
    print("   - No GUI dependencies")
    print("   - Text-based coordinate input")
    print("   - Works on any system")
    print()
    print("2. 🖱️  Simple GUI Annotator")
    print("   - Click and drag interface")
    print("   - Visual bounding boxes")
    print("   - Requires working OpenCV GUI")
    print()
    print("3. 🏷️  Labelme (Professional)")
    print("   - Full-featured annotation tool")
    print("   - May have dependency issues")
    print()
    print("4. 🚪 Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting Console Annotator...")
                try:
                    from console_annotator import main as console_main
                    console_main()
                except ImportError:
                    # Run as subprocess if import fails
                    cmd = [sys.executable, str(project_root / "console_annotator.py")]
                    subprocess.run(cmd)
                break
                
            elif choice == '2':
                print("\n🚀 Starting Simple GUI Annotator...")
                try:
                    from simple_annotator import main as gui_main
                    gui_main()
                except ImportError:
                    # Run as subprocess if import fails
                    cmd = [sys.executable, str(project_root / "simple_annotator.py")]
                    subprocess.run(cmd)
                break
                
            elif choice == '3':
                print("\n🚀 Starting Labelme...")
                try:
                    from start_labelme import main as labelme_main
                    labelme_main()
                except ImportError:
                    # Run as subprocess if import fails
                    cmd = [sys.executable, str(project_root / "start_labelme.py")]
                    subprocess.run(cmd)
                break
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Please enter 1, 2, 3, or 4")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try another option.")

if __name__ == "__main__":
    main()
