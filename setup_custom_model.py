#!/usr/bin/env python3
"""
Update ball_tracker2.py to use your custom trained model
"""

import shutil
from pathlib import Path

def update_ball_tracker_with_custom_model():
    """Update the main ball tracker to use the custom trained model"""
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    best_model_path = models_dir / "ball_laser_detector" / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"❌ Trained model not found at: {best_model_path}")
        print("Please make sure training completed successfully.")
        return False
    
    print(f"✅ Found trained model: {best_model_path}")
    print(f"Model size: {best_model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Copy the model to a convenient location
    custom_model_path = project_root / "custom_ball_laser_model.pt"
    shutil.copy2(best_model_path, custom_model_path)
    print(f"📋 Copied model to: {custom_model_path}")
    
    # Create a configuration file
    config_content = f"""# Custom Model Configuration
# Generated automatically after training

CUSTOM_MODEL_PATH = r"{custom_model_path}"
USE_CUSTOM_MODEL = True

# Training Results
# Model trained on your specific ball and laser data
# Should provide better accuracy than generic YOLO
"""
    
    config_file = project_root / "custom_model_config.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"⚙️  Created config file: {config_file}")
    
    # Instructions for integration
    print("\n" + "="*60)
    print("🎯 INTEGRATION INSTRUCTIONS")
    print("="*60)
    print()
    print("Your custom model is ready! To use it in your ball tracker:")
    print()
    print("1️⃣  MANUAL INTEGRATION:")
    print("   Edit yolo_ball_detector.py and change:")
    print("   FROM: YOLOBallDetector('yolov8n.pt')")
    print("   TO:   YOLOBallDetector('custom_ball_laser_model.pt')")
    print()
    print("2️⃣  TEST THE MODEL:")
    print("   Run: python test_custom_model.py --model custom_ball_laser_model.pt --mode camera")
    print()
    print("3️⃣  EXPECTED IMPROVEMENTS:")
    print("   - Better detection of YOUR specific ball")
    print("   - Better detection of YOUR laser pointer")
    print("   - Reduced false positives")
    print("   - More consistent tracking")
    print()
    print("4️⃣  PERFORMANCE COMPARISON:")
    print("   - Test both generic YOLO and your custom model")
    print("   - Your custom model should be more accurate for your specific setup")
    print()
    
    return True

def main():
    print(" Custom Model Integration Setup")
    print("="*50)
    
    success = update_ball_tracker_with_custom_model()
    if success:
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Test your model: python test_custom_model.py --model custom_ball_laser_model.pt --mode camera")
        print("2. Manually edit ball_tracker2.py to use 'custom_ball_laser_model.pt'")
        print("3. Run your tracker with the custom model!")

if __name__ == "__main__":
    main()
