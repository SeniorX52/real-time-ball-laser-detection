#!/usr/bin/env python3
"""
Quick test to check if your YOLO is using GPU
"""
from yolo_ball_detector import YOLOBallDetector
import torch

print("🔍 Testing Ball Detector GPU Usage")
print("=" * 50)

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n🎯 Initializing YOLO detector...")
detector = YOLOBallDetector('yolov8n.pt')

print(f"✅ Detector initialized successfully!")
print("Check the output above to see if YOLO is using CUDA.")
