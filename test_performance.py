#!/usr/bin/env python3
"""
Test GPU performance with actual ball detection
"""
from yolo_ball_detector import YOLOBallDetector
import cv2
import numpy as np
import time
import torch

def test_ball_detection_performance():
    print("🎯 Testing Ball Detection Performance")
    print("=" * 50)
    
    # Initialize detector
    detector = YOLOBallDetector('yolov8n.pt')
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # HSV ranges for yellow
    hsv_low = np.array([20, 50, 50])
    hsv_high = np.array([30, 255, 255])
    
    print("\n🚀 Performance Test (10 iterations):")
    times = []
    
    for i in range(10):
        start_time = time.time()
        
        # Test detection
        result = detector.detect_yellow_ball(test_image, hsv_low, hsv_high)
        
        end_time = time.time()
        detection_time = end_time - start_time
        times.append(detection_time)
        
        fps = 1.0 / detection_time if detection_time > 0 else 0
        print(f"  Frame {i+1}: {detection_time:.4f}s ({fps:.1f} FPS)")
    
    avg_time = np.mean(times)
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Min time: {min(times):.4f}s ({1.0/min(times):.1f} FPS)")
    print(f"  Max time: {max(times):.4f}s ({1.0/max(times):.1f} FPS)")
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

if __name__ == "__main__":
    test_ball_detection_performance()
