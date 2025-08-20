#!/usr/bin/env python3
"""
GPU Test for Ball Detection System
"""
import torch
from ultralytics import YOLO
import cv2
import time

def test_gpu_setup():
    print("🔍 GPU Setup Test")
    print("=" * 50)
    
    # Test PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test tensor operations
        print("\n🧪 Testing GPU tensor operations...")
        try:
            # Create tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()  # Wait for GPU operation to complete
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU tensor multiplication: {gpu_time:.4f}s")
            
            # Compare with CPU
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            start_time = time.time()
            z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start_time
            
            print(f"⏱️ CPU tensor multiplication: {cpu_time:.4f}s")
            print(f"🚀 GPU speedup: {cpu_time/gpu_time:.1f}x faster")
            
        except Exception as e:
            print(f"❌ GPU tensor test failed: {e}")
    
    # Test YOLO
    print("\n🎯 Testing YOLO GPU usage...")
    try:
        model = YOLO('yolov8n.pt')
        print(f"YOLO device: {model.device}")
        
        # Force to GPU if available
        if cuda_available:
            model.to('cuda')
            print(f"YOLO moved to: {model.device}")
        
        # Test inference with dummy image
        dummy_image = torch.randn(3, 640, 640)
        if cuda_available:
            dummy_image = dummy_image.cuda()
        
        print("🔄 Testing YOLO inference...")
        start_time = time.time()
        
        # Create a numpy array for YOLO (it expects BGR format)
        test_frame = (torch.randn(480, 640, 3) * 255).byte().numpy()
        
        results = model(test_frame, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"✅ YOLO inference time: {inference_time:.4f}s")
        print(f"📊 Detected {len(results[0].boxes) if results[0].boxes else 0} objects")
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_setup()
    
    print("\n" + "=" * 50)
    print("💡 Tips to fix GPU issues:")
    print("1. Check NVIDIA driver: nvidia-smi")
    print("2. Install CUDA-enabled PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("3. Restart VS Code/Python after installation")
