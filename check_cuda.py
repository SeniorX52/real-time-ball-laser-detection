#!/usr/bin/env python3
"""
CUDA and PyTorch GPU Setup Diagnostic Tool
"""
import sys
import os

def check_cuda_setup():
    print("🔍 CUDA/GPU Setup Diagnostic")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {'✅ YES' if cuda_available else '❌ NO'}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU devices count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("❌ CUDA not available - running on CPU only")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    print()
    
    # Check Ultralytics YOLO
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO installed")
        
        # Test device detection
        model = YOLO('yolov8n.pt')
        device = model.device
        print(f"YOLO default device: {device}")
        
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    
    print()
    
    # Check NVIDIA driver
    try:
        result = os.popen('nvidia-smi').read()
        if 'NVIDIA-SMI' in result:
            print("✅ NVIDIA driver detected")
            # Extract driver version
            lines = result.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"Driver info: {line.strip()}")
                    break
        else:
            print("❌ NVIDIA-SMI not found or not working")
    except:
        print("❌ Could not check NVIDIA driver")
    
    return cuda_available

def fix_suggestions():
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("=" * 50)
    
    print("1. 🎯 INSTALL CORRECT PyTorch VERSION:")
    print("   Visit: https://pytorch.org/get-started/locally/")
    print("   For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    
    print("2. 🔌 CHECK NVIDIA DRIVER:")
    print("   - Download latest driver from: https://www.nvidia.com/drivers")
    print("   - Restart computer after installation")
    print()
    
    print("3. 🧪 TEST CUDA:")
    print("   Run: nvidia-smi")
    print("   Should show GPU information")
    print()
    
    print("4. 🔄 REINSTALL IF NEEDED:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    cuda_works = check_cuda_setup()
    
    if not cuda_works:
        fix_suggestions()
        
    print("\n" + "=" * 50)
    print("🚀 Run this script after making changes to recheck!")
