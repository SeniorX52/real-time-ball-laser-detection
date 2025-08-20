try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"CUDA version built with: {torch.version.cuda}")
    else:
        print("CUDA version built with: CPU-only version")
        
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected by PyTorch")
        
except Exception as e:
    print(f"Error: {e}")
    
# Test Ultralytics
try:
    from ultralytics import YOLO
    print(f"Ultralytics imported successfully")
    model = YOLO('yolov8n.pt')
    print(f"YOLO device: {model.device}")
except Exception as e:
    print(f"Ultralytics error: {e}")
