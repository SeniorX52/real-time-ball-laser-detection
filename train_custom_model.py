#!/usr/bin/env python3
"""
Train a custom YOLO model on ball and laser detection data.
This script sets up and trains a YOLOv8 model using your labeled data.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil

class YOLOTrainer:
    def __init__(self, data_dir, model_dir):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset(self):
        """Prepare the dataset structure for YOLO training"""
        # Create YOLO dataset structure
        dataset_dir = self.data_dir / "yolo_dataset"
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        
        for split_dir in [train_dir, val_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        processed_dir = self.data_dir / "processed"
        images_dir = self.data_dir / "images"
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            return None
        
        # Split dataset (80% train, 20% val)
        import random
        random.shuffle(image_files)
        split_idx = int(0.8 * len(image_files))
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        print(f"Dataset split: {len(train_images)} training, {len(val_images)} validation")
        
        # Copy files to appropriate directories
        def copy_files(image_list, target_dir):
            for img_path in image_list:
                # Copy image
                target_img = target_dir / "images" / img_path.name
                shutil.copy2(img_path, target_img)
                
                # Copy corresponding label file
                label_name = img_path.stem + '.txt'
                label_path = processed_dir / label_name
                if label_path.exists():
                    target_label = target_dir / "labels" / label_name
                    shutil.copy2(label_path, target_label)
                else:
                    print(f"Warning: No label file for {img_path.name}")
        
        copy_files(train_images, train_dir)
        copy_files(val_images, val_dir)
        
        return dataset_dir
    
    def create_yaml_config(self, dataset_dir):
        """Create YAML configuration file for YOLO training"""
        config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                0: 'ball',
                1: 'laser'
            }
        }
        
        yaml_path = self.model_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created dataset config: {yaml_path}")
        return yaml_path
    
    def train_model(self, dataset_yaml, epochs=100, imgsz=640):
        """Train the YOLO model"""
        # Load a pre-trained YOLOv8 model
        model = YOLO('yolov8n.pt')  # Use nano model for faster training
        
        print(f"Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Dataset config: {dataset_yaml}")
        
        # Train the model
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            patience=50,
            save=True,
            device='cuda' if os.path.exists('/usr/bin/nvidia-smi') else 'cpu',
            project=str(self.model_dir),
            name='ball_laser_detector'
        )
        
        return results
    
    def export_model(self):
        """Export the trained model"""
        # Find the best weights
        weights_dir = self.model_dir / 'ball_laser_detector' / 'weights'
        best_weights = weights_dir / 'best.pt'
        
        if best_weights.exists():
            print(f"Best model weights saved at: {best_weights}")
            
            # Export to different formats
            model = YOLO(str(best_weights))
            
            # Export to ONNX for deployment
            try:
                model.export(format='onnx')
                print("Model exported to ONNX format")
            except Exception as e:
                print(f"ONNX export failed: {e}")
            
            return str(best_weights)
        else:
            print("No trained weights found!")
            return None

def main():
    project_root = Path(__file__).parent
    data_dir = project_root / "training_data"
    model_dir = project_root / "models"
    
    trainer = YOLOTrainer(data_dir, model_dir)
    
    # Check if we have processed annotations
    processed_dir = data_dir / "processed"
    if not processed_dir.exists() or not list(processed_dir.glob('*.txt')):
        print("No processed annotations found!")
        print("Please run convert_annotations.py first to convert your labelme annotations.")
        return
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset_dir = trainer.prepare_dataset()
    if dataset_dir is None:
        return
    
    # Create YAML config
    yaml_config = trainer.create_yaml_config(dataset_dir)
    
    # Train model
    print("\nStarting training...")
    results = trainer.train_model(yaml_config, epochs=100)
    
    # Export model
    print("\nExporting model...")
    best_model = trainer.export_model()
    
    if best_model:
        print(f"\nTraining complete! Best model: {best_model}")
        print("You can now use this model in your ball detection application.")

if __name__ == "__main__":
    main()
