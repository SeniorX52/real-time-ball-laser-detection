# AI Model Training Environment Setup

This environment is set up for training custom AI models using labelme for annotation and YOLO for object detection.

## Directory Structure

```
Ball_Detection/
├── training_data/
│   ├── images/          # Raw images and labelme JSON annotations
│   ├── annotations/     # (backup location for annotations)
│   └── processed/       # Converted YOLO format annotations
├── models/              # Trained model weights and configs
├── collect_training_data.py    # Capture frames for training
├── start_labelme.py            # Launch labelme for annotation
├── convert_annotations.py      # Convert labelme to YOLO format
├── train_custom_model.py       # Train custom YOLO model
└── test_custom_model.py        # Test trained model
```

## Workflow Steps

### 1. Data Collection
```bash
python collect_training_data.py
```
- Captures frames from your camera
- Press SPACE to save frames
- Saves images to `training_data/images/`

### 2. Data Annotation
```bash
python start_labelme.py
```
- Launches labelme with proper configuration
- Annotate your images with labels: "ball" and "laser"
- Saves JSON annotations alongside images

### 3. Convert Annotations
```bash
python convert_annotations.py
```
- Converts labelme JSON to YOLO format
- Creates bounding boxes from polygon annotations
- Saves processed data to `training_data/processed/`

### 4. Train Model
```bash
python train_custom_model.py
```
- Trains a custom YOLOv8 model on your data
- Automatically splits data into train/validation
- Saves trained model to `models/` directory

### 5. Test Model
```bash
# Test on camera
python test_custom_model.py --model models/ball_laser_detector/weights/best.pt --mode camera

# Test on image
python test_custom_model.py --model models/ball_laser_detector/weights/best.pt --mode image --image path/to/test.jpg
```

## Installed Packages

- **labelme**: For image annotation
- **tensorflow**: Deep learning framework
- **ultralytics**: YOLO implementation
- **opencv-python**: Computer vision
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **albumentations**: Data augmentation
- **imgaug**: Additional data augmentation
- **jupyter**: Notebook environment
- **matplotlib**: Plotting and visualization

## Tips for Better Training

1. **Diverse Data**: Collect images with different:
   - Lighting conditions
   - Ball positions
   - Backgrounds
   - Camera angles

2. **Balanced Dataset**: 
   - Aim for similar numbers of ball and laser samples
   - Include images with both objects, one object, or neither

3. **Quality Annotations**:
   - Draw tight bounding boxes around objects
   - Be consistent with labeling
   - Check annotations before training

4. **Training Parameters**:
   - Start with 100 epochs
   - Monitor validation loss
   - Adjust based on results

## Integration with Your Current App

Once trained, you can use your custom model in `ball_tracker2.py`:

```python
# Replace the generic YOLO model with your custom one
yolo_detector = YOLOBallDetector('models/ball_laser_detector/weights/best.pt')
```

Your custom model will be specifically trained on your ball and laser data, potentially providing better accuracy than the generic YOLO model.
