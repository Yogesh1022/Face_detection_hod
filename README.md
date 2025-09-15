# Head of Department (HOD) Face Detection System

This project implements a highly accurate face detection and recognition system specifically trained to identify a Head of Department (HOD) using deep learning techniques.

## Project Overview

The system uses MobileNetV2 architecture with transfer learning to achieve high accuracy while maintaining real-time performance. It's designed to work efficiently even on resource-constrained devices like Raspberry Pi.

## Project Structure

```
HOD_detection/
├── faces/                  # Positive samples (HOD face images)
├── frames/                # Extracted video frames
├── Frame_data/           # Source video files
├── test_faces/           # Test images of HOD
├── test_negative/        # Test images of other people
├── test_results/         # Visualization results
├── video_frame.py        # Frame extraction script
├── train_face_detection.py      # Original training script
├── train_face_detection_batched.py  # Memory-efficient training script
└── test_model.py         # Model testing script
```

## Setup and Installation

1. Create a Python virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install required packages:
```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn tqdm pillow matplotlib
```

## Usage Guide

### 1. Frame Extraction
Extract frames from videos using:
```bash
python video_frame.py
```

### 2. Model Training
Train the model using the memory-efficient batch processing:
```bash
python train_face_detection_batched.py
```

The training script:
- Uses MobileNetV2 with transfer learning
- Implements batch processing to handle large datasets
- Automatically generates synthetic negative samples
- Saves the best model during training

### 3. Testing the Model

#### Image Testing
Test the model on individual images:
```bash
python test_model.py
```

Results will be saved in the `test_results` directory with visual indicators:
- Green text: Positive detection
- Red text: Negative detection
- Confidence score shown as percentage

### Model Performance

Current model achieves:
- Positive Detection Accuracy: 100%
- Processing Speed: ~7 images/second on CPU
- Model Size: Optimized for edge devices

### Files Description

1. `video_frame.py`
   - Extracts frames from video files
   - Saves frames for training/testing

2. `train_face_detection_batched.py`
   - Implements memory-efficient training
   - Uses batch processing to handle large datasets
   - Includes data augmentation for negative samples

3. `test_model.py`
   - Tests model on images
   - Provides visual results with confidence scores
   - Calculates accuracy metrics

## Training Data Requirements

- Positive Samples: Face images of the target person (HOD)
- Negative Samples: Face images of other people
- Recommended: At least 1000 positive samples for robust training

## Best Practices

1. Data Collection:
   - Collect face images in different lighting conditions
   - Include various angles and expressions
   - Ensure good image quality

2. Training:
   - Use batch processing for large datasets
   - Monitor training progress with visualizations
   - Save model checkpoints regularly

3. Testing:
   - Test with diverse images
   - Validate in real-world conditions
   - Monitor false positive/negative rates

## Troubleshooting

Common issues and solutions:

1. Memory Errors:
   - Use `train_face_detection_batched.py` instead of the original script
   - Reduce batch size if needed
   - Enable memory-efficient data loading

2. Low Accuracy:
   - Increase training data diversity
   - Adjust learning rate
   - Try different data augmentation techniques

3. Slow Processing:
   - Reduce input image size
   - Use GPU acceleration if available
   - Optimize batch size for your hardware

## Future Improvements

1. Planned enhancements:
   - Real-time video processing
   - Multi-person detection
   - Mobile deployment optimizations

## License

This project is for educational and research purposes only. Not for commercial use without permission.

## Contributors

- Initial development and implementation
- Model optimization and testing
- Documentation and maintenance