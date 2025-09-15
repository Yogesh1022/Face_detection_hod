#!/usr/bin/env python3
"""
Highly Accurate Face Recognition Model for Single Person Detection
Using MobileNetV2 + Transfer Learning + MediaPipe Face Detection

This script creates a binary classifier to detect a specific person from 12,000 training images.
Designed for high precision (99%+ accuracy) with real-time capability on Raspberry Pi 4.

Required folder structure:
- train_data/          # 12,000 images of the target person
- negative_data/       # (Optional) Other people's faces for negative samples
- test.jpg            # Test image for inference

Dependencies:
pip install tensorflow opencv-python mediapipe numpy scikit-learn tqdm pillow matplotlib
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mediapipe as mp
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    def __init__(self, img_size=(224, 224), batch_size=32, epochs=20):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        logger.info("Face Recognition Model initialized")
    
    def detect_and_crop_face(self, image_path):
        """
        Detect and crop face from image using MediaPipe
        Returns cropped face or None if no face detected
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                # Get the first (largest) detection
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding and ensure boundaries
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)
                
                # Crop face
                face = image[y:y+height, x:x+width]
                return face
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def preprocess_image(self, face_image):
        """
        Preprocess face image for model input
        """
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_image, self.img_size)
            
            # Convert to RGB if needed
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            return face_normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def load_and_preprocess_data(self, positive_dir='train_data', negative_dir='negative_data'):
        """
        Load and preprocess training data
        """
        logger.info("Loading and preprocessing data...")
        
        X, y = [], []
        
        # Load positive samples (target person)
        if os.path.exists(positive_dir):
            positive_files = [f for f in os.listdir(positive_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            logger.info(f"Found {len(positive_files)} positive samples")
            
            for filename in tqdm(positive_files, desc="Processing positive samples"):
                filepath = os.path.join(positive_dir, filename)
                face = self.detect_and_crop_face(filepath)
                
                if face is not None:
                    processed_face = self.preprocess_image(face)
                    if processed_face is not None:
                        X.append(processed_face)
                        y.append(1)  # Positive class
        else:
            logger.error(f"Positive data directory '{positive_dir}' not found!")
            return None, None
        
        # Load negative samples (other people)
        negative_count = 0
        if os.path.exists(negative_dir):
            negative_files = [f for f in os.listdir(negative_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            logger.info(f"Found {len(negative_files)} negative samples")
            
            # Limit negative samples to balance dataset
            max_negatives = min(len(negative_files), len(X))
            negative_files = random.sample(negative_files, max_negatives)
            
            for filename in tqdm(negative_files, desc="Processing negative samples"):
                filepath = os.path.join(negative_dir, filename)
                face = self.detect_and_crop_face(filepath)
                
                if face is not None:
                    processed_face = self.preprocess_image(face)
                    if processed_face is not None:
                        X.append(processed_face)
                        y.append(0)  # Negative class
                        negative_count += 1
        else:
            logger.warning(f"Negative data directory '{negative_dir}' not found!")
            logger.info("Generating synthetic negative samples through augmentation...")
            
            # Create synthetic negatives through heavy augmentation
            negative_count = len(X) // 4  # 25% of positives as negatives
            synthetic_negatives = self.generate_synthetic_negatives(X[:negative_count])
            
            X.extend(synthetic_negatives)
            y.extend([0] * len(synthetic_negatives))
        
        logger.info(f"Total samples: {len(X)} (Positive: {len(X) - negative_count}, Negative: {negative_count})")
        
        return np.array(X), np.array(y)
    
    def generate_synthetic_negatives(self, positive_samples):
        """
        Generate synthetic negative samples through heavy augmentation
        """
        logger.info("Generating synthetic negative samples...")
        
        synthetic_negatives = []
        
        for sample in positive_samples:
            # Apply heavy distortions to make it a "negative" sample
            
            # Random rotation (large angles)
            angle = random.randint(90, 270)
            h, w = sample.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(sample, M, (w, h))
            
            # Random color distortion
            distorted = sample.copy()
            distorted[:, :, 0] = np.clip(distorted[:, :, 0] * random.uniform(0.3, 2.0), 0, 1)
            distorted[:, :, 1] = np.clip(distorted[:, :, 1] * random.uniform(0.3, 2.0), 0, 1)
            distorted[:, :, 2] = np.clip(distorted[:, :, 2] * random.uniform(0.3, 2.0), 0, 1)
            
            # Add noise
            noise = np.random.normal(0, 0.1, sample.shape)
            noisy = np.clip(sample + noise, 0, 1)
            
            # Random cropping and resizing
            crop_size = random.randint(150, 200)
            start_x = random.randint(0, 224 - crop_size)
            start_y = random.randint(0, 224 - crop_size)
            cropped = sample[start_y:start_y+crop_size, start_x:start_x+crop_size]
            resized = cv2.resize(cropped, (224, 224))
            
            synthetic_negatives.extend([rotated, distorted, noisy, resized])
        
        return synthetic_negatives
    
    def create_model(self):
        """
        Create MobileNetV2-based model for binary classification
        """
        logger.info("Creating MobileNetV2-based model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        self.model = model
        
        return model
    
    def fine_tune_model(self):
        """
        Unfreeze and fine-tune the last layers of the base model
        """
        logger.info("Fine-tuning model...")
        
        # Unfreeze the last few layers of MobileNetV2
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Fine-tuning from layer {fine_tune_at}")
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators with augmentation
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_model(self, X, y):
        """
        Train the face recognition model
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(X_train, y_train, X_val, y_val)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_face_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Calculate steps
        steps_per_epoch = len(X_train) // self.batch_size
        validation_steps = len(X_val) // self.batch_size
        
        # Train model (Phase 1: Feature extraction)
        logger.info("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs // 2,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tune model (Phase 2)
        logger.info("Phase 2: Fine-tuning...")
        self.fine_tune_model()
        
        history2 = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs // 2,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        # Evaluate model
        self.evaluate_model(X_val, y_val)
        
        # Save final model
        self.model.save('models/face_recognition_final.h5')
        logger.info("Model training completed and saved!")
        
        return self.history
    
    def evaluate_model(self, X_val, y_val):
        """
        Evaluate model performance
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_val, y_pred_binary)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_binary)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Classification report
        report = classification_report(y_val, y_pred_binary, target_names=['Not Person', 'Target Person'])
        logger.info(f"Classification Report:\n{report}")
        
        return accuracy, precision, recall, f1
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('logs/training_history.png')
        plt.show()
    
    def load_model(self, model_path='models/best_face_model.h5'):
        """
        Load trained model
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_single_image(self, image_path, threshold=0.9):
        """
        Predict if a single image contains the target person
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return None, None
        
        # Detect and crop face
        face = self.detect_and_crop_face(image_path)
        if face is None:
            logger.warning(f"No face detected in {image_path}")
            return None, None
        
        # Preprocess
        processed_face = self.preprocess_image(face)
        if processed_face is None:
            logger.error("Error preprocessing image")
            return None, None
        
        # Predict
        prediction = self.model.predict(np.expand_dims(processed_face, axis=0))[0][0]
        
        is_target_person = prediction >= threshold
        confidence = prediction if is_target_person else 1 - prediction
        
        logger.info(f"Prediction: {prediction:.4f}, Is target person: {is_target_person}, Confidence: {confidence:.4f}")
        
        return is_target_person, confidence
    
    def real_time_detection(self, camera_index=0, threshold=0.9):
        """
        Real-time face detection using camera (optimized for Raspberry Pi 4)
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return
        
        logger.info("Starting real-time detection...")
        logger.info("Press 'q' to quit, 's' to save current frame")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Raspberry Pi
        
        frame_count = 0
        process_every_n_frames = 3  # Process every 3rd frame for speed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % process_every_n_frames == 0:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Extract face
                        face_roi = frame[y:y+height, x:x+width]
                        
                        if face_roi.size > 0:
                            # Preprocess face
                            processed_face = self.preprocess_image(face_roi)
                            
                            if processed_face is not None:
                                # Predict
                                prediction = self.model.predict(
                                    np.expand_dims(processed_face, axis=0), 
                                    verbose=0
                                )[0][0]
                                
                                is_target = prediction >= threshold
                                confidence = prediction if is_target else 1 - prediction
                                
                                # Draw bounding box and label
                                color = (0, 255, 0) if is_target else (0, 0, 255)
                                label = f"TARGET: {confidence:.2f}" if is_target else f"OTHER: {confidence:.2f}"
                                
                                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                                cv2.putText(frame, label, (x, y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'logs/capture_{timestamp}.jpg', frame)
                logger.info(f"Frame saved as capture_{timestamp}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time detection stopped")
    
    def convert_to_tflite(self, model_path='models/best_face_model.h5', output_path='models/face_model.tflite'):
        """
        Convert model to TensorFlow Lite for Raspberry Pi optimization
        """
        try:
            # Load the model
            model = tf.keras.models.load_model(model_path)
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Optional: Use float16 quantization for smaller size
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save the model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TensorFlow Lite model saved to {output_path}")
            
            # Print model size
            model_size = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"TensorFlow Lite model size: {model_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting to TensorFlow Lite: {str(e)}")
            return False

def main():
    """
    Main function to train and test the face recognition model
    """
    logger.info("Starting Face Recognition Model Training")
    
    # Initialize model
    model = FaceRecognitionModel(
        img_size=(224, 224),
        batch_size=32,
        epochs=20
    )
    
    try:
        # Check if model already exists
        if os.path.exists('models/best_face_model.h5'):
            logger.info("Found existing model. Loading...")
            if model.load_model():
                logger.info("Model loaded successfully!")
            else:
                logger.info("Failed to load model. Training new model...")
                raise FileNotFoundError
        else:
            raise FileNotFoundError
            
    except FileNotFoundError:
        # Load and preprocess data
        X, y = model.load_and_preprocess_data()
        
        if X is None or len(X) == 0:
            logger.error("No training data found! Please ensure 'train_data' folder exists with images.")
            return
        
        # Create model
        model.create_model()
        
        # Train model
        history = model.train_model(X, y)
        
        # Plot training history
        model.plot_training_history()
        
        # Convert to TensorFlow Lite for Raspberry Pi
        model.convert_to_tflite()
    
    # Test on a single image
    if os.path.exists('test.jpg'):
        logger.info("Testing on test.jpg...")
        is_target, confidence = model.predict_single_image('test.jpg', threshold=0.9)
        
        if is_target is not None:
            result = "TARGET PERSON" if is_target else "NOT TARGET PERSON"
            logger.info(f"Result: {result} (Confidence: {confidence:.4f})")
        else:
            logger.warning("Could not process test.jpg")
    else:
        logger.warning("test.jpg not found for testing")
    
    # Uncomment for real-time detection
    # model.real_time_detection(camera_index=0, threshold=0.9)
    
    logger.info("Face Recognition Model setup completed!")

if __name__ == "__main__":
    main()