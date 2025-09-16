#!/usr/bin/env python3
"""
Highly Accurate Face Recognition Model for Single Person Detection
Using MobileNetV2 + Transfer Learning + MediaPipe Face Detection with Batch Processing

This script creates a binary classifier to detect a specific person from images.
Designed for high precision with memory-efficient batch processing.

Required folder structure:
- faces/             # Positive samples (faces to detect)
- negative_data/     # (Optional) Other faces for negative samples
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
# Removed mediapipe import as it's not used in this version
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import glob
from datetime import datetime
import json

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
        
    def build_model(self):
        """
        Build and compile the model
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

    def create_data_generator(self, batch_size=32):
        """
        Create a generator that yields batches of data
        """
        positive_paths = []
        negative_paths = []
        
        # Collect positive sample paths
        positive_dir = "faces"
        if os.path.exists(positive_dir):
            positive_paths = glob.glob(os.path.join(positive_dir, "*.jpg"))
            if not positive_paths:
                logger.error(f"No .jpg files found in '{positive_dir}' directory!")
                raise FileNotFoundError(f"No .jpg files found in '{positive_dir}' directory!")
        else:
            logger.error(f"Positive data directory '{positive_dir}' not found!")
            raise FileNotFoundError(f"Directory '{positive_dir}' not found!")
        
        # Collect negative sample paths
        negative_dir = "negative_data"
        if os.path.exists(negative_dir):
            negative_paths = glob.glob(os.path.join(negative_dir, "*.jpg"))
        else:
            logger.warning(f"Negative data directory '{negative_dir}' not found!")
            logger.info("Will generate synthetic negative samples during training...")
            
        total_positives = len(positive_paths)
        total_negatives = len(negative_paths)
        
        if total_negatives == 0:
            # We'll generate synthetic negatives on the fly
            total_negatives = total_positives // 4
        
        logger.info(f"Total samples: {total_positives + total_negatives} (Positive: {total_positives}, Negative: {total_negatives})")
        
        def generator():
            while True:
                # Initialize batch arrays
                batch_images = []
                batch_labels = []
                
                # Mix positive and negative samples in the batch
                while len(batch_images) < batch_size:
                    # Determine if we should add a positive or negative sample
                    if len(batch_images) < batch_size * 0.7:  # 70% positive samples
                        # Add positive sample
                        if positive_paths:
                            img_path = random.choice(positive_paths)
                            try:
                                img = cv2.imread(img_path)
                                img = cv2.resize(img, (224, 224))
                                img = img.astype(np.float32) / 255.0  # Normalize here
                                batch_images.append(img)
                                batch_labels.append(1)
                            except Exception as e:
                                logger.error(f"Error processing {img_path}: {e}")
                                continue
                    else:
                        # Add negative sample
                        if negative_paths:
                            img_path = random.choice(negative_paths)
                        else:
                            # Generate synthetic negative from a random positive
                            img_path = random.choice(positive_paths)
                            try:
                                img = cv2.imread(img_path)
                                img = cv2.resize(img, (224, 224))
                                if negative_paths:
                                    img = img.astype(np.float32) / 255.0
                                else:
                                    img = self.apply_augmentation(img).astype(np.float32) / 255.0
                                batch_images.append(img)
                                batch_labels.append(0)
                            except Exception as e:
                                logger.error(f"Error processing negative {img_path}: {e}")
                                continue
                
                yield np.array(batch_images), np.array(batch_labels)
        
        steps_per_epoch = (total_positives + total_negatives) // batch_size
        return generator(), steps_per_epoch

    def apply_augmentation(self, image):
        """
        Apply heavy augmentation to create synthetic negative samples
        """
        # Random rotation
        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness and contrast
        alpha = random.uniform(0.5, 1.5)  # Contrast
        beta = random.uniform(-30, 30)    # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        return image

    def train(self):
        """
        Train the model using batch processing
        """
        # Build the model if not already built
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_gen, steps_per_epoch = self.create_data_generator(self.batch_size)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                save_weights_only=False
            ),
            CustomModelSaver()  # This will save as .pkl after each epoch
        ]
        
        # Train the model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            callbacks=callbacks
        )
        
        return self.history

def main():
    try:
        # Initialize model with very small batch size to reduce memory usage
        model = FaceRecognitionModel(batch_size=8, epochs=20)
        
        # Train model
        history = model.train()
        
        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(history.history, f)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()