#!/usr/bin/env python3
"""
Convert TensorFlow/Keras .h5 model to pickle (.pkl) format
"""

import tensorflow as tf
import pickle
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_model_as_pickle(h5_path, pkl_path=None):
    """
    Load a .h5 model and save it as .pkl
    """
    logger.info("Starting model conversion...")
    try:
        # Default pkl path if none provided
        if pkl_path is None:
            pkl_path = h5_path.rsplit('.', 1)[0] + '.pkl'
        
        # Load the model
        logger.info(f"Loading model from {h5_path}")
        model = tf.keras.models.load_model(h5_path)
        
        # Get model weights as a dictionary
        weights = {}
        for layer in model.layers:
            weights[layer.name] = layer.get_weights()
        
        # Get model architecture
        model_config = model.get_config()
        
        # Create a dictionary with all model information
        model_data = {
            'class_name': model.__class__.__name__,
            'config': model_config,
            'weights': weights,
            'optimizer_config': model.optimizer.get_config() if model.optimizer else None
        }
        
        # Save as pickle
        logger.info(f"Saving model as pickle to {pkl_path}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Conversion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        return False

def load_model_from_pickle(pkl_path):
    """
    Load a model from pickle format
    """
    try:
        # Load pickle file
        logger.info(f"Loading model from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct model from architecture
        model = tf.keras.models.Model.from_config(model_data['config'])
        
        # Set weights
        for layer in model.layers:
            if layer.name in model_data['weights']:
                layer.set_weights(model_data['weights'][layer.name])
        
        # Compile model if optimizer config exists
        if model_data['optimizer_config']:
            optimizer = tf.keras.optimizers.deserialize(model_data['optimizer_config'])
            model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        
        logger.info("Model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def main():
    try:
        # Path to your .h5 model
        h5_model_path = 'best_model.h5'
        
        if not os.path.exists(h5_model_path):
            raise FileNotFoundError(f"Model file {h5_model_path} not found!")
        
        # Convert model
        success = save_model_as_pickle(h5_model_path)
        
        if success:
            # Test loading the converted model
            pkl_path = h5_model_path.rsplit('.', 1)[0] + '.pkl'
            test_model = load_model_from_pickle(pkl_path)
            
            if test_model is not None:
                logger.info("Model conversion and test loading successful!")
            else:
                logger.error("Failed to load the converted model!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()