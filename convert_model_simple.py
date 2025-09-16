#!/usr/bin/env python3
"""
Simplified model converter from .h5 to .pkl
"""

import tensorflow as tf
import pickle
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_model(h5_path='best_model.h5', pkl_path='best_model.pkl'):
    """
    Convert model from .h5 to .pkl format
    """
    try:
        logger.info(f"Loading model from {h5_path}")
        model = tf.keras.models.load_model(h5_path)
        
        # Save just the weights
        weights = model.get_weights()
        
        logger.info(f"Saving weights to {pkl_path}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(weights, f)
        
        logger.info("Model weights saved successfully!")
        return True
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        return False

if __name__ == "__main__":
    convert_model()