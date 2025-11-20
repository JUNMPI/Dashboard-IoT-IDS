"""
Model loading and inference utilities for IoT-IDS.

This module handles loading trained Autoencoder-FNN models and their
preprocessing artifacts (scalers, encoders) for threat classification.
"""

import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM MODEL LOADING TO HANDLE KERAS VERSION COMPATIBILITY
# =============================================================================

def _load_model_with_compatibility(model_path: Path) -> tf.keras.Model:
    """
    Load Keras model with compatibility handling for different versions.

    Handles compatibility issues when loading models saved with different
    Keras versions (batch_shape, DTypePolicy, etc.).

    Args:
        model_path: Path to the .h5 model file

    Returns:
        Loaded Keras model
    """
    # First attempt: Normal load
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully (normal load)")
        return model
    except Exception as e:
        logger.warning(f"Normal load failed: {str(e)[:100]}...")

    # Second attempt: Load weights only (most compatible approach)
    try:
        logger.info("Attempting to load using load_weights approach...")

        from tensorflow.keras import layers, Model

        # Define the Autoencoder-FNN architecture with LeakyReLU
        # Arquitectura exacta del modelo guardado

        # Input layer
        input_layer = layers.Input(shape=(16,), name='Tensor_de_Entrada_X')

        # Encoder: 16 -> 12 -> 8 -> 6 (latente)
        x = layers.Dense(12, activation='linear', name='dense')(input_layer)
        x = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu')(x)
        x = layers.Dense(8, activation='linear', name='dense_1')(x)
        x = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu_1')(x)
        latent = layers.Dense(6, activation='linear', name='Espacio_Latente_Z')(x)
        latent_activated = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu_2')(latent)

        # Decoder branch: 6 -> 8 -> 12 -> 16 (reconstrucción)
        decoder = layers.Dense(8, activation='linear', name='dense_2')(latent_activated)
        decoder = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu_3')(decoder)
        decoder = layers.Dense(12, activation='linear', name='dense_3')(decoder)
        decoder = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu_4')(decoder)
        reconstruction = layers.Dense(16, activation='linear', name='Salida_Reconstruccion_Xp')(decoder)

        # Classifier branch: 6 -> 64 -> 32 -> 6 (clasificación)
        classifier = layers.Dense(64, activation='linear', name='dense_4')(latent_activated)
        classifier = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu_5')(classifier)
        classifier = layers.Dense(32, activation='linear', name='dense_5')(classifier)
        classifier = layers.LeakyReLU(alpha=0.3, name='leaky_re_lu_6')(classifier)
        classification = layers.Dense(6, activation='softmax', name='Salida_Clasificacion_Yp')(classifier)

        # Create model
        model = Model(inputs=input_layer, outputs=[reconstruction, classification])

        # Load weights
        model.load_weights(str(model_path))

        logger.info("Model loaded successfully using manual architecture + load_weights")
        return model

    except Exception as e2:
        logger.warning(f"Load weights approach failed: {str(e2)[:100]}...")

    # Third attempt: With custom objects for compatibility
    try:
        logger.info("Attempting load with comprehensive custom objects...")

        # Mock DTypePolicy for compatibility
        class DTypePolicy:
            def __init__(self, name='float32'):
                self.name = name

            def get_config(self):
                return {'name': self.name}

        # Custom InputLayer
        def custom_input_layer(*args, **kwargs):
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    kwargs['input_shape'] = batch_shape[1:]
            return keras.layers.InputLayer(*args, **kwargs)

        # Custom Dense layer that handles DTypePolicy
        def custom_dense(*args, **kwargs):
            # Remove or convert DTypePolicy
            if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
                if kwargs['dtype'].get('class_name') == 'DTypePolicy':
                    dtype_name = kwargs['dtype'].get('config', {}).get('name', 'float32')
                    kwargs['dtype'] = dtype_name
            return keras.layers.Dense(*args, **kwargs)

        # Custom LeakyReLU that handles negative_slope -> alpha
        def custom_leaky_relu(*args, **kwargs):
            # Convert negative_slope to alpha
            if 'negative_slope' in kwargs:
                kwargs['alpha'] = kwargs.pop('negative_slope')
            # Remove or convert DTypePolicy
            if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
                if kwargs['dtype'].get('class_name') == 'DTypePolicy':
                    dtype_name = kwargs['dtype'].get('config', {}).get('name', 'float32')
                    kwargs['dtype'] = dtype_name
            return keras.layers.LeakyReLU(*args, **kwargs)

        custom_objects = {
            'DTypePolicy': DTypePolicy,
            'InputLayer': custom_input_layer,
            'Dense': custom_dense,
            'LeakyReLU': custom_leaky_relu,
        }

        with keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)

        logger.info("Model loaded successfully with custom objects")
        return model

    except Exception as e3:
        logger.error(f"All loading attempts failed. Final error: {e3}")
        raise Exception(f"Could not load model from {model_path}. Last error: {e3}")

# =============================================================================
# CONFIGURATION - Update these with your actual filenames
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
SYNTHETIC_DIR = MODELS_DIR / "synthetic"
REAL_DIR = MODELS_DIR / "real"

# Synthetic model artifacts (in models/synthetic/)
SYNTHETIC_MODEL_FILE = "modelo_ae_fnn_iot_synthetic.h5"
SYNTHETIC_SCALER_FILE = "scaler_synthetic.pkl"
SYNTHETIC_ENCODER_FILE = "label_encoder_synthetic.pkl"
SYNTHETIC_CLASSES_FILE = "class_names_synthetic.npy"
SYNTHETIC_METADATA_FILE = "model_metadata_synthetic.json"

# Real model artifacts (in models/real/)
REAL_MODEL_FILE = "modelo_ae_fnn_iot_REAL.h5"
REAL_SCALER_FILE = "scaler_REAL.pkl"
REAL_ENCODER_FILE = "label_encoder_REAL.pkl"
REAL_CLASSES_FILE = "class_names_REAL.npy"
REAL_METADATA_FILE = "model_metadata_REAL.json"

# Model configuration
N_FEATURES = 16  # PCA components (PC1-PC16)
N_CLASSES = 6    # Normal + 5 attack types (brute_force, ddos, mitm, scan, spoofing)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _load_pickle(filepath: Path) -> Any:
    """Load pickled object safely."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {e}")
        raise

def _load_numpy(filepath: Path) -> np.ndarray:
    """Load numpy array safely."""
    try:
        return np.load(filepath, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading numpy array from {filepath}: {e}")
        raise

def _load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON metadata safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        raise

def _verify_model_structure(model: tf.keras.Model) -> bool:
    """
    Verify model has expected AE-FNN structure.

    Args:
        model: Loaded Keras model

    Returns:
        True if structure is valid
    """
    try:
        # Check input shape
        input_shape = model.input_shape
        if input_shape[-1] != N_FEATURES:
            logger.warning(f"Expected {N_FEATURES} input features, got {input_shape[-1]}")
            return False

        # Check if model has multiple outputs (reconstruction + classification)
        outputs = model.output if isinstance(model.output, list) else [model.output]
        if len(outputs) < 2:
            logger.warning("Model should have 2 outputs (reconstruction + classification)")
            return False

        logger.info("Model structure verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying model structure: {e}")
        return False

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

@st.cache_resource
def load_synthetic_model() -> Tuple[tf.keras.Model, Any, Any, np.ndarray, Dict[str, Any]]:
    """
    Load synthetic model and all preprocessing artifacts.

    Returns:
        Tuple of (model, scaler, label_encoder, class_names, metadata)

    Raises:
        FileNotFoundError: If model files are missing
        ValueError: If model structure is invalid
    """
    logger.info("Loading synthetic model...")

    # Load model
    model_path = SYNTHETIC_DIR / SYNTHETIC_MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = _load_model_with_compatibility(model_path)

    # Verify structure
    if not _verify_model_structure(model):
        raise ValueError("Invalid model structure")

    # Load preprocessing artifacts
    scaler = _load_pickle(SYNTHETIC_DIR / SYNTHETIC_SCALER_FILE)
    label_encoder = _load_pickle(SYNTHETIC_DIR / SYNTHETIC_ENCODER_FILE)
    class_names = _load_numpy(SYNTHETIC_DIR / SYNTHETIC_CLASSES_FILE)
    metadata = _load_json(SYNTHETIC_DIR / SYNTHETIC_METADATA_FILE)

    logger.info("Synthetic model loaded successfully")
    return model, scaler, label_encoder, class_names, metadata

@st.cache_resource
def load_real_model() -> Tuple[tf.keras.Model, Any, Any, np.ndarray, Dict[str, Any]]:
    """
    Load real (CICIoT2023) model and all preprocessing artifacts.

    Returns:
        Tuple of (model, scaler, label_encoder, class_names, metadata)

    Raises:
        FileNotFoundError: If model files are missing
        ValueError: If model structure is invalid
    """
    logger.info("Loading real model...")

    # Load model
    model_path = REAL_DIR / REAL_MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = _load_model_with_compatibility(model_path)

    # Verify structure
    if not _verify_model_structure(model):
        raise ValueError("Invalid model structure")

    # Load preprocessing artifacts
    scaler = _load_pickle(REAL_DIR / REAL_SCALER_FILE)
    label_encoder = _load_pickle(REAL_DIR / REAL_ENCODER_FILE)
    class_names = _load_numpy(REAL_DIR / REAL_CLASSES_FILE)
    metadata = _load_json(REAL_DIR / REAL_METADATA_FILE)

    logger.info("Real model loaded successfully")
    return model, scaler, label_encoder, class_names, metadata

def load_model(model_type: str) -> Tuple[tf.keras.Model, Any, Any, np.ndarray, Dict[str, Any]]:
    """
    Load model based on type (synthetic or real).

    Args:
        model_type: Either 'synthetic' or 'real'

    Returns:
        Tuple of (model, scaler, label_encoder, class_names, metadata)

    Raises:
        ValueError: If model_type is invalid
    """
    if model_type == 'synthetic':
        return load_synthetic_model()
    elif model_type == 'real':
        return load_real_model()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'synthetic' or 'real'")

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def predict_sample(
    model: tf.keras.Model,
    scaler: Any,
    label_encoder: Any,
    class_names: np.ndarray,
    sample: np.ndarray
) -> Tuple[str, np.ndarray, float]:
    """
    Predict threat class for a single network traffic sample.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        class_names: Array of class names
        sample: Array of 16 PCA components (raw, not normalized)

    Returns:
        Tuple of (predicted_class, class_probabilities, confidence_percentage)

    Example:
        >>> sample = np.random.randn(16)
        >>> pred, probs, conf = predict_sample(model, scaler, encoder, names, sample)
        >>> print(f"Prediction: {pred} ({conf:.2f}% confidence)")
    """
    # Validate input
    if sample.shape != (N_FEATURES,):
        raise ValueError(f"Sample must have {N_FEATURES} features, got {sample.shape}")

    # Normalize
    sample_scaled = scaler.transform(sample.reshape(1, -1))

    # Predict (suppress verbose output)
    predictions = model.predict(sample_scaled, verbose=0)

    # Extract classification output (second output for AE-FNN)
    # Handle both single output and multi-output models
    if isinstance(predictions, list):
        class_probs = predictions[1]  # Classification output
    else:
        class_probs = predictions

    # Get predicted class
    predicted_idx = np.argmax(class_probs[0])
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0]

    # Calculate confidence
    confidence = float(np.max(class_probs[0]) * 100)

    return predicted_class, class_probs[0], confidence

def predict_batch(
    model: tf.keras.Model,
    scaler: Any,
    label_encoder: Any,
    class_names: np.ndarray,
    samples: np.ndarray,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict threat classes for multiple samples efficiently.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        class_names: Array of class names
        samples: Array of shape (n_samples, 16)
        batch_size: Batch size for prediction

    Returns:
        Tuple of (predictions_array, confidences_array)

    Example:
        >>> samples = np.random.randn(100, 16)
        >>> preds, confs = predict_batch(model, scaler, encoder, names, samples)
    """
    # Validate input
    if samples.ndim != 2 or samples.shape[1] != N_FEATURES:
        raise ValueError(f"Samples must have shape (n, {N_FEATURES}), got {samples.shape}")

    # Normalize batch
    samples_scaled = scaler.transform(samples)

    # Predict
    predictions = model.predict(samples_scaled, batch_size=batch_size, verbose=0)

    # Extract classification output
    if isinstance(predictions, list):
        class_probs = predictions[1]
    else:
        class_probs = predictions

    # Get predicted classes
    predicted_indices = np.argmax(class_probs, axis=1)
    predicted_classes = label_encoder.inverse_transform(predicted_indices)

    # Get confidences
    confidences = np.max(class_probs, axis=1) * 100

    return predicted_classes, confidences

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info(metadata: Dict[str, Any]) -> str:
    """
    Format model metadata for display.

    Args:
        metadata: Model metadata dictionary

    Returns:
        Formatted string with model information
    """
    info = []
    info.append(f"Accuracy: {metadata.get('accuracy', 'N/A')}")
    info.append(f"Precision: {metadata.get('precision', 'N/A')}")
    info.append(f"Recall: {metadata.get('recall', 'N/A')}")
    info.append(f"F1-Score: {metadata.get('f1_score', 'N/A')}")
    return " | ".join(info)

def check_model_files(model_type: str) -> Dict[str, bool]:
    """
    Check if all required model files exist.

    Args:
        model_type: Either 'synthetic' or 'real'

    Returns:
        Dictionary mapping filename to existence status
    """
    if model_type == 'synthetic':
        base_dir = SYNTHETIC_DIR
        files = {
            'model': SYNTHETIC_MODEL_FILE,
            'scaler': SYNTHETIC_SCALER_FILE,
            'encoder': SYNTHETIC_ENCODER_FILE,
            'classes': SYNTHETIC_CLASSES_FILE,
            'metadata': SYNTHETIC_METADATA_FILE
        }
    else:
        base_dir = REAL_DIR
        files = {
            'model': REAL_MODEL_FILE,
            'scaler': REAL_SCALER_FILE,
            'encoder': REAL_ENCODER_FILE,
            'classes': REAL_CLASSES_FILE,
            'metadata': REAL_METADATA_FILE
        }

    status = {}
    for key, filename in files.items():
        filepath = base_dir / filename
        status[key] = filepath.exists()

    return status
