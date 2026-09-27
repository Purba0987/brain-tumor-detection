"""
Multi-Model Ensemble & Consensus Voting Engine — TumorAI XAI
Executes multi-architecture inference across VGG16, ResNet50, and EfficientNetB0,
computing ensemble class probabilities, confidence scores, and inter-model consensus agreement.
"""

import os
import numpy as np
import tensorflow as tf
from modules.model_trainer import CLASSES, build_model

# Cache for loaded models
_LOADED_MODELS = {}

def get_model(arch, weights_path=None):
    """
    Returns or loads a model instance for the specified architecture.
    """
    global _LOADED_MODELS
    arch_key = arch.lower().strip()
    
    if arch_key in _LOADED_MODELS:
        return _LOADED_MODELS[arch_key]
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    path_map = {
        'vgg16': os.path.join(models_dir, 'vgg16_best.h5'),
        'resnet50': os.path.join(models_dir, 'resnet50_best.h5'),
        'efficientnetb0': os.path.join(models_dir, 'efficientnetb0_best.h5')
    }
    
    if weights_path is None:
        weights_path = path_map.get(arch_key, os.path.join(models_dir, f"{arch_key}_best.h5"))
        if not os.path.exists(weights_path) and arch_key == 'vgg16':
            weights_path = os.path.join(models_dir, 'model.h5')

    try:
        if os.path.exists(weights_path):
            print(f"Loading model '{arch_key}' from {weights_path}...")
            model = tf.keras.models.load_model(weights_path, compile=False)
        else:
            print(f"Weights file for '{arch_key}' not found at {weights_path}. Building initialized architecture...")
            model = build_model(architecture=arch_key)
            
        _LOADED_MODELS[arch_key] = model
        return model
    except Exception as e:
        print(f"Error loading model '{arch_key}': {e}. Building fallback model structure.")
        model = build_model(architecture=arch_key)
        _LOADED_MODELS[arch_key] = model
        return model

def predict_ensemble(img_array, primary_model=None):
    """
    Performs ensemble prediction across VGG16, ResNet50, and EfficientNetB0 models.
    
    Args:
        img_array: Preprocessed image tensor shape (1, 128, 128, 3)
        primary_model: Optional pre-loaded main model to avoid re-loading
        
    Returns:
        dict with ensemble_class, ensemble_confidence, consensus_score, 
             consensus_status, individual_predictions
    """
    architectures = ['vgg16', 'resnet50', 'efficientnetb0']
    individual_preds = {}
    model_top_classes = []
    prob_vectors = []

    for arch in architectures:
        try:
            if arch == 'vgg16' and primary_model is not None:
                model = primary_model
            else:
                model = get_model(arch)

            preds = model.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_class = CLASSES[top_idx]
            conf = float(preds[top_idx])

            individual_preds[arch] = {
                'predicted_class': top_class,
                'confidence': round(conf, 4),
                'probabilities': [round(float(p), 4) for p in preds]
            }
            model_top_classes.append(top_class)
            prob_vectors.append(preds)
        except Exception as e:
            print(f"Ensemble prediction error for {arch}: {e}")

    if not prob_vectors:
        return None

    # Weighted Ensemble Average (VGG16 weight 0.5, ResNet 0.25, EffNet 0.25)
    weights = [0.5, 0.25, 0.25][:len(prob_vectors)]
    weights = [w / sum(weights) for w in weights]
    
    ensemble_probs = np.zeros(len(CLASSES))
    for w, p in zip(weights, prob_vectors):
        ensemble_probs += w * p

    ensemble_top_idx = int(np.argmax(ensemble_probs))
    ensemble_top_class = CLASSES[ensemble_top_idx]
    ensemble_conf = float(ensemble_probs[ensemble_top_idx])

    # Consensus Agreement Ratio
    matching_votes = sum(1 for c in model_top_classes if c == ensemble_top_class)
    consensus_score = round((matching_votes / len(model_top_classes)) * 100.0, 1)

    if consensus_score >= 100.0:
        consensus_status = "Unanimous Agreement (3/3 models agree)"
        flagged = False
    elif consensus_score >= 60.0:
        consensus_status = "Majority Agreement (2/3 models agree)"
        flagged = False
    else:
        consensus_status = "Inter-Model Disagreement (⚠️ Radiologist review recommended)"
        flagged = True

    return {
        'ensemble_class': ensemble_top_class,
        'ensemble_confidence': round(ensemble_conf, 4),
        'consensus_score': consensus_score,
        'consensus_status': consensus_status,
        'flagged_for_review': flagged,
        'ensemble_probabilities': [round(float(p), 4) for p in ensemble_probs],
        'individual_predictions': individual_preds
    }
