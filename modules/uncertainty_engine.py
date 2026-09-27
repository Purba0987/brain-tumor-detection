import numpy as np
import tensorflow as tf

def predict_with_uncertainty(model, img_array, num_samples=20, uncertainty_threshold=0.025):
    """
    Performs Monte Carlo (MC) Dropout stochastic forward passes to estimate predictive confidence and epistemic uncertainty.
    
    Args:
        model: Trained Keras model with Dropout layers.
        img_array: Preprocessed image tensor of shape (1, H, W, 3).
        num_samples: Number of MC forward passes (default 20).
        uncertainty_threshold: Variance threshold for classifying Low vs High uncertainty.
        
    Returns:
        dict containing predicted_class, class_probabilities, confidence, uncertainty_variance, entropy, reliability_level, clinical_note.
    """
    predictions = []
    
    # Run N stochastic forward passes with dropout active
    for _ in range(num_samples):
        preds = model(img_array, training=True)  # training=True keeps dropout layers active
        predictions.append(preds.numpy()[0])

    predictions = np.array(predictions)  # Shape: (num_samples, num_classes)
    
    # Mean prediction across all MC runs
    mean_probs = np.mean(predictions, axis=0)
    pred_class_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_class_idx])

    # Epistemic uncertainty = Variance of predictions for the top predicted class
    variance = float(np.var(predictions[:, pred_class_idx]))
    
    # Total predictive entropy H(p)
    eps = 1e-10
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))

    # Categorize reliability
    if variance <= uncertainty_threshold and confidence >= 0.70:
        reliability = 'Low'
        clinical_note = "Model prediction is highly confident and stable."
    else:
        reliability = 'High'
        clinical_note = "High predictive variance detected. Further expert radiologist evaluation is recommended."

    return {
        'predicted_class_index': pred_class_idx,
        'class_probabilities': mean_probs.tolist(),
        'confidence': confidence,
        'uncertainty_variance': variance,
        'predictive_entropy': entropy,
        'reliability_level': reliability,
        'clinical_recommendation': clinical_note
    }
