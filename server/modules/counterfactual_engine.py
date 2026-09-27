"""
Counterfactual Explainable AI & Prediction Sensitivity Engine — TumorAI XAI
Computes contrastive sensitivity maps and region modification impact analysis
answering: 'What minimal image modifications would alter the model's diagnosis?'
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from modules.xai_engine import generate_gradcam

def compute_counterfactual_sensitivity(model, img_array, target_class_index=None, grid_size=8):
    """
    Computes spatial prediction sensitivity and counterfactual contrastive heatmap.
    
    Args:
        model: Loaded Keras model instance
        img_array: Preprocessed image tensor shape (1, H, W, 3)
        target_class_index: Target class index to evaluate
        grid_size: Spatial grid division size
        
    Returns:
        dict containing sensitivity_score, top_influential_regions, and contrastive heatmap path.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    h, w = img_array.shape[1], img_array.shape[2]

    base_preds = model(img_tensor).numpy()[0]
    if target_class_index is None:
        target_class_index = int(np.argmax(base_preds))

    base_score = float(base_preds[target_class_index])

    cf_map = np.zeros((h, w), dtype=np.float32)
    cell_h, cell_w = h // grid_size, w // grid_size
    region_impacts = []

    mean_bg = np.mean(img_array)

    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w

            perturbed = img_array.copy()
            # Mask out current region with mean background value
            perturbed[:, y_start:y_end, x_start:x_end, :] = mean_bg

            new_pred = model(tf.convert_to_tensor(perturbed, dtype=tf.float32)).numpy()[0][target_class_index]
            score_drop = base_score - float(new_pred)

            if score_drop > 0:
                cf_map[y_start:y_end, x_start:x_end] = score_drop
                region_impacts.append({
                    'grid_box': [int(x_start), int(y_start), int(cell_w), int(cell_h)],
                    'score_drop': round(float(score_drop), 4),
                    'drop_percentage': round(float((score_drop / max(base_score, 1e-5)) * 100.0), 1)
                })

    # Normalize counterfactual map
    if np.max(cf_map) > 0:
        cf_map_norm = cf_map / np.max(cf_map)
    else:
        cf_map_norm = cf_map

    # Overall sensitivity index (0 - 100%)
    max_drop = max([r['score_drop'] for r in region_impacts], default=0.0)
    sensitivity_score = round(min(100.0, float(max_drop * 120.0)), 1)

    # Sort top 3 influential spatial regions
    region_impacts.sort(key=lambda x: x['score_drop'], reverse=True)
    top_regions = region_impacts[:3]

    # Create contrastive heatmap overlay (Cyan/Purple colormap)
    img_rgb = (img_array[0] * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array[0].astype(np.uint8)
    cf_uint8 = (cf_map_norm * 255).astype(np.uint8)
    cf_colored = cv2.applyColorMap(cf_uint8, cv2.COLORMAP_COOL)
    cf_colored_rgb = cv2.cvtColor(cf_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 0.55, cf_colored_rgb, 0.45, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Draw top sensitive region boxes
    for idx, r in enumerate(top_regions):
        bx, by, bw_cell, bh_cell = r['grid_box']
        cv2.rectangle(overlay_bgr, (bx, by), (bx + bw_cell, by + bh_cell), (0, 255, 255), 1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotated_dir = os.path.join(base_dir, 'static', 'annotated')
    os.makedirs(annotated_dir, exist_ok=True)

    filename = f"counterfactual_{target_class_index}_{int(sensitivity_score)}.jpg"
    filepath = os.path.join(annotated_dir, filename)
    cv2.imwrite(filepath, overlay_bgr)

    return {
        'sensitivity_score': sensitivity_score,
        'sensitivity_level': "High Sensitivity" if sensitivity_score > 40 else "Moderate Sensitivity",
        'top_influential_regions': top_regions,
        'counterfactual_overlay_path': f"/static/annotated/{filename}",
        'counterfactual_file': filepath
    }
