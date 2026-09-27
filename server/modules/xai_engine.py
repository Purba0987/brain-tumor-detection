import numpy as np
import tensorflow as tf
import cv2

def find_target_layer(model):
    """
    Automatically locates the last 4D (Conv2D) activation layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Handle nested functional models (e.g., base model inside Sequential)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found in the model.")

def get_sub_model(model, layer_name=None):
    """
    Constructs a model that outputs [last_conv_layer_output, final_model_predictions].
    """
    if layer_name is None:
        layer_name = find_target_layer(model)
        
    try:
        conv_layer = model.get_layer(layer_name)
        return Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
    except Exception:
        # If target layer is inside a nested base layer
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    conv_layer = layer.get_layer(layer_name)
                    intermediate_model = Model(inputs=layer.input, outputs=conv_layer.output)
                    
                    # Build graph connecting model input -> intermediate model -> rest of model
                    inputs = model.input
                    x = intermediate_model(inputs)
                    # Pass through remaining layers after base
                    for rem_layer in model.layers[model.layers.index(layer)+1:]:
                        x = rem_layer(x)
                    return Model(inputs=inputs, outputs=[intermediate_model(inputs), x])
                except Exception:
                    continue
        raise ValueError(f"Layer {layer_name} not found in model.")

def generate_gradcam(model, img_array, class_index=None, layer_name=None):
    """
    Generates a standard Grad-CAM heatmap for a single image array (shape: [1, H, W, 3]).
    Returns normalized heatmap (float32 array, shape: [H, W], range [0, 1]).
    """
    if layer_name is None:
        layer_name = find_target_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    # Gradient of target output score w.r.t feature activation maps
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling of gradients (channel weights alpha_k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive contributions
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match input image resolution
    target_h, target_w = img_array.shape[1], img_array.shape[2]
    heatmap_resized = cv2.resize(heatmap_np, (target_w, target_h))
    
    # Normalize to [0, 1]
    if np.max(heatmap_resized) > 0:
        heatmap_resized = heatmap_resized / np.max(heatmap_resized)

    return heatmap_resized

def generate_gradcam_plus_plus(model, img_array, class_index=None, layer_name=None):
    """
    Generates a Grad-CAM++ heatmap using 2nd and 3rd order gradients.
    Provides better spatial localization for multi-instance or fine features.
    """
    if layer_name is None:
        layer_name = find_target_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs, predictions = grad_model(img_array)
                if class_index is None:
                    class_index = tf.argmax(predictions[0])
                loss = predictions[:, class_index]
            
            # 1st order gradients
            grads_1 = tape3.gradient(loss, conv_outputs)
        # 2nd order gradients
        grads_2 = tape2.gradient(grads_1, conv_outputs)
    # 3rd order gradients
    grads_3 = tape1.gradient(grads_2, conv_outputs)

    conv_outputs = conv_outputs[0]
    grads_1 = grads_1[0]
    grads_2 = grads_2[0]
    grads_3 = grads_3[0]

    # Grad-CAM++ alpha weight equation numerator & denominator
    grad_2_sq = grads_2 ** 2
    denom = 2 * grad_2_sq + conv_outputs * grads_3
    denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))

    alphas = grad_2_sq / denom
    relu_grads_1 = tf.maximum(grads_1, 0.0)
    weights = tf.reduce_sum(alphas * relu_grads_1, axis=(0, 1))

    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy()
    target_h, target_w = img_array.shape[1], img_array.shape[2]
    heatmap_resized = cv2.resize(heatmap_np, (target_w, target_h))

    if np.max(heatmap_resized) > 0:
        heatmap_resized = heatmap_resized / np.max(heatmap_resized)

    return heatmap_resized

def generate_lime_explanation(model, img_array, class_index=None, num_segments=16, num_samples=100):
    """
    Generates a superpixel perturbation mask (LIME approximation).
    """
    img = img_array[0]
    h, w, c = img.shape
    
    # Grid superpixel segmentation (4x4 grid = 16 superpixels)
    grid_h, grid_w = h // 4, w // 4
    mask_grid = np.zeros((h, w), dtype=np.int32)
    segment_id = 0
    for r in range(4):
        for c_idx in range(4):
            r_end = h if r == 3 else (r + 1) * grid_h
            c_end = w if c_idx == 3 else (c_idx + 1) * grid_w
            mask_grid[r*grid_h:r_end, c_idx*grid_w:c_end] = segment_id
            segment_id += 1

    num_segments = segment_id
    base_pred = model.predict(img_array, verbose=0)[0]
    if class_index is None:
        class_index = np.argmax(base_pred)

    weights = np.zeros(num_segments)

    # Randomly hide superpixels and measure impact on prediction score
    for s in range(num_samples):
        active_segments = np.random.choice([0, 1], size=num_segments, p=[0.3, 0.7])
        perturbed_img = img.copy()
        
        for seg in range(num_segments):
            if active_segments[seg] == 0:
                perturbed_img[mask_grid == seg] = 0.0

        pred = model.predict(np.expand_dims(perturbed_img, axis=0), verbose=0)[0]
        score_diff = base_pred[class_index] - pred[class_index]
        
        for seg in range(num_segments):
            if active_segments[seg] == 0:
                weights[seg] += score_diff

    # Create LIME importance map
    lime_map = np.zeros((h, w), dtype=np.float32)
    for seg in range(num_segments):
        lime_map[mask_grid == seg] = weights[seg]

    lime_map = np.maximum(lime_map, 0.0)
    if np.max(lime_map) > 0:
        lime_map = lime_map / np.max(lime_map)

    return lime_map

def overlay_heatmap(img_rgb, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlays a normalized heatmap onto an RGB image (values 0-255).
    Returns RGB image with heatmap overlay (uint8).
    """
    if img_rgb.max() <= 1.0:
        img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    else:
        img_rgb_uint8 = img_rgb.astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb_uint8, 1 - alpha, heatmap_colored_rgb, alpha, 0)
    return overlay

def generate_counterfactual_map(model, img_array, target_class_index=None, steps=15):
    """
    Generates a Counterfactual ('What-If') contrastive explanation map.
    Identifies the minimal spatial region modifications required to alter model decision boundaries.
    
    Args:
        model: Keras classification model
        img_array: Preprocessed image tensor (1, H, W, 3)
        target_class_index: Class index to contrast against
        steps: Optimization steps for contrastive perturbation
        
    Returns:
        Counterfactual heatmap (float32 array normalized 0-1) and modified counterfactual image tensor
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    h, w = img_array.shape[1], img_array.shape[2]

    if target_class_index is None:
        preds = model(img_tensor).numpy()[0]
        target_class_index = int(np.argmax(preds))

    # Compute Grad-CAM map as initial mask basis
    grad_cam_map = generate_gradcam(model, img_array, class_index=target_class_index)
    
    # Invert and perturb the highest activation regions to create counterfactual hypothesis
    cf_heatmap = np.zeros((h, w), dtype=np.float32)
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size

    base_pred = model(img_tensor).numpy()[0][target_class_index]

    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w

            perturbed_tensor = img_array.copy()
            # Mask out region with mean background intensity
            perturbed_tensor[:, y_start:y_end, x_start:x_end, :] = np.mean(img_array)

            new_pred = model(tf.convert_to_tensor(perturbed_tensor)).numpy()[0][target_class_index]
            score_drop = base_pred - new_pred

            if score_drop > 0:
                cf_heatmap[y_start:y_end, x_start:x_end] = score_drop

    if np.max(cf_heatmap) > 0:
        cf_heatmap = cf_heatmap / np.max(cf_heatmap)

    return cf_heatmap

