"""
Pre-Inference AI Data Quality & Out-Of-Distribution (OOD) Checker — TumorAI XAI
Audits incoming MRI scan resolution, brightness, contrast, blur, noise levels, 
and out-of-distribution flags prior to deep learning model inference.
"""

import cv2
import numpy as np

def check_image_quality(img_path_or_array):
    """
    Performs comprehensive pre-inference quality audit on an MRI scan.
    
    Args:
        img_path_or_array: File path string or RGB/Grayscale numpy array
        
    Returns:
        dict containing quality metrics, pass/fail status, OOD score, and warnings.
    """
    if isinstance(img_path_or_array, str):
        img_bgr = cv2.imread(img_path_or_array)
        if img_bgr is None:
            return {
                'passed': False,
                'quality_score': 0.0,
                'status': 'Error Loading Image',
                'warnings': ['File corrupted or unreadable image format.'],
                'ood_score': 100.0
            }
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        arr = img_path_or_array
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)

    h, w = gray.shape[:2]
    warnings = []

    # 1. Resolution Check
    min_dim = min(h, w)
    res_passed = min_dim >= 128
    if not res_passed:
        warnings.append(f"Low image resolution ({w}x{h}). Minimum recommended resolution is 128x128.")

    # 2. Brightness Check
    mean_brightness = float(np.mean(gray))
    brightness_passed = 15.0 <= mean_brightness <= 235.0
    if mean_brightness < 15.0:
        warnings.append("Image is severely underexposed/dark.")
    elif mean_brightness > 235.0:
        warnings.append("Image is severely overexposed/washed out.")

    # 3. Contrast Check (Standard Deviation)
    std_contrast = float(np.std(gray))
    contrast_passed = std_contrast >= 20.0
    if not contrast_passed:
        warnings.append("Low image contrast detected. Feature extraction may be degraded.")

    # 4. Blur / Sharpness Check (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_passed = laplacian_var >= 45.0
    if not sharpness_passed:
        warnings.append(f"Image blur detected (Laplacian variance {round(laplacian_var, 1)}).")

    # 5. Out-Of-Distribution (OOD) Estimation
    # Compute histogram uniformity and border intensity profile to detect non-MRI scans (e.g. photos/diagrams)
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    border_mean = float(np.mean(border_pixels))
    
    hist, _ = np.histogram(gray, bins=16, range=(0, 256))
    hist_norm = hist / np.sum(hist)
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))

    # Typical MRI background is dark (border_mean < 40) and entropy is bounded (2.5 - 3.8)
    ood_penalty = 0
    if border_mean > 50.0:
        ood_penalty += 35
        warnings.append("Out-of-Distribution warning: Non-standard background lighting or orientation.")
    if entropy < 1.5 or entropy > 3.9:
        ood_penalty += 25
        warnings.append("Out-of-Distribution warning: Abnormal pixel intensity distribution.")

    ood_score = round(min(100.0, float(ood_penalty + (0 if sharpness_passed else 20))), 1)
    
    # Calculate Overall Quality Score (0 - 100%)
    base_score = 100.0
    if not res_passed: base_score -= 20.0
    if not brightness_passed: base_score -= 15.0
    if not contrast_passed: base_score -= 15.0
    if not sharpness_passed: base_score -= 20.0
    base_score -= (ood_penalty * 0.3)
    
    quality_score = round(max(0.0, base_score), 1)
    passed = len(warnings) == 0 or (quality_score >= 60.0 and ood_score < 50.0)

    status = "Optimal Quality (Passed)" if passed else "Quality Warnings Flagged"

    return {
        'passed': passed,
        'quality_score': quality_score,
        'status': status,
        'metrics': {
            'resolution': f"{w}x{h}",
            'mean_brightness': round(mean_brightness, 1),
            'std_contrast': round(std_contrast, 1),
            'sharpness_laplacian': round(laplacian_var, 1),
            'pixel_entropy': round(entropy, 2)
        },
        'ood_score': ood_score,
        'warnings': warnings
    }
