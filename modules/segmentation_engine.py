"""
Automatic Tumor Segmentation & 3D Volume Calculation Engine — TumorAI XAI
Computes pixel-accurate spatial segmentation, boundary contour overlays, 
maximum linear diameter, and estimated 3D tumor volume (cm³) from MRI scans.
"""

import os
import cv2
import numpy as np

DEFAULT_PIXEL_SPACING_MM = 0.9375  # Standard MRI spatial resolution (mm/pixel)
DEFAULT_SLICE_THICKNESS_MM = 3.0   # Standard MRI slice thickness (mm)

def segment_tumor(image_path_or_array, mask_or_heatmap=None, pixel_spacing=None, slice_thickness=None):
    """
    Segments tumor region, extracts boundaries, and calculates 3D volume & metrics.
    
    Args:
        image_path_or_array: File path string or RGB numpy array (H, W, 3)
        mask_or_heatmap: Optional ROI mask or Grad-CAM heatmap array (0-1 float or uint8)
        pixel_spacing: Tuple of (spacing_x_mm, spacing_y_mm) or float mm/pixel
        slice_thickness: Slice thickness in mm
        
    Returns:
        dict containing segmentation metrics, boundary contours, and overlay path.
    """
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            raise FileNotFoundError(f"Image file not found: {image_path_or_array}")
        img_bgr = cv2.imread(image_path_or_array)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_path_or_array.copy()
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Resolution scaling calibration
    if pixel_spacing is None:
        px_x = px_y = DEFAULT_PIXEL_SPACING_MM
    elif isinstance(pixel_spacing, (tuple, list)):
        px_x, px_y = pixel_spacing[0], pixel_spacing[1]
    else:
        px_x = px_y = float(pixel_spacing)

    if slice_thickness is None:
        th_mm = DEFAULT_SLICE_THICKNESS_MM
    else:
        th_mm = float(slice_thickness)

    # Segmentation mask processing
    if mask_or_heatmap is not None:
        if mask_or_heatmap.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask_or_heatmap, (w, h))
        else:
            mask_resized = mask_or_heatmap.copy()
        
        if mask_resized.max() <= 1.0:
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
        else:
            mask_uint8 = mask_resized.astype(np.uint8)

        _, bin_mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Otsu thresholding + Morphological refinement on high-contrast regions
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological opening/closing to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)

    # Find largest tumor contour
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'has_lesion': False,
            'area_pixels': 0,
            'area_cm2': 0.0,
            'volume_cm3': 0.0,
            'max_diameter_cm': 0.0,
            'bounding_box': [0, 0, 0, 0],
            'lesion_location': 'No Lesion Detected',
            'contour_overlay_path': None
        }

    # Select largest contiguous contour as primary lesion
    largest_contour = max(contours, key=cv2.contourArea)
    area_pixels = cv2.contourArea(largest_contour)

    # Bounding box & Minimum Area Rect for max diameter
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    rect = cv2.minAreaRect(largest_contour)
    (rect_cx, rect_cy), (rect_w, rect_h), angle = rect

    # Convert dimensions from pixels to mm/cm
    width_mm = bw * px_x
    height_mm = bh * px_y
    max_diam_mm = max(rect_w * px_x, rect_h * px_y)
    max_diam_cm = round(max_diam_mm / 10.0, 2)

    # Calculate 2D Area in cm²
    area_mm2 = area_pixels * (px_x * px_y)
    area_cm2 = round(area_mm2 / 100.0, 2)

    # Estimate 3D Volume in cm³ using Ellipsoidal Model (4/3 * pi * a * b * c)
    # where a, b are semi-axes in cm, and c is estimated depth based on slice thickness & shape
    a_cm = (max(rect_w, rect_h) * px_x) / 20.0
    b_cm = (min(rect_w, rect_h) * px_y) / 20.0
    c_cm = max(a_cm * 0.8, (th_mm / 10.0) * 3)  # Depth approximation based on slice stack
    
    volume_cm3 = round((4.0 / 3.0) * np.pi * a_cm * b_cm * c_cm, 2)

    # Determine lesion anatomical quadrant location
    cx_ratio = (x + bw / 2.0) / w
    cy_ratio = (y + bh / 2.0) / h
    
    lateral = "Left" if cx_ratio < 0.5 else "Right"
    vertical = "Frontal / Anterior" if cy_ratio < 0.5 else "Posterior / Occipital"
    lesion_location = f"{lateral} {vertical} region"

    # Draw sharp colored boundary contour overlay
    overlay_bgr = img_bgr.copy()
    cv2.drawContours(overlay_bgr, [largest_contour], -1, (0, 0, 255), 2)  # Red outline
    
    # Semi-transparent fill inside contour
    fill_mask = np.zeros_like(overlay_bgr)
    cv2.drawContours(fill_mask, [largest_contour], -1, (0, 165, 255), -1)  # Orange fill
    overlay_bgr = cv2.addWeighted(overlay_bgr, 0.8, fill_mask, 0.2, 0)

    # Draw Bounding Box & Center Marker
    cv2.rectangle(overlay_bgr, (x, y), (x + bw, y + bh), (255, 255, 0), 1)

    # Save contour overlay image to static/annotated directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotated_dir = os.path.join(base_dir, 'static', 'annotated')
    os.makedirs(annotated_dir, exist_ok=True)
    
    overlay_filename = f"seg_overlay_{int(cx_ratio*10000)}_{x}_{y}.jpg"
    overlay_path = os.path.join(annotated_dir, overlay_filename)
    cv2.imwrite(overlay_path, overlay_bgr)

    is_calibrated = (pixel_spacing is not None)

    if is_calibrated:
        volume_display = f"{volume_cm3:.2f} cm³"
        diameter_display = f"{max_diam_cm:.2f} cm"
        area_display = f"{area_cm2:.2f} cm²"
    else:
        volume_display = "Not validated (requires DICOM metadata)"
        diameter_display = "Not validated"
        area_display = "Not validated"

    return {
        'has_lesion': True,
        'is_calibrated': is_calibrated,
        'area_pixels': int(area_pixels),
        'area_cm2': area_cm2,
        'volume_cm3': volume_cm3,
        'max_diameter_cm': max_diam_cm,
        'volume_display': volume_display,
        'diameter_display': diameter_display,
        'area_display': area_display,
        'bounding_box': [int(x), int(y), int(bw), int(bh)],
        'lesion_location': f"AI-estimated:\n{lateral.lower()} {vertical.lower()} region",
        'contour_overlay_path': f"/static/annotated/{overlay_filename}",
        'contour_overlay_file': overlay_path
    }
