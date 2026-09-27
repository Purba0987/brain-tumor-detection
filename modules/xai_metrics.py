import numpy as np

def binarize_heatmap(heatmap, threshold_percentile=80):
    """
    Thresholds a normalized float heatmap (range 0-1) to create a binary prediction mask.
    Only activations above the specified percentile (default 80th percentile / top 20% brightest) are set to 1.
    """
    thresh = np.percentile(heatmap, threshold_percentile)
    binary_mask = (heatmap >= thresh).astype(np.uint8)
    return binary_mask

def calculate_iou(predicted_mask, ground_truth_mask):
    """
    Computes Intersection over Union (IoU) / Jaccard Index.
    """
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)

def calculate_dice_coefficient(predicted_mask, ground_truth_mask):
    """
    Computes the Dice Similarity Coefficient (DSC).
    """
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    total_area = pred.sum() + gt.sum()

    if total_area == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((2.0 * intersection) / total_area)

def calculate_localization_accuracy(heatmap, ground_truth_mask):
    """
    Checks if the maximum heatmap activation peak falls inside the ground-truth tumor region.
    Returns 1.0 if peak is inside mask, else 0.0.
    """
    gt = (ground_truth_mask > 0).astype(np.uint8)
    if gt.sum() == 0:
        return 1.0  # Empty mask baseline

    peak_coords = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    is_inside = gt[peak_coords[0], peak_coords[1]] > 0
    return 1.0 if is_inside else 0.0

def evaluate_xai_batch(heatmaps, ground_truth_masks, threshold_percentile=80):
    """
    Computes average IoU, Dice Coefficient, and Localization Accuracy across a set of XAI heatmaps and ground truth masks.
    """
    ious = []
    dices = []
    loc_accs = []

    for heatmap, gt_mask in zip(heatmaps, ground_truth_masks):
        pred_mask = binarize_heatmap(heatmap, threshold_percentile=threshold_percentile)
        iou = calculate_iou(pred_mask, gt_mask)
        dice = calculate_dice_coefficient(pred_mask, gt_mask)
        loc_acc = calculate_localization_accuracy(heatmap, gt_mask)

        ious.append(iou)
        dices.append(dice)
        loc_accs.append(loc_acc)

    return {
        'mean_iou': float(np.mean(ious)),
        'mean_dice': float(np.mean(dices)),
        'localization_accuracy': float(np.mean(loc_accs)),
        'threshold_percentile_used': threshold_percentile
    }
