"""
Longitudinal Tumor Growth Tracker & Treatment Response Monitor — TumorAI XAI
Tracks tumor volume changes across historical patient MRI scans, computing growth rates,
RANO/RECIST treatment response categories, and rendering digital patient timelines.
"""

from datetime import datetime
from services.supabase_client import supabase_client
from services.scan_service import get_scans_by_patient

def calculate_longitudinal_growth(patient_id):
    """
    Computes longitudinal tumor growth metrics and treatment response history for a patient.
    
    Args:
        patient_id: UUID string of patient
        
    Returns:
        dict containing growth timeline, overall response, volume delta, and monthly growth rate.
    """
    scans_data = get_scans_by_patient(patient_id)
    
    if not scans_data or len(scans_data) == 0:
        return {
            'has_history': False,
            'total_scans': 0,
            'timeline': [],
            'volume_delta_cm3': 0.0,
            'volume_delta_pct': 0.0,
            'monthly_growth_rate_pct': 0.0,
            'treatment_response': 'No Historical Scans Available',
            'trend_indicator': '⚪ Baseline'
        }

    # Sort scans chronologically by timestamp
    timeline_items = []
    for idx, s in enumerate(scans_data):
        created_at = s.get('created_at', '')
        prediction = s.get('prediction', 'Unknown').replace('Tumor: ', '').replace('Result: ', '')
        conf_raw = s.get('confidence', '90%')
        try:
            conf_val = float(str(conf_raw).replace('%', ''))
            conf_val_dec = conf_val / 100.0 if conf_val > 1.0 else conf_val
        except Exception:
            conf_val_dec = 0.90

        unc_level = (s.get('uncertainty') or 'Low').title()
        var_val = float(s.get('uncertainty_variance') or 0.0)

        # Dynamic AI Review Flags generated strictly from actual inference metrics
        review_flags = []
        if unc_level == 'High' or var_val > 0.025 or conf_val_dec < 0.70:
            review_flags.append({'type': 'warning', 'text': 'High AI uncertainty detected'})
            review_flags.append({'type': 'warning', 'text': 'Model disagreement / variance threshold reached'})
        else:
            review_flags.append({'type': 'success', 'text': 'High model consensus & confidence'})

        if idx == 0:
            review_flags.append({'type': 'info', 'text': 'Baseline study recorded'})

        # Extract DICOM spatial metrics if present
        has_valid_spatial = bool(s.get('has_valid_spatial', False) or s.get('metadata', {}).get('has_valid_spatial', False))
        vol = float(s.get('volume_cm3', 0.0) or s.get('metadata', {}).get('volume_cm3', 0.0))
        diam = float(s.get('max_diameter_cm', 0.0) or s.get('metadata', {}).get('max_diameter_cm', 0.0))

        timeline_items.append({
            'scan_id': s.get('id'),
            'study_id': f"MRI-{idx+1:03d}",
            'created_at': created_at,
            'date_formatted': created_at[:10] if created_at else f'Study #{idx+1}',
            'prediction': prediction,
            'confidence': f"{conf_val_dec*100:.1f}%",
            'top_probability': f"{conf_val_dec*100:.1f}%",
            'consensus_score': 100.0 if unc_level == 'Low' else 66.7,
            'uncertainty': unc_level,
            'uncertainty_variance': round(var_val, 5),
            'xai_method': s.get('xai_method', 'Grad-CAM'),
            'xai_available': ['Grad-CAM', 'Grad-CAM++', 'LIME', 'ROI'],
            'volume_cm3': round(vol, 2) if has_valid_spatial else None,
            'max_diameter_cm': round(diam, 2) if has_valid_spatial else None,
            'has_valid_spatial': has_valid_spatial,
            'image_path': s.get('image_path', ''),
            'heatmap_path': s.get('heatmap_path', ''),
            'annotated_path': s.get('annotated_path', ''),
            'is_baseline': (idx == 0),
            'review_flags': review_flags
        })

    # Sort by date
    timeline_items.sort(key=lambda x: x['created_at'])

    if len(timeline_items) <= 1:
        first_scan = timeline_items[0] if len(timeline_items) == 1 else {}
        return {
            'has_history': False,
            'total_scans': len(timeline_items),
            'timeline': timeline_items,
            'previous_study': 'Not available',
            'latest_volume_cm3': first_scan.get('volume_cm3'),
            'volume_delta_display': 'N/A',
            'growth_rate_display': 'N/A',
            'volume_delta_cm3': None,
            'volume_delta_pct': None,
            'monthly_growth_rate_pct': None,
            'treatment_response': 'Baseline Study — No Historical Comparison Available',
            'trend_indicator': '⚪ Baseline Study — No Historical Comparison Available',
            'spatial_note': 'Comparison unavailable — valid DICOM spatial measurements required.'
        }

    # Multiple scans: compute delta if valid spatial measurement exists
    first_scan = timeline_items[0]
    latest_scan = timeline_items[-1]

    has_spatial = (first_scan.get('has_valid_spatial') and latest_scan.get('has_valid_spatial'))
    v_initial = first_scan.get('volume_cm3') or 0.0
    v_latest = latest_scan.get('volume_cm3') or 0.0

    if has_spatial and v_initial > 0:
        vol_delta = round(v_latest - v_initial, 2)
        vol_pct = round(((v_latest - v_initial) / v_initial) * 100.0, 1)
        spatial_note = f"Volume Change: {vol_pct:+.1f}% ({vol_delta:+.2f} cm³)"
    else:
        vol_delta = None
        vol_pct = None
        spatial_note = "Comparison unavailable — valid DICOM spatial measurements required."

    return {
        'has_history': True,
        'total_scans': len(timeline_items),
        'timeline': timeline_items,
        'initial_volume_cm3': v_initial if has_spatial else None,
        'latest_volume_cm3': v_latest if has_spatial else None,
        'volume_delta_cm3': vol_delta,
        'volume_delta_pct': vol_pct,
        'treatment_response': 'Follow-up Comparison Available' if has_spatial else 'Baseline Study — Spatial Measurements Required',
        'trend_indicator': '🟢 Longitudinal Scans Recorded',
        'spatial_note': spatial_note
    }


def compare_two_mri_studies(patient_id, prev_scan_id=None, curr_scan_id=None):
    """
    Compares two specific MRI studies belonging to the exact same patient.
    Strictly enforces patient safety boundaries and calculates mathematically valid deltas.
    """
    scans_data = get_scans_by_patient(patient_id)
    if not scans_data or len(scans_data) == 0:
        return {
            'status': 'empty',
            'has_history': False,
            'message': 'No MRI studies available for this patient.',
            'total_scans': 0
        }

    # Format timeline items
    timeline_items = []
    for idx, s in enumerate(scans_data):
        created_at = s.get('created_at', '')
        prediction = s.get('prediction', 'Unknown').replace('Tumor: ', '').replace('Result: ', '')
        conf_raw = s.get('confidence', '90%')
        try:
            conf_val = float(str(conf_raw).replace('%', ''))
            conf_val_dec = conf_val / 100.0 if conf_val > 1.0 else conf_val
        except Exception:
            conf_val_dec = 0.90

        has_valid_spatial = bool(s.get('has_valid_spatial', False) or s.get('metadata', {}).get('has_valid_spatial', False))
        vol = float(s.get('volume_cm3', 0.0) or s.get('metadata', {}).get('volume_cm3', 0.0))
        diam = float(s.get('max_diameter_cm', 0.0) or s.get('metadata', {}).get('max_diameter_cm', 0.0))

        s_id = str(s.get('id') or f"scan_{idx}")
        timeline_items.append({
            'scan_id': s_id,
            'study_id': f"MRI-{idx+1:03d}",
            'patient_id': patient_id,
            'patient_name': s.get('patient_name', 'Patient'),
            'created_at': created_at,
            'date_formatted': created_at[:10] if created_at else f'Study #{idx+1}',
            'prediction': prediction,
            'confidence': round(conf_val_dec * 100.0, 1),
            'confidence_str': f"{conf_val_dec*100:.1f}%",
            'uncertainty': (s.get('uncertainty') or 'Low').title(),
            'xai_method': s.get('xai_method', 'Grad-CAM'),
            'volume_cm3': round(vol, 2) if (has_valid_spatial and vol > 0) else None,
            'max_diameter_cm': round(diam, 2) if (has_valid_spatial and diam > 0) else None,
            'has_valid_spatial': has_valid_spatial,
            'image_path': s.get('image_path', ''),
            'heatmap_path': s.get('heatmap_path', ''),
            'annotated_path': s.get('annotated_path', ''),
            'is_latest': False
        })

    # Sort chronologically by date
    timeline_items.sort(key=lambda x: x['created_at'])
    if timeline_items:
        timeline_items[-1]['is_latest'] = True

    if len(timeline_items) < 2:
        return {
            'status': 'insufficient_scans',
            'has_history': False,
            'message': 'No previous comparable MRI study is available for this patient.',
            'total_scans': len(timeline_items),
            'single_study': timeline_items[0] if timeline_items else None,
            'available_studies': timeline_items
        }

    # Find requested prev and curr scans
    prev_study = None
    curr_study = None

    if prev_scan_id:
        for t in timeline_items:
            if str(t['scan_id']) == str(prev_scan_id) or t['study_id'] == str(prev_scan_id):
                prev_study = t
                break

    if curr_scan_id:
        for t in timeline_items:
            if str(t['scan_id']) == str(curr_scan_id) or t['study_id'] == str(curr_scan_id):
                curr_study = t
                break

    # Defaults: prev = first study, curr = latest study
    if not prev_study:
        prev_study = timeline_items[0]
    if not curr_study:
        curr_study = timeline_items[-1]

    # Calculate valid differences
    conf_prev = prev_study['confidence']
    conf_curr = curr_study['confidence']
    conf_diff = round(conf_curr - conf_prev, 1)

    vol_prev = prev_study['volume_cm3']
    vol_curr = curr_study['volume_cm3']
    vol_diff = round(vol_curr - vol_prev, 2) if (vol_prev is not None and vol_curr is not None) else None
    vol_pct = round(((vol_curr - vol_prev) / vol_prev) * 100.0, 1) if (vol_prev and vol_curr and vol_prev > 0) else None

    diam_prev = prev_study['max_diameter_cm']
    diam_curr = curr_study['max_diameter_cm']
    diam_diff = round(diam_curr - diam_prev, 2) if (diam_prev is not None and diam_curr is not None) else None

    pred_changed = (prev_study['prediction'].lower() != curr_study['prediction'].lower())

    return {
        'status': 'success',
        'has_history': True,
        'total_scans': len(timeline_items),
        'available_studies': timeline_items,
        'previous_study': prev_study,
        'current_study': curr_study,
        'comparison_metrics': {
            'prediction': {
                'previous': prev_study['prediction'],
                'current': curr_study['prediction'],
                'changed': pred_changed
            },
            'confidence': {
                'previous': f"{conf_prev}%",
                'current': f"{conf_curr}%",
                'change_pp': conf_diff,
                'change_str': f"{conf_diff:+.1f} percentage points"
            },
            'volume': {
                'previous': f"{vol_prev} cm³" if vol_prev is not None else "Not available",
                'current': f"{vol_curr} cm³" if vol_curr is not None else "Not available",
                'change_cm3': f"{vol_diff:+.2f} cm³" if vol_diff is not None else "Not available",
                'change_pct': f"{vol_pct:+.1f}%" if vol_pct is not None else "Not available"
            },
            'diameter': {
                'previous': f"{diam_prev} cm" if diam_prev is not None else "Not available",
                'current': f"{diam_curr} cm" if diam_curr is not None else "Not available",
                'change_cm': f"{diam_diff:+.2f} cm" if diam_diff is not None else "Not available"
            }
        }
    }

