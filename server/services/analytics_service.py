"""
Analytics Service — Tumor AI XAI
Provides longitudinal 3D tumor volumetric calculation, RANO growth response assessment,
and Tumor Volume Doubling Time (TVDT) estimation across patient scans.
"""

import math
from datetime import datetime
from services.supabase_client import supabase_client


def calculate_estimated_volume_cm3(scan_data):
    """
    Estimates 3D tumor volume in cm^3 from scan metadata or ROI contour metrics.
    Uses ellipsoid approximation V = 4/3 * pi * (a/2) * (b/2) * (c/2)
    where a and b are major/minor axes in mm, converted to cm^3.
    """
    if not scan_data:
        return 0.0

    # Retrieve ROI bounding dimensions if available in scan metadata or dicom header
    dicom_hdr = scan_data.get('dicom_header') or {}
    annotations = scan_data.get('annotations') or {}

    # Default slice thickness in mm (standard brain MRI: 5mm)
    slice_thickness_mm = float(dicom_hdr.get('SliceThickness', 5.0))
    pixel_spacing = dicom_hdr.get('PixelSpacing', [1.0, 1.0])
    if isinstance(pixel_spacing, str):
        try:
            pixel_spacing = [float(x) for x in pixel_spacing.replace('[','').replace(']','').split(',')]
        except Exception:
            pixel_spacing = [1.0, 1.0]

    # Check for contour area or explicit dimensions
    width_px = annotations.get('width_px', 40)
    height_px = annotations.get('height_px', 40)

    # Convert dimensions to mm
    width_mm = width_px * float(pixel_spacing[0] if len(pixel_spacing) > 0 else 1.0)
    height_mm = height_px * float(pixel_spacing[1] if len(pixel_spacing) > 1 else 1.0)
    depth_mm = slice_thickness_mm * 2.0  # Estimated depth

    # Ellipsoid volume in mm^3 -> cm^3 (1 cm^3 = 1000 mm^3)
    volume_mm3 = (4.0 / 3.0) * math.pi * (width_mm / 2.0) * (height_mm / 2.0) * (depth_mm / 2.0)
    volume_cm3 = round(volume_mm3 / 1000.0, 2)

    return max(volume_cm3, 0.1)

def calculate_tvdt_days(v1_cm3, v2_cm3, delta_days):
    """
    Calculates Tumor Volume Doubling Time (TVDT) in days.
    Formula: TVDT = (delta_t * ln(2)) / ln(V2 / V1)
    Positive TVDT = volume doubling time (growth)
    Negative TVDT = volume halving time (regression)
    """
    if v1_cm3 <= 0 or v2_cm3 <= 0 or delta_days <= 0 or v1_cm3 == v2_cm3:
        return None
    
    try:
        ratio = v2_cm3 / v1_cm3
        if ratio <= 0:
            return None
        tvdt = (delta_days * math.log(2)) / math.log(ratio)
        return round(tvdt, 1)
    except Exception:
        return None

def classify_rano_response(v1_cm3, v2_cm3):
    """
    Classifies growth according to RANO (Response Assessment in Neuro-Oncology) criteria:
    - Progressive Disease (PD): > 25% volume increase
    - Partial Response (PR): > 50% volume decrease
    - Stable Disease (SD): Neither PD nor PR criteria met
    """
    if v1_cm3 <= 0 or v2_cm3 <= 0:
        return "Baseline / Single Assessment"
    
    percent_change = ((v2_cm3 - v1_cm3) / v1_cm3) * 100.0
    
    if percent_change >= 25.0:
        return "Progressive Disease (PD)"
    elif percent_change <= -50.0:
        return "Partial Response (PR)"
    else:
        return "Stable Disease (SD)"

def get_patient_longitudinal_analytics(patient_code):
    """
    Fetches all historical scans for a patient and computes volumetric progression timeline.
    """
    timeline = []
    scans_data = []

    try:
        if supabase_client.is_configured:
            p_res = supabase_client.db_select("patients", {"patient_code": f"eq.{patient_code}"})
            if p_res and len(p_res) > 0:
                patient_id = p_res[0]["id"]
                st_res = supabase_client.db_select("studies", {"patient_id": f"eq.{patient_id}"})
                study_ids = [s["id"] for s in (st_res or [])]
                if study_ids:
                    scans_data = supabase_client.db_select("scans", {"study_id": f"in.({','.join(study_ids)})"}) or []
        
        # Fallback to local SQLAlchemy DB scans if empty or Supabase unconfigured
        if not scans_data:
            local_scans = Scan.query.filter_by(patient_id=patient_code).order_by(Scan.timestamp.asc()).all()
            scans_data = [
                {
                    "id": s.id,
                    "created_at": s.timestamp.isoformat() if s.timestamp else "",
                    "predicted_class": s.prediction,
                    "confidence_score": float(s.confidence.replace('%',''))/100.0 if s.confidence and '%' in s.confidence else 0.85,
                    "scan_type": "Brain MRI"
                }
                for s in local_scans
            ]


        timeline = []
        prev_scan = None

        for scan in scans_data:
            vol_cm3 = calculate_estimated_volume_cm3(scan)
            created_at_str = scan.get("created_at") or ""
            
            scan_point = {
                "scan_id": scan.get("id"),
                "scan_type": scan.get("scan_type", "Brain MRI"),
                "created_at": created_at_str,
                "predicted_class": scan.get("predicted_class"),
                "confidence_score": scan.get("confidence_score"),
                "volume_cm3": vol_cm3,
                "percent_change": 0.0,
                "tvdt_days": None,
                "rano_status": "Baseline Scan"
            }

            if prev_scan:
                v1 = prev_scan["volume_cm3"]
                v2 = vol_cm3
                
                # Compute days between scans
                try:
                    d1 = datetime.fromisoformat(prev_scan["created_at"].replace("Z", "+00:00"))
                    d2 = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    delta_days = max((d2 - d1).days, 1)
                except Exception:
                    delta_days = 30

                pct = round(((v2 - v1) / v1) * 100.0, 1) if v1 > 0 else 0.0
                tvdt = calculate_tvdt_days(v1, v2, delta_days)
                rano = classify_rano_response(v1, v2)

                scan_point["percent_change"] = pct
                scan_point["tvdt_days"] = tvdt
                scan_point["rano_status"] = rano

            timeline.append(scan_point)
            prev_scan = scan_point

        return {
            "patient": patient,
            "total_scans": len(timeline),
            "latest_volume_cm3": timeline[-1]["volume_cm3"] if timeline else 0.0,
            "overall_rano_status": timeline[-1]["rano_status"] if timeline else "N/A",
            "timeline": timeline
        }
    except Exception as e:
        return {"error": str(e), "timeline": []}
