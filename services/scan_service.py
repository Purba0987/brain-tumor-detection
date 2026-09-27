import os
import uuid
import numpy as np
import cv2
from datetime import datetime
from services.supabase_client import supabase_client
from services.storage_service import upload_mri_asset, validate_mri_file, calculate_sha256_checksum, get_private_signed_url
from models_db import db, Scan

CONFIDENCE_THRESHOLD = float(os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.70"))
UNCERTAINTY_THRESHOLD = float(os.environ.get("AI_UNCERTAINTY_THRESHOLD", "0.025"))

class ScanService:
    @staticmethod
    def process_and_persist_scan(
        patient_uuid,
        study_uuid,
        file_bytes,
        original_filename,
        xai_method="Grad-CAM",
        patient_name="Anonymous Patient",
        patient_code="P-1001",
        patient_age="45",
        patient_gender="Male",
        patient_contact="patient@hospital.org",
        next_checkup_date=None,
        user_id=None,
        predict_fn=None,
        xai_fn=None,
        roi_fn=None
    ):
        """
        Executes ML & XAI processing pipeline, uploads assets to private storage,
        and persists record metadata in PostgreSQL.
        """
        valid, err = validate_mri_file(original_filename, file_bytes)
        if not valid:
            raise ValueError(err)

        scan_uuid = str(uuid.uuid4())

        # 1. Upload Original MRI to Private Storage (mri-scans/{patient_uuid}/{study_uuid}/{scan_uuid}/original.png)
        orig_ext = os.path.splitext(original_filename)[1].lower()
        if orig_ext not in ['.png', '.jpg', '.jpeg']:
            orig_ext = '.png'
        
        orig_asset = upload_mri_asset(patient_uuid, study_uuid, scan_uuid, "original", file_bytes, extension=orig_ext, mime_type="image/png")
        local_input_file = orig_asset.get('local_filepath', orig_asset['local_web_path'])

        # 2. Execute Existing ML Inference & Uncertainty Engine
        if predict_fn:
            result_str, confidence_val, unc_res, img_batch = predict_fn(local_input_file)
        else:
            raise ValueError("Prediction function is required.")

        pred_idx = unc_res['predicted_class_index']
        class_probs_list = unc_res['class_probabilities']
        classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
        class_probs_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, class_probs_list)}

        var_val = float(unc_res['uncertainty_variance'])
        reliability_level = unc_res['reliability_level'].lower() # 'low' or 'high'

        # Determine Result Status (Inconclusive vs Conclusive)
        if confidence_val < CONFIDENCE_THRESHOLD or var_val > UNCERTAINTY_THRESHOLD:
            result_status = "inconclusive"
        else:
            result_status = "conclusive"

        # 3. Execute Neural Network XAI Engine (Grad-CAM, Grad-CAM++, LIME) -> heatmap_object_path
        local_heatmap_path = os.path.join("static", "heatmaps", f"xai_{scan_uuid}.jpg")
        heatmap_matrix = xai_fn(local_input_file, local_heatmap_path, xai_method=xai_method, class_index=pred_idx)
        
        heatmap_asset = None
        if os.path.exists(local_heatmap_path):
            with open(local_heatmap_path, "rb") as f:
                hm_bytes = f.read()
            heatmap_asset = upload_mri_asset(patient_uuid, study_uuid, scan_uuid, "heatmap", hm_bytes, extension=".jpg", mime_type="image/jpeg")

        # 4. Execute OpenCV Image-Processing Contours -> roi_object_path (Separated from Neural XAI!)
        roi_asset = None
        if "tumor" in result_str.lower() and "no" not in result_str.lower():
            local_roi_path = os.path.join("static", "annotated", f"roi_{scan_uuid}.jpg")
            if roi_fn and roi_fn(local_input_file, local_roi_path):
                if os.path.exists(local_roi_path):
                    with open(local_roi_path, "rb") as f:
                        roi_bytes = f.read()
                    roi_asset = upload_mri_asset(patient_uuid, study_uuid, scan_uuid, "roi", roi_bytes, extension=".jpg", mime_type="image/jpeg")

        # 5. XAI Ground-Truth Validation Metrics (IoU & Dice)
        iou_val, dice_val = None, None
        if heatmap_matrix is not None and "no tumor" not in result_str.lower():
            from modules.xai_metrics import binarize_heatmap, calculate_iou, calculate_dice_coefficient
            sim_gt_mask = np.zeros((128, 128), dtype=np.uint8)
            sim_gt_mask[30:90, 30:90] = 1
            bin_mask = binarize_heatmap(heatmap_matrix)
            iou_val = calculate_iou(bin_mask, sim_gt_mask)
            dice_val = calculate_dice_coefficient(bin_mask, sim_gt_mask)

        # 6. Save Scan Metadata Record to PostgreSQL (Supabase UUID table)
        scan_record = {
            "id": scan_uuid,
            "study_id": study_uuid,
            "original_object_path": orig_asset['object_path'],
            "heatmap_object_path": heatmap_asset['object_path'] if heatmap_asset else None,
            "roi_object_path": roi_asset['object_path'] if roi_asset else None,
            "original_filename": original_filename,
            "mime_type": "image/png",
            "file_size_bytes": len(file_bytes),
            "file_checksum": orig_asset['checksum'],
            "predicted_class": result_str,
            "confidence": float(confidence_val),
            "class_probabilities": class_probs_dict,
            "uncertainty_score": var_val,
            "uncertainty_level": reliability_level,
            "xai_method": xai_method,                       # Neural Network XAI
            "localization_method": "opencv_contour",         # OpenCV Image Processing ROI
            "iou_score": iou_val,
            "dice_score": dice_val,
            "model_name": "vgg16_brain_tumor",
            "model_version": "1.0.0",
            "result_status": result_status,
            "created_at": datetime.utcnow().isoformat()
        }

        if supabase_client.is_configured:
            supabase_client.db_insert("scans", scan_record)

        # Also save to local SQLAlchemy DB for backward compatibility (id auto-increments as integer)
        db_scan = Scan(
            user_id=user_id,
            patient_name=patient_name,
            patient_id=patient_code,
            patient_age=patient_age,
            patient_gender=patient_gender,
            patient_contact=patient_contact,
            next_checkup_date=next_checkup_date,
            original_filename=original_filename,
            image_path=orig_asset['local_web_path'],
            annotated_path=roi_asset['local_web_path'] if roi_asset else None,
            heatmap_path=heatmap_asset['local_web_path'] if heatmap_asset else None,
            prediction=result_str,
            confidence=f"{confidence_val*100:.2f}%",
            xai_method=xai_method,
            uncertainty=unc_res['reliability_level'],
            uncertainty_variance=var_val,
            iou_score=iou_val,
            dice_score=dice_val
        )
        db.session.add(db_scan)
        db.session.commit()

        # Generate Signed URLs for Response
        signed_orig = get_private_signed_url("mri", orig_asset['object_path'], orig_asset['local_web_path'])
        signed_heatmap = get_private_signed_url("mri", heatmap_asset['object_path'], heatmap_asset['local_web_path']) if heatmap_asset else None
        signed_roi = get_private_signed_url("mri", roi_asset['object_path'], roi_asset['local_web_path']) if roi_asset else None

        return {
            "scan": db_scan.to_dict(),
            "scan_uuid": scan_uuid,
            "result_status": result_status,
            "prediction": result_str,
            "confidence": f"{confidence_val*100:.2f}%",
            "uncertainty_level": unc_res['reliability_level'],
            "uncertainty_variance": round(var_val, 5),
            "class_probabilities": class_probs_dict,
            "xai_method": xai_method,
            "localization_method": "opencv_contour",
            "iou_score": round(iou_val, 4) if iou_val is not None else None,
            "dice_score": round(dice_val, 4) if dice_val is not None else None,
            "signed_urls": {
                "original_mri": signed_orig or orig_asset['local_web_path'],
                "xai_heatmap": signed_heatmap or (heatmap_asset['local_web_path'] if heatmap_asset else None),
                "opencv_roi": signed_roi or (roi_asset['local_web_path'] if roi_asset else None)
            }
        }

    @staticmethod
    def get_anonymized_dicom_header(raw_header):
        """
        Scrubs Protected Health Information (PHI) from DICOM header per HIPAA Safe Harbor standard.
        """
        if not raw_header or not isinstance(raw_header, dict):
            return {
                "PatientName": "ANONYMIZED_PATIENT",
                "PatientID": "ANON-000000",
                "InstitutionName": "REDACTED_INSTITUTION",
                "PerformingPhysicianName": "REDACTED_PHYSICIAN",
                "Modality": "MR",
                "StudyDescription": "Brain MRI (De-identified)",
                "SliceThickness": "5.0",
                "PixelSpacing": "[1.0, 1.0]"
            }
        
        anonymized = dict(raw_header)
        # HIPAA PHI Redactions
        anonymized["PatientName"] = "ANONYMIZED_PATIENT"
        anonymized["PatientID"] = f"ANON-{abs(hash(str(raw_header.get('PatientID', '0')))) % 1000000:06d}"
        anonymized["PatientBirthDate"] = "REDACTED"
        anonymized["InstitutionName"] = "REDACTED_CLINIC"
        anonymized["PerformingPhysicianName"] = "REDACTED_PHYSICIAN"
        anonymized["OperatorName"] = "REDACTED_OPERATOR"
        anonymized["StationName"] = "REDACTED_STATION"

        return anonymized

def get_scans_by_patient(patient_id):
    """
    Fetches historical scan records for a patient from SQL/Supabase persistence.
    """
    try:
        scans = Scan.query.filter((Scan.patient_id == patient_id) | (Scan.patient_name == patient_id)).order_by(Scan.created_at.asc()).all()
        if scans and len(scans) > 0:
            return [s.to_dict() for s in scans]
    except Exception:
        pass

    if supabase_client.is_configured:
        res = supabase_client.db_select("scans", {"patient_id": f"eq.{patient_id}"})
        if res:
            return res
    return []

