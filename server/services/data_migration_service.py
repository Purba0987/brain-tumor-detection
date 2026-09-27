import uuid
from datetime import datetime
from models_db import db, Scan as LegacyScan
from services.supabase_client import supabase_client
from services.patient_service import PatientService
from services.study_service import StudyService

class DataMigrationService:
    @staticmethod
    def migrate_legacy_scans_to_supabase():
        """
        Migrates existing scans from SQLite/Neon local tables to the new Supabase PostgreSQL schema.
        Prevents duplicate records.
        """
        if not supabase_client.is_configured:
            print("[Migration] Supabase client is not configured; skipping legacy data migration.")
            return False

        legacy_records = LegacyScan.query.all()
        print(f"[Migration] Found {len(legacy_records)} legacy scan records to check for migration.")

        migrated_count = 0
        for leg in legacy_records:
            patient_code = leg.patient_id or f"P-LEGACY-{leg.id}"
            patient = PatientService.get_or_create_patient(patient_code=patient_code, full_name=leg.patient_name or "Anonymous Patient")
            patient_uuid = patient['id']

            study = StudyService.get_or_create_study(patient_id=patient_uuid, description=f"Legacy Study for {patient_code}")
            study_uuid = study['id']

            # Check if scan already exists in Supabase scans table
            existing_scan = supabase_client.db_select("scans", {"study_id": f"eq.{study_uuid}", "original_filename": f"eq.{leg.original_filename}"})
            if existing_scan and len(existing_scan) > 0:
                continue  # Skip already migrated scan

            # Construct scan record for Supabase
            scan_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"legacy_{leg.id}"))
            supabase_scan = {
                "id": scan_uuid,
                "study_id": study_uuid,
                "original_object_path": f"{patient_uuid}/{study_uuid}/{scan_uuid}/original.png",
                "heatmap_object_path": f"{patient_uuid}/{study_uuid}/{scan_uuid}/heatmap.jpg" if leg.heatmap_path else None,
                "roi_object_path": f"{patient_uuid}/{study_uuid}/{scan_uuid}/roi.jpg" if leg.annotated_path else None,
                "original_filename": leg.original_filename,
                "mime_type": "image/png",
                "file_size_bytes": 1024,
                "file_checksum": "legacy_checksum_migrated",
                "predicted_class": leg.prediction,
                "confidence": float(leg.confidence.replace('%', '').strip()) / 100.0 if '%' in leg.confidence else float(leg.confidence or 0.85),
                "uncertainty_score": leg.uncertainty_variance or 0.0,
                "uncertainty_level": leg.uncertainty.lower() if leg.uncertainty else "unknown",
                "xai_method": leg.xai_method or "Grad-CAM",
                "localization_method": "opencv_contour",
                "iou_score": leg.iou_score,
                "dice_score": leg.dice_score,
                "model_name": "vgg16_brain_tumor",
                "model_version": "1.0.0",
                "result_status": "conclusive",
                "created_at": leg.created_at.isoformat() if hasattr(leg.created_at, 'isoformat') else datetime.utcnow().isoformat()
            }

            inserted = supabase_client.db_insert("scans", supabase_scan)
            if inserted:
                migrated_count += 1

        print(f"[Migration] Migration complete. Migrated {migrated_count} legacy records to Supabase PostgreSQL.")
        return True
