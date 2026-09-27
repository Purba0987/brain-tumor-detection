import os
import uuid
from datetime import datetime
from services.supabase_client import supabase_client
from services.storage_service import upload_pdf_report, validate_pdf_file, get_private_signed_url
from services.auth_service import is_radiologist, get_current_user_role

REPORT_STATUSES = ['draft', 'generated', 'needs_review', 'approved', 'amended', 'archived']

class ReportService:
    @staticmethod
    def get_latest_report_version(study_id):
        if supabase_client.is_configured:
            reports = supabase_client.db_select("reports", {"study_id": f"eq.{study_id}", "order": "version.desc", "limit": 1})
            if reports and len(reports) > 0:
                return reports[0]['version']
        return 0

    @staticmethod
    def generate_and_store_report(
        patient_uuid,
        study_uuid,
        pdf_bytes,
        patient_name="Anonymous Patient",
        patient_id_code="P-1001",
        generated_by=None
    ):
        """
        Validates, uploads, and records versioned PDF report in private storage and PostgreSQL.
        """
        filename = f"report_{patient_id_code}.pdf"
        valid, err = validate_pdf_file(filename, pdf_bytes)
        if not valid:
            raise ValueError(err)

        current_v = ReportService.get_latest_report_version(study_uuid)
        new_v = current_v + 1

        report_uuid = str(uuid.uuid4())
        report_number = f"REP-{patient_id_code}-V{new_v}-{int(datetime.utcnow().timestamp()) % 100000}"

        # Upload PDF to Private Storage (patient-reports/{patient_uuid}/{study_uuid}/report-v{version}.pdf)
        stored_asset = upload_pdf_report(patient_uuid, study_uuid, version=new_v, pdf_bytes=pdf_bytes)

        report_record = {
            "id": report_uuid,
            "study_id": study_uuid,
            "report_number": report_number,
            "version": new_v,
            "object_path": stored_asset['object_path'],
            "original_filename": filename,
            "mime_type": "application/pdf",
            "file_size_bytes": stored_asset['size_bytes'],
            "file_checksum": stored_asset['checksum'],
            "status": "generated",
            "summary": f"XAI Clinical Medical Diagnostic Report V{new_v}",
            "generated_by": generated_by,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        if supabase_client.is_configured:
            supabase_client.db_insert("reports", report_record)

        signed_download_url = get_private_signed_url("report", stored_asset['object_path'], ttl_seconds=300)

        return {
            "report_id": report_uuid,
            "report_number": report_number,
            "version": new_v,
            "status": "generated",
            "object_path": stored_asset['object_path'],
            "file_checksum": stored_asset['checksum'],
            "signed_download_url": signed_download_url
        }

    @staticmethod
    def update_report_status(report_id, new_status, user_id=None):
        if new_status not in REPORT_STATUSES:
            raise ValueError(f"Invalid report status '{new_status}'. Allowed: {REPORT_STATUSES}")

        # Only Radiologists/Admins can approve reports
        if new_status == 'approved' and not is_radiologist():
            raise PermissionError("Only authorized Radiologists or Administrators can approve medical reports.")

        # Check immutability of existing approved report
        if supabase_client.is_configured:
            existing = supabase_client.db_select("reports", {"id": f"eq.{report_id}"})
            if existing and len(existing) > 0:
                if existing[0]['status'] == 'approved' and new_status != 'amended':
                    raise ValueError("Approved report versions are immutable and cannot be edited. Create an amended version instead.")

        update_payload = {"status": new_status, "updated_at": datetime.utcnow().isoformat()}
        if new_status == 'approved':
            update_payload["approved_by"] = user_id
            update_payload["approved_at"] = datetime.utcnow().isoformat()
        elif new_status == 'needs_review':
            update_payload["reviewed_by"] = user_id
            update_payload["reviewed_at"] = datetime.utcnow().isoformat()

        if supabase_client.is_configured:
            return supabase_client.db_update("reports", {"id": f"eq.{report_id}"}, update_payload)
        return True

    @staticmethod
    def get_report_signed_url(report_id):
        if supabase_client.is_configured:
            reports = supabase_client.db_select("reports", {"id": f"eq.{report_id}"})
            if reports and len(reports) > 0:
                obj_path = reports[0]['object_path']
                return get_private_signed_url("report", obj_path, ttl_seconds=300)
        return None
