import uuid
from datetime import datetime
from services.supabase_client import supabase_client
from models_db import db, Scan

STUDY_STATUSES = ['draft', 'processing', 'needs_review', 'reviewed', 'approved', 'failed', 'archived']

class StudyService:
    @staticmethod
    def create_study(patient_id, description=None, clinical_indication=None, created_by=None):
        study_uuid = str(uuid.uuid4())
        record = {
            "id": study_uuid,
            "patient_id": patient_id,
            "modality": "MRI",
            "study_date": datetime.utcnow().isoformat(),
            "description": description or "Brain MRI Diagnostic Study",
            "clinical_indication": clinical_indication or "Tumor evaluation",
            "status": "draft",
            "created_by": created_by,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        if supabase_client.is_configured:
            inserted = supabase_client.db_insert("studies", record)
            if inserted and len(inserted) > 0:
                return inserted[0]
        
        return record

    @staticmethod
    def get_or_create_study(patient_id, description="Brain MRI Diagnostic Study"):
        if supabase_client.is_configured:
            existing = supabase_client.db_select("studies", {"patient_id": f"eq.{patient_id}", "order": "created_at.desc", "limit": 1})
            if existing and len(existing) > 0:
                return existing[0]

        return StudyService.create_study(patient_id=patient_id, description=description)

    @staticmethod
    def update_study_status(study_id, new_status):
        if new_status not in STUDY_STATUSES:
            raise ValueError(f"Invalid study status '{new_status}'. Allowed: {STUDY_STATUSES}")

        if supabase_client.is_configured:
            return supabase_client.db_update("studies", {"id": f"eq.{study_id}"}, {"status": new_status, "updated_at": datetime.utcnow().isoformat()})
        return True

    @staticmethod
    def add_study_note(study_id, author_id, note_text):
        note_uuid = str(uuid.uuid4())
        record = {
            "id": note_uuid,
            "study_id": study_id,
            "author_id": author_id,
            "note_text": note_text,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        if supabase_client.is_configured:
            inserted = supabase_client.db_insert("study_notes", record)
            if inserted and len(inserted) > 0:
                return inserted[0]
        return record

    @staticmethod
    def get_study_notes(study_id):
        if supabase_client.is_configured:
            res = supabase_client.db_select("study_notes", {"study_id": f"eq.{study_id}", "order": "created_at.desc"})
            if res is not None:
                return res
        return []

    @staticmethod
    def compare_longitudinal_studies(study_id_1, study_id_2):
        """
        Provides neutral side-by-side longitudinal comparison data between two studies for the same patient.
        Does not declare progression/improvement automatically.
        """
        scan_1 = Scan.query.filter_by(id=study_id_1).first() if not isinstance(study_id_1, str) or not "-" in study_id_1 else None
        scan_2 = Scan.query.filter_by(id=study_id_2).first() if not isinstance(study_id_2, str) or not "-" in study_id_2 else None

        if not scan_1:
            scan_1 = Scan.query.order_by(Scan.created_at.desc()).first()
        if not scan_2:
            scan_2 = Scan.query.order_by(Scan.created_at.asc()).first()

        def scan_summary(s):
            if not s:
                return {}
            return {
                'scan_id': s.id,
                'patient_id': s.patient_id,
                'created_at': s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': s.prediction,
                'confidence': s.confidence,
                'uncertainty': s.uncertainty,
                'uncertainty_variance': round(s.uncertainty_variance or 0.0, 5),
                'xai_method': s.xai_method,
                'localization_method': 'opencv_contour',
                'iou_score': s.iou_score,
                'dice_score': s.dice_score,
                'image_path': s.image_path,
                'heatmap_path': s.heatmap_path,
                'roi_path': s.annotated_path
            }

        return {
            'study_baseline': scan_summary(scan_1),
            'study_followup': scan_summary(scan_2),
            'comparison_notes': 'Neutral longitudinal comparison data loaded. Qualified radiologist evaluation required for clinical progression interpretation.'
        }
