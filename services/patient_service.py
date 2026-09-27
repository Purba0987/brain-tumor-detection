from datetime import datetime
from models_db import db, Scan, User, Patient
from services.supabase_client import supabase_client
from services.auth_service import anonymize_patient_data, get_current_user_role

class PatientService:
    @staticmethod
    def generate_next_patient_id():
        count = Patient.query.count() + 1
        return f"P{count:03d}"

    @staticmethod
    def create_patient(patient_code, full_name=None, age=None, sex=None, phone=None, email=None, clinical_notes=None, created_by=None):
        patient_code = (patient_code or '').strip()
        if not patient_code:
            patient_code = PatientService.generate_next_patient_id()

        full_name = (full_name or 'Anonymous Patient').strip()
        
        # Check if patient already exists in local DB
        existing_p = Patient.query.filter_by(patient_id=patient_code).first()
        if existing_p:
            if full_name and full_name != 'Anonymous Patient':
                existing_p.full_name = full_name
            if age:
                existing_p.age = str(age)
            if sex:
                existing_p.gender = sex
            if phone or email:
                existing_p.contact = phone or email
            db.session.commit()
            return existing_p.to_dict()

        new_p = Patient(
            patient_id=patient_code,
            full_name=full_name,
            age=str(age) if age else '45',
            gender=sex or 'Male',
            contact=phone or email or '9876543210',
            created_by_doctor_id=created_by
        )
        db.session.add(new_p)
        db.session.commit()
        return new_p.to_dict()

    @staticmethod
    def get_or_create_patient(patient_code, full_name="Anonymous Patient", age="45", gender="Male", contact="patient@hospital.org", doctor_id=None):
        patient_code = (patient_code or '').strip()
        if not patient_code:
            patient_code = PatientService.generate_next_patient_id()
            
        existing_p = Patient.query.filter_by(patient_id=patient_code).first()
        if existing_p:
            return existing_p.to_dict()
        
        return PatientService.create_patient(
            patient_code=patient_code,
            full_name=full_name,
            age=age,
            sex=gender,
            phone=contact,
            created_by=doctor_id
        )

    @staticmethod
    def get_next_visit_info(patient_code):
        patient_code = (patient_code or '').strip()
        existing_scans = Scan.query.filter(
            (Scan.patient_id.ilike(f"%{patient_code}%")) | (Scan.patient_name.ilike(f"%{patient_code}%"))
        ).all()
        next_visit_num = len(existing_scans) + 1
        now = datetime.now()
        visit_time_str = now.strftime('%I:%M %p')
        return next_visit_num, visit_time_str

    @staticmethod
    def list_patients(search_query=None, is_archived=False, limit=20, offset=0):
        user_role = get_current_user_role()
        
        if supabase_client.is_configured:
            params = {"is_archived": f"eq.{str(is_archived).lower()}", "limit": limit, "offset": offset, "order": "created_at.desc"}
            if search_query:
                params["patient_code"] = f"ilike.%{search_query}%"
            res = supabase_client.db_select("patients", params)
            if res is not None:
                return [anonymize_patient_data(p, user_role) for p in res]

        # Fallback to local scans database records
        scans = Scan.query.order_by(Scan.created_at.desc()).limit(limit).offset(offset).all()
        patients_map = {}
        for s in scans:
            code = s.patient_id or "P-1001"
            if code not in patients_map:
                p_dict = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, code)),
                    "patient_code": code,
                    "full_name": s.patient_name or "Anonymous Patient",
                    "created_at": s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    "is_archived": False
                }
                patients_map[code] = anonymize_patient_data(p_dict, user_role)

        return list(patients_map.values())

    @staticmethod
    def archive_patient(patient_id):
        if supabase_client.is_configured:
            return supabase_client.db_update("patients", {"id": f"eq.{patient_id}"}, {"is_archived": True, "updated_at": datetime.utcnow().isoformat()})
        return True

    @staticmethod
    def archive_scan(scan_id):
        scan = Scan.query.get(scan_id)
        if scan:
            scan.is_archived = True
            db.session.commit()
            return True
        return False

    @staticmethod
    def restore_scan(scan_id):
        scan = Scan.query.get(scan_id)
        if scan:
            scan.is_archived = False
            db.session.commit()
            return True
        return False

    @staticmethod
    def search_patients(search_query):
        search_query = (search_query or '').strip()
        if not search_query:
            patients = Patient.query.order_by(Patient.created_at.desc()).all()
        else:
            q = f"%{search_query}%"
            patients = Patient.query.filter(
                (Patient.patient_id.ilike(q)) |
                (Patient.full_name.ilike(q)) |
                (Patient.contact.ilike(q))
            ).order_by(Patient.created_at.desc()).all()

        results = []
        for p in patients:
            doc_name = p.doctor.username if p.doctor else 'Dr. System'
            latest_scan = Scan.query.filter(Scan.patient_id == p.patient_id).order_by(Scan.created_at.desc()).first()
            last_checkup = latest_scan.created_at.strftime('%d %b %Y') if (latest_scan and latest_scan.created_at) else 'No visit yet'
            next_checkup = (latest_scan.next_checkup_date if latest_scan else None) or 'Not scheduled'
            scan_count = Scan.query.filter(Scan.patient_id == p.patient_id).count()

            results.append({
                'id': p.id,
                'patient_id': p.patient_id,
                'patient_code': p.patient_id,
                'full_name': p.full_name,
                'patient_name': p.full_name,
                'age': p.age or '45',
                'gender': p.gender or 'Male',
                'contact': p.contact or 'N/A',
                'doctor_name': doc_name,
                'last_checkup_date': last_checkup,
                'next_checkup_date': next_checkup,
                'total_scans': scan_count,
                'total_visits': scan_count,
                'registration_date': p.created_at.strftime('%d %b %Y') if p.created_at else 'N/A'
            })

        if not results and search_query:
            q = f"%{search_query}%"
            scans = Scan.query.filter(
                (Scan.patient_id.ilike(q)) | (Scan.patient_name.ilike(q))
            ).order_by(Scan.created_at.desc()).all()
            seen_codes = set()
            for s in scans:
                code = s.patient_id or 'P-1001'
                if code not in seen_codes:
                    seen_codes.add(code)
                    doc_name = s.owner.username if s.owner else 'Dr. System'
                    results.append({
                        'id': s.id,
                        'patient_id': code,
                        'patient_code': code,
                        'full_name': s.patient_name or 'Anonymous Patient',
                        'patient_name': s.patient_name or 'Anonymous Patient',
                        'age': getattr(s, 'patient_age', '') or '45',
                        'gender': getattr(s, 'patient_gender', '') or 'Male',
                        'contact': getattr(s, 'patient_contact', '') or 'N/A',
                        'doctor_name': doc_name,
                        'last_checkup_date': s.created_at.strftime('%d %b %Y') if s.created_at else 'No visit yet',
                        'next_checkup_date': s.next_checkup_date or 'Not scheduled',
                        'total_scans': 1,
                        'total_visits': 1,
                        'registration_date': s.created_at.strftime('%d %b %Y') if s.created_at else 'N/A'
                    })

        return results

    @staticmethod
    def get_patient_timeline(patient_code, include_archived=False):
        patient_code = (patient_code or '').strip()
        from sqlalchemy import func, or_
        
        patient_rec = None
        if patient_code:
            patient_rec = Patient.query.filter(Patient.patient_id.ilike(patient_code)).first()
            if not patient_rec:
                clean_code = patient_code.lstrip('#').strip()
                if clean_code.isdigit():
                    scan_by_id = Scan.query.get(int(clean_code))
                    if scan_by_id and scan_by_id.patient_id:
                        patient_rec = Patient.query.filter(Patient.patient_id.ilike(scan_by_id.patient_id)).first()

        if not patient_rec and not patient_code:
            patient_rec = Patient.query.order_by(Patient.created_at.desc()).first()

        target_code = patient_rec.patient_id if patient_rec else (patient_code or 'P001')
        patient_name = patient_rec.full_name if patient_rec else (patient_code or 'Unknown Patient')
        doctor_name = patient_rec.doctor.username if (patient_rec and patient_rec.doctor) else 'Dr. System'
        p_age = patient_rec.age if patient_rec else '45'
        p_gender = patient_rec.gender if patient_rec else 'Male'
        p_contact = patient_rec.contact if patient_rec else 'N/A'
        reg_date = patient_rec.created_at.strftime('%d %B %Y') if (patient_rec and patient_rec.created_at) else None

        if patient_rec:
            query = Scan.query.filter(Scan.patient_id == patient_rec.patient_id)
        else:
            query = Scan.query.filter((Scan.patient_id.ilike(target_code)) | (Scan.patient_name.ilike(target_code)))

        if not include_archived:
            query = query.filter((Scan.is_archived == False) | (Scan.is_archived == None))

        scans = query.order_by(Scan.created_at.desc()).all() # Newest first
        scan_dicts = [s.to_dict() for s in scans]
        total_scans = len(scan_dicts)

        if total_scans > 0:
            first_study_date = scans[-1].created_at.strftime('%d %B %Y') if scans[-1].created_at else 'Baseline'
            latest_study_date = scans[0].created_at.strftime('%d %B %Y') if scans[0].created_at else 'Latest'
            latest_next_checkup = scans[0].next_checkup_date or 'Not scheduled'
            if not patient_rec and scan_dicts[0].get('patient_name'):
                patient_name = scan_dicts[0]['patient_name']
            if not patient_rec and scan_dicts[0].get('doctor_name'):
                doctor_name = scan_dicts[0]['doctor_name']
        else:
            first_study_date = 'No study yet'
            latest_study_date = 'No study yet'
            latest_next_checkup = 'Not scheduled'

        trajectory_status = "Baseline Study / Single Record" if total_scans <= 1 else "Active Patient Journey"
        confidence_trend = "N/A"
        volume_delta_display = "N/A"
        rano_status = "Baseline — No Historical Comparison"

        if total_scans > 1:
            first = scan_dicts[-1]
            last = scan_dicts[0]
            first_conf = float(str(first.get('confidence', '0')).replace('%', '').strip() or 0)
            last_conf = float(str(last.get('confidence', '0')).replace('%', '').strip() or 0)
            confidence_trend = f"{first_conf:.1f}% → {last_conf:.1f}%"

        return {
            'status': 'success',
            'patient_code': target_code,
            'patient_id': target_code,
            'patient_name': patient_name,
            'patient_age': p_age or '45',
            'patient_gender': p_gender or 'Male',
            'patient_contact': p_contact or 'N/A',
            'doctor_name': doctor_name,
            'registration_date': reg_date or 'N/A',
            'total_scans': total_scans,
            'total_visits': total_scans,
            'first_study_date': first_study_date,
            'latest_study_date': latest_study_date,
            'next_checkup_date': latest_next_checkup,
            'trajectory_status': trajectory_status,
            'confidence_trend': confidence_trend,
            'volume_delta_display': volume_delta_display,
            'rano_status': rano_status,
            'timeline': scan_dicts
        }
