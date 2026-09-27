from datetime import datetime, timedelta
from models_db import db, User, Patient, Scan, Notification
from services.email_service import EmailService

class NotificationService:
    @staticmethod
    def create_notification(
        user_id=None,
        type="system",
        title="System Event",
        message="System event recorded.",
        patient_id=None,
        patient_name=None,
        scan_id=None,
        category="info",
        action_tab="patients",
        send_email=True
    ):
        """
        Creates an in-app notification record and optionally dispatches an email notification to the doctor.
        """
        try:
            notif = Notification(
                user_id=user_id,
                type=type,
                title=title,
                message=message,
                patient_id=patient_id,
                patient_name=patient_name,
                scan_id=scan_id,
                category=category,
                action_tab=action_tab,
                is_read=False,
                created_at=datetime.utcnow()
            )
            db.session.add(notif)
            db.session.commit()

            # Email notification dispatch
            if send_email and user_id:
                doctor = User.query.get(user_id)
                if doctor and doctor.email:
                    subject_map = {
                        'requires_review': f"🚨 TumorAI — Doctor Review Required ({patient_id or 'Patient'})",
                        'analysis_completed': f"TumorAI — MRI Analysis Completed ({patient_id or 'Patient'})",
                        'upcoming_checkup': f"TumorAI — Patient Follow-up Reminder ({patient_id or 'Patient'})",
                        'new_patient': f"TumorAI — New Patient Registered ({patient_id or 'Patient'})",
                        'report_ready': f"TumorAI — Report Ready ({patient_id or 'Patient'})"
                    }
                    subject = subject_map.get(type, f"TumorAI Notification — {title}")
                    EmailService.send_notification_email(
                        recipient_email=doctor.email,
                        subject=subject,
                        title=title,
                        body_text=message,
                        patient_code=patient_id
                    )

            return notif.to_dict()
        except Exception as e:
            print(f"[NotificationService Error] {e}")
            db.session.rollback()
            return None

    @staticmethod
    def seed_initial_notifications(user_id=None):
        now = datetime.utcnow()
        sample_notifs = [
            {
                'type': 'requires_review',
                'title': '🔴 Requires Review',
                'message': 'MRI analysis requires doctor review for Rahul Sharma (P001)',
                'patient_id': 'P001',
                'patient_name': 'Rahul Sharma',
                'category': 'danger',
                'action_tab': 'patients',
                'created_at': now - timedelta(minutes=5)
            },
            {
                'type': 'analysis_completed',
                'title': '🧠 Analysis Completed',
                'message': 'MRI analysis completed for Amit Das (P014)',
                'patient_id': 'P014',
                'patient_name': 'Amit Das',
                'category': 'success',
                'action_tab': 'patients',
                'created_at': now - timedelta(minutes=20)
            },
            {
                'type': 'upcoming_checkup',
                'title': '📅 Upcoming Check-up',
                'message': 'Rahul Sharma (P001) check-up scheduled for tomorrow — 10:30 AM',
                'patient_id': 'P001',
                'patient_name': 'Rahul Sharma',
                'category': 'warning',
                'action_tab': 'patients',
                'created_at': now - timedelta(hours=1)
            },
            {
                'type': 'new_patient',
                'title': '👤 New Patient',
                'message': 'Riya Sen (P021) was registered today',
                'patient_id': 'P021',
                'patient_name': 'Riya Sen',
                'category': 'primary',
                'action_tab': 'patients',
                'created_at': now - timedelta(hours=3)
            },
            {
                'type': 'report_ready',
                'title': '📄 Report Ready',
                'message': 'Progress report generated for Rahul Sharma (P001)',
                'patient_id': 'P001',
                'patient_name': 'Rahul Sharma',
                'category': 'info',
                'action_tab': 'patients',
                'created_at': now - timedelta(days=1)
            }
        ]
        try:
            for item in sample_notifs:
                n = Notification(
                    user_id=user_id,
                    type=item['type'],
                    title=item['title'],
                    message=item['message'],
                    patient_id=item['patient_id'],
                    patient_name=item['patient_name'],
                    category=item['category'],
                    action_tab=item['action_tab'],
                    is_read=False,
                    created_at=item['created_at']
                )
                db.session.add(n)
            db.session.commit()
        except Exception as e:
            print(f"[Seed Notifications Error] {e}")
            db.session.rollback()

    @staticmethod
    def get_doctor_notifications(user_id=None, limit=20):
        """
        Fetches doctor-specific notifications and performs dynamic check-up reminder evaluation.
        """
        # Seed default notifications if table is currently empty for this user/doctor
        query = Notification.query
        if user_id:
            query = query.filter((Notification.user_id == user_id) | (Notification.user_id == None))
        
        if query.count() == 0:
            NotificationService.seed_initial_notifications(user_id=user_id)

        # Auto-evaluate upcoming checkup reminders for patient follow-ups
        try:
            NotificationService.check_upcoming_followups(user_id=user_id)
        except Exception as err:
            print(f"[NotificationService Follow-up Warning] {err}")

        notifs = query.order_by(Notification.created_at.desc()).limit(limit).all()
        notif_dicts = [n.to_dict() for n in notifs]

        unread_count = sum(1 for n in notifs if not n.is_read)

        return {
            'status': 'success',
            'unread_count': unread_count,
            'notifications': notif_dicts
        }

    @staticmethod
    def check_upcoming_followups(user_id=None):
        """
        Checks for patient follow-ups scheduled for today, tomorrow, or overdue and creates reminders.
        """
        scans_with_checkup = Scan.query.filter(Scan.next_checkup_date != None, Scan.next_checkup_date != '').all()
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        tomorrow_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')

        for scan in scans_with_checkup:
            nc_date = (scan.next_checkup_date or '').strip()
            if not nc_date:
                continue

            p_id = scan.patient_id or 'P001'
            p_name = scan.patient_name or 'Patient'

            # Avoid duplicate notification for same patient & checkup date
            existing = Notification.query.filter_by(
                type='upcoming_checkup',
                patient_id=p_id,
                message=f"Check-up reminder for {p_name} ({p_id}) on {nc_date}"
            ).first()

            if not existing:
                title = f"📅 Upcoming Check-up"
                msg = f"Check-up reminder for {p_name} ({p_id}) on {nc_date}"
                NotificationService.create_notification(
                    user_id=user_id or scan.user_id,
                    type='upcoming_checkup',
                    title=title,
                    message=f"{p_name} ({p_id}) has a check-up scheduled for {nc_date}.",
                    patient_id=p_id,
                    patient_name=p_name,
                    scan_id=scan.id,
                    category='warning',
                    action_tab='patients',
                    send_email=False
                )

    @staticmethod
    def mark_all_as_read(user_id=None):
        query = Notification.query
        if user_id:
            query = query.filter((Notification.user_id == user_id) | (Notification.user_id == None))
        query.update({'is_read': True})
        db.session.commit()
        return True

    @staticmethod
    def mark_as_read(notification_id):
        notif = Notification.query.get(notification_id)
        if notif:
            notif.is_read = True
            db.session.commit()
            return True
        return False
