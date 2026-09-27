from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(30), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='Doctor') # 'Doctor', 'Radiologist', 'Patient'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    scans = db.relationship('Scan', backref='owner', lazy=True, cascade="all, delete-orphan")
    chat_logs = db.relationship('ChatLog', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), unique=True, nullable=False) # e.g. 'P001'
    full_name = db.Column(db.String(100), nullable=False) # e.g. 'Rahul Sharma'
    age = db.Column(db.String(20), nullable=True) # e.g. '45'
    gender = db.Column(db.String(20), nullable=True) # e.g. 'Male'
    contact = db.Column(db.String(100), nullable=True) # e.g. '9876543210'
    created_by_doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship('User', backref='registered_patients', lazy=True)

    def to_dict(self):
        doc_name = self.doctor.username if self.doctor else 'Dr. System'
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'patient_code': self.patient_id,
            'full_name': self.full_name,
            'patient_name': self.full_name,
            'age': self.age or '45',
            'gender': self.gender or 'Male',
            'contact': self.contact or 'N/A',
            'doctor_name': doc_name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'created_date': self.created_at.strftime('%d %B %Y') if hasattr(self.created_at, 'strftime') else str(self.created_at)
        }


class Scan(db.Model):
    __tablename__ = 'scans'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    patient_name = db.Column(db.String(100), default='Anonymous')
    patient_id = db.Column(db.String(50), default='P-1001')
    patient_age = db.Column(db.String(20), nullable=True, default='45')
    patient_gender = db.Column(db.String(20), nullable=True, default='Male')
    patient_contact = db.Column(db.String(100), nullable=True, default='patient@hospital.org')
    visit_number = db.Column(db.Integer, nullable=True, default=1)
    visit_time = db.Column(db.String(30), nullable=True)
    next_checkup_date = db.Column(db.String(50), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    annotated_path = db.Column(db.String(255), nullable=True)
    heatmap_path = db.Column(db.String(255), nullable=True)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    xai_method = db.Column(db.String(30), default='Grad-CAM')
    uncertainty = db.Column(db.String(20), default='Low')
    uncertainty_variance = db.Column(db.Float, default=0.0)
    iou_score = db.Column(db.Float, nullable=True)
    dice_score = db.Column(db.Float, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    clinical_status = db.Column(db.String(50), default='Pending Review')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_archived = db.Column(db.Boolean, default=False)
    
    chat_logs = db.relationship('ChatLog', backref='scan', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        doc_name = self.owner.username if self.owner else 'Dr. System'
        doc_email = self.owner.email if self.owner else ''
        created_dt = self.created_at or datetime.utcnow()
        v_time = getattr(self, 'visit_time', None) or created_dt.strftime('%I:%M %p')
        v_num = getattr(self, 'visit_number', 1) or 1
        return {
            'id': self.id,
            'scan_id': self.id,
            'user_id': self.user_id,
            'doctor_name': doc_name,
            'doctor_email': doc_email,
            'patient_name': self.patient_name,
            'patient_id': self.patient_id,
            'patient_age': getattr(self, 'patient_age', '') or '45',
            'patient_gender': getattr(self, 'patient_gender', '') or 'Male',
            'patient_contact': getattr(self, 'patient_contact', '') or 'patient@hospital.org',
            'visit_number': v_num,
            'visit_time': v_time,
            'next_checkup_date': getattr(self, 'next_checkup_date', None) or '',
            'visit_date': created_dt.strftime('%d %B %Y'),
            'visit_date_short': created_dt.strftime('%Y-%m-%d'),
            'original_filename': self.original_filename,
            'image_path': self.image_path,
            'annotated_path': self.annotated_path,
            'heatmap_path': self.heatmap_path,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'xai_method': self.xai_method or 'Grad-CAM',
            'uncertainty': self.uncertainty or 'Low',
            'uncertainty_variance': round(self.uncertainty_variance or 0.0, 5),
            'iou_score': round(self.iou_score, 4) if self.iou_score is not None else None,
            'dice_score': round(self.dice_score, 4) if self.dice_score is not None else None,
            'notes': self.notes or '',
            'clinical_status': self.clinical_status or 'Pending Review',
            'is_archived': bool(getattr(self, 'is_archived', False)),
            'created_at': created_dt.strftime('%Y-%m-%d %H:%M:%S')
        }

    def __repr__(self):
        return f'<Scan {self.id} - {self.prediction}>'


class ChatLog(db.Model):
    __tablename__ = 'chat_logs'

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    sender = db.Column(db.String(20), nullable=False) # 'user' or 'bot'
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'sender': self.sender,
            'message': self.message,
            'timestamp': self.timestamp.strftime('%H:%M:%S')
        }


class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    type = db.Column(db.String(50), nullable=False, default='system')
    category = db.Column(db.String(20), default='info')
    title = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    patient_id = db.Column(db.String(50), nullable=True)
    patient_name = db.Column(db.String(100), nullable=True)
    scan_id = db.Column(db.Integer, nullable=True)
    is_read = db.Column(db.Boolean, default=False)
    action_tab = db.Column(db.String(50), default='patients')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='notifications', lazy=True)

    def get_time_ago(self, dt):
        now = datetime.utcnow()
        diff = now - dt
        seconds = max(0, diff.total_seconds())
        if seconds < 60:
            return 'Just now'
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f'{mins} minute{"s" if mins > 1 else ""} ago'
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f'{hours} hour{"s" if hours > 1 else ""} ago'
        elif seconds < 172800:
            return 'Yesterday'
        else:
            days = int(seconds / 86400)
            return f'{days} days ago'

    def to_dict(self):
        created_dt = self.created_at or datetime.utcnow()
        return {
            'id': self.id,
            'user_id': self.user_id,
            'type': self.type,
            'category': self.category or 'info',
            'title': self.title,
            'message': self.message,
            'patient_id': self.patient_id,
            'patient_name': self.patient_name,
            'scan_id': self.scan_id,
            'is_read': bool(self.is_read),
            'action_tab': self.action_tab or 'patients',
            'created_at': created_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'time_ago': self.get_time_ago(created_dt)
        }

