from functools import wraps
from flask import request, jsonify
from flask_login import current_user

ROLES = ['admin', 'radiologist', 'clinician', 'researcher']

def get_current_user_role():
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        role = getattr(current_user, 'role', 'Clinician').lower()
        if role == 'doctor':
            return 'clinician'
        return role
    return 'clinician'  # Default fallback role for guest/local testing

def is_admin():
    return get_current_user_role() == 'admin'

def is_radiologist():
    return get_current_user_role() in ['admin', 'radiologist']

def is_clinician():
    return get_current_user_role() in ['admin', 'radiologist', 'clinician']

def is_researcher():
    return get_current_user_role() == 'researcher'

def require_role(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_role = get_current_user_role()
            if user_role not in allowed_roles and user_role != 'admin':
                return jsonify({'error': 'Unauthorized: Insufficient role permissions for this operation.'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def anonymize_patient_data(patient_dict, user_role=None):
    """
    Strips direct patient identifiers for Researchers.
    Researchers see only 'Anonymized Patient' and patient_code.
    """
    if user_role is None:
        user_role = get_current_user_role()

    if user_role == 'researcher':
        cleaned = dict(patient_dict)
        cleaned['full_name'] = 'Anonymized Patient'
        cleaned['date_of_birth'] = None
        cleaned['sex'] = None
        cleaned['phone'] = None
        cleaned['email'] = None
        cleaned['clinical_notes'] = '[REDACTED FOR RESEARCH ACCESS]'
        cleaned['is_anonymized'] = True
        return cleaned
    return patient_dict
