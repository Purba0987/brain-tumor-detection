import os
import hashlib
import uuid
from services.supabase_client import supabase_client

ALLOWED_MRI_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm'}
ALLOWED_REPORT_EXTENSIONS = {'.pdf'}

MAX_MRI_SIZE_BYTES = int(os.environ.get("MAX_MRI_UPLOAD_MB", "15")) * 1024 * 1024
MAX_REPORT_SIZE_BYTES = int(os.environ.get("MAX_REPORT_UPLOAD_MB", "20")) * 1024 * 1024

def calculate_sha256_checksum(file_bytes):
    """Calculates SHA-256 checksum hex string for file content validation."""
    return hashlib.sha256(file_bytes).hexdigest()

def validate_mri_file(filename, file_bytes):
    if not file_bytes or len(file_bytes) == 0:
        return False, "File is empty."
    
    if len(file_bytes) > MAX_MRI_SIZE_BYTES:
        return False, f"File size exceeds limit of {MAX_MRI_SIZE_BYTES // (1024*1024)}MB."

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_MRI_EXTENSIONS:
        return False, f"Unsupported image extension '{ext}'. Allowed: PNG, JPG, JPEG, DCM."

    # Prevent path traversal in filename
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, "Invalid/unsafe filename path traversal attempt."

    return True, None

def validate_pdf_file(filename, file_bytes):
    if not file_bytes or len(file_bytes) == 0:
        return False, "Report PDF is empty."

    if len(file_bytes) > MAX_REPORT_SIZE_BYTES:
        return False, f"Report PDF size exceeds limit of {MAX_REPORT_SIZE_BYTES // (1024*1024)}MB."

    ext = os.path.splitext(filename)[1].lower()
    if ext != '.pdf':
        return False, "Report file must have a .pdf extension."

    if not file_bytes.startswith(b'%PDF'):
        return False, "Invalid PDF header signature."

    return True, None

def generate_mri_object_paths(patient_uuid, study_uuid, scan_uuid=None, extension=".png"):
    if not scan_uuid:
        scan_uuid = str(uuid.uuid4())
    
    prefix = f"{patient_uuid}/{study_uuid}/{scan_uuid}"
    return {
        'scan_uuid': scan_uuid,
        'original_path': f"{prefix}/original{extension}",
        'heatmap_path': f"{prefix}/heatmap{extension}",
        'roi_path': f"{prefix}/roi{extension}",
        'mask_path': f"{prefix}/mask{extension}"
    }

def generate_report_object_path(patient_uuid, study_uuid, version=1):
    return f"{patient_uuid}/{study_uuid}/report-v{version}.pdf"

def upload_mri_asset(patient_uuid, study_uuid, scan_uuid, asset_type, file_bytes, extension=".png", mime_type="image/png"):
    bucket = supabase_client.mri_bucket
    object_path = f"{patient_uuid}/{study_uuid}/{scan_uuid}/{asset_type}{extension}"
    
    # Save locally to static folder as fallback
    local_subfolder = "uploads" if asset_type == "original" else ("heatmaps" if asset_type == "heatmap" else "annotated")
    local_dir = os.path.join("static", local_subfolder)
    os.makedirs(local_dir, exist_ok=True)
    local_filepath = os.path.join(local_dir, f"{scan_uuid}_{asset_type}{extension}")
    with open(local_filepath, "wb") as f:
        f.write(file_bytes)

    # Upload to Supabase Private Bucket if configured
    if supabase_client.is_configured:
        supabase_client.storage_upload(bucket, object_path, file_bytes, mime_type=mime_type)

    checksum = calculate_sha256_checksum(file_bytes)
    return {
        'object_path': object_path,
        'local_filepath': local_filepath,
        'local_web_path': f"/static/{local_subfolder}/{scan_uuid}_{asset_type}{extension}",
        'checksum': checksum,
        'size_bytes': len(file_bytes)
    }

def upload_pdf_report(patient_uuid, study_uuid, version, pdf_bytes):
    bucket = supabase_client.report_bucket
    object_path = generate_report_object_path(patient_uuid, study_uuid, version=version)
    
    # Upload to Supabase Private Reports Bucket if configured
    if supabase_client.is_configured:
        supabase_client.storage_upload(bucket, object_path, pdf_bytes, mime_type="application/pdf")

    checksum = calculate_sha256_checksum(pdf_bytes)
    return {
        'object_path': object_path,
        'checksum': checksum,
        'size_bytes': len(pdf_bytes)
    }

def get_private_signed_url(bucket_type, object_path, local_web_path=None, ttl_seconds=300):
    if not object_path:
        return local_web_path
    if not supabase_client.is_configured:
        return local_web_path
    bucket = supabase_client.report_bucket if bucket_type == "report" else supabase_client.mri_bucket
    res = supabase_client.create_signed_url(bucket, object_path, ttl_seconds=ttl_seconds)
    return res or local_web_path
