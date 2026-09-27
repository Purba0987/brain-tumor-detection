import os
import requests
from dotenv import load_dotenv

load_dotenv()

class SupabaseClientWrapper:
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
        self.service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.mri_bucket = os.environ.get("SUPABASE_MRI_BUCKET", "mri-scans")
        self.report_bucket = os.environ.get("SUPABASE_REPORT_BUCKET", "patient-reports")
        self.signed_url_ttl = int(os.environ.get("SIGNED_URL_TTL_SECONDS", "300"))

    @property
    def is_configured(self):
        return bool(self.url and self.anon_key and "your-project-ref" not in self.url)

    def get_headers(self, use_service_role=True, bearer_token=None):
        key = self.service_role_key if (use_service_role and self.service_role_key) else self.anon_key
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {bearer_token or key}",
            "Content-Type": "application/json"
        }
        return headers

    def db_select(self, table, query_params=None, bearer_token=None):
        if not self.is_configured:
            return None
        endpoint = f"{self.url}/rest/v1/{table}"
        headers = self.get_headers(use_service_role=True, bearer_token=bearer_token)
        try:
            resp = requests.get(endpoint, headers=headers, params=query_params or {}, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[SupabaseDB] Select error: {e}")
        return None

    def db_insert(self, table, data, bearer_token=None):
        if not self.is_configured:
            return None
        endpoint = f"{self.url}/rest/v1/{table}"
        headers = self.get_headers(use_service_role=True, bearer_token=bearer_token)
        headers["Prefer"] = "return=representation"
        try:
            resp = requests.post(endpoint, headers=headers, json=data, timeout=10)
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                print(f"[SupabaseDB] Insert failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[SupabaseDB] Insert error: {e}")
        return None

    def db_update(self, table, match_params, data, bearer_token=None):
        if not self.is_configured:
            return None
        endpoint = f"{self.url}/rest/v1/{table}"
        headers = self.get_headers(use_service_role=True, bearer_token=bearer_token)
        headers["Prefer"] = "return=representation"
        try:
            resp = requests.patch(endpoint, headers=headers, params=match_params, json=data, timeout=10)
            if resp.status_code in (200, 204):
                return resp.json()
        except Exception as e:
            print(f"[SupabaseDB] Update error: {e}")
        return None

    def storage_upload(self, bucket, object_path, file_bytes, mime_type="application/octet-stream"):
        if not self.is_configured:
            return False
        endpoint = f"{self.url}/storage/v1/object/{bucket}/{object_path.lstrip('/')}"
        headers = {
            "apikey": self.service_role_key or self.anon_key,
            "Authorization": f"Bearer {self.service_role_key or self.anon_key}",
            "Content-Type": mime_type,
            "x-upsert": "true"
        }
        try:
            resp = requests.post(endpoint, headers=headers, data=file_bytes, timeout=15)
            if resp.status_code in (200, 201):
                return True
            else:
                print(f"[SupabaseStorage] Upload failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[SupabaseStorage] Upload error: {e}")
        return False

    def create_signed_url(self, bucket, object_path, ttl_seconds=300):
        if not self.is_configured:
            # Fallback to local static route URL for offline/test mode
            return f"/static/{object_path}"
        
        endpoint = f"{self.url}/storage/v1/object/sign/{bucket}/{object_path.lstrip('/')}"
        headers = self.get_headers(use_service_role=True)
        payload = {"expiresIn": ttl_seconds}
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                signed_path = data.get("signedURL", "")
                if signed_path.startswith("/"):
                    return f"{self.url}/storage/v1{signed_path}"
                return signed_path
        except Exception as e:
            print(f"[SupabaseStorage] Signed URL error: {e}")
        return f"/static/{object_path}"

# Global Singleton Supabase Client
supabase_client = SupabaseClientWrapper()
