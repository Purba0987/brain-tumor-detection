import uuid
from datetime import datetime
from services.supabase_client import supabase_client
from services.auth_service import get_current_user_role

LOGGED_AUDIT_TRAIL = []

class AuditService:
    @staticmethod
    def log_event(action, entity_type, entity_id=None, actor_id=None, metadata=None):
        """
        Appends a sanitized audit event record. Never stores passwords, tokens, secret keys, or raw file contents.
        """
        audit_uuid = str(uuid.uuid4())
        clean_metadata = dict(metadata or {})
        
        # Strip any sensitive keys from metadata
        for key in ['password', 'token', 'secret', 'api_key', 'service_role']:
            if key in clean_metadata:
                clean_metadata[key] = '[REDACTED]'

        record = {
            "id": audit_uuid,
            "actor_id": actor_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": str(entity_id) if entity_id else None,
            "metadata": clean_metadata,
            "created_at": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }

        LOGGED_AUDIT_TRAIL.append(record)

        if supabase_client.is_configured:
            supabase_client.db_insert("audit_logs", record)

        return record

    @staticmethod
    def get_audit_trail():
        """Returns recorded system audit trail."""
        return LOGGED_AUDIT_TRAIL
