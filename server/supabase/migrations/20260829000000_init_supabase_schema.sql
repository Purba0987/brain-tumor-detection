-- =====================================================================
-- Supabase PostgreSQL Database Migration for TumorAI XAI Platform
-- Version: 20260829000000
-- Description: Creates profiles, patients, studies, scans, reports,
--              study_notes, audit_logs tables with RLS and indexes.
-- =====================================================================

-- 1. Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Profiles Table (Linked to Supabase Auth)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'radiologist', 'clinician', 'researcher')),
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Patients Master Table
CREATE TABLE IF NOT EXISTS public.patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_code TEXT NOT NULL UNIQUE,
    full_name TEXT NULL,
    date_of_birth DATE NULL,
    sex TEXT NULL,
    phone TEXT NULL,
    email TEXT NULL,
    clinical_notes TEXT NULL,
    is_anonymized BOOLEAN NOT NULL DEFAULT false,
    is_archived BOOLEAN NOT NULL DEFAULT false,
    created_by UUID NULL REFERENCES public.profiles(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Studies Table
CREATE TABLE IF NOT EXISTS public.studies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    study_uid TEXT NULL,
    modality TEXT NOT NULL DEFAULT 'MRI',
    study_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT NULL,
    clinical_indication TEXT NULL,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'processing', 'needs_review', 'reviewed', 'approved', 'failed', 'archived')),
    created_by UUID NULL REFERENCES public.profiles(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Scans Table (Inference, Neural XAI & Image Processing ROI Data)
CREATE TABLE IF NOT EXISTS public.scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id UUID NOT NULL REFERENCES public.studies(id) ON DELETE CASCADE,
    slice_number INT NULL,
    sequence_name TEXT NULL,
    original_object_path TEXT NOT NULL,
    heatmap_object_path TEXT NULL, -- Neural Network XAI Heatmap (Grad-CAM/Grad-CAM++/LIME)
    roi_object_path TEXT NULL,     -- Image Processing ROI (OpenCV Contour/Thresholding)
    segmentation_mask_object_path TEXT NULL,
    localization_object_path TEXT NULL,
    original_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_checksum TEXT NOT NULL, -- SHA-256
    predicted_class TEXT NULL,
    confidence NUMERIC NULL CHECK (confidence >= 0 AND confidence <= 1),
    class_probabilities JSONB NULL,
    uncertainty_score NUMERIC NULL CHECK (uncertainty_score >= 0 AND uncertainty_score <= 1),
    uncertainty_level TEXT NOT NULL DEFAULT 'unknown' CHECK (uncertainty_level IN ('low', 'medium', 'high', 'unknown')),
    xai_method TEXT NULL, -- 'Grad-CAM', 'Grad-CAM++', 'LIME'
    localization_method TEXT NULL DEFAULT 'opencv_contour', -- 'opencv_contour'
    iou_score NUMERIC NULL CHECK (iou_score >= 0 AND iou_score <= 1),
    dice_score NUMERIC NULL CHECK (dice_score >= 0 AND dice_score <= 1),
    localization_accuracy NUMERIC NULL,
    model_name TEXT NOT NULL DEFAULT 'vgg16_brain_tumor',
    model_version TEXT NOT NULL DEFAULT '1.0.0',
    processing_time_ms INT NULL,
    result_status TEXT NOT NULL DEFAULT 'conclusive' CHECK (result_status IN ('conclusive', 'inconclusive', 'failed')),
    processing_error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Reports Table (Versioned PDF Medical Reports)
CREATE TABLE IF NOT EXISTS public.reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id UUID NOT NULL REFERENCES public.studies(id) ON DELETE CASCADE,
    report_number TEXT NOT NULL UNIQUE,
    version INT NOT NULL DEFAULT 1,
    object_path TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'application/pdf',
    file_size_bytes BIGINT NOT NULL,
    file_checksum TEXT NOT NULL, -- SHA-256
    status TEXT NOT NULL DEFAULT 'generated' CHECK (status IN ('draft', 'generated', 'needs_review', 'approved', 'amended', 'archived')),
    summary TEXT NULL,
    generated_by UUID NULL REFERENCES public.profiles(id) ON DELETE SET NULL,
    reviewed_by UUID NULL REFERENCES public.profiles(id) ON DELETE SET NULL,
    approved_by UUID NULL REFERENCES public.profiles(id) ON DELETE SET NULL,
    reviewed_at TIMESTAMPTZ NULL,
    approved_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_study_version UNIQUE (study_id, version)
);

-- 7. Study Notes Table (Radiologist/Clinician Clinical Notes)
CREATE TABLE IF NOT EXISTS public.study_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id UUID NOT NULL REFERENCES public.studies(id) ON DELETE CASCADE,
    author_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    note_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 8. Audit Logs Table (Append-Only Audit Trail)
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id UUID NULL REFERENCES public.profiles(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id UUID NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 9. Database Performance Indexes
CREATE INDEX IF NOT EXISTS idx_patients_code ON public.patients(patient_code);
CREATE INDEX IF NOT EXISTS idx_patients_name_lower ON public.patients(LOWER(full_name));
CREATE INDEX IF NOT EXISTS idx_patients_created_at ON public.patients(created_at);
CREATE INDEX IF NOT EXISTS idx_patients_archived ON public.patients(is_archived);

CREATE INDEX IF NOT EXISTS idx_studies_patient_id ON public.studies(patient_id);
CREATE INDEX IF NOT EXISTS idx_studies_date ON public.studies(study_date);
CREATE INDEX IF NOT EXISTS idx_studies_status ON public.studies(status);

CREATE INDEX IF NOT EXISTS idx_scans_study_id ON public.scans(study_id);
CREATE INDEX IF NOT EXISTS idx_scans_pred_class ON public.scans(predicted_class);
CREATE INDEX IF NOT EXISTS idx_scans_uncertainty ON public.scans(uncertainty_level);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON public.scans(created_at);

CREATE INDEX IF NOT EXISTS idx_reports_study_id ON public.reports(study_id);
CREATE INDEX IF NOT EXISTS idx_reports_status ON public.reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_number ON public.reports(report_number);

CREATE INDEX IF NOT EXISTS idx_audit_actor ON public.audit_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON public.audit_logs(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON public.audit_logs(created_at);

-- 10. Enable Row-Level Security (RLS)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.study_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- Service Role Policy (Full Access for Backend Service Key)
CREATE POLICY service_role_all_profiles ON public.profiles FOR ALL USING (true);
CREATE POLICY service_role_all_patients ON public.patients FOR ALL USING (true);
CREATE POLICY service_role_all_studies ON public.studies FOR ALL USING (true);
CREATE POLICY service_role_all_scans ON public.scans FOR ALL USING (true);
CREATE POLICY service_role_all_reports ON public.reports FOR ALL USING (true);
CREATE POLICY service_role_all_notes ON public.study_notes FOR ALL USING (true);
CREATE POLICY service_role_all_audit ON public.audit_logs FOR ALL USING (true);
