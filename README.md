# TumorAI XAI — Persistent Patient History & Secure Medical Report Platform

Enterprise Explainable AI (XAI) System for Brain Tumor Radiology Diagnosis, incorporating Deep Convolutional Neural Networks (VGG16), Grad-CAM / Grad-CAM++ / LIME Saliency Maps, Monte Carlo Uncertainty Estimation, Supabase PostgreSQL Persistence, Private Image & PDF Storage, and Role-Based Access Control.

---

## 1. Architecture Overview

- **Classifier Backbone**: VGG16 Transfer Learning (Input `128x128x3`, classes: `glioma`, `meningioma`, `pituitary`, `notumor`).
- **Neural Network XAI Engine**:
  - `Grad-CAM`: Feature map gradient activation mapping (`block5_conv3`).
  - `Grad-CAM++`: 2nd & 3rd-order gradient spatial localization.
  - `LIME`: Superpixel grid perturbation & contribution explanations.
- **Image-Processing ROI**: OpenCV Contour thresholding (`localization_method="opencv_contour"`, stored separately as `roi_object_path`).
- **XAI Spatial Validation**: IoU (Intersection over Union) & Dice Similarity Coefficient against ground-truth masks.
- **Epistemic Uncertainty**: Monte Carlo (MC) Dropout stochastic inference ($N=10$) computing variance & predictive entropy.
- **Database Persistence**: Supabase PostgreSQL (`profiles`, `patients`, `studies`, `scans`, `reports`, `study_notes`, `audit_logs`).
- **Private File Storage**: Supabase Storage Buckets (`mri-scans`, `patient-reports`) with 300-second short-lived signed URLs.

---

## 2. Step-by-Step Manual Supabase Setup Guide

Follow these step-by-step instructions to connect your manual Supabase account:

### Step 1: Create a Supabase Project
1. Log into your account at [supabase.com](https://supabase.com).
2. Click **New Project**, select your organization, name your project (e.g., `tumorai-xai-db`), set a database password, and choose a region.

### Step 2: Copy API Credentials
1. Go to **Project Settings** -> **API**.
2. Copy the **Project URL** (e.g., `https://your-project-ref.supabase.co`).
3. Copy the **anon / public key**.
4. Copy the **service_role key** (Keep secret! Only used by the Flask backend).

### Step 3: Run the SQL Database Migration
1. In the Supabase Dashboard, navigate to the **SQL Editor**.
2. Click **New Query**, open the project migration file:  
   [`supabase/migrations/20260829000000_init_supabase_schema.sql`](file:///c:/Users/Purba%20Hanra/brain%20tumor/supabase/migrations/20260829000000_init_supabase_schema.sql)
3. Copy and paste the complete SQL script into the query editor and click **Run**.
4. Verify that tables `profiles`, `patients`, `studies`, `scans`, `reports`, `study_notes`, and `audit_logs` are created.

### Step 4: Create Private Storage Buckets
1. In the Supabase Dashboard, go to **Storage**.
2. Click **New Bucket** and name it: `mri-scans`
   - Set **Public bucket** to **OFF** (Private).
3. Click **New Bucket** and name it: `patient-reports`
   - Set **Public bucket** to **OFF** (Private).

### Step 5: Configure Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` in your code editor and manually paste your credentials:
   ```env
   SUPABASE_URL=https://your-project-ref.supabase.co
   SUPABASE_ANON_KEY=your-actual-anon-key
   SUPABASE_SERVICE_ROLE_KEY=your-actual-service-role-key
   SUPABASE_MRI_BUCKET=mri-scans
   SUPABASE_REPORT_BUCKET=patient-reports
   SIGNED_URL_TTL_SECONDS=300
   ```

---

## 3. Starting and Running the Application

### Launch Local Backend Server
```powershell
.\venv\Scripts\python.exe main.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### Running Automated Test Suite
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Key Security & Compliance Disclaimers

- **Private Buckets & Temporary Signed URLs**: All raw MRI scans, heatmaps, ROIs, and PDF reports are stored in private buckets. Access is granted only via short-lived signed URLs (TTL 300s).
- **Service Role Key Security**: `SUPABASE_SERVICE_ROLE_KEY` is loaded strictly on the Flask backend and is never exposed to the browser.
- **HIPAA / GDPR Disclaimer**: This software is intended strictly for research and clinical decision support purposes. Using Supabase Free does not automatically guarantee HIPAA or GDPR compliance. Real-world production deployment requires formal regulatory, legal, and security review.
