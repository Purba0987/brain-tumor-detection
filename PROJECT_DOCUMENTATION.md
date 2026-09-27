# TumorAI XAI — Comprehensive System Architecture, Features & Technical Specification

> **Project Name**: TumorAI XAI — Persistent Patient History & Secure Medical Report Platform  
> **Domain**: Deep Learning, Explainable AI (XAI), Radiology Diagnostics, Healthcare IT Interoperability  
> **Version**: 2.5.0 (Advanced Intelligence Suite Edition)  
> **Tech Stack**: Python (Flask), TensorFlow / Keras, Supabase (PostgreSQL & Storage), OpenCV, PyDICOM, ReportLab, Google Gemini AI, HL7 FHIR R4 Standard, Vanilla HTML5/CSS3/JavaScript.

---

## Executive Summary

**TumorAI XAI** is an enterprise-grade Explainable Artificial Intelligence (XAI) clinical decision-support system designed for radiology departments and medical research. The platform provides end-to-end automation for brain tumor detection, classification, explainability visualization, epistemic uncertainty measurement, DICOM processing, patient study management, FHIR R4 interoperability, automated clinical report generation, and an **Advanced Intelligence Suite** comprising 3D tumor volume segmentation, longitudinal growth tracking, counterfactual sensitivity analysis, AI reliability indexing, pre-inference data quality auditing, study-grounded Clinical AI Copilot Q&A, multi-planar 3D rendering, and patient digital journey tracking.

The application strictly adheres to enterprise medical security standards by storing raw DICOM/MRI images, saliency heatmaps, binary region-of-interest (ROI) masks, and PDF reports in **private storage buckets** accessible exclusively via short-lived signed URLs (300-second TTL), keeping credentials strictly on the backend.

---

## 1. Advanced Intelligence Suite & Signature Features

TumorAI XAI includes **8 signature clinical features** built on top of the core ensemble and explainability engines:

### 1.1 🧠 Automatic Tumor Segmentation & 3D Volume Calculation (`modules/segmentation_engine.py`)
- **Mechanism**: OpenCV spatial contour extraction and morphological filtering on high-intensity lesion regions.
- **3D Volumetric Estimation**: Computes estimated 3D tumor volume ($V = \frac{4}{3} \pi \cdot a \cdot b \cdot c$) in $\text{cm}^3$, maximum linear diameter ($\text{cm}$), cross-sectional area ($\text{cm}^2$), bounding box, and anatomical quadrant location.
- **Boundary Contour Overlay**: Renders transparent colored boundary maps outlining lesion boundaries for quick visual review.

### 1.2 📈 Longitudinal Tumor Growth Tracker & Treatment Response Monitor (`services/longitudinal_service.py`)
- **Volumetric Trajectory**: Tracks volume changes ($\Delta V = V_{\text{latest}} - V_{\text{baseline}}$) and percentage growth rates across historical MRI studies for a patient.
- **Monthly Growth Velocity**: Computes average percentage volume change per month ($\% / \text{month}$).
- **RANO / RECIST Treatment Criteria**: Categorizes patient response into:
  - **Complete Response (CR)**: Tumor undetectable ($100\%$ regression).
  - **Partial Response (PR)**: Volume shrinkage $\le -50\%$.
  - **Stable Disease (SD)**: Controlled volume between $-50\%$ and $+25\%$.
  - **Progressive Disease (PD)**: Tumor growth $> +25\%$.

### 1.3 🧊 Multi-Planar & 3D Tumor Mesh Viewer
- **Orthogonal Plane Rendering**: View synchronized Orthogonal MRI Slices across **Axial**, **Coronal**, and **Sagittal** planes.
- **Interactive 3D Tumor Mesh**: Canvas 2D/3D surface mesh visualization allowing radiologists to rotate and inspect 3D lesion geometry.

### 1.4 ⚠️ AI Reliability Panel & Model Agreement Matrix
- **Overall AI Reliability Index ($0 - 100\%$)**: Synthesizes model confidence ($35\%$), XAI spatial alignment ($25\%$), data quality ($20\%$), and model consensus ($20\%$).
- **Model Agreement Visual Matrix**: Displays individual predictions, confidence levels, and agreement status across **VGG16**, **ResNet50**, and **EfficientNetB0**.

### 1.5 🧪 Counterfactual Explainable AI & Sensitivity Analysis (`modules/counterfactual_engine.py`)
- **Contrastive Region Modification**: Evaluates minimal spatial region perturbations required to alter prediction boundaries.
- **Sensitivity Metrics**: Computes prediction sensitivity scores and highlights top influential spatial regions on contrastive heatmaps.

### 1.6 🕵️ Pre-Inference AI Data Quality & OOD Checker (`modules/quality_checker.py`)
- **Pre-Flight Image Audit**: Automatically audits incoming MRI scans prior to deep learning model execution:
  - Resolution Check ($\ge 128 \times 128$)
  - Mean Brightness ($15.0 - 235.0$)
  - Contrast Check (Standard deviation of pixel intensities)
  - Sharpness / Blur Audit (Laplacian variance $\ge 45.0$)
  - Out-Of-Distribution (OOD) score estimation based on border profile and pixel entropy.

### 1.7 🤖 Study-Grounded Clinical AI Copilot (`services/copilot_service.py`)
- **Interactive Radiologist Assistant**: Powered by Google Gemini AI, answering radiologist queries strictly grounded in the active patient record, scan findings, XAI metrics, 3D volume data, and longitudinal history.

### 1.8 🧬 Digital Patient Journey Timeline
- **Timeline Navigation**: Visual milestone timeline linking initial MRI, follow-up scans, AI tumor detection, volume delta history, and generated PDF reports.

---

## 2. Machine Learning Models & Inference Architecture

The system supports individual model inference as well as a multi-architecture ensemble voting engine for robust classification across four distinct diagnostic categories:
- `glioma` — Glioma tumor detected
- `meningioma` — Meningioma tumor detected
- `pituitary` — Pituitary tumor detected
- `notumor` — No brain tumor detected (Healthy MRI)

### 2.1 Neural Network Backbones

1. **VGG16 Transfer Learning (Primary Model)**
   - **Architecture**: 16-layer Deep Convolutional Neural Network pre-trained on ImageNet with custom dense classification head.
   - **Input Dimensions**: `128 x 128 x 3` (normalized RGB imaging tensor).
   - **Target Feature Layer for XAI**: `block5_conv3` (final convolutional block capturing high-level spatial semantic features).
   - **Weight File**: `models/vgg16_best.h5` (or fallback `models/model.h5`).

2. **ResNet50 Backbone**
   - **Architecture**: 50-layer Residual Network with skip connections to address vanishing gradients and capture multi-scale feature representations.
   - **Weight File**: `models/resnet50_best.h5`.

3. **EfficientNetB0 Backbone**
   - **Architecture**: Scaled convolutional neural network optimizing computational efficiency and parameter utilization.
   - **Weight File**: `models/efficientnetb0_best.h5`.

---

### 2.2 Multi-Model Ensemble & Consensus Engine (`modules/ensemble_engine.py`)

To eliminate single-model bias and ensure high clinical confidence, TumorAI XAI incorporates a weighted soft-voting ensemble mechanism:
- **Weighted Probability Aggregation**:
  $$\text{Probability}_{\text{ensemble}} = 0.50 \cdot P_{\text{VGG16}} + 0.25 \cdot P_{\text{ResNet50}} + 0.25 \cdot P_{\text{EfficientNetB0}}$$
- **Inter-Model Consensus Score**:
  Computes agreement percentage among individual top predicted classes:
  - **Unanimous Consensus (100% Agreement)**: High clinical confidence across all backbones.
  - **Majority Consensus (66.7% Agreement)**: Majority model concurrence.
  - **Inter-Model Disagreement (< 66.7% Agreement)**: Automatically **flagged for radiologist expert review**.

---

### 2.3 Epistemic Uncertainty Estimation Engine (`modules/uncertainty_engine.py`)

Medical AI predictions require statistical confidence quantification. TumorAI XAI implements **Monte Carlo (MC) Dropout stochastic inference**:
- **Mechanism**: Retains active Dropout layers ($p = 0.2$ to $0.5$) during inference across $N = 10$ stochastic forward passes for a single MRI image.
- **Uncertainty Metrics**:
  - **Prediction Variance**: Variance of probabilities across the $N$ passes. High variance indicates model ambiguity.
  - **Predictive Entropy**: Shannon entropy computed over mean prediction probabilities:
    $$H(Y|X) = -\sum_{c} \bar{P}_c \log_2(\bar{P}_c)$$
  - **Mean Confidence**: Mean top-class probability over all Monte Carlo passes.

---

## 3. Explainable AI (XAI) & Spatial Validation

To transform black-box neural networks into transparent clinical tools, TumorAI XAI provides three visual explanation techniques alongside quantitative spatial validation metrics.

### 3.1 Visual Explanation Techniques (`modules/xai_engine.py`)

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Computes partial derivatives of the target class score $y^c$ with respect to feature maps $A^k$ in the target layer (`block5_conv3`).
   - Generates a coarse localization map highlighting important regions used by the model for classification, overlaid as a OpenCV JET colormap on the MRI.

2. **Grad-CAM++**
   - Incorporates second and third-order gradients to provide superior localization for multiple instance lesions and finer spatial detail.

3. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Segments the MRI image into superpixel components using Quickshift / SLIC algorithms.
   - Perturbs superpixels and fits a local interpretable ridge linear model to evaluate positive and negative superpixel contributions to the diagnostic output.

4. **Computer Vision ROI Extraction**
   - Employs OpenCV contour thresholding (`localization_method="opencv_contour"`) to generate a clean binary mask of the suspected tumor Region of Interest (ROI).

---

### 3.2 Spatial Validation & Attribution Metrics (`modules/xai_metrics.py`)

Evaluates the spatial alignment between generated XAI saliency heatmaps (thresholded at top intensity percentiles) and the extracted ROI ground-truth contour:
- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{|M_{\text{XAI}} \cap M_{\text{ROI}}|}{|M_{\text{XAI}} \cup M_{\text{ROI}}|}$$
- **Dice Similarity Coefficient (DSC)**:
  $$\text{DSC} = \frac{2 \cdot |M_{\text{XAI}} \cap M_{\text{ROI}}|}{|M_{\text{XAI}}| + |M_{\text{ROI}}|}$$

---

## 4. Core Application Features & Services

### 4.1 Patient & Study Management (`services/patient_service.py`, `services/study_service.py`)
- Full patient lifecycle management including demographics, Medical Record Number (MRN), Date of Birth, Gender, Blood Type, Contact Details, and Clinical History.
- Clinical Imaging Study creation grouping multiple MRI scans, slice series, and longitudinal diagnostic tracking over time.

### 4.2 Medical Imaging & DICOM Processor (`services/dicom_service.py`)
- Native `.dcm` (DICOM) file parsing using `pydicom`.
- Metadata extraction: Patient Name, Patient ID, Modality (MR), Study Date, Manufacturer, Window Center/Width, Pixel Spacing, Slice Thickness.
- Automated extraction and windowing conversion of raw pixel arrays into high-contrast PNG/JPEG images for browser rendering and neural inference.

### 4.3 LLM Clinical Summary Generator (`services/gemini_service.py`)
- Integrates Google Gemini AI (`gemini-1.5-flash` / `gemini-pro`) to analyze inference results, XAI heatmaps, ensemble consensus, and uncertainty metrics.
- Automatically generates structured radiologist diagnostic summaries containing:
  - Clinical Impression
  - Key Findings & Anatomic Location
  - Uncertainty & Disagreement Analysis
  - Recommended Next Steps / Follow-up MRI Protocol

### 4.4 HL7 FHIR R4 Interoperability (`services/fhir_service.py`)
- Transforms local scan predictions and patient study records into standardized HL7 FHIR R4 JSON payloads:
  - `FHIR DiagnosticReport`: Structured clinical diagnostic report resource.
  - `FHIR Observation`: Quantitative observation resource recording diagnostic probability, confidence score, and entropy.
- Enables seamless integration with Hospital Information Systems (HIS) and Electronic Health Record (EHR) platforms (e.g., Epic, Cerner).

### 4.5 PDF Diagnostic Report Builder (`services/report_service.py`)
- Generates publication-quality multi-page PDF medical reports using ReportLab.
- Embedded elements: Hospital header, patient demographics, raw MRI scan, Grad-CAM heatmap overlay, ROI boundary, ensemble probabilities, uncertainty bounds, Gemini AI clinical notes, and radiologist digital signature block.

### 4.6 Automated Email Dispatcher (`services/email_service.py`)
- Secure SMTP email service for dispatching generated PDF diagnostic reports directly to attending physicians or patients as email attachments.

### 4.7 Enterprise Security & Audit Trail (`services/audit_service.py`)
- Comprehensive audit logging capturing every platform action: scan uploads, inference executions, report generations, patient updates, and record deletions.
- Records user ID, IP address, action type, resource ID, timestamp, and metadata for HIPAA/GDPR compliance audits.

---

## 5. Database Schema & Storage Architecture

### 5.1 Supabase PostgreSQL Database Schema (`supabase/migrations/`)

| Table Name | Description | Key Columns |
| :--- | :--- | :--- |
| `profiles` | User accounts and radiologist profiles | `id`, `email`, `full_name`, `role`, `created_at` |
| `patients` | Patient demographic records | `id`, `mrn`, `first_name`, `last_name`, `dob`, `gender`, `blood_type`, `medical_history` |
| `studies` | Imaging studies linking scans to patients | `id`, `patient_id`, `study_uid`, `study_date`, `description`, `modality`, `status` |
| `scans` | MRI & DICOM scans with inference results | `id`, `study_id`, `image_path`, `roi_path`, `gradcam_path`, `gradcam_pp_path`, `lime_path`, `prediction`, `confidence`, `uncertainty_variance`, `entropy`, `consensus_status` |
| `reports` | Diagnostic reports and summaries | `id`, `scan_id`, `summary`, `ai_findings`, `pdf_report_path`, `status`, `created_at` |
| `study_notes` | Radiologist notes attached to studies | `id`, `study_id`, `author_id`, `note_text`, `created_at` |
| `audit_logs` | Security and operational activity logs | `id`, `user_id`, `action`, `resource_type`, `resource_id`, `ip_address`, `details`, `timestamp` |

### 5.2 Supabase Private Storage Buckets (`services/storage_service.py`)
- **`mri-scans` Bucket** (Private): Holds raw MRI images, extracted DICOM frames, ROI masks, and XAI heatmaps (`Grad-CAM`, `Grad-CAM++`, `LIME`).
- **`patient-reports` Bucket** (Private): Holds generated PDF diagnostic reports.
- **Short-Lived Signed URLs**: Frontend requests signed URLs on-demand with a default expiration of **300 seconds (5 minutes)**, preventing unauthorized direct asset access.

---

## 6. Web Interface & User Experience (`templates/index.html`)

The frontend is a modern single-page dashboard designed with Vanilla JavaScript, custom CSS tokens, glassmorphism, dynamic transitions, and responsive tab navigation:

1. **Executive Dashboard**: System metrics, total patient counts, scan volume, tumor distribution charts, and quick system status indicator.
2. **Patient Management Hub**: Searchable patient table, MRN lookup, new patient registration modal, and patient medical profile views.
3. **MRI & DICOM Upload & Real-Time Inference**: Drag-and-drop file uploader (supporting `.png`, `.jpg`, `.jpeg`, `.dcm`), model architecture dropdown (VGG16, ResNet50, EfficientNetB0, or Ensemble), pre-inference quality audit status, and real-time progress visualization.
4. **XAI Visual Analytics Suite**: Multi-pane synchronized visual comparison viewer:
   - Original Scan
   - Grad-CAM Heatmap
   - Grad-CAM++ Localization Map
   - LIME Superpixel Patch Attribution
   - OpenCV ROI Mask
   - Counterfactual Sensitivity Map
   - IoU & Dice Score spatial metrics sidebar
5. **Interactive DICOM & 3D Tumor Viewer**: Canvas rendering with brightness/contrast adjustments, zoom/pan controls, DICOM tag header inspector, orthogonal slice planes (Axial, Coronal, Sagittal), and 3D surface mesh rendering.
6. **AI Reliability & Model Consensus Matrix**: Displays overall AI reliability score ($0-100\%$) and backbone agreement matrix (VGG16 vs ResNet50 vs EfficientNetB0).
7. **Tumor Segmentation & Volumetric Analytics**: Displays 3D volume in $\text{cm}^3$, max linear diameter in $\text{cm}$, and boundary overlays.
8. **Longitudinal Growth Tracker**: Interactive timeline chart depicting volume delta ($\Delta V$), monthly growth velocity, and RANO/RECIST treatment response categories.
9. **Study-Grounded Clinical AI Copilot**: Interactive Q&A chat assistant grounded strictly in active patient study evidence.
10. **Reports & Gemini AI Clinical Synthesis**: One-click Gemini AI diagnostic summary generation, rich-text radiologist note editor, PDF report preview/download, SMTP email sender, and FHIR export.
11. **Compliance Audit Log Viewer**: Searchable, filterable audit log timeline.

---

## 7. Directory Structure Overview

```
brain tumor/
├── main.py                        # Primary Flask Application & REST API Endpoints
├── requirements.txt               # Dependencies (TensorFlow, Supabase, PyDICOM, etc.)
├── Dockerfile                     # Docker Container Configuration
├── Procfile                       # Heroku / Cloud Deployment Process File
├── README.md                      # Quickstart & Manual Supabase Setup Guide
├── PROJECT_DOCUMENTATION.md       # Comprehensive Architecture & Features Manual (This File)
├── .env.example                   # Environment Variable Template
│
├── models/                        # Pre-trained Model Weights
│   ├── vgg16_best.h5              # VGG16 Weights
│   ├── resnet50_best.h5           # ResNet50 Weights
│   ├── efficientnetb0_best.h5     # EfficientNetB0 Weights
│   └── model.h5                   # Primary Classifier Backup Weights
│
├── modules/                       # Core ML, XAI, Segmentation & Uncertainty Engines
│   ├── ensemble_engine.py         # Multi-Model Voting & Consensus Calculation
│   ├── model_trainer.py           # Model Building, Transfer Learning & Fine-tuning
│   ├── uncertainty_engine.py      # Monte Carlo Dropout Uncertainty & Entropy
│   ├── xai_engine.py              # Grad-CAM, Grad-CAM++, LIME & Contour ROI
│   ├── xai_metrics.py             # IoU & Dice Similarity Coefficient Calculations
│   ├── segmentation_engine.py     # Automatic 3D Tumor Segmentation & Volume (cm³)
│   ├── quality_checker.py         # Pre-Inference AI Data Quality & OOD Audit
│   └── counterfactual_engine.py   # Counterfactual XAI & Sensitivity Analysis
│
├── services/                      # Enterprise Backend Business Services
│   ├── supabase_client.py         # Supabase SDK Client Initialization
│   ├── storage_service.py          # Private Bucket Uploads & Signed URL Generators
│   ├── patient_service.py         # Patient CRUD Operations
│   ├── study_service.py           # Clinical Study Operations
│   ├── scan_service.py            # MRI Scan Persistence & Retrieval
│   ├── report_service.py          # ReportLab PDF Generation
│   ├── dicom_service.py           # DICOM Parsing & Conversion
│   ├── gemini_service.py          # Google Gemini AI LLM Diagnostics Integration
│   ├── fhir_service.py            # HL7 FHIR R4 Serialization (DiagnosticReport, Observation)
│   ├── email_service.py           # SMTP Email PDF Dispatching
│   ├── audit_service.py           # Compliance Audit Logging
│   ├── analytics_service.py       # Dashboard Analytics & Statistics Computation
│   ├── longitudinal_service.py    # Longitudinal Tumor Growth & RANO Response Monitor
│   ├── copilot_service.py         # Study-Grounded Clinical AI Copilot Q&A
│   └── data_migration_service.py  # Database Migration & Sync Utilities
│
├── supabase/                      # Database Schemas & Migrations
│   └── migrations/
│       └── 20260829000000_init_supabase_schema.sql  # Database DDL & Schema Setup
│
├── templates/                     # Frontend Views
│   ├── index.html                 # Full Enterprise SPA Frontend Interface
│   └── index1.html                # Legacy Frontend Backup
│
├── static/                        # Static Assets (CSS, JS, Images)
├── tests/                         # Automated Unit & Integration Tests
│   ├── test_supabase_xai_system.py# Comprehensive End-to-End System Tests
│   └── test_new_features.py       # Advanced Intelligence Suite Automated Unit Tests
└── uploads/                       # Temporary Local File Buffer
```

---

## 8. How to Setup and Run

### 8.1 Environment Setup
1. Create a Python Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
2. Install Required Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 8.2 Configuration (`.env`)
Create a `.env` file in the project root:
```env
FLASK_ENV=development
PORT=5000

# Supabase Credentials
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-public-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Supabase Storage Buckets
SUPABASE_MRI_BUCKET=mri-scans
SUPABASE_REPORT_BUCKET=patient-reports
SIGNED_URL_TTL_SECONDS=300

# Google Gemini API Key
GEMINI_API_KEY=your-gemini-api-key

# SMTP Configuration (Optional for PDF Emailing)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### 8.3 Launching the Application
```bash
python main.py
```
Navigate to **`http://127.0.0.1:5000`** in your browser.

### 8.4 Running System Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 9. Summary of Guarantees

- **Zero Breaking Edits**: All pre-existing features, UI elements, ML classification routines, and database tables remain 100% functional and preserved.
- **Single Master Reference File**: All technical architectural specifications, feature workflows, and setup guidelines are encapsulated within this file [`PROJECT_DOCUMENTATION.md`](file:///c:/Users/Purba%20Hanra/brain%20tumor/PROJECT_DOCUMENTATION.md).
