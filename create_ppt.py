import sys
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Theme Colors
BG_WHITE = RGBColor(255, 255, 255)
BG_OFFWHITE = RGBColor(248, 250, 252)
COLOR_TEAL = RGBColor(13, 148, 136)       # #0D9488
COLOR_BLUE = RGBColor(29, 78, 216)       # #1D4ED8
COLOR_GREEN = RGBColor(16, 185, 129)     # #10B981
COLOR_SLATE_DARK = RGBColor(15, 23, 42)  # #0F172A
COLOR_MUTED = RGBColor(100, 116, 139)    # #64748B
COLOR_CARD_BG = RGBColor(241, 245, 249)  # #F1F5F9
COLOR_CARD_BORDER = RGBColor(203, 213, 225) # #CBD5E1
COLOR_ACCENT_BG = RGBColor(240, 253, 250)   # #F0FDFA

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="TUMORAI XAI PLATFORM"):
        # Header Box
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.0))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Category tag
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = COLOR_TEAL
        
        # Title
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_SLATE_DARK
        
        # Top Accent Bar
        accent_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.04))
        accent_line.fill.solid()
        accent_line.fill.fore_color.rgb = COLOR_TEAL
        accent_line.line.color.rgb = COLOR_TEAL

    def add_card(slide, left, top, width, height, title, items, border_color=None, bg_color=None):
        if border_color is None:
            border_color = COLOR_TEAL
        if bg_color is None:
            bg_color = BG_OFFWHITE

        # Card background
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)
        
        # Text
        tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.15), width - Inches(0.3), height - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_SLATE_DARK
        
        for item in items:
            p = tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_SLATE_DARK
            p.space_before = Pt(4)
        return card

    # ==================== SLIDE 1: Title Slide ====================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_SLATE_DARK
    bg1.line.fill.background()

    # Title Card
    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(3.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "TumorAI XAI"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL

    p_sub = tf1.add_paragraph()
    p_sub.text = "Explainable Brain MRI Analysis & Longitudinal Intelligence"
    p_sub.font.size = Pt(24)
    p_sub.font.color.rgb = RGBColor(241, 245, 249)
    p_sub.space_before = Pt(10)

    p_desc = tf1.add_paragraph()
    p_desc.text = "Enterprise Diagnostic Decision-Support Platform with Multi-Model AI Ensemble, Spatial XAI, Epistemic Uncertainty Quantification, 3D Volumetric Segmentation & FHIR R4 Integration"
    p_desc.font.size = Pt(13)
    p_desc.font.color.rgb = COLOR_MUTED
    p_desc.space_before = Pt(20)

    # Footer Tag
    tb1_ft = slide1.shapes.add_textbox(Inches(1.0), Inches(6.2), Inches(11.333), Inches(0.5))
    tf1_ft = tb1_ft.text_frame
    p_ft = tf1_ft.paragraphs[0]
    p_ft.text = "Version 2.5.0 | Medical AI & Clinical Decision Support Suite"
    p_ft.font.size = Pt(11)
    p_ft.font.color.rgb = COLOR_GREEN

    # ==================== SLIDE 2: Project Overview & Objectives ====================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Project Overview & Core Objectives", "EXECUTIVE SUMMARY")
    
    add_card(slide2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), 
             "🎯 Platform Overview", [
                 "Enterprise-grade Explainable AI (XAI) clinical decision-support system for brain MRI diagnostics.",
                 "Automated multi-class classification: Glioma, Meningioma, Pituitary Tumor, and Healthy (No Tumor).",
                 "Integrated spatial explainability heatmaps (Grad-CAM, Grad-CAM++, LIME) and ROI boundary extraction.",
                 "Provides stochastic epistemic uncertainty quantification via Monte Carlo Dropout & Shannon Entropy.",
                 "Grounded Clinical AI Copilot powered by Google Gemini LLM for diagnostic impression synthesis."
             ], COLOR_TEAL, COLOR_ACCENT_BG)

    add_card(slide2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), 
             "🚀 Key Clinical Objectives", [
                 "Transparent Diagnostic AI: Replace 'black-box' predictions with spatial visual explanations and IoU/Dice metrics.",
                 "Longitudinal Volumetric Tracking: Compute 3D tumor volume (cm³) and measure growth velocity across patient visits.",
                 "Standardized Treatment Criteria: Categorize patient status using RANO / RECIST criteria (CR, PR, SD, PD).",
                 "Healthcare Interoperability: Export findings as standardized HL7 FHIR R4 payloads (DiagnosticReport & Observation).",
                 "Enterprise Security: Enforce HIPAA/GDPR audit trails and short-lived (300s TTL) signed URLs for medical image assets."
             ], COLOR_BLUE, BG_OFFWHITE)

    # ==================== SLIDE 3: Problem Statement ====================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Clinical Problem Statement & Key Gaps", "BACKGROUND & NEED")

    add_card(slide3, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "1. Diagnostic Workload & Fatigue", [
                 "High volume of complex brain MRI scans creates diagnostic backlogs for radiologists.",
                 "Subtle tumor margins and early lesion growth are prone to intra/inter-observer variability.",
                 "Manual 3D volume estimation from 2D slices is time-consuming and subjective."
             ], COLOR_TEAL)

    add_card(slide3, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "2. The Black-Box AI Trust Barrier", [
                 "Standard deep learning classifiers lack spatial rationale for their predictions.",
                 "No visual feedback showing which image regions drove the diagnostic classification.",
                 "Absence of confidence bounds or uncertainty indicators during out-of-distribution or noisy scans."
             ], COLOR_BLUE)

    add_card(slide3, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "3. Disconnected Clinical Systems", [
                 "Lack of integrated longitudinal growth tracking across historical MRI follow-ups.",
                 "Isolated AI tools fail to export standardized HL7 FHIR R4 records to HIS/EHR systems.",
                 "Security risks from unencrypted asset links and non-compliant audit mechanisms."
             ], COLOR_GREEN)

    # ==================== SLIDE 4: System Architecture Diagram ====================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "System Architecture & Layer Interaction", "TECHNICAL DIAGRAM")

    # Layer Cards
    # Frontend
    add_card(slide4, Inches(0.8), Inches(1.6), Inches(2.7), Inches(5.2),
             "1. Frontend Layer", [
                 "Vanilla HTML5 / CSS3 / ES6 JS",
                 "Responsive SPA Dashboard",
                 "Orthogonal 3D Slice Viewer",
                 "Interactive XAI Canvas",
                 "Longitudinal Growth Chart",
                 "Gemini AI Copilot Chat UI",
                 "PDF Report Preview"
             ], COLOR_TEAL)

    # Backend APIs
    add_card(slide4, Inches(3.8), Inches(1.6), Inches(2.7), Inches(5.2),
             "2. Backend REST API", [
                 "Flask Web Framework (Python)",
                 "Flask-SQLAlchemy DB ORM",
                 "Flask-Login Auth Manager",
                 "PyDICOM Imaging Parser",
                 "ReportLab PDF Generator",
                 "SMTP Email Dispatcher",
                 "FHIR R4 Serializer"
             ], COLOR_BLUE)

    # Core AI/XAI Engines
    add_card(slide4, Inches(6.8), Inches(1.6), Inches(2.7), Inches(5.2),
             "3. AI & XAI Engines", [
                 "Weighted Ensemble Voting",
                 "VGG16 / ResNet50 / EfficientNet",
                 "MC Dropout Uncertainty",
                 "Grad-CAM / Grad-CAM++",
                 "LIME Superpixel Explainer",
                 "Contour 3D Segmentation",
                 "Pre-Inference Quality Audit"
             ], COLOR_GREEN)

    # DB & Cloud Storage
    add_card(slide4, Inches(9.8), Inches(1.6), Inches(2.7), Inches(5.2),
             "4. Database & Storage", [
                 "Supabase PostgreSQL DB",
                 "Patients, Scans, Users Tables",
                 "Audit & Chat Logs Tables",
                 "Private Buckets: mri-scans",
                 "Private Buckets: patient-reports",
                 "300s TTL Signed Storage URLs",
                 "Google Gemini 1.5 Flash API"
             ], COLOR_SLATE_DARK)

    # ==================== SLIDE 5: Complete End-to-End Workflow ====================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "End-to-End Diagnostic & Clinical Workflow", "SYSTEM PIPELINE")

    steps = [
        ("1. MRI Upload", "PNG / JPG / DICOM drag & drop; DICOM tag parsing via PyDICOM."),
        ("2. Pre-Flight Audit", "Quality audit check (Resolution, contrast, Laplacian blur, OOD score)."),
        ("3. AI Ensemble", "Soft-weighted ensemble forward pass across VGG16, ResNet50 & EfficientNetB0."),
        ("4. XAI & 3D Volume", "Grad-CAM / LIME heatmap generation, morphological contouring & 3D volume calculation."),
        ("5. Reliability Analysis", "MC Dropout 10-pass uncertainty variance, Shannon Entropy & Consensus matrix."),
        ("6. Patient DB Storage", "Save record to Supabase DB; upload assets to private bucket with 300s TTL URLs."),
        ("7. Synthesis & Report", "Gemini LLM clinical synthesis, FHIR export & ReportLab PDF email dispatch.")
    ]

    top_pos = 1.6
    for i, (stitle, sdesc) in enumerate(steps):
        step_card = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(top_pos), Inches(11.733), Inches(0.68))
        step_card.fill.solid()
        step_card.fill.fore_color.rgb = COLOR_ACCENT_BG if i % 2 == 0 else BG_OFFWHITE
        step_card.line.color.rgb = COLOR_TEAL if i % 2 == 0 else COLOR_BLUE
        step_card.line.width = Pt(1)

        tb = slide5.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.08), Inches(11.3), Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{stitle}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_SLATE_DARK
        
        run = p.add_run()
        run.text = sdesc
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = COLOR_MUTED

        top_pos += 0.76

    # ==================== SLIDE 6: Backend Architecture & Services ====================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Backend Architecture: Modules & Core Services", "COMPONENT BREAKDOWN")

    add_card(slide6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "⚙️ AI & Analytics Modules (`modules/`)", [
                 "ensemble_engine.py: Soft-voting ensemble (50% VGG16, 25% ResNet50, 25% EfficientNetB0) & model agreement matrix.",
                 "uncertainty_engine.py: MC Dropout (10 passes), Shannon Entropy H(Y|X), prediction variance calculation.",
                 "xai_engine.py & xai_metrics.py: Grad-CAM (block5_conv3), Grad-CAM++, LIME superpixels, IoU & Dice spatial scores.",
                 "segmentation_engine.py: Morphological contour extraction, 3D volume calculation (V = 4/3 π a b c), max linear diameter.",
                 "quality_checker.py: Pre-inference audit (Resolution, brightness, contrast, Laplacian blur threshold ≥45.0, OOD).",
                 "counterfactual_engine.py: Sensitivity perturbation analysis and decision boundary evaluation."
             ], COLOR_TEAL)

    add_card(slide6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             "🏢 Business Services (`services/`)", [
                 "patient_service.py & study_service.py: Patient MRN registration, demographic tracking, clinical study creation.",
                 "dicom_service.py: Native .dcm header parsing & windowing conversion to high-contrast array.",
                 "gemini_service.py: Google Gemini LLM API integration for automated diagnostic summary generation.",
                 "fhir_service.py: HL7 FHIR R4 JSON serialization (DiagnosticReport & Observation resources).",
                 "report_service.py: Multi-page ReportLab PDF report generator with embedded maps & signature block.",
                 "longitudinal_service.py: Growth velocity calculation & RANO/RECIST treatment response categorization.",
                 "copilot_service.py: Patient-grounded conversational Q&A assistant.",
                 "audit_service.py: Security and HIPAA/GDPR operational audit logging."
             ], COLOR_BLUE)

    # ==================== SLIDE 7: Patient Management & Longitudinal Tracking ====================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Patient Management & Longitudinal Growth Tracking", "LONGITUDINAL INTELLIGENCE")

    add_card(slide7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "👤 Patient & Visit Record Management", [
                 "Unique Patient Record Indexing: Managed via unique MRN (Medical Record Number) and internal patient_id.",
                 "Comprehensive Demographics: Stores patient full name, age, gender, contact details, blood type, and clinical history.",
                 "Doctor Assignment: Tracks attending physician via created_by_doctor_id.",
                 "Multi-Visit Scan Linking: Groups multiple MRI scans under patient profile with visit_number, visit_time, visit_date, and next_checkup_date.",
                 "Archival Controls: Allows soft archival and restoration of past scan visits."
             ], COLOR_TEAL)

    add_card(slide7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             "📈 Longitudinal Volumetric Growth Analytics", [
                 "Volumetric Trajectory ΔV: Tracks volume deltas between baseline scan and current follow-up scan (ΔV = V_latest - V_baseline).",
                 "Monthly Growth Velocity: Calculates average percentage volume change per month (% / month).",
                 "RANO / RECIST Clinical Criteria Classification:",
                 "  • Complete Response (CR): 100% tumor volume regression.",
                 "  • Partial Response (PR): Volume shrinkage ≤ -50%.",
                 "  • Stable Disease (SD): Tumor volume within -50% to +25%.",
                 "  • Progressive Disease (PD): Volume growth > +25% (Triggers High Risk Alert)."
             ], COLOR_GREEN)

    # ==================== SLIDE 8: AI & XAI Features ====================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "AI Model Ensemble & XAI Visual Suite", "PREDICTION & TRANSPARENCY")

    add_card(slide8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "🤖 Deep Learning & Soft Ensemble Engine", [
                 "Multi-Model Ensemble Backbones: VGG16 (Primary), ResNet50, and EfficientNetB0.",
                 "Weighted Soft Voting Formula: Probability_ensemble = 0.50 P_VGG16 + 0.25 P_ResNet50 + 0.25 P_EfficientNetB0.",
                 "Model Consensus Matrix: Tracks inter-model agreement (Unanimous 100%, Majority 66.7%, Disagreement <66.7%).",
                 "Monte Carlo Dropout Uncertainty: 10 stochastic passes measure prediction variance and Shannon Entropy H(Y|X).",
                 "AI Reliability Index (0–100%): Synthesizes confidence (35%), XAI alignment (25%), quality score (20%), and consensus (20%)."
             ], COLOR_TEAL)

    add_card(slide8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             "🎨 Explainable AI (XAI) Analytics Suite", [
                 "Grad-CAM Localization: Computes gradients at block5_conv3 to produce coarse spatial intensity heatmaps.",
                 "Grad-CAM++: Higher-order spatial gradients for fine-grained multi-lesion localization.",
                 "LIME Superpixel Attribution: Segments MRI into superpixels and fits local linear model to show positive/negative contributions.",
                 "Contour ROI Mask: OpenCV morphological contouring extracts precise 2D region of interest boundary.",
                 "Quantitative Spatial Validation: Computes IoU (Intersection over Union) and Dice Similarity Coefficient (DSC) between heatmaps and ROI."
             ], COLOR_BLUE)

    # ==================== SLIDE 9: Clinical Workspace Features ====================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Key Clinical Workspace & Radiologist Tools", "WORKSPACE FEATURES")

    # 4 grid cards
    add_card(slide9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.45),
             "🔍 Dual MRI Comparison & 3D Viewer", [
                 "Side-by-side slice comparison between baseline & follow-up scans.",
                 "Multi-Planar orthogonal rendering (Axial, Coronal, Sagittal).",
                 "Interactive 3D surface mesh rendering for geometric inspection."
             ], COLOR_TEAL)

    add_card(slide9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.45),
             "📝 Clinical Notes & Status Workflow", [
                 "Rich-text radiologist clinical note editor.",
                 "Status tracking: Pending Review, Reviewed, Verified.",
                 "Automated follow-up scheduling with next_checkup_date."
             ], COLOR_BLUE)

    add_card(slide9, Inches(0.8), Inches(4.35), Inches(5.6), Inches(2.45),
             "⚡ Real-Time System Notifications", [
                 "Automated alerts for model disagreement (<66.7% consensus).",
                 "High uncertainty warning when MC Dropout variance > threshold.",
                 "Progressive Disease alerts when tumor growth exceeds +25%."
             ], COLOR_GREEN)

    add_card(slide9, Inches(6.8), Inches(4.35), Inches(5.7), Inches(2.45),
             "🤖 AI Copilot & Automated PDF Reports", [
                 "Study-grounded Gemini LLM Assistant answering radiologist queries.",
                 "One-click ReportLab PDF generation with embedded maps & signature.",
                 "SMTP email dispatcher sending PDFs directly to physicians."
             ], COLOR_SLATE_DARK)

    # ==================== SLIDE 10: Database Schema & Data Flow ====================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Database Schema & Patient-Study Data Flow", "DATA PERSISTENCE")

    add_card(slide10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "🗄️ Relational Entity Schema (`models_db.py`)", [
                 "users: id, username, email, phone, password_hash, role (Doctor/Radiologist/Patient), created_at.",
                 "patients: id, patient_id (MRN), full_name, age, gender, contact, created_by_doctor_id, created_at.",
                 "scans: id, user_id, patient_id, patient_name, visit_number, visit_time, next_checkup_date, image_path, heatmap_path, prediction, confidence, uncertainty_variance, iou_score, dice_score, clinical_status.",
                 "chat_logs: id, scan_id, user_id, sender, message, timestamp.",
                 "notifications: id, user_id, type, title, message, patient_id, scan_id, is_read, action_tab.",
                 "audit_logs: id, user_id, action, resource_type, resource_id, ip_address, timestamp."
             ], COLOR_TEAL)

    add_card(slide10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             "🔗 Entity Relationships & Storage Flow", [
                 "User (Doctor) 1 : N Patient (Doctor creates & manages patients).",
                 "Patient 1 : N Scan (Linked by patient_id across multi-visit follow-ups).",
                 "User 1 : N Scan (Doctor owns uploaded scan records).",
                 "Scan 1 : N ChatLog (Conversational context bound to specific scan).",
                 "User 1 : N Notification (User alerts & system notifications).",
                 "Private Bucket Isolation: Raw scans & heatmaps stored in mri-scans, reports in patient-reports.",
                 "Short-Lived Signed URLs: Access granted exclusively via backend-generated signed URLs (300s TTL)."
             ], COLOR_BLUE)

    # ==================== SLIDE 11: Security & Privacy ====================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Security, Doctor Access Control & Privacy", "COMPLIANCE & GOVERNANCE")

    add_card(slide11, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "🔐 Access Control & Auth", [
                 "Role-Based Access Control (RBAC): Differentiates Doctor, Radiologist, and Patient capabilities.",
                 "Flask-Login Session Management with secure cookie handling.",
                 "Bcrypt password hashing (generate_password_hash / check_password_hash)."
             ], COLOR_TEAL)

    add_card(slide11, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "🛡️ Asset Protection & TTL", [
                 "Supabase Private Buckets (mri-scans, patient-reports) prevent public HTTP access.",
                 "Short-Lived Signed URLs with strict 300-second (5-minute) expiration TTL.",
                 "Backend Credential Isolation: API keys and database credentials strictly server-side."
             ], COLOR_BLUE)

    add_card(slide11, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "📋 Audit & FHIR Standards", [
                 "HIPAA/GDPR Audit Logging (audit_logs table): Records user ID, IP address, resource ID, action, and timestamp.",
                 "HL7 FHIR R4 Interoperability: Generates anonymized/sanitized FHIR DiagnosticReport & Observation payloads.",
                 "Zero raw DICOM tag leakage to public clients."
             ], COLOR_GREEN)

    # ==================== SLIDE 12: Results, Limitations & Future Scope ====================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Results, System Limitations & Future Scope", "CONCLUSION & ROADMAP")

    add_card(slide12, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "🏆 Key Benchmark Results", [
                 "High Diagnostic Performance across Glioma, Meningioma, Pituitary, and No Tumor classes.",
                 "Robust Soft-Voting Ensemble consensus reducing single-model false positives.",
                 "High IoU & Dice spatial alignment between Grad-CAM heatmaps and contour ROI.",
                 "Sub-second inference time per scan with real-time UI feedback."
             ], COLOR_TEAL)

    add_card(slide12, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "⚠️ Current System Limitations", [
                 "Single-Slice 2D Input: Currently processes individual 2D axial/coronal slices rather than full 3D NIfTI volumes.",
                 "Scanner Domain Shift: Performance depends on standard MRI slice contrast and pixel quality.",
                 "Requires PyDICOM windowing tuning for non-standard vendor DICOM files."
             ], COLOR_BLUE)

    add_card(slide12, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2),
             "🔮 Future Development Scope", [
                 "3D Volumetric NIfTI Support: Process full multi-slice 3D MRI series natively.",
                 "PACS Server Integration: Native PACS server integration via DICOM web protocols (C-STORE, C-FIND, WADO-RS).",
                 "Multi-Center Federated Learning: Train ensemble models across multi-hospital data without centralizing patient records."
             ], COLOR_GREEN)

    # Save presentation
    output_path = os.path.join(os.getcwd(), "TumorAI_XAI_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    build_presentation()
