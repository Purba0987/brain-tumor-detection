"""
Gemini AI Radiologist Assistant Service
Generates structured multi-modal AI clinical impression reports using Google Gemini API,
with a robust fallback clinical rule engine when an API key is not present.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_radiologist_impression(prediction_data, patient_data=None):
    """
    Generates an AI Radiologist Impression report based on model predictions,
    XAI metrics, uncertainty variance, and optional patient history.
    """
    if not prediction_data:
        prediction_data = {}
    predicted_class = str(prediction_data.get('predicted_class') or prediction_data.get('prediction') or 'unknown')

    raw_conf = prediction_data.get('confidence', 0.92)
    if isinstance(raw_conf, str):
        try:
            confidence = float(raw_conf.replace('%', '').strip())
            if confidence > 1.0:
                confidence /= 100.0
        except ValueError:
            confidence = 0.92
    else:
        try:
            confidence = float(raw_conf)
            if confidence > 1.0:
                confidence /= 100.0
        except (ValueError, TypeError):
            confidence = 0.92

    uncertainty = str(prediction_data.get('uncertainty', 'Low'))
    raw_var = prediction_data.get('uncertainty_variance', 0.012)
    try:
        uncertainty_variance = float(str(raw_var).replace('%', '').strip())
    except (ValueError, TypeError):
        uncertainty_variance = 0.012

    iou_score = prediction_data.get('iou_score', None)
    dice_score = prediction_data.get('dice_score', None)
    xai_method = str(prediction_data.get('xai_method', 'Grad-CAM'))
    symptoms = patient_data.get('symptoms', 'None reported') if patient_data else 'None reported'
    age = patient_data.get('age', 'Unspecified') if patient_data else 'Unspecified'
    gender = patient_data.get('gender', 'Unspecified') if patient_data else 'Unspecified'

    # Check for Gemini API key
    gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

    if gemini_api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
            prompt = f"""You are an expert Neuro-Radiologist AI assistant evaluating a Brain MRI Scan report.
Generate a structured, professional, HIPAA-compliant Radiologist Clinical Impression & Differential Diagnosis based on these parameters:

- Predicted Diagnosis: {predicted_class.upper()}
- Model Confidence: {confidence*100:.1f}%
- Epistemic Uncertainty Variance: {uncertainty_variance:.4f} ({uncertainty} Risk)
- XAI Method & Quality: {xai_method} (IoU: {iou_score if iou_score is not None else 'N/A'}, Dice: {dice_score if dice_score is not None else 'N/A'})
- Patient Demographics: Age {age}, Gender {gender}
- Patient Symptoms: {symptoms}

Please structure your response with:
1. Executive Summary & Diagnostic Confidence
2. Key Imaging & Saliency Findings
3. Differential Diagnosis & Risk Assessment
4. Recommended Clinical Next Steps
Keep it concise, rigorous, and professional.
"""
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                text_content = result['candidates'][0]['content']['parts'][0]['text']
                return {
                    'status': 'success',
                    'provider': 'Gemini 1.5 Flash',
                    'impression': text_content
                }
        except Exception as e:
            print(f"Gemini API call failed, using fallback engine: {e}")

    # Fallback Rule Engine if API key is absent or request fails
    return generate_fallback_impression(predicted_class, confidence, uncertainty, uncertainty_variance, iou_score, dice_score, symptoms)

def generate_fallback_impression(predicted_class, confidence, uncertainty, uncertainty_variance, iou_score, dice_score, symptoms):
    cls_upper = predicted_class.upper()
    conf_pct = f"{confidence * 100:.1f}%"
    
    if predicted_class.lower() == 'notumor':
        summary = f"No primary intracranial tumor detected. High diagnostic confidence ({conf_pct})."
        findings = "Normal brain parenchymal architecture without focal space-occupying mass, abnormal contrast enhancement, or midline shift."
        differential = "Normal MRI study. Consider non-neoplastic causes if symptoms persist."
        steps = "Routine clinical follow-up. Repeat MRI if focal neurological deficits develop."
    elif predicted_class.lower() == 'glioma':
        summary = f"Hyperintense mass lesion compatible with High/Low-Grade Glioma detected with {conf_pct} confidence."
        findings = f"Saliency localization ({uncertainty} Epistemic Risk) highlights localized intra-axial cerebral parenchymal involvement."
        differential = "Primary Glioma (Astrocytoma / Glioblastoma vs. Oligodendroglioma). Differential includes brain metastasis or cerebral abscess."
        steps = "Urgent Contrast-Enhanced 3T MRI with MR Spectroscopy, Diffusion Tensor Imaging (DTI), and neurosurgical evaluation for biopsy/resection."
    elif predicted_class.lower() == 'meningioma':
        summary = f"Dural-based extra-axial lesion consistent with Meningioma identified ({conf_pct} confidence)."
        findings = "XAI maps show well-demarcated extra-axial mass with characteristic dural attachment."
        differential = "Meningioma (WHO Grade I/II). Differential includes dural metastasis, schwannoma, or solitary fibrous tumor."
        steps = "Neuro-oncological consultation. Evaluate tumor size, mass effect on adjacent cortex, and suitability for surgical resection or stereotactic radiosurgery."
    else:  # pituitary
        summary = f"Sellar/suprasellar mass consistent with Pituitary Adenoma/Macroadenoma detected ({conf_pct} confidence)."
        findings = "Localized activation centered in the sellar region adjacent to the optic chiasm."
        differential = "Pituitary Adenoma (functioning vs. non-functioning). Differential includes craniopharyngioma or Rathke cleft cyst."
        steps = "Comprehensive endocrine panel (prolactin, ACTH, GH, TSH), formal visual field perimetry, and neurosurgical evaluation."

    impression_md = f"""### AI Radiologist Impression & Differential Assessment

**Executive Summary:**  
{summary}

**Key Imaging & Saliency Findings:**  
- **Primary Diagnosis:** `{cls_upper}` (Confidence: **{conf_pct}**)
- **Uncertainty Level:** **{uncertainty}** (Variance: `{uncertainty_variance:.4f}`)
- **Saliency Alignment:** {findings}

**Differential Diagnosis:**  
{differential}

**Recommended Next Steps:**  
{steps}

---
*Disclaimer: Generated by TumorAI Clinical Assistant for decision support purposes.*
"""
    return {
        'status': 'success',
        'provider': 'Clinical Rule Engine (Offline)',
        'impression': impression_md
    }
