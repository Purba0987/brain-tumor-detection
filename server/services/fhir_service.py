"""
FHIR Service — Tumor AI XAI
Converts scan diagnostics, patient metadata, and AI predictions into HL7 FHIR R4 standard JSON resources.
"""

from datetime import datetime, timezone

def generate_fhir_diagnostic_report(scan_data, patient_data=None, report_data=None):
    """
    Generates an HL7 FHIR R4 DiagnosticReport JSON resource representation of a scan & XAI analysis.
    """
    scan_id = scan_data.get('id', 'unknown')
    patient_code = patient_data.get('patient_code', 'P-UNKNOWN') if patient_data else 'P-UNKNOWN'
    patient_name = patient_data.get('name', 'Anonymous Patient') if patient_data else 'Anonymous Patient'
    
    predicted_class = scan_data.get('predicted_class', 'unknown')
    confidence_score = scan_data.get('confidence_score', 0.0)
    xai_method = scan_data.get('xai_method', 'Grad-CAM')
    created_at = scan_data.get('created_at', datetime.now(timezone.utc).isoformat())

    # Build FHIR Observation resource for the AI prediction
    observation_resource = {
        "resourceType": "Observation",
        "id": f"obs-{scan_id}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "24725-4",
                    "display": "Brain MRI Report"
                }
            ],
            "text": "Brain Tumor Convolutional Neural Network Analysis"
        },
        "subject": {
            "display": f"{patient_name} ({patient_code})"
        },
        "effectiveDateTime": created_at,
        "valueCodeableConcept": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "126952004" if predicted_class != "notumor" else "260413007",
                    "display": f"Tumor Category: {predicted_class.upper()}"
                }
            ],
            "text": predicted_class
        },
        "component": [
            {
                "code": {
                    "text": "Confidence Score"
                },
                "valueQuantity": {
                    "value": round(float(confidence_score) * 100.0, 2) if confidence_score else 0.0,
                    "unit": "%",
                    "system": "http://unitsofmeasure.org",
                    "code": "%"
                }
            },
            {
                "code": {
                    "text": "XAI Explanation Algorithm"
                },
                "valueString": xai_method
            }
        ]
    }

    # Build FHIR DiagnosticReport resource
    diagnostic_report_resource = {
        "resourceType": "DiagnosticReport",
        "id": f"report-{scan_id}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic Imaging Report"
                }
            ],
            "text": "TumorAI Explainable Brain MRI Diagnostic Report"
        },
        "subject": {
            "reference": f"Patient/{patient_code}",
            "display": patient_name
        },
        "effectiveDateTime": created_at,
        "issued": created_at,
        "performer": [
            {
                "display": "TumorAI XAI System (VGG16 Backbone)"
            }
        ],
        "result": [
            {
                "reference": f"Observation/obs-{scan_id}",
                "display": f"Classification: {predicted_class}"
            }
        ],
        "conclusion": f"Brain MRI analyzed via {xai_method} XAI model. Classification: {predicted_class.upper()} with confidence {round(float(confidence_score or 0)*100, 1)}%.",
        "contained": [observation_resource]
    }

    return diagnostic_report_resource
