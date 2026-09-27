"""
DICOM Processing & Automated Anonymization Service — TumorAI XAI
Parses medical DICOM (.dcm) files, anonymizes PHI metadata tags (HIPAA/GDPR compliance),
and extracts image frames for deep learning CNN inference.
"""

import os
import io
import numpy as np
from PIL import Image

def process_and_anonymize_dicom(file_bytes, filename="scan.dcm"):
    """
    Parses a DICOM file from raw bytes, strips PHI metadata,
    extracts the image frame converted to PIL Image, and returns anonymized metadata.
    """
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError

        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        
        # Extract PHI before anonymizing for logging/audit (if needed)
        original_patient_id = getattr(ds, 'PatientID', 'ANONYMOUS')
        
        # Anonymize Sensitive PHI Tags
        phi_tags = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAddress', 'InstitutionName', 'ReferringPhysicianName',
            'PerformingPhysicianName', 'OperatorsName'
        ]
        
        anonymized_hdr = {}
        for tag in phi_tags:
            if hasattr(ds, tag):
                anonymized_hdr[tag] = f"[ANONYMIZED_PHI_{tag}]"
                setattr(ds, tag, f"ANONYMIZED_{tag.upper()}")

        # Extract Non-PHI DICOM metadata useful for clinical volumetric calculations
        slice_thickness = getattr(ds, 'SliceThickness', 5.0)
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        modality = getattr(ds, 'Modality', 'MR')
        study_description = getattr(ds, 'StudyDescription', 'Brain MRI Scan')

        # Convert Pixel Array to Normalized RGB Image
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array.astype(float)
            
            # Apply Windowing / Normalization
            rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
            rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
            pixel_array = pixel_array * rescale_slope + rescale_intercept
            
            # Normalize to 0-255
            p_min, p_max = np.min(pixel_array), np.max(pixel_array)
            if p_max > p_min:
                normalized = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(pixel_array, dtype=np.uint8)

            # Convert 2D grayscale slice to 3-channel RGB PIL image
            if normalized.ndim == 2:
                pil_img = Image.fromarray(normalized).convert('RGB')
            else:
                pil_img = Image.fromarray(normalized[0]).convert('RGB')
        else:
            raise ValueError("No pixel image array found in DICOM file.")

        return {
            'status': 'success',
            'is_dicom': True,
            'image': pil_img,
            'metadata': {
                'Modality': str(modality),
                'StudyDescription': str(study_description),
                'SliceThickness': float(slice_thickness),
                'PixelSpacing': [float(x) for x in pixel_spacing],
                'AnonymizedPHI': anonymized_hdr
            }
        }
    except ImportError:
        print("pydicom library not installed. Falling back to basic file stream parser.")
        return fallback_dicom_parser(file_bytes, filename)
    except Exception as e:
        print(f"Error processing DICOM file: {e}")
        return fallback_dicom_parser(file_bytes, filename)

def fallback_dicom_parser(file_bytes, filename):
    """
    Fallback parser when pydicom is absent or file is standard image format with .dcm extension.
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        return {
            'status': 'success',
            'is_dicom': False,
            'image': pil_img,
            'metadata': {
                'Modality': 'MR',
                'StudyDescription': 'Brain MRI Scan (Fallback Image Extractor)',
                'SliceThickness': 5.0,
                'PixelSpacing': [1.0, 1.0],
                'AnonymizedPHI': {'PatientName': '[ANONYMIZED]'}
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': f"Failed to parse DICOM or image file: {str(e)}"}
