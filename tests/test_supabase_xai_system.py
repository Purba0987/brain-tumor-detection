import os
import sys
import unittest
import numpy as np
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.storage_service import validate_mri_file, validate_pdf_file, calculate_sha256_checksum, generate_mri_object_paths
from services.auth_service import anonymize_patient_data
from services.patient_service import PatientService
from services.study_service import StudyService
from services.report_service import ReportService
from services.audit_service import AuditService
from modules.model_trainer import build_model
from modules.xai_engine import generate_gradcam, generate_gradcam_plus_plus
from modules.xai_metrics import calculate_iou, calculate_dice_coefficient
from modules.uncertainty_engine import predict_with_uncertainty

class TestSupabaseXAISystem(unittest.TestCase):
    def setUp(self):
        self.model = build_model(architecture='vgg16', input_shape=(128, 128, 3), num_classes=4)

    def test_file_validation(self):
        valid, err = validate_mri_file("test.png", b"fake_mri_bytes")
        self.assertTrue(valid)
        self.assertIsNone(err)

        valid_bad, err_bad = validate_mri_file("test.exe", b"fake_bytes")
        self.assertFalse(valid_bad)
        self.assertIn("Unsupported image extension", err_bad)

        valid_pdf, err_pdf = validate_pdf_file("report.pdf", b"%PDF-1.4 header bytes")
        self.assertTrue(valid_pdf)

    def test_sha256_checksum(self):
        content = b"sample_mri_data_stream"
        checksum = calculate_sha256_checksum(content)
        self.assertEqual(len(checksum), 64)

    def test_researcher_anonymization(self):
        raw_patient = {
            "id": "p-123",
            "patient_code": "P-999",
            "full_name": "John Doe",
            "phone": "+123456789",
            "email": "johndoe@example.com"
        }
        anon = anonymize_patient_data(raw_patient, user_role='researcher')
        self.assertEqual(anon['full_name'], 'Anonymized Patient')
        self.assertIsNone(anon['phone'])
        self.assertIsNone(anon['email'])

        regular = anonymize_patient_data(raw_patient, user_role='radiologist')
        self.assertEqual(regular['full_name'], 'John Doe')

    def test_object_path_generation(self):
        paths = generate_mri_object_paths("patient-123", "study-456")
        self.assertTrue(paths['original_path'].startswith("patient-123/study-456/"))
        self.assertTrue(paths['original_path'].endswith("/original.png"))

    def test_ml_xai_pipeline_regression(self):
        dummy_img = np.random.uniform(0, 255, (1, 128, 128, 3)).astype(np.float32) / 255.0
        
        unc_res = predict_with_uncertainty(self.model, dummy_img, num_samples=5)
        self.assertIn('confidence', unc_res)
        self.assertIn('uncertainty_variance', unc_res)

        heatmap = generate_gradcam(self.model, dummy_img, class_index=0)
        self.assertEqual(heatmap.shape, (128, 128))

        heatmap_pp = generate_gradcam_plus_plus(self.model, dummy_img, class_index=0)
        self.assertEqual(heatmap_pp.shape, (128, 128))

    def test_iou_and_dice(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[2:8, 2:8] = 1
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[2:8, 2:8] = 1

        iou = calculate_iou(m1, m2)
        dice = calculate_dice_coefficient(m1, m2)
        self.assertEqual(iou, 1.0)
        self.assertEqual(dice, 1.0)

    def test_audit_logging(self):
        evt = AuditService.log_event("patient_created", "patients", entity_id="p-100", metadata={"patient_code": "P-100"})
        self.assertEqual(evt['action'], "patient_created")
        self.assertEqual(evt['entity_type'], "patients")

if __name__ == '__main__':
    unittest.main()
