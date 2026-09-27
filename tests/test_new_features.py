"""
Automated System & Integration Tests for TumorAI XAI Advanced Intelligence Suite
Tests Segmentation, Quality Auditing, Counterfactual XAI, Longitudinal Tracking, and Clinical Copilot APIs.
"""

import unittest
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.segmentation_engine import segment_tumor
from modules.quality_checker import check_image_quality
from services.longitudinal_service import calculate_longitudinal_growth
from services.copilot_service import answer_copilot_query

class TestAdvancedIntelligenceSuite(unittest.TestCase):

    def setUp(self):
        # Create synthetic RGB MRI image (128x128x3) with a circular lesion center
        self.synthetic_mri = np.zeros((128, 128, 3), dtype=np.uint8) + 40
        # Add hyperintense lesion region
        y, x = np.ogrid[:128, :128]
        mask = (x - 64)**2 + (y - 64)**2 <= 20**2
        self.synthetic_mri[mask] = [210, 210, 210]

    def test_segmentation_engine(self):
        result = segment_tumor(self.synthetic_mri)
        self.assertTrue(result['has_lesion'])
        self.assertGreater(result['area_pixels'], 0)
        self.assertGreater(result['volume_cm3'], 0.0)
        self.assertGreater(result['max_diameter_cm'], 0.0)
        self.assertIsNotNone(result['contour_overlay_path'])

    def test_quality_checker_valid_image(self):
        result = check_image_quality(self.synthetic_mri)
        self.assertTrue(result['passed'])
        self.assertGreaterEqual(result['quality_score'], 50.0)
        self.assertIn('metrics', result)

    def test_quality_checker_dark_image(self):
        dark_mri = np.zeros((64, 64, 3), dtype=np.uint8)
        result = check_image_quality(dark_mri)
        self.assertFalse(result['passed'])
        self.assertGreater(len(result['warnings']), 0)

    def test_longitudinal_growth_single_scan(self):
        res = calculate_longitudinal_growth("P-TEST-999")
        self.assertIn('has_history', res)
        self.assertIn('treatment_response', res)

    def test_copilot_service(self):
        query = "Why did the model classify this as glioma?"
        context = {
            'patient': {'mrn': 'P-TEST', 'age': '45', 'gender': 'Female'},
            'scan': {'prediction': 'glioma', 'confidence': 0.94, 'uncertainty_variance': 0.001},
            'segmentation': {'volume_cm3': 12.6, 'max_diameter_cm': 3.4, 'lesion_location': 'Left frontal'},
            'longitudinal': {'treatment_response': 'Baseline Assessment'}
        }
        res = answer_copilot_query(query, context)
        self.assertEqual(res['status'], 'success')
        self.assertIn('answer', res)

    def test_copilot_service_string_confidence(self):
        query = "tumor volume"
        context = {
            'patient': {'mrn': 'P-TEST-STR', 'age': '54', 'gender': 'Male'},
            'scan': {'prediction': 'Tumor: Glioma', 'confidence': '94.82%', 'uncertainty_variance': '0.0012'},
            'segmentation': {'volume_cm3': '12.6 cm³', 'max_diameter_cm': '3.4 cm', 'lesion_location': 'Left frontal'},
            'longitudinal': {'volume_delta_cm3': '+1.2 cm³', 'volume_delta_pct': '+10.5%', 'treatment_response': 'Stable Disease'}
        }
        res = answer_copilot_query(query, context)
        self.assertEqual(res['status'], 'success')
        self.assertIn('12.60 cm³', res['answer'])

    def test_copilot_service_empty_context(self):
        query = "What is the diagnosis?"
        res = answer_copilot_query(query, None)
        self.assertEqual(res['status'], 'success')
        self.assertIn('answer', res)

if __name__ == '__main__':
    unittest.main()
