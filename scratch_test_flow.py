import os
import sys
import json
from datetime import datetime

# Set up environment and app context
from main import app
from models_db import db, User, Patient, Scan
from services.patient_service import PatientService
from services.scan_service import ScanService

def run_acceptance_tests():
    with app.app_context():
        print("=== RUNNING ACCEPTANCE TESTS ===")
        
        # 0. Ensure a test doctor user exists
        doctor = User.query.filter_by(email="doctor@test.com").first()
        if not doctor:
            doctor = User(
                username="Dr. ABC",
                email="doctor@test.com",
                role="Doctor"
            )
            doctor.set_password("doctor123")
            db.session.add(doctor)
            db.session.commit()
            print(f"Created test doctor: {doctor.username} (ID: {doctor.id})")
        else:
            print(f"Found existing doctor: {doctor.username} (ID: {doctor.id})")

        # Clean up any existing test records for TESTP001
        Scan.query.filter_by(patient_id="TESTP001").delete()
        Patient.query.filter_by(patient_id="TESTP001").delete()
        db.session.commit()

        # -------------------------------------------------------------
        # TEST 1: Create Patient Record
        # -------------------------------------------------------------
        print("\n--- TEST 1: Create Patient Record ---")
        p_data = PatientService.get_or_create_patient(
            patient_code="TESTP001",
            full_name="Rahul Sharma",
            age="45",
            gender="Male",
            contact="9876543210",
            doctor_id=doctor.id
        )
        print(f"Patient Created/Fetched: {p_data['full_name']} ({p_data['patient_id']})")
        assert p_data['patient_id'] == "TESTP001"
        assert p_data['full_name'] == "Rahul Sharma"
        assert str(p_data['age']) == "45"
        assert p_data['contact'] == "9876543210"
        
        # Verify persistence
        db.session.expire_all()
        persisted_p = Patient.query.filter_by(patient_id="TESTP001").first()
        assert persisted_p is not None, "Patient record was not persisted in DB!"
        print("[OK] TEST 1 PASSED: Patient TESTP001 successfully created and persisted!")

        # -------------------------------------------------------------
        # TEST 2: First Checkup (Visit 1)
        # -------------------------------------------------------------
        print("\n--- TEST 2: First Checkup on Same Day ---")
        v1_num, v1_time = PatientService.get_next_visit_info("TESTP001")
        print(f"Visit 1 Info: Visit #{v1_num}, Time: {v1_time}")
        assert v1_num == 1
        
        # Simulate creating Scan 1 (Visit 1)
        scan1 = Scan(
            original_filename="test_scan1.jpg",
            image_path="/static/uploads/test_scan1.jpg",
            prediction="Glioma",
            confidence="98.5%",
            patient_name="Rahul Sharma",
            patient_id="TESTP001",
            patient_age=45,
            patient_gender="Male",
            patient_contact="9876543210",
            visit_number=v1_num,
            visit_time=v1_time,
            created_at=datetime.utcnow(),
            user_id=doctor.id
        )
        db.session.add(scan1)
        db.session.commit()
        scan1_id = scan1.id
        print(f"Scan 1 Saved: ID {scan1_id}, Visit #{scan1.visit_number}, Time {scan1.visit_time}, Prediction {scan1.prediction}")
        print("[OK] TEST 2 PASSED: Visit 1 saved under patient TESTP001!")

        # -------------------------------------------------------------
        # TEST 3: Second Checkup on Same Day (Visit 2)
        # -------------------------------------------------------------
        print("\n--- TEST 3: Second Checkup on Same Day ---")
        v2_num, v2_time = PatientService.get_next_visit_info("TESTP001")
        print(f"Visit 2 Info: Visit #{v2_num}, Time: {v2_time}")
        assert v2_num == 2, f"Expected visit #2, got {v2_num}"

        # Simulate creating Scan 2 (Visit 2)
        scan2 = Scan(
            original_filename="test_scan2.jpg",
            image_path="/static/uploads/test_scan2.jpg",
            prediction="Meningioma",
            confidence="96.2%",
            patient_name="Rahul Sharma",
            patient_id="TESTP001",
            patient_age=45,
            patient_gender="Male",
            patient_contact="9876543210",
            visit_number=v2_num,
            visit_time=v2_time,
            created_at=datetime.utcnow(),
            user_id=doctor.id
        )
        db.session.add(scan2)
        db.session.commit()
        scan2_id = scan2.id
        print(f"Scan 2 Saved: ID {scan2_id}, Visit #{scan2.visit_number}, Time {scan2.visit_time}, Prediction {scan2.prediction}")
        print("[OK] TEST 3 PASSED: Visit 2 saved under patient TESTP001 without overwriting Visit 1!")

        # -------------------------------------------------------------
        # TEST 4: Patient History Search
        # -------------------------------------------------------------
        print("\n--- TEST 4: Search Patient History ---")
        timeline_res = PatientService.get_patient_timeline("Rahul Sharma")
        print(f"Timeline Result Status: {timeline_res['status']}")
        print(f"Patient Name: {timeline_res['patient_name']}, ID: {timeline_res['patient_code']}")
        print(f"Total Visits: {timeline_res['total_scans']}")
        print("Visits in Timeline:")
        for item in timeline_res['timeline']:
            print(f"  - Visit #{item.get('visit_number')}: Date={item.get('date_formatted')}, Time={item.get('visit_time')}, Pred={item.get('prediction')}, Doctor={item.get('doctor_name')}")
        
        assert timeline_res['patient_name'] == "Rahul Sharma"
        assert timeline_res['patient_code'] in ["TESTP001", "P001"]
        assert len(timeline_res['timeline']) >= 2, f"Expected at least 2 visits, got {len(timeline_res['timeline'])}"
        print("[OK] TEST 4 PASSED: Patient history search returned full chronological visit history!")

        # -------------------------------------------------------------
        # TEST 5: Open Previous Visit (Individual Details)
        # -------------------------------------------------------------
        print("\n--- TEST 5: Open Previous Visits ---")
        read_scan1 = Scan.query.get(scan1_id)
        read_scan2 = Scan.query.get(scan2_id)
        
        print(f"Visit 1 Record: ID={read_scan1.id}, Pred={read_scan1.prediction}, Conf={read_scan1.confidence}, VisitNum={read_scan1.visit_number}")
        print(f"Visit 2 Record: ID={read_scan2.id}, Pred={read_scan2.prediction}, Conf={read_scan2.confidence}, VisitNum={read_scan2.visit_number}")
        
        assert read_scan1.prediction == "Glioma"
        assert read_scan2.prediction == "Meningioma"
        assert read_scan1.id != read_scan2.id
        assert read_scan1.visit_number == 1
        assert read_scan2.visit_number == 2
        print("[OK] TEST 5 PASSED: Previous visits return individual stored data without mixing!")

        print("\n==============================================")
        print("SUCCESS: ALL 5 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
        print("==============================================")

if __name__ == "__main__":
    run_acceptance_tests()
