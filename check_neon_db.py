import os
from dotenv import load_dotenv
from flask import Flask
from models_db import db, User, Scan

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    users = User.query.all()
    scans = Scan.query.all()
    print(f"Total Users in Neon DB: {len(users)}")
    for u in users:
        print(f" - User: {u.username} ({u.email}), Role: {u.role}")
    print(f"Total Scans in Neon DB: {len(scans)}")
    for s in scans:
        print(f" - Scan #{s.id}: {s.patient_name} -> {s.prediction}")
