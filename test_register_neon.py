import os
from dotenv import load_dotenv
from flask import Flask
from models_db import db, User

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    # Create test user if not exists
    if not User.query.filter_by(username='dr_john').first():
        u = User(username='dr_john', email='drjohn@hospital.com', role='Doctor')
        u.set_password('securepassword123')
        db.session.add(u)
        db.session.commit()
        print("Successfully registered test user 'dr_john' into Neon PostgreSQL database!")
    else:
        print("User 'dr_john' already exists in Neon database!")

    # Print all users in Neon DB
    users = User.query.all()
    print(f"Current Users in Neon DB ({len(users)}):")
    for user in users:
        print(f"  - ID: {user.id} | Username: {user.username} | Email: {user.email} | Role: {user.role}")
