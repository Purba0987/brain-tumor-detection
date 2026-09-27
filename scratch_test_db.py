import os
from dotenv import load_dotenv
from flask import Flask
from models_db import db, User, Scan, ChatLog
import sqlalchemy

load_dotenv()

app = Flask(__name__)
database_url = os.environ.get('DATABASE_URL')

def init_db(app):
    if database_url:
        try:
            # Test PostgreSQL connection first
            test_engine = sqlalchemy.create_engine(database_url, connect_args={'connect_timeout': 3})
            with test_engine.connect() as conn:
                pass
            app.config['SQLALCHEMY_DATABASE_URI'] = database_url
            print("Successfully connected to PostgreSQL database!")
        except Exception as e:
            print("PostgreSQL connection refused or unavailable. Falling back to SQLite database.")
            app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tumorai.db'
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tumorai.db'

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database schema initialized successfully on:", app.config['SQLALCHEMY_DATABASE_URI'])

init_db(app)
