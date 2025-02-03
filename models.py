# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    password = db.Column(db.String(200), nullable=False)
    birthday = db.Column(db.String(10), nullable=False)
    profile_pic = db.Column(db.String(150), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class Plant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_name = db.Column(db.String(150), unique=True, nullable=False)
    scientific_name = db.Column(db.String(150), nullable=False)
    common_uses = db.Column(db.Text, nullable=False)
    phytochemicals = db.Column(db.Text, nullable=False)

class SystemStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    csv_imported = db.Column(db.Boolean, default=False)


class PhytochemicalCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    molecular_formula = db.Column(db.String(50))
    molecular_weight = db.Column(db.Float)
    canonical_smiles = db.Column(db.String(500))
    last_updated = db.Column(db.DateTime, default=datetime.now(timezone.utc))

class UserFormulation(db.Model):
    """New Table to Store User Formulations from the Scent Route"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ingredients = db.Column(db.Text, nullable=False)  # Store list of ingredients as a string
    compatibility_score = db.Column(db.Integer, nullable=False)  # Store compatibility score
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp