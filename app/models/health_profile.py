from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class HealthProfile(Base):
    __tablename__ = "health_profiles"
    
    profile_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.user_id'), unique=True)
    height = Column(Float)
    weight = Column(Float)
    bmi = Column(Float)
    has_diabetes = Column(Boolean, default=False)
    has_hypertension = Column(Boolean, default=False)
    smoking_status = Column(String)
    alcohol_consumption = Column(String)
    exercise_frequency = Column(String)
    family_history = Column(JSON)
    current_medications = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="health_profile")
