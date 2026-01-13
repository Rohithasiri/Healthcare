from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    phone_number = Column(String)
    date_of_birth = Column(DateTime)
    gender = Column(String)
    user_type = Column(String, default="patient")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    health_profile = relationship("HealthProfile", back_populates="user", uselist=False)
    screening_records = relationship("ScreeningRecord", back_populates="user")
    cholesterol_records = relationship("CholesterolRecord", back_populates="user")
    risk_assessments = relationship("RiskAssessment", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
