from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    
    assessment_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.user_id'))
    screening_id = Column(Integer, ForeignKey('screening_records.record_id'), nullable=True)
    cholesterol_id = Column(Integer, ForeignKey('cholesterol_records.cholesterol_id'), nullable=True)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float)
    risk_category = Column(String)
    contributing_factors = Column(JSON)
    bp_risk_component = Column(Float)
    cholesterol_risk_component = Column(Float)
    lifestyle_risk_component = Column(Float)
    family_history_risk_component = Column(Float)
    model_confidence = Column(Float)
    
    user = relationship("User", back_populates="risk_assessments")
