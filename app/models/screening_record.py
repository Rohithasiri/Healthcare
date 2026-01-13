from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class ScreeningRecord(Base):
    __tablename__ = "screening_records"
    
    record_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.user_id'))
    screening_date = Column(DateTime, default=datetime.utcnow)
    face_video_path = Column(String)
    finger_video_path = Column(String)
    heart_rate = Column(Float)
    heart_rate_variability = Column(Float)
    estimated_systolic_bp = Column(Float)
    estimated_diastolic_bp = Column(Float)
    ppg_quality_score = Column(Float)
    screening_status = Column(String, default="pending")
    
    user = relationship("User", back_populates="screening_records")
