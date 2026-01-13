from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class CholesterolRecord(Base):
    __tablename__ = "cholesterol_records"
    
    cholesterol_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.user_id'))
    test_date = Column(DateTime)
    report_image_path = Column(String)
    total_cholesterol = Column(Float)
    ldl_cholesterol = Column(Float)
    hdl_cholesterol = Column(Float)
    triglycerides = Column(Float)
    vldl_cholesterol = Column(Float)
    total_hdl_ratio = Column(Float)
    lab_name = Column(String)
    extraction_method = Column(String)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="cholesterol_records")
