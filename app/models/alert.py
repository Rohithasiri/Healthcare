from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class Alert(Base):
    __tablename__ = "alerts"
    
    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('users.user_id'))
    alert_type = Column(String)
    title = Column(String)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)
    action_required = Column(Boolean, default=False)
    action_deadline = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="alerts")
