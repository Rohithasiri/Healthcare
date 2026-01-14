from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.health_profile import HealthProfile
from app.models.user import User
from app.utils.auth import get_current_user
from typing import List
from datetime import datetime

router = APIRouter(prefix="/api/profile", tags=["Health Profile"])

class HealthProfileCreate(BaseModel):
    height: float
    weight: float
    has_diabetes: bool = False
    has_hypertension: bool = False
    smoking_status: str
    alcohol_consumption: str
    exercise_frequency: str
    family_history: List[str] = []
    current_medications: List[str] = []

@router.post("/create")
def create_health_profile(
    profile_data: HealthProfileCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    existing_profile = db.query(HealthProfile).filter(
        HealthProfile.user_id == current_user["user_id"]
    ).first()
    
    height_m = profile_data.height / 100
    bmi = profile_data.weight / (height_m ** 2)
    
    if existing_profile:
        existing_profile.height = profile_data.height
        existing_profile.weight = profile_data.weight
        existing_profile.bmi = round(bmi, 2)
        existing_profile.has_diabetes = profile_data.has_diabetes
        existing_profile.has_hypertension = profile_data.has_hypertension
        existing_profile.smoking_status = profile_data.smoking_status
        existing_profile.alcohol_consumption = profile_data.alcohol_consumption
        existing_profile.exercise_frequency = profile_data.exercise_frequency
        existing_profile.family_history = profile_data.family_history
        existing_profile.current_medications = profile_data.current_medications
        existing_profile.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(existing_profile)
        
        return {
            "message": "Health profile updated successfully",
            "profile_id": existing_profile.profile_id,
            "bmi": existing_profile.bmi
        }
    else:
        new_profile = HealthProfile(
            user_id=current_user["user_id"],
            height=profile_data.height,
            weight=profile_data.weight,
            bmi=round(bmi, 2),
            has_diabetes=profile_data.has_diabetes,
            has_hypertension=profile_data.has_hypertension,
            smoking_status=profile_data.smoking_status,
            alcohol_consumption=profile_data.alcohol_consumption,
            exercise_frequency=profile_data.exercise_frequency,
            family_history=profile_data.family_history,
            current_medications=profile_data.current_medications
        )
        
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        
        return {
            "message": "Health profile created successfully",
            "profile_id": new_profile.profile_id,
            "bmi": new_profile.bmi
        }

@router.get("/get")
def get_health_profile(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(HealthProfile).filter(
        HealthProfile.user_id == current_user["user_id"]
    ).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Health profile not found"
        )
    
    user = db.query(User).filter(User.user_id == current_user["user_id"]).first()
    age = None
    if user and user.date_of_birth:
        age = (datetime.utcnow() - user.date_of_birth).days // 365
    
    return {
        "profile_id": profile.profile_id,
        "height": profile.height,
        "weight": profile.weight,
        "bmi": profile.bmi,
        "age": age,
        "has_diabetes": profile.has_diabetes,
        "has_hypertension": profile.has_hypertension,
        "smoking_status": profile.smoking_status,
        "alcohol_consumption": profile.alcohol_consumption,
        "exercise_frequency": profile.exercise_frequency,
        "family_history": profile.family_history,
        "current_medications": profile.current_medications,
        "updated_at": profile.updated_at
    }
