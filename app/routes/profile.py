from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.health_profile import HealthProfile
from app.models.cholesterol_record import CholesterolRecord
from app.models.user import User
from app.utils.auth import get_current_user
from typing import List, Optional
from datetime import datetime, date

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

class CholesterolRecordCreate(BaseModel):
    test_date: str  # YYYY-MM-DD format
    total_cholesterol: float
    ldl_cholesterol: Optional[float] = None
    hdl_cholesterol: Optional[float] = None
    triglycerides: Optional[float] = None
    # Removed test_location and notes - not in your model

@router.post("/create")
def create_health_profile(
    profile_data: HealthProfileCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update health profile"""
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
    """Get current user's health profile"""
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


# ========== CHOLESTEROL ENDPOINTS ==========

@router.post("/cholesterol/add")
def add_cholesterol_record(
    record_data: CholesterolRecordCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new cholesterol test record"""
    try:
        # Parse the test date
        test_date = datetime.strptime(record_data.test_date, "%Y-%m-%d").date()
        
        # Create new cholesterol record (only with fields that exist in model)
        new_record = CholesterolRecord(
            user_id=current_user["user_id"],
            test_date=test_date,
            total_cholesterol=record_data.total_cholesterol,
            ldl_cholesterol=record_data.ldl_cholesterol,
            hdl_cholesterol=record_data.hdl_cholesterol,
            triglycerides=record_data.triglycerides
        )
        
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        
        # Get the ID field name dynamically (could be 'id', 'record_id', 'cholesterol_id', etc.)
        id_field = None
        for attr in ['id', 'record_id', 'cholesterol_id']:
            if hasattr(new_record, attr):
                id_field = getattr(new_record, attr)
                break
        
        return {
            "message": "Cholesterol record added successfully",
            "record_id": id_field,
            "test_date": new_record.test_date.strftime("%Y-%m-%d"),
            "total_cholesterol": new_record.total_cholesterol,
            "hdl_cholesterol": new_record.hdl_cholesterol,
            "ldl_cholesterol": new_record.ldl_cholesterol
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding cholesterol record: {str(e)}"
        )


@router.get("/cholesterol/history")
def get_cholesterol_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get cholesterol test history for current user"""
    records = db.query(CholesterolRecord).filter(
        CholesterolRecord.user_id == current_user["user_id"]
    ).order_by(CholesterolRecord.test_date.desc()).limit(limit).all()
    
    if not records:
        return {
            "message": "No cholesterol records found",
            "records": []
        }
    
    return {
        "records": [
            {
                "record_id": getattr(r, 'id', getattr(r, 'record_id', None)),
                "test_date": r.test_date.strftime("%Y-%m-%d"),
                "total_cholesterol": r.total_cholesterol,
                "ldl_cholesterol": r.ldl_cholesterol,
                "hdl_cholesterol": r.hdl_cholesterol,
                "triglycerides": r.triglycerides,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in records
        ]
    }


@router.get("/cholesterol/latest")
def get_latest_cholesterol(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the most recent cholesterol test record"""
    record = db.query(CholesterolRecord).filter(
        CholesterolRecord.user_id == current_user["user_id"]
    ).order_by(CholesterolRecord.test_date.desc()).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No cholesterol records found"
        )
    
    return {
        "record_id": getattr(record, 'id', getattr(record, 'record_id', None)),
        "test_date": record.test_date.strftime("%Y-%m-%d"),
        "total_cholesterol": record.total_cholesterol,
        "ldl_cholesterol": record.ldl_cholesterol,
        "hdl_cholesterol": record.hdl_cholesterol,
        "triglycerides": record.triglycerides,
        "created_at": record.created_at.isoformat() if record.created_at else None
    }


@router.delete("/cholesterol/{record_id}")
def delete_cholesterol_record(
    record_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a cholesterol record"""
    record = db.query(CholesterolRecord).filter(
        CholesterolRecord.record_id == record_id,
        CholesterolRecord.user_id == current_user["user_id"]
    ).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cholesterol record not found"
        )
    
    db.delete(record)
    db.commit()
    
    return {"message": "Cholesterol record deleted successfully"}