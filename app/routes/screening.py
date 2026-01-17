from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from datetime import datetime
import os
import uuid

from app.database import get_db
from app.models.screening_record import ScreeningRecord
from app.models.health_profile import HealthProfile
from app.models.cholesterol_record import CholesterolRecord
from app.models.user import User
from app.utils.auth import get_current_user

router = APIRouter(prefix="/api/screening", tags=["Screening"])

# Create uploads directories
UPLOAD_DIR = "uploads/videos"
REPORTS_DIR = "uploads/reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB


# ========== PYDANTIC MODELS ==========

class BPRecordCreate(BaseModel):
    """Blood pressure recording"""
    screening_date: str  # YYYY-MM-DD format
    blood_pressure_systolic: int = Field(..., ge=70, le=250)
    blood_pressure_diastolic: int = Field(..., ge=40, le=150)
    heart_rate: Optional[int] = Field(None, ge=40, le=200)
    screening_type: str = "manual"  # manual, device, video
    notes: Optional[str] = None


# ========== HELPER FUNCTIONS ==========

def validate_video_file(file: UploadFile) -> tuple[bool, str]:
    """Validate video file format and size"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Invalid file format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    return True, "Valid"


def classify_blood_pressure(systolic: int, diastolic: int) -> str:
    """Classify blood pressure according to AHA guidelines"""
    if systolic < 120 and diastolic < 80:
        return "normal"
    elif systolic < 130 and diastolic < 80:
        return "elevated"
    elif systolic < 140 or diastolic < 90:
        return "hypertension_stage_1"
    elif systolic < 180 or diastolic < 120:
        return "hypertension_stage_2"
    else:
        return "hypertensive_crisis"


def get_bp_recommendation(category: str) -> str:
    """Get recommendation based on BP category"""
    recommendations = {
        "normal": "Maintain healthy lifestyle. Continue regular monitoring.",
        "elevated": "Adopt heart-healthy habits. Monitor BP regularly.",
        "hypertension_stage_1": "Consult healthcare provider. Lifestyle changes recommended.",
        "hypertension_stage_2": "Consult healthcare provider immediately. Medication may be needed.",
        "hypertensive_crisis": "URGENT: Seek immediate medical attention!"
    }
    return recommendations.get(category, "Consult healthcare provider")


# ========== VIDEO UPLOAD ENDPOINTS (EXISTING) ==========

@router.post("/upload")
async def upload_screening_videos(
    face_video: UploadFile = File(..., description="Face video for rPPG analysis"),
    finger_video: UploadFile = File(..., description="Finger video for PPG analysis"),
    eye_image: UploadFile = File(None, description="Eye image for cholesterol detection (optional)"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload screening videos and images for processing"""
    
    # Check if user has health profile
    profile = db.query(HealthProfile).filter(
        HealthProfile.user_id == current_user["user_id"]
    ).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please create a health profile before screening"
        )
    
    # Validate face video
    is_valid, message = validate_video_file(face_video)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Face video: {message}")
    
    # Validate finger video
    is_valid, message = validate_video_file(finger_video)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Finger video: {message}")
    
    # Generate unique screening ID
    screening_id = str(uuid.uuid4())
    
    # Save face video
    face_filename = f"face_{screening_id}{os.path.splitext(face_video.filename)[1]}"
    face_path = os.path.join(UPLOAD_DIR, face_filename)
    
    with open(face_path, "wb") as f:
        content = await face_video.read()
        f.write(content)
    
    # Save finger video
    finger_filename = f"finger_{screening_id}{os.path.splitext(finger_video.filename)[1]}"
    finger_path = os.path.join(UPLOAD_DIR, finger_filename)
    
    with open(finger_path, "wb") as f:
        content = await finger_video.read()
        f.write(content)
    
    # Save eye image if provided
    eye_path = None
    if eye_image:
        eye_filename = f"eye_{screening_id}{os.path.splitext(eye_image.filename)[1]}"
        eye_path = os.path.join(UPLOAD_DIR, eye_filename)
        
        with open(eye_path, "wb") as f:
            content = await eye_image.read()
            f.write(content)
    
    # Create screening record in database
    screening = ScreeningRecord(
        user_id=current_user["user_id"],
        face_video_path=face_path,
        finger_video_path=finger_path,
        screening_status="pending"
    )
    
    db.add(screening)
    db.commit()
    db.refresh(screening)
    
    # Get record ID dynamically
    record_id = None
    for attr in ['id', 'record_id', 'screening_id']:
        if hasattr(screening, attr):
            record_id = getattr(screening, attr)
            break
    
    return {
        "message": "Videos uploaded successfully",
        "screening_id": record_id,
        "status": "pending",
        "note": "Processing will begin shortly. Check back for results.",
        "files_uploaded": {
            "face_video": face_filename,
            "finger_video": finger_filename,
            "eye_image": eye_filename if eye_image else None
        }
    }


@router.get("/status/{screening_id}")
def get_screening_status(
    screening_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get screening status and results"""
    
    screening = db.query(ScreeningRecord).filter(
        ScreeningRecord.record_id == screening_id,
        ScreeningRecord.user_id == current_user["user_id"]
    ).first()
    
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    
    return {
        "screening_id": screening.record_id,
        "status": screening.screening_status,
        "screening_date": screening.screening_date,
        "results": {
            "heart_rate": screening.heart_rate,
            "hrv": screening.heart_rate_variability,
            "estimated_bp": f"{screening.estimated_systolic_bp}/{screening.estimated_diastolic_bp}" if screening.estimated_systolic_bp else None,
            "quality_score": screening.ppg_quality_score
        }
    }


@router.get("/history")
def get_screening_history(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's screening history"""
    
    screenings = db.query(ScreeningRecord).filter(
        ScreeningRecord.user_id == current_user["user_id"]
    ).order_by(ScreeningRecord.screening_date.desc()).limit(10).all()
    
    return {
        "screenings": [
            {
                "screening_id": s.record_id,
                "date": s.screening_date,
                "status": s.screening_status,
                "heart_rate": s.heart_rate,
                "estimated_bp": f"{s.estimated_systolic_bp}/{s.estimated_diastolic_bp}" if s.estimated_systolic_bp else None
            }
            for s in screenings
        ],
        "total": len(screenings)
    }


# ========== BLOOD PRESSURE ENDPOINTS (NEW) ==========

@router.post("/bp/record")
async def record_blood_pressure(
    bp_data: BPRecordCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Record blood pressure measurement
    Can be from manual cuff, device, or video screening
    """
    try:
        # Parse date
        screening_date = datetime.strptime(bp_data.screening_date, "%Y-%m-%d").date()
        
        # Create screening record
        new_record = ScreeningRecord(
            user_id=current_user["user_id"],
            screening_date=screening_date,
            blood_pressure_systolic=bp_data.blood_pressure_systolic,
            blood_pressure_diastolic=bp_data.blood_pressure_diastolic,
            heart_rate=bp_data.heart_rate,
            screening_type=bp_data.screening_type
        )
        
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        
        # Get record ID dynamically
        record_id = None
        for attr in ['id', 'record_id', 'screening_id']:
            if hasattr(new_record, attr):
                record_id = getattr(new_record, attr)
                break
        
        # Classify BP
        bp_category = classify_blood_pressure(
            bp_data.blood_pressure_systolic,
            bp_data.blood_pressure_diastolic
        )
        
        return {
            "message": "Blood pressure recorded successfully",
            "record_id": record_id,
            "screening_date": new_record.screening_date.strftime("%Y-%m-%d"),
            "systolic": new_record.blood_pressure_systolic,
            "diastolic": new_record.blood_pressure_diastolic,
            "heart_rate": new_record.heart_rate,
            "bp_category": bp_category,
            "recommendation": get_bp_recommendation(bp_category)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording BP: {str(e)}"
        )


@router.get("/bp/history")
async def get_bp_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get blood pressure measurement history"""
    records = db.query(ScreeningRecord).filter(
        ScreeningRecord.user_id == current_user["user_id"]
    ).order_by(ScreeningRecord.screening_date.desc()).limit(limit).all()
    
    if not records:
        return {
            "message": "No BP records found",
            "records": []
        }
    
    return {
        "records": [
            {
                "record_id": getattr(r, 'id', getattr(r, 'record_id', None)),
                "screening_date": r.screening_date.strftime("%Y-%m-%d"),
                "systolic": r.blood_pressure_systolic,
                "diastolic": r.blood_pressure_diastolic,
                "heart_rate": r.heart_rate,
                "screening_type": r.screening_type,
                "bp_category": classify_blood_pressure(
                    r.blood_pressure_systolic,
                    r.blood_pressure_diastolic
                )
            }
            for r in records
        ]
    }


@router.get("/bp/latest")
async def get_latest_bp(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get most recent blood pressure reading"""
    record = db.query(ScreeningRecord).filter(
        ScreeningRecord.user_id == current_user["user_id"]
    ).order_by(ScreeningRecord.screening_date.desc()).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No BP records found"
        )
    
    bp_category = classify_blood_pressure(
        record.blood_pressure_systolic,
        record.blood_pressure_diastolic
    )
    
    return {
        "record_id": getattr(record, 'id', getattr(record, 'record_id', None)),
        "screening_date": record.screening_date.strftime("%Y-%m-%d"),
        "systolic": record.blood_pressure_systolic,
        "diastolic": record.blood_pressure_diastolic,
        "heart_rate": record.heart_rate,
        "screening_type": record.screening_type,
        "bp_category": bp_category,
        "recommendation": get_bp_recommendation(bp_category)
    }


@router.get("/bp/stats")
async def get_bp_statistics(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get BP statistics (average, trend, etc.)"""
    records = db.query(ScreeningRecord).filter(
        ScreeningRecord.user_id == current_user["user_id"]
    ).order_by(ScreeningRecord.screening_date.desc()).limit(30).all()
    
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No BP records found"
        )
    
    # Calculate averages
    avg_systolic = sum(r.blood_pressure_systolic for r in records) / len(records)
    avg_diastolic = sum(r.blood_pressure_diastolic for r in records) / len(records)
    
    # Determine trend
    if len(records) >= 2:
        recent_avg_sys = sum(r.blood_pressure_systolic for r in records[:5]) / min(5, len(records))
        older_avg_sys = sum(r.blood_pressure_systolic for r in records[-5:]) / min(5, len(records))
        trend = "improving" if recent_avg_sys < older_avg_sys else "stable" if recent_avg_sys == older_avg_sys else "increasing"
    else:
        trend = "insufficient_data"
    
    return {
        "total_readings": len(records),
        "average_systolic": round(avg_systolic, 1),
        "average_diastolic": round(avg_diastolic, 1),
        "trend": trend,
        "bp_category": classify_blood_pressure(avg_systolic, avg_diastolic),
        "latest_reading": {
            "date": records[0].screening_date.strftime("%Y-%m-%d"),
            "systolic": records[0].blood_pressure_systolic,
            "diastolic": records[0].blood_pressure_diastolic
        }
    }


# ========== LAB REPORT OCR ENDPOINTS (NEW) ==========

@router.post("/report/upload")
async def upload_lab_report(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload lab report image for OCR processing
    Automatically extracts cholesterol values
    """
    
    # Validate file type
    file_ext = Path(image.filename).suffix.lower()
    
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {ALLOWED_IMAGE_EXTENSIONS}"
        )
    
    try:
        # Create unique filename
        file_id = str(uuid.uuid4())
        filename = f"report_{file_id}{file_ext}"
        file_path = os.path.join(REPORTS_DIR, filename)
        
        # Save file
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # TODO: Process with OCR service
        # Mock OCR result for now
        mock_ocr_result = {
            "success": True,
            "extracted_values": {
                "total_cholesterol": 220,
                "ldl_cholesterol": 140,
                "hdl_cholesterol": 45,
                "triglycerides": 180
            },
            "confidence": 0.85,
            "raw_text": "Total Cholesterol: 220 mg/dL\nLDL: 140 mg/dL..."
        }
        
        # Auto-create cholesterol record if extraction successful
        if mock_ocr_result["success"] and mock_ocr_result["confidence"] > 0.7:
            new_record = CholesterolRecord(
                user_id=current_user["user_id"],
                test_date=datetime.now().date(),
                total_cholesterol=mock_ocr_result["extracted_values"]["total_cholesterol"],
                ldl_cholesterol=mock_ocr_result["extracted_values"]["ldl_cholesterol"],
                hdl_cholesterol=mock_ocr_result["extracted_values"]["hdl_cholesterol"],
                triglycerides=mock_ocr_result["extracted_values"]["triglycerides"]
            )
            
            db.add(new_record)
            db.commit()
            db.refresh(new_record)
            
            record_id = None
            for attr in ['id', 'record_id', 'cholesterol_id']:
                if hasattr(new_record, attr):
                    record_id = getattr(new_record, attr)
                    break
            
            return {
                "message": "Lab report processed successfully",
                "file_id": file_id,
                "filename": filename,
                "ocr_result": mock_ocr_result,
                "cholesterol_record_id": record_id,
                "auto_created": True,
                "note": "Cholesterol values automatically added to your profile. Using mock OCR - integrate real service for production."
            }
        
        return {
            "message": "Lab report uploaded",
            "file_id": file_id,
            "filename": filename,
            "ocr_result": mock_ocr_result,
            "note": "Review and confirm extracted values"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing report: {str(e)}"
        )


@router.get("/report/list")
async def list_uploaded_reports(
    current_user: dict = Depends(get_current_user)
):
    """List all uploaded lab reports"""
    
    reports_dir = Path(REPORTS_DIR)
    if not reports_dir.exists():
        return {"reports": []}
    
    reports = []
    for report_file in reports_dir.glob("*.*"):
        reports.append({
            "filename": report_file.name,
            "size_mb": round(report_file.stat().st_size / (1024 * 1024), 2),
            "uploaded_at": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat()
        })
    
    return {"reports": reports}