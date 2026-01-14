from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.screening_record import ScreeningRecord
from app.models.health_profile import HealthProfile
from app.models.user import User
from app.utils.auth import get_current_user
import os
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/screening", tags=["Screening"])

# Create uploads directory
UPLOAD_DIR = "uploads/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

def validate_video_file(file: UploadFile) -> tuple[bool, str]:
    """Validate video file format and size"""
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Invalid file format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    
    # Note: Size check would need to read the file, skip for now
    # In production, implement proper size checking
    
    return True, "Valid"

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
    
    return {
        "message": "Videos uploaded successfully",
        "screening_id": screening.record_id,
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
