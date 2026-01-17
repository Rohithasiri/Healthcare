"""
rPPG (Remote Photoplethysmography) Service
Analyzes video to extract heart rate and cardiovascular signals
Primary: PhysNet from rPPG-Toolbox
Backup: FFT-based analysis
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import primary processor
try:
    from rppg_processor import RPPGProcessor
    PRIMARY_AVAILABLE = True
    logger.info("Primary rPPG processor (PhysNet) available")
except ImportError as e:
    logger.warning(f"Primary processor not available: {e}")
    PRIMARY_AVAILABLE = False

# Try to import backup processor
try:
    from backup_processor import BackupRPPGProcessor
    BACKUP_AVAILABLE = True
    logger.info("Backup rPPG processor available")
except ImportError as e:
    logger.warning(f"Backup processor not available: {e}")
    BACKUP_AVAILABLE = False

if not PRIMARY_AVAILABLE and not BACKUP_AVAILABLE:
    logger.error("No rPPG processors available!")

app = FastAPI(
    title="rPPG Service",
    description="Remote Photoplethysmography analysis service using PhysNet",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
primary_processor = None
backup_processor = None

if PRIMARY_AVAILABLE:
    try:
        # Try to load pre-trained model if available
        model_path = os.getenv('PHYSNET_MODEL_PATH', None)
        primary_processor = RPPGProcessor(model_path=model_path)
        logger.info("Primary processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize primary processor: {e}")
        PRIMARY_AVAILABLE = False

if BACKUP_AVAILABLE:
    try:
        backup_processor = BackupRPPGProcessor()
        logger.info("Backup processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize backup processor: {e}")
        BACKUP_AVAILABLE = False


class rPPGRequest(BaseModel):
    """Request model for rPPG analysis"""
    video_url: Optional[str] = None
    user_id: Optional[str] = None


class rPPGResponse(BaseModel):
    """Response model for rPPG analysis"""
    success: bool
    heart_rate: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    hrv_pnn50: Optional[float] = None
    estimated_systolic_bp: Optional[float] = None
    estimated_diastolic_bp: Optional[float] = None
    signal_quality_score: Optional[float] = None
    message: str
    method: Optional[str] = None
    fps: Optional[float] = None
    frames_processed: Optional[int] = None


def validate_video_file(filename: str) -> bool:
    """Validate video file format"""
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "rPPG Service",
        "version": "1.0.0",
        "status": "running",
        "primary_processor": PRIMARY_AVAILABLE,
        "backup_processor": BACKUP_AVAILABLE
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if (PRIMARY_AVAILABLE or BACKUP_AVAILABLE) else "unhealthy"
    return {
        "status": status,
        "service": "rppg",
        "primary_available": PRIMARY_AVAILABLE,
        "backup_available": BACKUP_AVAILABLE
    }


@app.post("/analyze", response_model=rPPGResponse)
async def analyze_rppg(request: rPPGRequest):
    """
    Analyze video for rPPG signals (from URL)
    
    Args:
        request: rPPG request with video URL
        
    Returns:
        rPPG analysis results
    """
    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    
    # TODO: Download video from URL and process
    # For now, return error
    raise HTTPException(
        status_code=501,
        detail="URL-based analysis not yet implemented. Use /analyze-video endpoint."
    )


@app.post("/analyze-video", response_model=rPPGResponse)
async def analyze_video_file(file: UploadFile = File(...)):
    """
    Analyze uploaded video file for rPPG signals
    
    Args:
        file: Video file to analyze (mp4, avi, mov formats)
        
    Returns:
        rPPG analysis results with heart rate, HRV, BP estimates, and quality score
    """
    # Validate file format
    if not validate_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: mp4, avi, mov, mkv, webm. Got: {Path(file.filename).suffix}"
        )
    
    # Save uploaded file to temporary location
    temp_file = None
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file = tmp.name
            
            # Write uploaded content
            content = await file.read()
            tmp.write(content)
            tmp.flush()
        
        logger.info(f"Saved uploaded file to: {temp_file} ({len(content)} bytes)")
        
        # Process video with primary processor first
        result = None
        method_used = None
        
        if PRIMARY_AVAILABLE and primary_processor:
            try:
                logger.info("Attempting analysis with primary processor (PhysNet)")
                result = primary_processor.process_video(temp_file)
                method_used = "physnet"
                logger.info("Primary processor succeeded")
            except Exception as e:
                logger.warning(f"Primary processor failed: {e}. Trying backup...")
                result = None
        
        # Fallback to backup processor
        if result is None and BACKUP_AVAILABLE and backup_processor:
            try:
                logger.info("Attempting analysis with backup processor (FFT)")
                result = backup_processor.process_video(temp_file)
                method_used = result.get('method', 'backup_fft')
                logger.info("Backup processor succeeded")
            except Exception as e:
                logger.error(f"Backup processor also failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Video processing failed: {str(e)}"
                )
        
        if result is None:
            raise HTTPException(
                status_code=503,
                detail="No rPPG processors available"
            )
        
        # Check signal quality
        quality_score = result.get('signal_quality_score', 0.0)
        if quality_score < 0.3:
            logger.warning(f"Low signal quality detected: {quality_score}")
        
        # Check for face detection issues
        frames_processed = result.get('frames_processed', 0)
        if frames_processed < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient frames with detected faces: {frames_processed}. "
                       "Ensure good lighting and face visibility."
            )
        
        # Build response
        response = rPPGResponse(
            success=result.get('success', True),
            heart_rate=result.get('heart_rate'),
            hrv_sdnn=result.get('hrv_sdnn'),
            hrv_rmssd=result.get('hrv_rmssd'),
            hrv_pnn50=result.get('hrv_pnn50'),
            estimated_systolic_bp=result.get('estimated_systolic_bp'),
            estimated_diastolic_bp=result.get('estimated_diastolic_bp'),
            signal_quality_score=quality_score,
            method=method_used,
            fps=result.get('fps'),
            frames_processed=frames_processed,
            message="Analysis completed successfully"
        )
        
        logger.info(f"Analysis complete: HR={response.heart_rate}, Quality={quality_score:.2f}")
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        error_msg = str(e)
        
        if "too short" in error_msg.lower():
            raise HTTPException(status_code=400, detail=error_msg)
        elif "face" in error_msg.lower() or "detected" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Face detection failed: {error_msg}. "
                       "Ensure good lighting, face visibility, and stable camera position."
            )
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    
    except Exception as e:
        logger.error(f"Unexpected error during video analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
