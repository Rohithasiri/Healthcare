from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.database import engine, Base

# Import models
from app.models.user import User
from app.models.health_profile import HealthProfile
from app.models.screening_record import ScreeningRecord
from app.models.cholesterol_record import CholesterolRecord
from app.models.risk_assessment import RiskAssessment
from app.models.alert import Alert

# Import routes
from app.routes.auth import router as auth_router
from app.routes.profile import router as profile_router
from app.routes.screening import router as screening_router
from app.routes.risk import router as risk_router  # ← ADD THIS LINE

# Create database tables
Base.metadata.create_all(bind=engine)

# Create uploads directory if it doesn't exist
os.makedirs("uploads/videos", exist_ok=True)
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/reports", exist_ok=True)

app = FastAPI(
    title="Heart Screening API",
    description="AI-powered cardiovascular risk screening system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(screening_router)
app.include_router(risk_router)  # ← ADD THIS LINE

@app.get("/")
def root():
    return {
        "message": "Heart Screening API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected"}


@app.get("/api/services/status")
async def check_ml_services():
    """Check status of all ML microservices"""
    from app.services.ml_client import ml_client
    
    try:
        status = await ml_client.get_all_services_status()
        return {
            "backend": "healthy",
            "microservices": status,
            "all_services_up": all(status.values())
        }
    except Exception as e:
        return {
            "backend": "healthy",
            "microservices": {
                "rppg": False,
                "ocr": False,
                "eye_analysis": False,
                "risk_prediction": False
            },
            "all_services_up": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)