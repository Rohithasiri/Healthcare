from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# Create database tables
Base.metadata.create_all(bind=engine)

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

# Include routers
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(screening_router)

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
