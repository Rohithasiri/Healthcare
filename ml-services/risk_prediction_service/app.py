"""
Risk Prediction Service
Predicts cardiovascular disease risk based on multiple factors
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

app = FastAPI(
    title="Risk Prediction Service",
    description="Cardiovascular disease risk prediction service",
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


class RiskFactors(BaseModel):
    """Risk factors model"""
    age: int
    gender: str
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    cholesterol_total: Optional[float] = None
    cholesterol_hdl: Optional[float] = None
    cholesterol_ldl: Optional[float] = None
    glucose: Optional[float] = None
    bmi: Optional[float] = None
    smoking: bool = False
    diabetes: bool = False
    family_history: bool = False
    heart_rate: Optional[float] = None
    vessel_density: Optional[float] = None
    av_ratio: Optional[float] = None


class RiskPredictionRequest(BaseModel):
    """Request model for risk prediction"""
    user_id: Optional[str] = None
    factors: RiskFactors


class RiskPredictionResponse(BaseModel):
    """Response model for risk prediction"""
    success: bool
    risk_score: float
    risk_category: str
    factors_contributing: List[str]
    recommendations: List[str]
    message: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Risk Prediction Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "risk_prediction"}


@app.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(request: RiskPredictionRequest):
    """
    Predict cardiovascular disease risk
    
    Args:
        request: Risk prediction request with factors
        
    Returns:
        Risk prediction results
    """
    try:
        factors = request.factors
        
        # TODO: Implement ML model inference
        # This is a placeholder calculation
        risk_score = 0.0
        
        # Simple risk calculation (placeholder)
        if factors.age > 50:
            risk_score += 0.1
        if factors.blood_pressure_systolic > 140:
            risk_score += 0.15
        if factors.cholesterol_total and factors.cholesterol_total > 200:
            risk_score += 0.1
        if factors.smoking:
            risk_score += 0.15
        if factors.diabetes:
            risk_score += 0.2
        if factors.family_history:
            risk_score += 0.1
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk category
        if risk_score < 0.2:
            category = "Low"
        elif risk_score < 0.5:
            category = "Moderate"
        elif risk_score < 0.75:
            category = "High"
        else:
            category = "Very High"
        
        # Generate recommendations
        recommendations = []
        if factors.blood_pressure_systolic > 140:
            recommendations.append("Monitor blood pressure regularly")
        if factors.cholesterol_total and factors.cholesterol_total > 200:
            recommendations.append("Consider cholesterol-lowering diet")
        if factors.smoking:
            recommendations.append("Quit smoking to reduce risk")
        if factors.bmi and factors.bmi > 25:
            recommendations.append("Maintain healthy weight through diet and exercise")
        
        return RiskPredictionResponse(
            success=True,
            risk_score=round(risk_score, 3),
            risk_category=category,
            factors_contributing=["Age", "Blood Pressure", "Cholesterol"],
            recommendations=recommendations,
            message="Risk prediction completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
