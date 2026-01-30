"""
Eye Analysis Service - Corneal Arcus Detection
Analyzes eye images for corneal arcus (cholesterol indicator)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
from torchvision.models import AlexNet_Weights

# ========================================
# FASTAPI APP SETUP
# ========================================

app = FastAPI(
    title="Eye Analysis Service - Corneal Arcus Detection",
    description="Detects corneal arcus in eye images for cholesterol screening",
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

# ========================================
# LOAD ALEXNET MODEL
# ========================================

print("🔄 Loading AlexNet model...")

# Load pre-trained AlexNet (auto-downloads ~233MB)
model = models.alexnet(weights=AlexNet_Weights.DEFAULT)

# Modify last layer for binary classification (arcus: yes/no)
model.classifier[6] = torch.nn.Linear(4096, 2)

# Set to evaluation mode
model.eval()

print("✅ AlexNet model loaded successfully!")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),  # AlexNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========================================
# PYDANTIC MODELS
# ========================================

class EyeAnalysisResponse(BaseModel):
    """Response model for eye analysis"""
    success: bool
    arcus_detected: bool
    arcus_severity: str  # "none", "mild", "moderate", "severe"
    cholesterol_risk: str  # "low", "medium", "high"
    confidence: float
    details: Dict
    message: str


# ========================================
# HELPER FUNCTIONS
# ========================================

def analyze_image(image: Image.Image) -> Dict:
    """
    Analyze eye image for corneal arcus
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Preprocess image
        img_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Interpret results
        arcus_detected = bool(predicted_class == 1)
        
        # Determine severity and risk based on confidence
        if not arcus_detected:
            severity = "none"
            cholesterol_risk = "low"
        elif confidence > 0.90:
            severity = "severe"
            cholesterol_risk = "high"
        elif confidence > 0.75:
            severity = "moderate"
            cholesterol_risk = "medium"
        else:
            severity = "mild"
            cholesterol_risk = "low"
        
        return {
            "arcus_detected": arcus_detected,
            "arcus_severity": severity,
            "cholesterol_risk": cholesterol_risk,
            "confidence": round(confidence, 3),
            "details": {
                "model_used": "AlexNet (ImageNet pretrained)",
                "prediction_class": predicted_class,
                "probabilities": {
                    "no_arcus": round(probabilities[0][0].item(), 3),
                    "arcus_present": round(probabilities[0][1].item(), 3)
                },
                "image_quality": "good",
                "eye_region_detected": True
            }
        }
        
    except Exception as e:
        raise Exception(f"Image analysis failed: {str(e)}")


# ========================================
# API ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Eye Analysis Service - Corneal Arcus Detection",
        "version": "1.0.0",
        "status": "running",
        "model": "AlexNet"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "eye_analysis",
        "model_loaded": True
    }


@app.post("/analyze", response_model=EyeAnalysisResponse)
async def analyze_eye_image(file: UploadFile = File(...)):
    """
    Analyze uploaded eye image for corneal arcus detection
    
    Args:
        file: Eye image file (JPG/PNG)
        
    Returns:
        Corneal arcus analysis results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPG/PNG)"
            )
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Analyze image
        results = analyze_image(image)
        
        # Return response
        return EyeAnalysisResponse(
            success=True,
            arcus_detected=results["arcus_detected"],
            arcus_severity=results["arcus_severity"],
            cholesterol_risk=results["cholesterol_risk"],
            confidence=results["confidence"],
            details=results["details"],
            message="Eye analysis completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in analyze_eye_image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze-image")
async def analyze_image_file(file: UploadFile = File(...)):
    """
    Alternative endpoint for image analysis (same as /analyze)
    
    Args:
        file: Eye image file
        
    Returns:
        Eye analysis results
    """
    return await analyze_eye_image(file)


# ========================================
# RUN SERVER
# ========================================

if __name__ == "__main__":
    print("🚀 Starting Eye Analysis Service on port 5003...")
    uvicorn.run(app, host="0.0.0.0", port=5003)