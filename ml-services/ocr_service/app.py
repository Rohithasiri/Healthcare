"""
OCR Service
Extracts text from medical documents and reports using OCR
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn

app = FastAPI(
    title="OCR Service",
    description="Optical Character Recognition service for medical documents",
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


class OCRRequest(BaseModel):
    """Request model for OCR"""
    image_url: Optional[str] = None
    document_type: Optional[str] = "general"
    user_id: Optional[str] = None


class OCRResponse(BaseModel):
    """Response model for OCR"""
    success: bool
    extracted_text: Optional[str] = None
    structured_data: Optional[Dict] = None
    confidence: Optional[float] = None
    message: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "OCR Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ocr"}


@app.post("/extract", response_model=OCRResponse)
async def extract_text(request: OCRRequest):
    """
    Extract text from document image
    
    Args:
        request: OCR request with image URL
        
    Returns:
        Extracted text and structured data
    """
    try:
        # TODO: Implement OCR logic using PaddleOCR or Tesseract
        # This is a placeholder response
        return OCRResponse(
            success=True,
            extracted_text="Sample extracted text from document",
            structured_data={
                "blood_pressure": "120/80",
                "cholesterol": "200 mg/dL",
                "glucose": "95 mg/dL"
            },
            confidence=0.92,
            message="Text extraction completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-from-file")
async def extract_from_file(file: UploadFile = File(...)):
    """
    Extract text from uploaded document image
    
    Args:
        file: Image file to process
        
    Returns:
        Extracted text and structured data
    """
    try:
        # TODO: Implement file processing
        # This is a placeholder response
        return {
            "success": True,
            "extracted_text": "Sample extracted text from document",
            "structured_data": {
                "blood_pressure": "120/80",
                "cholesterol": "200 mg/dL",
                "glucose": "95 mg/dL"
            },
            "confidence": 0.92,
            "message": "Text extraction completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
