from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import sys
import os
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model', 'training'))

from config import settings
from inference import predict_breast_cancer_for_api

# Set up logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="API for processing MRI images and detecting breast cancer with bounding box annotations",
    version=settings.VERSION
)

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    breast_density: int = Form(3, description="Breast density value (1-4)"),
    left_or_right_breast: str = Form("LEFT", description="Either 'LEFT' or 'RIGHT'"),
    subtlety: int = Form(3, description="Subtlety rating (1-5)"),
    mass_margins: str = Form("SPICULATED", description="One of: 'CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'"),
    model_path: Optional[str] = Form(None, description="Path to the trained model (optional)"),
    force_malignant: bool = Form(False, description="Force malignant prediction for testing")
):
    """
    Upload and process MRI image for breast cancer classification
    
    Parameters:
    - file: Image file to analyze
    - breast_density: Breast density value (1-4)
    - left_or_right_breast: Either 'LEFT' or 'RIGHT'
    - subtlety: Subtlety rating (1-5)
    - mass_margins: One of: 'CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'
    - model_path: Optional path to the trained model
    - force_malignant: Force malignant prediction for testing purposes
    
    Returns:
    - Classification results
    - Confidence scores
    - Generated result image (base64 encoded)
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate parameters
        if breast_density not in range(1, 5):
            raise HTTPException(status_code=400, detail="Breast density must be between 1 and 4")
        
        if left_or_right_breast not in ["LEFT", "RIGHT"]:
            raise HTTPException(status_code=400, detail="left_or_right_breast must be 'LEFT' or 'RIGHT'")
        
        if subtlety not in range(1, 6):
            raise HTTPException(status_code=400, detail="Subtlety must be between 1 and 5")
        
        allowed_margins = ['CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED']
        if mass_margins not in allowed_margins:
            raise HTTPException(status_code=400, detail=f"mass_margins must be one of: {allowed_margins}")
        
        # Read and validate image
        contents = await file.read()
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {file.filename}, Size: {image.size}")
        logger.info(f"Parameters - Density: {breast_density}, Side: {left_or_right_breast}, Subtlety: {subtlety}, Margins: {mass_margins}")
        
        # Use default model path if not provided
        if model_path is None or model_path == "":
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoint', 'multimodal_breast_cancer_model_20250618_034854.keras')
        
        # Process with inference model
        try:
            prediction_prob, classification, result_image_base64 = predict_breast_cancer_for_api(
                image=image,
                breast_density=breast_density,
                left_or_right_breast=left_or_right_breast,
                subtlety=subtlety,
                mass_margins=mass_margins,
                model_path=model_path,
                force_malignant=force_malignant
            )
            
            result = {
                "filename": file.filename,
                "image_size": list(image.size),
                "classification": {
                    "predicted_class": classification,
                    "probability": float(prediction_prob),
                    "confidence": float(abs(prediction_prob - 0.5) * 2),
                    "classes": {
                        "BENIGN": float(1 - prediction_prob),
                        "MALIGNANT": float(prediction_prob)
                    }
                },
                "input_parameters": {
                    "breast_density": breast_density,
                    "left_or_right_breast": left_or_right_breast,
                    "subtlety": subtlety,
                    "mass_margins": mass_margins
                },
                "result_image": f"data:image/png;base64,{result_image_base64}",
                "interpretation": {
                    "risk_level": "HIGH" if classification == "MALIGNANT" else "LOW",
                    "recommendation": "Recommend immediate medical consultation" if classification == "MALIGNANT" else "Consider routine follow-up"
                },
                "metadata": {
                    "model_version": settings.MODEL_VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "force_malignant": force_malignant
                }
            }
            
            return JSONResponse(content=result)
            
        except Exception as model_error:
            logger.error(f"Model inference error: {str(model_error)}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(model_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upload-image-simple")
async def upload_image_simple(
    file: UploadFile = File(...),
    force_malignant: bool = Form(False, description="Force malignant prediction for testing")
):
    """
    Simple version of upload-image with default parameters for quick testing
    
    Uses default parameters:
    - breast_density: 3
    - left_or_right_breast: "LEFT"
    - subtlety: 3
    - mass_margins: "SPICULATED"
    """
    return await upload_image(
        file=file,
        breast_density=3,
        left_or_right_breast="LEFT",
        subtlety=3,
        mass_margins="SPICULATED",
        model_path=None,
        force_malignant=force_malignant
    )

@app.get("/")
async def root():
    """API information and health check"""
    return {
        "message": "Breast Cancer MRI Classification API",
        "version": settings.VERSION,
        "endpoints": {
            "/upload-image": "Full endpoint with all parameters",
            "/upload-image-simple": "Simplified endpoint with default parameters",
            "/docs": "API documentation"
        },
        "parameters": {
            "breast_density": "1-4 (integer)",
            "left_or_right_breast": "LEFT or RIGHT (string)",
            "subtlety": "1-5 (integer)",
            "mass_margins": "CIRCUMSCRIBED, ILL_DEFINED, SPICULATED, MICROLOBULATED, or OBSCURED (string)"
        }
    }

def validate_image(contents: bytes, max_size: int) -> bool:
    #TODO: Implement this function
    """
    Validate image contents and size
    
    Args:
        contents: Raw image bytes
        max_size: Maximum allowed file size in bytes
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    pass

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG
    ) 