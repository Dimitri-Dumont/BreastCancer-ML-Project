from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
from typing import Dict, Any, List
import logging

from config import settings
from model_service import model_service

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

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("Starting up Breast Cancer MRI Classification API")
    await model_service.load_model()


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process MRI image for breast cancer classification
    
    Returns:
    - Original image
    - Processed image with bounding boxes
    - Classification results
    - Confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        contents = await file.read()
        
        if not validate_image(contents, settings.MAX_FILE_SIZE):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image or file too large (max {settings.MAX_FILE_SIZE // (1024*1024)}MB)"
            )
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {file.filename}, Size: {image.size}")
        
        # Process with model
        result = await model_service.predict_single_image(image, file.filename)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process-batch")
async def process_batch_images(files: List[UploadFile] = File(...)):
    """
    Process multiple MRI images in batch
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    images_to_process = []
    
    # Validate all files first
    for file in files:
        try:
            if not file.content_type.startswith("image/"):
                continue
                
            contents = await file.read()
            
            if not validate_image(contents, settings.MAX_FILE_SIZE):
                continue
                
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            images_to_process.append((image, file.filename))
            
        except Exception as e:
            logger.error(f"Error validating {file.filename}: {str(e)}")
            continue
    
    if not images_to_process:
        raise HTTPException(status_code=400, detail="No valid images found in batch")
    
    # Process all images
    results = await model_service.predict_batch_images(images_to_process)
    
    return {
        "batch_results": results, 
        "processed_count": len(results),
        "total_submitted": len(files)
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