import os
from typing import List
from decouple import config

class Settings:
    # API Configuration
    APP_NAME: str = "Breast Cancer MRI Classification API"
    VERSION: str = "1.0.0"
    DEBUG: bool = config("DEBUG", default=False, cast=bool)
    
    # Server Configuration
    HOST: str = config("HOST", default="0.0.0.0")
    PORT: int = config("PORT", default=8000, cast=int)
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = config("MAX_FILE_SIZE", default=10 * 1024 * 1024, cast=int)  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"]
    UPLOAD_DIR: str = config("UPLOAD_DIR", default="uploads")
    
    # Model Configuration
    MODEL_PATH: str = config("MODEL_PATH", default="models/breast_cancer_cnn.pth")
    MODEL_VERSION: str = config("MODEL_VERSION", default="1.0.0")
    BATCH_SIZE: int = config("BATCH_SIZE", default=1, cast=int)
    
    # Image Processing
    TARGET_IMAGE_SIZE: tuple = (224, 224)  # Standard input size for most CNNs
    
    # Logging
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")
    
   
# Create settings instance
settings = Settings()

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True) 