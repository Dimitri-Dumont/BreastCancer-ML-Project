# Breast Cancer MRI Classification API

A FastAPI-based REST API for processing MRI images and detecting breast cancer using CNN models. This API accepts MRI images, processes them through a trained CNN model, and returns classification results 

## Features

- ✅ Image upload and validation
- ✅ Single image processing


## Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On Unix/MacOS
source venv/bin/activate

# Install dependencies

cd BreastCancer-ML-Project/api
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

Create a `.env` file in the api directory for custom configuration:

```env
DEBUG=True
HOST=0.0.0.0
PORT=8000
MAX_FILE_SIZE=10485760
LOG_LEVEL=INFO
MODEL_PATH=models/breast_cancer_cnn.pth
```

### 3. Start the Server

```bash
# Using Python directly
python main.py

# Or using the startup script
python run.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

### Endpoints

#### Image Processing
- **POST** `/upload-image` - Process a single MRI image



```


