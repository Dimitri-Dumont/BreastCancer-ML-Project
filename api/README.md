# Breast Cancer MRI Classification API

A FastAPI-based REST API for processing MRI images and detecting breast cancer using CNN models. This API accepts MRI images, processes them through a trained CNN model, and returns classification results along with annotated images showing suspicious regions with bounding boxes.

## Features

- ✅ Image upload and validation
- ✅ Single image processing
- ✅ Batch image processing
- ✅ Bounding box detection and annotation
- ✅ Logging and monitoring

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
- **POST** `/process-batch` - Process multiple images in batch (max 10)

#### Model Management (TODO: optional extra mile haven't implemented)
- **GET** `/model-info` - Get model information and metadata
- **POST** `/model/reload` - Reload the model (useful for updates)

### Usage Examples

#### Single Image Upload

```bash
curl -X POST "http://localhost:8000/upload-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/mri_image.jpg"
```

#### Response Format

```json
{
  "filename": "mri_image.jpg",
  "image_size": [512, 512],
  "original_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "processed_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "classification": {
    "predicted_class": "benign",
    "confidence": 0.85,
    "classes": {
      "benign": 0.85,
      "malignant": 0.15
    }
  },
  "bounding_boxes": [
    {
      "x": 150,
      "y": 100,
      "width": 80,
      "height": 60,
      "confidence": 0.92,
      "class": "suspicious_region"
    }
  ],
  "metadata": {
    "processing_time_ms": 150,
    "model_version": "v1.0.0",
    "timestamp": "2024-01-01T12:00:00Z",
    "num_detections": 1
  }
}
```

## Integration with Your CNN Model

The API is currently set up with mock predictions. To be eventually replaced with our model
