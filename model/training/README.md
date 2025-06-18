# Multimodal Breast Cancer Classification

A deep learning project that combines mammography images with clinical tabular data for breast cancer classification using a multimodal ResNet-based architecture.

## Project Overview

This project implements a multimodal machine learning approach to classify breast masses as **MALIGNANT** or **BENIGN** using the CBIS-DDSM dataset. The model currently achieves ~80% validation accuracy by combining visual features from mammography images with clinical tabular features.

## Data Preprocessing

### Image Processing Pipeline
- **DICOM to JPEG Mapping**: Created a extensive mapping system using `dicom_info.csv` to convert DICOM UIDs to corresponding JPEG file paths
- **Image Loading**: Loaded 1,636 mammography images successfully 
- **Preprocessing**: Resized images to 224×224×3, normalized pixel values to [0,1] range
- **Format Handling**: Support for both DICOM and JPEG formats with robust error handling

### Tabular Data Integration
- **Feature Selection**: Selected relevant clinical features: `breast_density`, `subtlety`, `left_or_right_breast`, `mass_margins`
- **Encoding**: Applied one-hot encoding to categorical features, resulting in 23-dimensional tabular feature vectors
- **Data Alignment**: Ensured perfect alignment between images, tabular features, and labels

## Model Architecture

### Multimodal ResNet-Based Design
```
┌─────────────────┐    ┌─────────────────┐
│   Image Input   │    │  Tabular Input  │
│   (224×224×3)   │    │     (23,)       │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
    ┌─────▼─────┐           ┌────▼────┐
    │  ResNet50 │           │ Dense   │
    │(pretrained)│           │ (64→32) │
    └─────┬─────┘           └────┬────┘
          │                      │
    ┌─────▼─────┐                │
    │Global Avg │                │
    │   Pool    │                │
    └─────┬─────┘                │
          │                      │
          └─────────┬────────────┘
                    │
              ┌─────▼─────┐
              │Concatenate│
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │Dense (64) │
              │+ Dropout  │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │ Sigmoid   │
              │(Binary)   │
              └───────────┘
```

**Key Components:**
- **Image Branch**: ResNet50 (ImageNet pretrained) + Global Average Pooling
- **Tabular Branch**: Dense layers (64→32 neurons) with dropout
- **Fusion Layer**: Concatenation of image and tabular features
- **Classification Head**: Dense layer with sigmoid activation for binary classification

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Download CBIS-DDSM dataset from Kaggle
# Note: You'll need kaggle CLI configured with your API credentials

# Install kaggle CLI if not already installed
pip install kaggle

# Download and extract dataset
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset
unzip cbis-ddsm-breast-cancer-image-dataset.zip -d archive/
```

**Alternative Manual Download:**
1. Visit: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data
2. Download the dataset zip file
3. Extract to `archive/` directory in your project root

### 4. Run the Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run MultiModal_PreProc_Training.ipynb
# Execute cells sequentially from top to bottom
```

## Training Results

- **Dataset Size**: 1,636 samples (863 benign, 773 malignant)
- **Train/Test Split**: 80/20 (1,308 train, 328 test)
- **Final Validation Accuracy**: ~80.5%
- **Training Time**: ~30 minutes (10 epochs)

## Model Outputs

After training, the following files are saved in the `BreastCancer-ML-Project/model/checkpoint` directory:
- `multimodal_breast_cancer_weights_*.weights.h5` - Model weights only
- `multimodal_breast_cancer_model_*.h5` - Complete model (legacy format)
- `multimodal_breast_cancer_model_*.keras` - Complete model (Keras 3 format) recommended for model deployment
- `training_history_*.pkl` - Training metrics and loss curves

## Requirements

See `requirements.txt` for complete dependency list. Key requirements:
- Python ≥3.8
- TensorFlow ≥2.13.0
- OpenCV ≥4.8.0
- scikit-learn ≥1.3.0
- pandas ≥2.0.0
- pydicom ≥2.4.0

## Dataset Citation

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- Original dataset from Digital Database for Screening Mammography (DDSM)
- Preprocessed and curated version available on Kaggle
- Used for research and educational purposes

## Next Steps

- Experiment with different fusion strategies
- Add data augmentation for improved generalization
- Implement cross-validation for more robust evaluation
- Deploy model using Flask/FastAPI for clinical decision support
