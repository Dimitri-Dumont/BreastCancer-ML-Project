# Breast Cancer MRI Classification - End-to-End ML Project

A comprehensive machine learning project that combines **Exploratory Data Analysis (EDA)**, **multimodal deep learning**, and **full-stack web development** to create an intelligent breast cancer classification system using mammography images and clinical data.

## Project Overview

This project implements a complete pipeline from data analysis to production deployment:

1. **📊 Exploratory Data Analysis** - In-depth analysis of the CBIS-DDSM breast cancer dataset
2. **🧠 Multimodal CNN Training** - ResNet-based deep learning model combining image and tabular data  
3. **🌐 Full-Stack Web Application** - React frontend with FastAPI backend for real-time classification

## 🗂️ Project Structure

```
BreastCancer-ML-Project/
├── 📓 EDA & Training Notebooks     # Jupyter notebooks for analysis and training
├── 🤖 model/                       # Model training, checkpoints, and artifacts
│   ├── training/                   # Training scripts and detailed documentation
│   └── checkpoint/                 # Saved model weights and architectures
├── 🚀 api/                         # FastAPI backend service
│   ├── main.py                     # API endpoints and image processing
│   └── requirements.txt            # Backend dependencies
├── 💻 ui/boob/                     # React + Vite frontend
│   ├── src/                        # React components and UI logic
│   └── package.json                # Frontend dependencies
└── 📁 archive/                     # Raw CBIS-DDSM dataset
```

## 🔬 Dataset & Methodology

**Dataset:** [CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data)
- **Size:** 1,636 mammography images (863 benign, 773 malignant)
- **Format:** DICOM and JPEG images with clinical metadata
- **Features:** Combines visual mammography data with clinical tabular features

**Model Architecture:**
- **Baseline Exploration:** Initial experiments with Random Forest classifier on clinical tabular data
- **Deep Learning Approach:** ResNet50 (ImageNet pretrained) for image feature extraction
- **Multimodal Fusion:** Combines image features with clinical data (breast density, subtlety, margins, etc.)
- **Performance:** ~80.5% validation accuracy with multimodal CNN
- **Classification:** Binary (MALIGNANT vs BENIGN)

## ✨ Key Features

### 🔍 **Exploratory Data Analysis & Model Exploration**
- Comprehensive dataset analysis and visualization
- DICOM to JPEG mapping and preprocessing pipeline
- Statistical analysis of clinical features and image distributions
- Initial Random Forest baseline experiments on tabular clinical data

### 🧠 **Advanced ML Model**
- **Multimodal Architecture:** Combines CNN image features with clinical tabular data
- **Transfer Learning:** ResNet50 backbone pretrained on ImageNet
- **Robust Preprocessing:** DICOM handling, image normalization, and feature engineering
- **Production Ready:** Saved model artifacts in multiple formats (.keras, .h5, .weights.h5)

### 🌐 **Full-Stack Web Application**
- **Frontend:** Modern React + Vite SPA with drag-and-drop interface
- **Backend:** FastAPI with async image processing and ML inference
- **User Experience:** Intuitive drag-and-drop MRI image upload
- **Real-time Results:** Instant classification feedback with confidence scores

## 📈 Model Performance

- **Training Accuracy:** ~80.5%
- **Dataset Split:** 80/20 train/test (1,308 train, 328 test samples)
- **Training Time:** ~30 minutes (10 epochs)
- **Architecture:** Multimodal ResNet50 + Dense layers + Dropout regularization

## 🛠️ Technical Stack

**Machine Learning:**
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- scikit-learn for preprocessing
- pydicom for medical image handling

**Backend:**
- FastAPI for high-performance API

**Frontend:**  
- React 19.1.0 with modern hooks
- Vite 6.3.5 for fast development

## 📚 Documentation

- **📖 [Model Training Guide](model/training/README.md)** - Detailed training process and architecture
- **🚀 [API Documentation](api/README.md)** - Backend setup and endpoint details  
- **💻 [Frontend Guide](ui/boob/README.md)** - React app setup and development

## 🎯 Use Cases

- **Clinical Decision Support:** Assist radiologists in mammography interpretation
- **Medical Education:** Training tool for medical students and residents
- **Research Platform:** Framework for experimenting with multimodal medical AI
- **Screening Programs:** Automated pre-screening for high-volume mammography workflows

## 🔮 Future Enhancements

- Cross-validation for more robust model evaluation
- Data augmentation for improved generalization
- Additional fusion strategies (attention mechanisms, late fusion)
- Integration with PACS systems for clinical deployment
- Explainable AI features for clinical interpretability

---

**Dataset Citation:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM) - Digital Database for Screening Mammography
