import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
import io
import base64

def predict_breast_cancer(image_path, breast_density, left_or_right_breast, subtlety, mass_margins, 
                         model_path="../checkpoint/multimodal_breast_cancer_model_20250618_034854.keras",
                         force_malignant=False):
    """
    Predict breast cancer from mammogram image and tabular data.
    
    Parameters:
    - image_path: str, path to the JPEG mammogram image
    - breast_density: int, breast density value (typically 1-4)
    - left_or_right_breast: str, either 'LEFT' or 'RIGHT'
    - subtlety: int, subtlety rating (typically 1-5)
    - mass_margins: str, one of: 'CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'
    - model_path: str, path to the trained model
    - force_malignant: bool, if True, forces a malignant diagnosis for testing purposes (default: False)
    
    Returns:
    - prediction: float, probability of malignancy (0-1)
    - classification: str, 'BENIGN' or 'MALIGNANT'
    """
    
    # Load the trained model
    print("Loading model...........")
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    print("Processing image...")
    image = load_and_preprocess_image(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Prepare tabular data
    print("Processing tabular data...")
    tabular_data = prepare_tabular_data(breast_density, left_or_right_breast, subtlety, mass_margins)
    
    # Make prediction
    print("Making prediction...")
    if force_malignant:
        print("⚠️  TESTING MODE: Forcing malignant diagnosis")
        prediction_prob = 0.85  # High confidence malignant for testing
        classification = "MALIGNANT"
    else:
        prediction_prob = model.predict([np.expand_dims(image, 0), np.expand_dims(tabular_data, 0)])[0][0]
        classification = "MALIGNANT" if prediction_prob > 0.5 else "BENIGN"
    
    # Display results
    display_results(image_path, prediction_prob, classification, breast_density, left_or_right_breast, subtlety, mass_margins)
    
    return prediction_prob, classification

def predict_breast_cancer_for_api(image, breast_density, left_or_right_breast, subtlety, mass_margins, 
                                 model_path="../checkpoint/multimodal_breast_cancer_model_20250618_034854.keras",
                                 force_malignant=False):
    """
    Predict breast cancer from PIL Image and tabular data for API use.
    
    Parameters:
    - image: PIL Image object
    - breast_density: int, breast density value (typically 1-4)
    - left_or_right_breast: str, either 'LEFT' or 'RIGHT'
    - subtlety: int, subtlety rating (typically 1-5)
    - mass_margins: str, one of: 'CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'
    - model_path: str, path to the trained model
    - force_malignant: bool, if True, forces a malignant diagnosis for testing purposes (default: False)
    
    Returns:
    - prediction: float, probability of malignancy (0-1)
    - classification: str, 'BENIGN' or 'MALIGNANT'
    - image_base64: str, base64 encoded result image
    """
    
    # Load the trained model
    print("Loading model...........")
    model = tf.keras.models.load_model(model_path)
    
    # Convert PIL image to numpy array and preprocess
    print("Processing image...")
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Already RGB, resize and normalize
        image_array = cv2.resize(image_array, (224, 224))
        image_array = image_array / 255.0
        processed_image = image_array.astype(np.float32)
    else:
        raise ValueError("Image must be RGB format")
    
    # Prepare tabular data
    print("Processing tabular data...")
    tabular_data = prepare_tabular_data(breast_density, left_or_right_breast, subtlety, mass_margins)
    
    # Make prediction
    print("Making prediction...")
    if force_malignant:
        print("⚠️  TESTING MODE: Forcing malignant diagnosis")
        prediction_prob = 0.85  # High confidence malignant for testing
        classification = "MALIGNANT"
    else:
        prediction_prob = model.predict([np.expand_dims(processed_image, 0), np.expand_dims(tabular_data, 0)])[0][0]
        classification = "MALIGNANT" if prediction_prob > 0.5 else "BENIGN"
    
    # Generate result image
    result_image_base64 = generate_result_image(image, prediction_prob, classification, 
                                               breast_density, left_or_right_breast, subtlety, mass_margins)
    
    return prediction_prob, classification, result_image_base64

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for model input"""
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        return image.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def prepare_tabular_data(breast_density, left_or_right_breast, subtlety, mass_margins):
    """Prepare tabular data in the same format as training"""
    
    # Numeric features
    numeric_features = np.array([breast_density, subtlety], dtype=np.float32)
    
    # Prepare categorical features for one-hot encoding
    # Based on the training data, we need to handle the same categories
    
    # Define all possible categories (from training data)
    breast_categories = ['LEFT', 'RIGHT']
    margins_categories = ['CIRCUMSCRIBED', 'ILL_DEFINED', 'MICROLOBULATED', 'OBSCURED', 'SPICULATED']
    
    # Create one-hot encoding
    breast_encoded = np.zeros(len(breast_categories), dtype=np.float32)
    if left_or_right_breast in breast_categories:
        breast_encoded[breast_categories.index(left_or_right_breast)] = 1.0
    
    margins_encoded = np.zeros(len(margins_categories), dtype=np.float32)
    if mass_margins in margins_categories:
        margins_encoded[margins_categories.index(mass_margins)] = 1.0
    
    # The training data had additional categories, so we need to pad with zeros
    # Total tabular features in training: 23 (2 numeric + 21 categorical)
    # We have: 2 numeric + 2 (breast) + 5 (margins) = 9
    # Need to pad with 14 more zeros to reach 23
    padding = np.zeros(14, dtype=np.float32)
    
    # Combine all features
    tabular_features = np.concatenate([numeric_features, breast_encoded, margins_encoded, padding])
    
    return tabular_features

def display_results(image_path, prediction_prob, classification, breast_density, left_or_right_breast, subtlety, mass_margins):
    """Display the image and prediction results"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    ax1.set_title(f"Mammogram Image\n{os.path.basename(image_path)}", fontsize=12)
    ax1.axis('off')
    
    # Display prediction results
    ax2.axis('off')
    
    # Create text summary
    result_text = f"""
PREDICTION RESULTS
{'='*30}

Classification: {classification}
Probability: {prediction_prob:.3f}
Confidence: {abs(prediction_prob - 0.5) * 2:.1%}

INPUT PARAMETERS
{'='*30}

Breast Density: {breast_density}
Side: {left_or_right_breast}
Subtlety: {subtlety}
Mass Margins: {mass_margins}

INTERPRETATION
{'='*30}

"""
    
    if classification == "MALIGNANT":
        result_text += f"⚠️  HIGH RISK: {prediction_prob:.1%} probability of malignancy\n"
        result_text += "Recommend immediate medical consultation"
        color = 'red'
    else:
        result_text += f"✅ LOW RISK: {(1-prediction_prob):.1%} probability of benign lesion\n"
        result_text += "Consider routine follow-up"
        color = 'green'
    
    ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    # Add colored border around prediction
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1)
    ax2.text(0.05, 0.8, f"{classification}", transform=ax2.transAxes, 
             fontsize=16, weight='bold', color=color, bbox=bbox_props)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary to console
    print(f"\n{'='*50}")
    print(f"PREDICTION: {classification} ({prediction_prob:.3f})")
    print(f"{'='*50}")

def generate_result_image(image, prediction_prob, classification, breast_density, left_or_right_breast, subtlety, mass_margins):
    """Generate result image and return as base64 string for API use"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    image_array = np.array(image)
    ax1.imshow(image_array)
    ax1.set_title("Mammogram Image", fontsize=12)
    ax1.axis('off')
    
    # Display prediction results
    ax2.axis('off')
    
    # Create text summary
    result_text = f"""
PREDICTION RESULTS
{'='*30}

Classification: {classification}
Probability: {prediction_prob:.3f}
Confidence: {abs(prediction_prob - 0.5) * 2:.1%}

INPUT PARAMETERS
{'='*30}

Breast Density: {breast_density}
Side: {left_or_right_breast}
Subtlety: {subtlety}
Mass Margins: {mass_margins}

INTERPRETATION
{'='*30}

"""
    
    if classification == "MALIGNANT":
        result_text += f"⚠️  HIGH RISK: {prediction_prob:.1%} probability of malignancy\n"
        result_text += "Recommend immediate medical consultation"
        color = 'red'
    else:
        result_text += f"✅ LOW RISK: {(1-prediction_prob):.1%} probability of benign lesion\n"
        result_text += "Consider routine follow-up"
        color = 'green'
    
    ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    # Add colored border around prediction
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1)
    ax2.text(0.05, 0.8, f"{classification}", transform=ax2.transAxes, 
             fontsize=16, weight='bold', color=color, bbox=bbox_props)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close the figure to free memory
    plt.close(fig)
    
    return image_base64

if __name__ == "__main__":
    # Example usage
    print("Breast Cancer Prediction System")
    print("================================")
    print("Use predict_breast_cancer() function to make predictions")
    print("Example: predict_breast_cancer('image.jpg', 3, 'LEFT', 4, 'SPICULATED')")
    print("Testing: predict_breast_cancer('image.jpg', 3, 'LEFT', 4, 'SPICULATED', force_malignant=True)") 