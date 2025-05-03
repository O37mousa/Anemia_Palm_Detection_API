import os
import sys
import numpy as np
import cv2
import logging
import io
from PIL import Image
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from skimage.color import rgb2lab
from skimage import img_as_float
from scipy.stats import skew, kurtosis
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from io import BytesIO

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Palm Anemia Detection API")

# Dynamically determine the base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to artifacts folder and model file
ARTIFACTS_FOLDER_PATH = os.path.join(BASE_DIR, "artifacts")
MODEL_FILENAME = "random_forest_classifier_palm.pkl"
rf_model_path = os.path.join(ARTIFACTS_FOLDER_PATH, MODEL_FILENAME)


# Global variable for the model
model = None

# Define class names
class_names = ['anemic', 'Non-anemic']

# Define image size
IMAGE_SIZE = (224, 224)

# ------------------- Feature Extraction Functions - EXACTLY MATCHING NOTEBOOK -------------------
def extract_color_features(images):
    """
    Extract color features (LAB color space and histograms) from a list of images.
    """
    features = []
    for img in images:
        # Normalize the image to the range [0, 1]
        img = img_as_float(img)

        # Convert image to LAB color space
        lab = rgb2lab(img)

        # Extract L, A, B channels
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Calculate statistical moments (mean, std, skew, kurtosis) for each channel
        l_mean, l_std, l_skew, l_kurt = np.mean(L), np.std(L), skew(L.flatten()), kurtosis(L.flatten())
        a_mean, a_std, a_skew, a_kurt = np.mean(A), np.std(A), skew(A.flatten()), kurtosis(A.flatten())
        b_mean, b_std, b_skew, b_kurt = np.mean(B), np.std(B), skew(B.flatten()), kurtosis(B.flatten())

        # Calculate histograms for each channel
        l_hist, _ = np.histogram(L.flatten(), bins=256, range=(0, 100), density=True)
        a_hist, _ = np.histogram(A.flatten(), bins=256, range=(-128, 128), density=True)
        b_hist, _ = np.histogram(B.flatten(), bins=256, range=(-128, 128), density=True)

        # Flatten histograms and concatenate with other features
        hist_features = np.concatenate([l_hist, a_hist, b_hist])

        # Combine all extracted features into a single feature vector
        feature_vector = np.array([
            l_mean, l_std, l_skew, l_kurt,
            a_mean, a_std, a_skew, a_kurt,
            b_mean, b_std, b_skew, b_kurt
        ])
        feature_vector = np.concatenate([feature_vector, hist_features])

        features.append(feature_vector)

    return np.array(features)

# ------------------- Model Loading -------------------
def load_model():
    """Load the ML model and return it"""
    global model
    try:
        logger.info(f"Attempting to load model from: {rf_model_path}")
        
        if not os.path.exists(rf_model_path):
            logger.error(f"Model file not found at: {rf_model_path}")
            return False
            
        model = joblib.load(rf_model_path)
        logger.info(f"Model loaded successfully. Type: {type(model)}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return False

# ------------------- FastAPI Endpoints -------------------
@app.on_event("startup")
async def startup_event():
    """Load the ML model on startup"""
    success = load_model()
    if not success:
        logger.warning("Application started, but model failed to load")

@app.get("/")
def read_root():
    return {
        "app_name": "Palm Anemia Detection API",
        "message": "Anemia Detection API for Palm is running!"
    }

@app.post("/detect/forest")
async def detect_rf(image: UploadFile = File(...)):
    """Detect anemia from palm image using Random Forest model (in-memory processing)"""
    if model is None:
        success = load_model()
        if not success:
            raise HTTPException(status_code=503, detail="Model failed to load")

    try:
        # Read image bytes directly from memory
        image_bytes = await image.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image_cv is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Convert to RGB and resize
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, IMAGE_SIZE)

        # Preprocess and extract features
        image_normalized = preprocess_input(image_resized.astype("float32"))
        features = extract_color_features([image_normalized])
        logger.info(f"Extracted features shape: {features.shape}")

        # Make prediction
        prediction = model.predict(features)
        prediction_value = int(prediction[0])
        label = class_names[prediction_value]
        probabilities = model.predict_proba(features)[0].tolist()

        # Build response
        response = {
            "filename": image.filename,
            "final_prediction": prediction_value,
            "label": label,
            "probabilities": probabilities,
            "class_probabilities": {
                "anemic": float(probabilities[0]),
                "Non-anemic": float(probabilities[1])
            }
        }

        logger.info(f"Prediction complete: {response}")
        return response

    except Exception as e:
        logger.error(f"Error processing image or making prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Anemia detection failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
