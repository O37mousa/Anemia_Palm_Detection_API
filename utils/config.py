# ----------------- Imports -----------------
import os
from dotenv import load_dotenv
import joblib
import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input # type: ignore
from skimage.color import rgb2lab
from skimage import img_as_float
from scipy.stats import skew, kurtosis
import logging

# ----------------- Logging Configuration -----------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------- Load Environment Variables -----------------
load_dotenv(override=True)

# Application metadata from environment
APP_NAME = os.getenv("APP_NAME", "Palm Anemia Detection API")
APP_VERSION = os.getenv("APP_VERSION")
SECRET_KEY_TOKEN = os.getenv("SECRET_KEY_TOKEN")

# ----------------- Define Artifact Paths -----------------
# Dynamically determine base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_FOLDER_PATH = os.path.join(BASE_DIR, "artifacts")
MODEL_FILENAME = "random_forest_classifier_palm.pkl"
rf_model_path = os.path.join(ARTIFACTS_FOLDER_PATH, MODEL_FILENAME)

# ----------------- Global Variables -----------------
model = None  # Will hold the loaded Random Forest model
class_names = ['anemic', 'Non-anemic']
IMAGE_SIZE = (224, 224)  # Resize input images to this size

# ----------------- Image Enhancement -----------------
def enhance_contrast(image):
    """
    Apply contrast enhancement using CLAHE in LAB color space.
    """
    image = cv2.GaussianBlur(image, (3, 3), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return image

# ----------------- Preprocessing Wrapper -----------------
def preprocess_image_array(image_array):
    """
    Resize and enhance contrast of image; returns as an array.
    """
    image_resized = cv2.resize(image_array, (224, 224))
    enhanced_image = enhance_contrast(image_resized)
    return np.array([enhanced_image])

# Dictionary to access preprocessing functions
preprocessing = {
    "preprocess_image_array": preprocess_image_array
}

# ------------------- Feature Extraction Functions - EXACTLY MATCHING NOTEBOOK -------------------
def extract_color_features(images):
    """
    Extract color features (LAB color space and histograms) from a list of images.
    """
    features = []
    for img in images:
        img = img_as_float(img)  # Normalize image to [0, 1]
        lab = rgb2lab(img)  # Convert to LAB color space
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Compute statistical moments
        l_mean, l_std, l_skew, l_kurt = np.mean(L), np.std(L), skew(L.flatten()), kurtosis(L.flatten())
        a_mean, a_std, a_skew, a_kurt = np.mean(A), np.std(A), skew(A.flatten()), kurtosis(A.flatten())
        b_mean, b_std, b_skew, b_kurt = np.mean(B), np.std(B), skew(B.flatten()), kurtosis(B.flatten())

        # Compute histograms for each channel
        l_hist, _ = np.histogram(L.flatten(), bins=256, range=(0, 100), density=True)
        a_hist, _ = np.histogram(A.flatten(), bins=256, range=(-128, 128), density=True)
        b_hist, _ = np.histogram(B.flatten(), bins=256, range=(-128, 128), density=True)

        # Combine all features
        hist_features = np.concatenate([l_hist, a_hist, b_hist])
        stats = np.array([
            l_mean, l_std, l_skew, l_kurt,
            a_mean, a_std, a_skew, a_kurt,
            b_mean, b_std, b_skew, b_kurt
        ])
        feature_vector = np.concatenate([stats, hist_features])
        features.append(feature_vector)

    return np.array(features)

# ------------------- Model Loading -------------------
def load_model():
    """
    Load the ML model from file and store in global variable.
    """
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


