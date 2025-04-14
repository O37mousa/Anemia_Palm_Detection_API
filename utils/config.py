import os
from dotenv import load_dotenv
import joblib
import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
from skimage.color import rgb2lab
from skimage import img_as_float
from scipy.stats import skew, kurtosis

# ----------------- Load Environment Variables -----------------
load_dotenv(override=True)

APP_NAME = os.getenv("APP_NAME", "Palm Anemia Detection API")
APP_VERSION = os.getenv("APP_VERSION")
SECRET_KEY_TOKEN = os.getenv("SECRET_KEY_TOKEN")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_FOLDER_PATH = os.path.join(BASE_DIR, "artifacts")

# ----------------- Image Enhancement -----------------
def enhance_contrast(image):
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Convert to LAB and apply CLAHE on L channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced

# ----------------- Preprocessing Functions -----------------
def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = enhance_contrast(img)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img

def read_upload_file(uploaded_file):
    contents = uploaded_file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img

def preprocess_uploaded_image(image, target_size=(224, 224)):
    if image is None:
        raise ValueError("No image data provided for preprocessing.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = enhance_contrast(image)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    return image

def extract_color_features(images):
    features = []
    for img in images:
        img = img_as_float(img)
        img[np.isnan(img)] = 0  # Fix NaNs
        img[np.isinf(img)] = 0  # Fix infs

        # Convert to LAB color space
        lab = rgb2lab(img)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Basic stats
        stats = [
            np.mean(L), np.std(L), skew(L.flatten()), kurtosis(L.flatten()),
            np.mean(A), np.std(A), skew(A.flatten()), kurtosis(A.flatten()),
            np.mean(B), np.std(B), skew(B.flatten()), kurtosis(B.flatten())
        ]

        # Histograms
        l_hist, _ = np.histogram(L.flatten(), bins=32, range=(0, 100), density=True)
        a_hist, _ = np.histogram(A.flatten(), bins=32, range=(-128, 128), density=True)
        b_hist, _ = np.histogram(B.flatten(), bins=32, range=(-128, 128), density=True)

        feature_vector = np.concatenate([stats, l_hist, a_hist, b_hist])
        features.append(feature_vector)

    return np.array(features)

def preprocess_image_array(image_array, target_size=(224, 224)):
    image = cv2.resize(image_array, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = enhance_contrast(image)
    image = image.astype(np.float32)
    image = img_as_float(image)
    features = extract_color_features([image])
    return features

# ----------------- Group Preprocessing -----------------
preprocessing = {
    "preprocess_image": preprocess_image,
    "read_upload_file": read_upload_file,
    "preprocess_uploaded_image": preprocess_uploaded_image,
    "extract_color_features": extract_color_features,
    "preprocess_image_array": preprocess_image_array,
    "enhance_contrast": enhance_contrast
}

# ----------------- Load Model -----------------
rf_model = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, "random_forest_classifier_palm.pkl"))
