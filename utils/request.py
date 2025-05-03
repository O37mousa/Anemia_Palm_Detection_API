import sys
import os
import cv2
import joblib
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2lab
from scipy.stats import skew, kurtosis


# Add the project root to the Python path
sys.path.append(os.path.abspath("."))

# Import the feature extractor
# from utils.config import preprocess_image_array

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

        # Histograms (increase number of bins to 64)
        l_hist, _ = np.histogram(L.flatten(), bins=256, range=(0, 100), density=True)
        a_hist, _ = np.histogram(A.flatten(), bins=256, range=(-128, 128), density=True)
        b_hist, _ = np.histogram(B.flatten(), bins=256, range=(-128, 128), density=True)

        feature_vector = np.concatenate([stats, l_hist, a_hist, b_hist])

        # Debugging: print feature vector length
        print(f"[DEBUG] Feature vector length: {len(feature_vector)}")

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


# Load the trained model
model_path = r"C:\System\Omar\Omar Nile University\Year 4\Grad\Mobile Application\Anemia_Palm_Detection_API\artifacts\random_forest_classifier_palm.pkl"
model = joblib.load(model_path)

# Load and preprocess the image
img_path = r"C:\System\Omar\Omar Nile University\Year 4\Grad\Mobile Application\Anemia_Palm_Detection_API\dataset\Palm dataset\Anemic\Anemic-260 (2).png"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at path: {img_path}")

features = preprocess_image_array(img)

# Check and reshape if needed
if features.ndim == 1:
    features = features.reshape(1, -1)

# Model prediction
prediction = model.predict(features)
label = "anemic" if prediction[0] == 1 else "non-anemic"

# Output result
print(f"Prediction: {prediction[0]} ({label})")
