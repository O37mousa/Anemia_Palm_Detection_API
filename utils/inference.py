import numpy as np
from fastapi import UploadFile
from utils.config import preprocessing
import cv2
from PIL import Image
import io

def detect_new(image: UploadFile, rf_model):
    """
    Inference function for palm anemia detection using uploaded image.
    """
    try:
        # Read uploaded image into memory
        image_bytes = image.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        # Convert to OpenCV format (BGR for consistency with cv2)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Check if the preprocessing function exists
        if "preprocess_image_array" not in preprocessing:
            return {"error": "Preprocessing function 'preprocess_image_array' not found in config."}

        # Extract features
        features = preprocessing["preprocess_image_array"](image_cv2)

        # If features are 1D, reshape to 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Check if the feature vector length is correct
        if features.shape[1] != 780:
            return {"error": f"Feature vector mismatch: got {features.shape[1]}, expected 780"}

        # Model prediction
        prediction = rf_model.predict(features)

        # Determine label
        prediction_value = int(prediction[0])
        label = "anemic" if prediction_value == 1 else "non-anemic"

        return {
            "final_prediction": prediction_value,
            "label": label
        }

    except Exception as e:
        return {"error": f"Anemia detection failed: {str(e)}"}
