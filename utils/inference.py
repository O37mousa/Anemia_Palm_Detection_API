
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

        # Extract features using the proper preprocessing pipeline
        features = preprocessing["preprocess_image_array"](image_cv2)

        if features.shape[1] != 780:
            return {"error": f"Feature vector mismatch: got {features.shape[1]}, expected 780"}

        prediction = rf_model.predict(features)
        prediction_value = int(prediction[0])
        label = "anemic" if prediction_value == 1 else "non-anemic"

        return {
            "final_prediction": prediction_value,
            "label": label
        }

    except Exception as e:
        return {"error": f"Anemia detection failed: {str(e)}"}
