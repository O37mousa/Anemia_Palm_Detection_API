from fastapi import FastAPI, HTTPException, UploadFile, File
from utils.config import (
    APP_NAME,
    rf_model
)
from utils.inference import detect_new

app = FastAPI(title=APP_NAME)

@app.get("/", tags=['Nails Anemia Detection'])
async def home() -> dict:
    return {
        "app_name": APP_NAME,
        "message": "Anemia Detection API for Palm is running!"
    } 

@app.post("/detect/forest", tags=["Models"], description="Detection of Anemia using RF")
async def detect_rf(image: UploadFile = File(...)) -> dict:
    try:
        
        # call the function
        response = detect_new(
            image=image,
            rf_model=rf_model
        )
        return response     
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There's a problem in anemic detection, {str(e)}")