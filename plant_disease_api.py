from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import os
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant_disease_model.keras')
IMG_SIZE = (128, 128)

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Get class names from the training directory
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'plantvillage_data')
try:
    class_names = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d)) and d not in ['train', 'validation']
    ])
    logger.info(f"Found {len(class_names)} classes: {class_names}")
except Exception as e:
    logger.error(f"Error loading class names: {str(e)}")
    raise

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")

        # Load and preprocess image
        try:
            img = Image.open(io.BytesIO(contents))
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img = img.convert('RGB')
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

        # Convert to array and normalize
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        # Make prediction
        try:
            preds = model.predict(x, verbose=0)
            pred_class = class_names[np.argmax(preds[0])]
            confidence = float(np.max(preds[0]))
            logger.info(f"Prediction successful: {pred_class} with confidence {confidence:.2f}")
            return {"predicted_class": pred_class, "confidence": confidence}
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Plant Disease Detection API is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
