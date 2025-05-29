import os
# Set environment variables to suppress warnings and disable CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import io
from PIL import Image
import logging
import cv2
import tensorflow as tf
import time
from datetime import datetime
import json
from typing import Dict, Any, Optional
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases from leaf images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant_disease_model.keras')
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
RATE_LIMIT = 100  # requests per minute
REQUEST_TIMEOUT = 30  # seconds

# Rate limiting
request_times: Dict[str, list] = {}

def check_rate_limit(client_ip: str) -> bool:
    """Check if the client has exceeded the rate limit."""
    current_time = time.time()
    if client_ip not in request_times:
        request_times[client_ip] = []
    
    # Remove old requests
    request_times[client_ip] = [t for t in request_times[client_ip] if current_time - t < 60]
    
    if len(request_times[client_ip]) >= RATE_LIMIT:
        return False
    
    request_times[client_ip].append(current_time)
    return True

def create_custom_input_layer(config):
    """Create a custom input layer with proper configuration."""
    try:
        if 'batch_shape' in config:
            input_shape = config['batch_shape'][1:]
        else:
            input_shape = config.get('input_shape', (None, None, 3))
        
        return InputLayer(
            input_shape=input_shape,
            dtype=config.get('dtype', 'float32'),
            name=config.get('name', 'input_layer')
        )
    except Exception as e:
        logger.error(f"Error creating input layer: {str(e)}")
        return InputLayer(
            input_shape=(224, 224, 3),
            dtype='float32',
            name='input_layer'
        )

def load_model_safely():
    """Safely load the model with proper error handling and version compatibility."""
    try:
        custom_objects = {
            'InputLayer': create_custom_input_layer
        }
        model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        logger.info("Model loaded successfully with custom objects")
    except Exception as e1:
        logger.warning(f"First loading attempt failed: {str(e1)}")
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={
                    'InputLayer': lambda config: InputLayer(
                        input_shape=(224, 224, 3),
                        dtype='float32',
                        name='input_layer'
                    )
                }
            )
            logger.info("Model loaded successfully with legacy format")
        except Exception as e2:
            logger.error(f"Second loading attempt failed: {str(e2)}")
            try:
                model = tf.keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects=None
                )
                logger.info("Model loaded successfully with minimal configuration")
            except Exception as e3:
                logger.error(f"All loading attempts failed: {str(e3)}")
                raise

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load model
try:
    model = load_model_safely()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Get class names
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

def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics for monitoring."""
    return {
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
        "cpu_percent": psutil.Process().cpu_percent(),
        "timestamp": datetime.now().isoformat()
    }

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to response."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        metrics = get_system_metrics()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "classes": len(class_names),
            "image_size": IMG_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "system_metrics": metrics
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Predict plant disease from uploaded image."""
    try:
        # Check rate limit
        client_ip = request.client.host
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
            )

        # Process image
        try:
            img = Image.open(io.BytesIO(contents))
            if not check_image_quality(img):
                raise HTTPException(
                    status_code=400,
                    detail="Image quality too low. Please ensure good lighting and focus."
                )
            
            img_array = preprocess_image(img)
            if img_array is None:
                raise HTTPException(status_code=400, detail="Error preprocessing image")
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

        # Make prediction
        try:
            preds = model.predict(img_array, verbose=0)
            pred_class_idx = np.argmax(preds[0])
            confidence = float(preds[0][pred_class_idx])
            
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"Low confidence prediction: {confidence:.2f}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Low confidence prediction ({confidence:.2%}). Please retake the image."
                )
            
            pred_class = class_names[pred_class_idx]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(preds[0])[-3:][::-1]
            top_3_predictions = {
                class_names[i]: float(preds[0][i])
                for i in top_3_idx
            }
            
            logger.info(f"Prediction successful: {pred_class} with confidence {confidence:.2f}")
            
            # Clean up memory
            del img_array
            gc.collect()
            
            return {
                "predicted_class": pred_class,
                "confidence": confidence,
                "top_3_predictions": top_3_predictions,
                "all_predictions": {
                    class_name: float(conf)
                    for class_name, conf in zip(class_names, preds[0])
                },
                "processing_time": time.time() - request.state.start_time
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Plant Disease Detection API is running.",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Upload image for disease prediction",
            "/docs": "API documentation",
            "/redoc": "Alternative API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=2,
        timeout_keep_alive=75,
        log_level="info"
    )
