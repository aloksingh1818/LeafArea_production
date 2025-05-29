import os
# Set environment variables to suppress warnings and disable CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Configure TensorFlow for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

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
IMG_SIZE = (224, 224)  # Increased image size for better accuracy
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence threshold

def load_model_safely():
    """Safely load the model with proper error handling and version compatibility."""
    try:
        # First try loading with custom objects and safe_mode
        custom_objects = {
            'InputLayer': lambda config: InputLayer(
                input_shape=config.get('input_shape', (None, None, 3)),
                dtype=config.get('dtype', 'float32'),
                name=config.get('name', 'input_layer')
            )
        }
        model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False, safe_mode=True)
        logger.info("Model loaded successfully with custom objects and safe mode")
    except Exception as e1:
        logger.warning(f"First loading attempt failed: {str(e1)}")
        try:
            # Try loading with legacy format and custom input shape
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={
                    'InputLayer': lambda config: InputLayer(
                        input_shape=(128, 128, 3),
                        dtype='float32',
                        name='input_layer'
                    )
                }
            )
            logger.info("Model loaded successfully with legacy format")
        except Exception as e2:
            logger.error(f"Second loading attempt failed: {str(e2)}")
            try:
                # Try loading with minimal configuration
                model = tf.keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects=None
                )
                logger.info("Model loaded successfully with minimal configuration")
            except Exception as e3:
                logger.error(f"All loading attempts failed: {str(e3)}")
                raise

    # Recompile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

try:
    model = load_model_safely()
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

def preprocess_image(img):
    """Enhanced image preprocessing for better prediction accuracy."""
    try:
        # Resize image
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return None

def check_image_quality(img):
    """Check image quality before prediction."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # Check focus
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        focus_score = min(1.0, laplacian_var / 1000.0)
        
        # Check lighting
        brightness = np.mean(gray)
        lighting_score = 1.0 - abs(brightness - 128) / 128
        
        # Check contrast
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 100.0)
        
        # Overall quality score
        quality_score = (focus_score + lighting_score + contrast_score) / 3
        
        return quality_score >= 0.7  # Return True if quality is good enough
    except Exception as e:
        logger.error(f"Error in quality check: {str(e)}")
        return False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(class_names),
        "image_size": IMG_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")

        # Load and check image
        try:
            img = Image.open(io.BytesIO(contents))
            
            # Check image quality
            if not check_image_quality(img):
                raise HTTPException(
                    status_code=400,
                    detail="Image quality too low. Please ensure good lighting and focus."
                )
            
            # Preprocess image
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
            
            # Check confidence threshold
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
            
            return {
                "predicted_class": pred_class,
                "confidence": confidence,
                "top_3_predictions": top_3_predictions,
                "all_predictions": {
                    class_name: float(conf)
                    for class_name, conf in zip(class_names, preds[0])
                }
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
def root():
    return {
        "message": "Plant Disease Detection API is running.",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Upload image for disease prediction",
            "/": "API information"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
