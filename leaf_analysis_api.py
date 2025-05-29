import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model (update path if needed)
MODEL_PATH = 'plant_disease_model.h5'
model = load_model(MODEL_PATH)

# Load class names dynamically (assumes subfolders in train dir)
TRAIN_DIR = 'plantvillage_data/train'
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

app = Flask(__name__)

def measure_leaf_area(image, calibration_object_diameter_mm=24.26):
    """
    Measures leaf area in mm^2 using OpenCV.
    Assumes a round calibration object (e.g., Indian 5 rupee coin, 24.26mm diameter) is present.
    Returns: area_mm2, mask (for visualization)
    """
    # Convert PIL image to OpenCV
    img = np.array(image)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Find circles (Hough)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        # Assume largest circle is calibration object
        calib_circle = max(circles, key=lambda c: c[2])
        calib_radius_px = calib_circle[2]
        px_per_mm = calib_radius_px * 2 / calibration_object_diameter_mm
    else:
        return None, None  # Calibration object not found
    # Threshold for leaf (simple green detection)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Morphology to clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_area_px = cv2.contourArea(leaf_contour)
    area_mm2 = leaf_area_px / (px_per_mm ** 2)
    return area_mm2, mask

def predict_disease(image):
    # Resize and preprocess
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return class_names[idx], float(preds[0][idx]), preds[0].tolist()

@app.route('/analyze_leaf', methods=['POST'])
def analyze_leaf():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    # Area measurement
    area_mm2, mask = measure_leaf_area(image)
    if area_mm2 is None:
        return jsonify({'error': 'Could not find calibration object or leaf'}), 400
    # Disease prediction
    disease, confidence, all_probs = predict_disease(image)
    return jsonify({
        'area_mm2': area_mm2,
        'area_cm2': area_mm2 / 100.0,
        'disease': disease,
        'confidence': confidence,
        'all_probs': all_probs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
