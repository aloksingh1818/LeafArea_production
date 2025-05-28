# Foliage Pixel Probe

A mobile application for analyzing and identifying plant foliage using image processing and machine learning techniques.

## Features

- Plant identification through image capture
- Detailed plant information and care instructions
- Offline database support
- Cross-platform compatibility (iOS and Android)

## Development

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Android Studio (for Android development)
- Xcode (for iOS development, macOS only)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

### Building for Production

```bash
npm run build
```

### Running on Mobile Devices

For Android:
```bash
npm run capacitor:sync
npm run capacitor:open:android
```

For iOS:
```bash
npm run capacitor:sync
npm run capacitor:open:ios
```

## Plant Disease Model (Keras)

This project includes a deep learning model for plant disease classification using the PlantVillage dataset. The model is trained using TensorFlow/Keras and saved as `plant_disease_model.keras`.

### Model Training

- Training script: `train_plantvillage_cnn.py`
- Data: `plantvillage_data/` (organized by class)
- To retrain the model:
  ```bash
  python train_plantvillage_cnn.py
  ```
- The script will print validation accuracy and loss after training.

### Model Evaluation

After training, the script evaluates the model on the validation set and prints accuracy and loss.

### Model API (FastAPI)

A FastAPI backend is provided for real-time predictions:
- Script: `plant_disease_api.py`
- To run the API:
  ```bash
  uvicorn plant_disease_api:app --reload --host 0.0.0.0 --port 8000
  ```
- Endpoint: `POST /predict` (upload an image to get prediction)

### Frontend

A basic frontend for image upload and prediction display can be added (see project roadmap).

### Data Augmentation

The training script uses Keras `ImageDataGenerator` for data augmentation (rotation, shift, shear, zoom, flip).

### Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs basic checks on push and pull requests.

### Project Roadmap

- [x] Model training and evaluation
- [x] Model API deployment
- [ ] Frontend UI for predictions
- [x] Data augmentation
- [x] CI/CD pipeline

---

## Plant Disease Prediction API

### Running the API

1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn tensorflow numpy python-multipart
   ```
2. Start the API server:
   ```bash
   uvicorn plant_disease_api:app --host 0.0.0.0 --port 8000 --reload
   ```

### API Endpoints
- `POST /predict` — Upload an image file (form field: `file`). Returns predicted class and confidence.
- `GET /health` — Health check endpoint.

#### Example request (Python):
```python
import requests
url = "http://localhost:8000/predict"
with open("path_to_image.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})
print(response.json())
```

### Model Details
- Input size: 128x128 RGB image
- Classes: (see plantvillage_data subfolders)
- Accuracy: ~77% (see `classification_report.txt`)

### Evaluation & Analysis
- Run `python3 evaluate_plant_disease_model.py` to evaluate the model and save reports.
- Run `python3 analyze_misclassifications.py` to analyze misclassified images.

### Batch Prediction
- Use `python3 batch_predict_plant_disease_api.py` to predict a folder of images and save results.

---

## Data Augmentation (for training)
- See `train_plantvillage_cnn.py` for augmentation options (rotation, flip, zoom, etc.).
- Add more images to `plantvillage_data` for improved robustness.

---

## Continuous Integration
- Add a `.github/workflows/python-app.yml` for CI/CD (example below):
```yaml
name: Python application
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: pip install fastapi uvicorn tensorflow numpy python-multipart
    - name: Run evaluation
      run: python3 evaluate_plant_disease_model.py
```

## License

MIT License
