# PlantVillage Disease Detection - Production-Ready Guide

## Project Overview
This project provides an end-to-end pipeline for plant disease detection using deep learning. It covers everything from data preparation, model training (with a focus on achieving 95%+ accuracy), evaluation, prediction, UI integration, and production deployment.

---

## Table of Contents
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction (Batch & Single)](#prediction-batch--single)
- [UI Integration](#ui-integration)
- [Production Deployment](#production-deployment)
- [Advanced Tips for High Accuracy](#advanced-tips-for-high-accuracy)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [AI Agent Prompt](#ai-agent-prompt)
- [Local Hosting & Development Hints](#local-hosting--development-hints)

---

## Requirements
- Python 3.8+
- pip
- [CUDA-enabled GPU (recommended for training)]
- Git
- Node.js & npm (for UI)

### Python Packages
Install all required Python packages:
```bash
pip install -r requirements.txt
```

---

## Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/aloksingh1818/LeafArea_production.git
   cd LeafArea_production
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install Node.js dependencies for the UI:**
   ```bash
   cd <ui-folder>  # e.g., 'frontend' or 'app' if present
   npm install
   cd ..
   ```

---

## Data Preparation
1. **Download and organize the dataset:**
   - Place your raw PlantVillage dataset in the `plantvillage_data/` directory.
   - The structure should be:
     ```
     plantvillage_data/
       ├── Class1/
       ├── Class2/
       └── ...
     ```
2. **Verify and split the dataset:**
   ```bash
   python verify_and_split_dataset.py
   ```
   - This will:
     - Check image quality
     - Split into `train` and `validation` folders
     - Resize images
     - Generate analysis reports

---

## Model Training
1. **Train the model:**
   ```bash
   python train_plantvillage_cnn.py
   ```
   - The script uses EfficientNet (B0/B3) and advanced augmentation for high accuracy.
   - Training logs and best model are saved in the `models/` and `logs/` directories.
   - **Tip:** Use a GPU for much faster training.
2. **Monitor training:**
   - Use TensorBoard for real-time monitoring:
     ```bash
     tensorboard --logdir logs/
     ```

---

## Model Evaluation
- After training, review:
  - `logs/plots/training_history.png` for accuracy/loss curves
  - `logs/evaluation_results.csv` for detailed predictions
  - `models/best_model.h5` for the best model
- Aim for **95%+ validation accuracy**. If not achieved, see [Advanced Tips](#advanced-tips-for-high-accuracy).

---

## Prediction (Batch & Single)
- Use `predict.py` to make predictions:
  - **Single image:**
    ```bash
    python predict.py --image path/to/image.jpg
    ```
  - **Batch prediction:**
    ```bash
    python predict.py --dir path/to/images/
    ```
  - Or edit the `main()` function in `predict.py` for custom usage.

---

## UI Integration
- If a UI is present (e.g., React, Vite, or other):
  1. **Start the UI:**
     ```bash
     cd <ui-folder>
     npm run dev
     ```
  2. **Connect the UI to the backend API** (see `plant_disease_api.py` or similar).
  3. **Upload images and view predictions in the browser.**

---

## Production Deployment
1. **Backend API:**
   - Use `plant_disease_api.py` (Flask/FastAPI) to serve predictions.
   - Deploy on a cloud VM, Docker, or serverless platform.
   - Example (Flask):
     ```bash
     python plant_disease_api.py
     ```
2. **Frontend:**
   - Deploy the UI to Vercel, Netlify, or your own server.
3. **Model Optimization:**
   - Convert the model to TensorFlow Lite or ONNX for edge/mobile deployment if needed.
4. **CI/CD:**
   - Set up GitHub Actions for automated testing and deployment.

---

## Advanced Tips for High Accuracy
- **Use EfficientNetB3 or higher for best results.**
- **Increase image size** (e.g., 224x224 or 300x300) if you have enough GPU memory.
- **Tune augmentation**: Try stronger augmentation for more robust models.
- **Use learning rate scheduling and early stopping.**
- **Try transfer learning**: Unfreeze more layers of EfficientNet for fine-tuning.
- **Use mixed precision training** for faster and more memory-efficient training (if supported by your GPU).
- **Monitor for overfitting**: Use dropout, regularization, and data augmentation.
- **Ensemble models**: Combine predictions from multiple models for even higher accuracy.

---

## Troubleshooting
- **Low accuracy?**
  - Check for class imbalance, data quality, or try a deeper model.
  - Review logs and plots for signs of overfitting/underfitting.
- **Out of memory?**
  - Reduce batch size or image size.
- **Deployment issues?**
  - Check API logs, CORS settings, and model path.

---

## License
This project is licensed under the MIT License.

## AI Agent Prompt

To complete the remaining parts of this project, including advanced features and a better UI dashboard, use the following prompt with your AI agent:

```
You are a powerful AI coding assistant. Your task is to complete the PlantVillage Disease Detection project with the following requirements:

1. **Model Training & Accuracy:**
   - Ensure the model achieves 95%+ validation accuracy using EfficientNetB3 or higher.
   - Implement advanced data augmentation, learning rate scheduling, and early stopping.
   - Use mixed precision training if supported by the GPU.
   - Consider ensemble methods for even higher accuracy.

2. **UI Dashboard:**
   - Create a modern, responsive UI using React/Vite or a similar framework.
   - Include features for image upload, real-time prediction, and detailed results display.
   - Show training metrics, model performance, and prediction confidence.
   - Implement a user-friendly dashboard with charts, graphs, and interactive elements.

3. **Backend API:**
   - Develop a robust API using Flask or FastAPI to serve model predictions.
   - Ensure proper error handling, logging, and security measures.
   - Optimize the API for production deployment.

4. **Deployment:**
   - Deploy the backend API on a cloud VM, Docker, or serverless platform.
   - Deploy the frontend UI to Vercel, Netlify, or a similar service.
   - Set up CI/CD pipelines for automated testing and deployment.

5. **Documentation:**
   - Update the README with detailed instructions for setup, training, and deployment.
   - Include troubleshooting tips and advanced usage guidelines.

6. **Testing:**
   - Implement unit tests for the model, API, and UI components.
   - Ensure the project is production-ready and scalable.

Please proceed step-by-step, ensuring each component is correctly implemented and integrated. Focus on high accuracy, user experience, and production readiness.
```

Use this prompt to guide the AI agent in completing the project with all the required features and improvements.

## Local Hosting & Development Hints

### Local Hosting
To host the project locally:

1. **Backend API:**
   - Run the Flask/FastAPI server:
     ```bash
     python plant_disease_api.py
     ```
   - The API will be available at `http://localhost:5000` (Flask) or `http://localhost:8000` (FastAPI).

2. **Frontend UI:**
   - Navigate to the UI directory:
     ```bash
     cd <ui-folder>
     npm run dev
     ```
   - The UI will be available at `http://localhost:3000` (or the port specified in your Vite/React setup).

### Ensuring Features Remain Intact
- **Area Calculation:** Ensure the area calculation logic is correctly integrated into the UI and API. Test with sample images to verify accuracy.
- **Model Accuracy:** Regularly evaluate the model using the validation set to ensure it maintains high accuracy (95%+).

### Hints for AI Developers
- **Step-by-Step Implementation:**
  1. **Data Preparation:** Use `verify_and_split_dataset.py` to ensure data quality and proper splitting.
  2. **Model Training:** Follow the training script (`train_plantvillage_cnn.py`) and monitor logs for accuracy improvements.
  3. **UI Development:** Use React/Vite for a modern UI. Integrate the API for real-time predictions.
  4. **API Development:** Ensure robust error handling and logging in the API.
  5. **Testing:** Implement unit tests for each component to ensure reliability.
  6. **Deployment:** Use Docker for containerization and CI/CD for automated deployment.

- **Common Pitfalls:**
  - Ensure all dependencies are correctly installed and up-to-date.
  - Monitor GPU usage during training to avoid out-of-memory errors.
  - Regularly backup your model and data to prevent loss.

Use these hints to guide your development process and ensure a smooth implementation of the project.
