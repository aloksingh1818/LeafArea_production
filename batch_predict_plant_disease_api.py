import requests
import os
import csv

API_URL = "http://localhost:8000/predict"
# Change this to the class you want to test, or use all folders
image_dir = "plantvillage_data/Pepper__bell___Bacterial_spot"

results = []

for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    image_path = os.path.join(image_dir, fname)
    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        try:
            response = requests.post(API_URL, files=files, timeout=10)
            if response.status_code == 200:
                pred = response.json()
                results.append({
                    "filename": fname,
                    "predicted_class": pred.get("predicted_class"),
                    "confidence": pred.get("confidence")
                })
            else:
                results.append({
                    "filename": fname,
                    "predicted_class": "ERROR",
                    "confidence": None
                })
        except Exception as e:
            results.append({
                "filename": fname,
                "predicted_class": f"EXCEPTION: {e}",
                "confidence": None
            })

with open("batch_predictions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "predicted_class", "confidence"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved predictions for {len(results)} images to batch_predictions.csv")
