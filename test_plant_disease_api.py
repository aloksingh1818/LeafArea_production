import requests

# Update this path to point to a real test image from your dataset
image_path = "plantvillage_data/Pepper__bell___Bacterial_spot/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG"

url = "http://localhost:8000/predict"

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.json())
